from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


CLASSIFICATION_PREPROCESSING_CURRENT = "current_scale_only"
CLASSIFICATION_PREPROCESSING_IMAGENET = "imagenet_mean_std"
CLASSIFICATION_PREPROCESSING_MODES = frozenset({
    CLASSIFICATION_PREPROCESSING_CURRENT,
    CLASSIFICATION_PREPROCESSING_IMAGENET,
})
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


def canonical_classification_preprocessing(value: Any = None) -> str:
    """Return one explicit, cache-safe DeepX classification preprocessing arm.

    The historical behaviour remains the default.  Deliberately avoid fuzzy
    aliases here: a typo must not silently select a different quantized model.
    """
    mode = str(value or CLASSIFICATION_PREPROCESSING_CURRENT).strip().lower()
    if mode not in CLASSIFICATION_PREPROCESSING_MODES:
        raise ValueError(
            "DeepX classification_preprocessing must be one of "
            + ", ".join(sorted(CLASSIFICATION_PREPROCESSING_MODES))
        )
    return mode


def resolve_profile_classification_preprocessing(config: Mapping[str, Any] | None = None) -> str:
    """Productive profile default; the historical low-level A/B API stays explicit."""
    return canonical_classification_preprocessing(dict(config or {}).get("classification_preprocessing") or CLASSIFICATION_PREPROCESSING_IMAGENET)


def declared_deepx_task(model_contract: Mapping[str, Any], output_contract: Mapping[str, Any], *, plan_task: str = "") -> str:
    """Resolve declared task evidence; missing/contradictory tasks are never claims."""
    values = [plan_task]
    for contract in (model_contract, output_contract):
        if not isinstance(contract, Mapping):
            continue
        values.extend(contract.get(key) for key in ("task", "task_hint", "benchmark_task", "model_task"))
        for key in ("model", "input", "preprocessing", "preprocessing_contract"):
            nested = contract.get(key)
            if isinstance(nested, Mapping):
                values.extend(nested.get(field) for field in ("task", "task_hint"))
    declared = {str(value).strip().lower() for value in values if value is not None and str(value).strip().lower() not in {"", "auto", "unknown"}}
    if not declared:
        raise ValueError("deepx_task_contract_missing")
    if not declared.issubset({"classification", "detection"}):
        raise ValueError("deepx_task_contract_unsupported:" + ",".join(sorted(declared)))
    if len(declared) != 1:
        raise ValueError("deepx_task_contract_conflict:" + ",".join(sorted(declared)))
    return next(iter(declared))


def classification_profile_admission(profile: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed for legacy classification claims; never rewrite the profile."""
    cfg = dict(profile.get("deepx_build") or {})
    mode = resolve_profile_classification_preprocessing(cfg)
    task = str(model.get("task") or model.get("task_hint") or "").lower()
    legacy = task == "classification" and mode == CLASSIFICATION_PREPROCESSING_CURRENT
    # Named modes are an explicit user decision; an unqualified legacy profile
    # cannot silently become a scientific claim, even when metadata is absent.
    run_mode = profile.get("run_mode") or profile.get("execution_preset") or {}
    mode_id = str(run_mode.get("id") or run_mode.get("mode") or run_mode.get("mode_id") or "") if isinstance(run_mode, Mapping) else str(run_mode)
    final = mode_id.lower() in {"final", "thesis_final", "final_scientific"}
    purpose = str(profile.get("purpose") or "").strip().lower()
    diagnostic = bool(cfg.get("diagnostic_only") is True or purpose == "deepx_preprocessing_ab")
    reason = "deepx_legacy_classification_preprocessing" if legacy else ""
    return {"classification_preprocessing": mode,
            "legacy_diagnostic": legacy,
            "allowed": not legacy or (diagnostic and not final),
            "claim_eligible": not legacy,
            "scientific_claim_exclusion_reason": reason,
            "reason": reason,
            "required_setting": "deepx_build.classification_preprocessing: imagenet_mean_std" if legacy else "",
            "diagnostic_only": bool(legacy and diagnostic),
            "counts_as_benchmark": not legacy}


def deepx_classification_preprocessing_contract(value: Any = None) -> dict[str, Any]:
    """Describe the complete numeric A/B contract without vendor guesswork.

    DX-COM's locally proven image-loader operations stay identical in both
    arms.  The corrected arm performs ImageNet normalization in a generated
    ONNX adapter, where standard ONNX Sub/Div semantics are checkable with ORT.
    """
    mode = canonical_classification_preprocessing(value)
    corrected = mode == CLASSIFICATION_PREPROCESSING_IMAGENET
    return {
        "schema": "onnx-splitpoint/deepx-classification-preprocessing",
        "schema_version": 1,
        "mode": mode,
        "runtime_image": {
            "dtype": "uint8",
            "layout": "HWC",
            "color_space": "RGB",
            "value_range": [0, 255],
        },
        "dxcom_loader": {
            "scale_divisor": 255.0,
            "output_layout": "NCHW",
            "output_dtype": "float32",
            "color_space": "RGB",
            "operations_contract": "resize_div255_bgr2rgb_transpose_expanddim",
        },
        "build_onnx_adapter": {
            "kind": "imagenet_rgb_mean_std" if corrected else "identity",
            "implementation": (
                "onnx_sub_then_div_before_original_graph"
                if corrected else "source_onnx_unchanged"
            ),
            "input_domain": "rgb_float32_0_1",
            "mean": list(IMAGENET_RGB_MEAN) if corrected else [],
            "std": list(IMAGENET_RGB_STD) if corrected else [],
            "broadcast_shape": [1, 3, 1, 1] if corrected else [],
        },
        "source_model_input_domain": (
            "rgb_float32_imagenet_mean_std"
            if corrected else "rgb_float32_0_1"
        ),
    }


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def materialize_imagenet_normalized_build_onnx(
    *,
    source_onnx: str | Path,
    output_dir: str | Path,
    input_name: str = "",
) -> tuple[Path, dict[str, Any]]:
    """Create a content-addressed ONNX adapter for the corrected A/B arm.

    The adapter accepts the loader's RGB float32 ``[0, 1]`` NCHW tensor and
    prepends ``Sub(mean)`` followed by ``Div(std)`` to the original graph.  The
    source model is never modified.  No unverified DX-COM preprocessing opcode
    is used.
    """
    source = Path(source_onnx).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"DeepX source ONNX not found: {source}")
    destination_root = Path(output_dir).expanduser()
    destination_root.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np  # type: ignore
        import onnx  # type: ignore
        from onnx import TensorProto, helper, numpy_helper  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure is explicit
        raise RuntimeError(
            "ONNX and NumPy are required for the DeepX ImageNet build adapter"
        ) from exc

    model = onnx.load(str(source), load_external_data=True)
    graph = model.graph
    initializers = {str(item.name) for item in graph.initializer}
    graph_inputs = [item for item in graph.input if str(item.name) not in initializers]
    selected = None
    if input_name:
        selected = next(
            (item for item in graph_inputs if str(item.name) == str(input_name)),
            None,
        )
    elif len(graph_inputs) == 1:
        selected = graph_inputs[0]
    if selected is None:
        raise ValueError("DeepX ImageNet adapter requires one unambiguous ONNX input")
    resolved_input_name = str(selected.name)
    try:
        element_type = int(selected.type.tensor_type.elem_type)
    except Exception as exc:
        raise ValueError("DeepX ImageNet adapter input tensor type is unavailable") from exc
    if element_type != int(TensorProto.FLOAT):
        raise ValueError("DeepX ImageNet adapter requires a float32 source-model input")

    used_names: set[str] = set(initializers)
    used_names.update(str(value.name) for value in graph.input)
    used_names.update(str(value.name) for value in graph.output)
    used_names.update(str(value.name) for value in graph.value_info)
    for node in graph.node:
        used_names.update(str(name) for name in node.input if str(name))
        used_names.update(str(name) for name in node.output if str(name))

    def unique(base: str) -> str:
        candidate = base
        index = 1
        while candidate in used_names:
            candidate = f"{base}_{index}"
            index += 1
        used_names.add(candidate)
        return candidate

    original_input = unique(f"{resolved_input_name}__imagenet_normalized")
    centered = unique(f"{resolved_input_name}__imagenet_centered")
    mean_name = unique("__onnx_splitpoint_imagenet_rgb_mean")
    std_name = unique("__onnx_splitpoint_imagenet_rgb_std")

    consumer_count = 0
    for node in graph.node:
        for index, name in enumerate(node.input):
            if str(name) == resolved_input_name:
                node.input[index] = original_input
                consumer_count += 1
    if consumer_count <= 0:
        raise ValueError("DeepX ImageNet adapter source input has no graph consumers")
    if any(str(value.name) == resolved_input_name for value in graph.output):
        raise ValueError("DeepX ImageNet adapter does not support input-as-output graphs")

    mean = np.asarray(IMAGENET_RGB_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(IMAGENET_RGB_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    graph.initializer.extend([
        numpy_helper.from_array(mean, name=mean_name),
        numpy_helper.from_array(std, name=std_name),
    ])
    existing_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend([
        helper.make_node(
            "Sub", [resolved_input_name, mean_name], [centered],
            name="onnx_splitpoint_imagenet_subtract_mean",
        ),
        helper.make_node(
            "Div", [centered, std_name], [original_input],
            name="onnx_splitpoint_imagenet_divide_std",
        ),
        *existing_nodes,
    ])

    contract = deepx_classification_preprocessing_contract(
        CLASSIFICATION_PREPROCESSING_IMAGENET
    )
    source_sha256 = _sha256_file(source)
    adapter_identity = {
        "schema": "onnx-splitpoint/deepx-build-onnx-adapter",
        "schema_version": 1,
        "source_onnx_sha256": source_sha256,
        "input_name": resolved_input_name,
        "consumer_count": consumer_count,
        "preprocessing": contract,
    }
    metadata_key = "onnx_splitpoint.deepx_build_adapter_sha256"
    metadata_value = _canonical_json_sha256(adapter_identity)
    kept_metadata = [item for item in model.metadata_props if item.key != metadata_key]
    del model.metadata_props[:]
    model.metadata_props.extend(kept_metadata)
    metadata = model.metadata_props.add()
    metadata.key = metadata_key
    metadata.value = metadata_value
    onnx.checker.check_model(model)

    staging = destination_root / f".{metadata_value}.tmp.onnx"
    onnx.save_model(model, str(staging), save_as_external_data=False)
    build_sha256 = _sha256_file(staging)
    destination = destination_root / f"imagenet_mean_std_{build_sha256}.onnx"
    if destination.is_file():
        if _sha256_file(destination) != build_sha256:
            raise RuntimeError("content-addressed DeepX build ONNX collision")
        staging.unlink()
    else:
        os.replace(staging, destination)
    receipt = {
        **adapter_identity,
        "adapter_contract_sha256": metadata_value,
        "build_onnx_path": str(destination),
        "build_onnx_sha256": build_sha256,
        "build_onnx_bytes": destination.stat().st_size,
    }
    return destination, receipt


def image_model_dxcom_config(
    *,
    task: str,
    input_name: str = "images",
    input_shape: Sequence[int] = (1, 3, 640, 640),
    calibration_dir: str | Path,
    calibration_num: int = 100,
    calibration_method: str = "ema",
    image_size: int | None = None,
    classification_preprocessing: str = CLASSIFICATION_PREPROCESSING_CURRENT,
) -> dict[str, Any]:
    """Create a task-bound DX-COM config for image-model ONNX exports.

    Classification uses a direct width/height resize. Detection mirrors the
    tested YOLO path with aspect-ratio preserving pad/114.  Requiring the task
    here prevents a DXNN compiled with detection geometry from later being
    labelled as a classification resize contract (or vice versa).
    """
    task_name = str(task or "").strip().lower()
    if task_name not in {"classification", "detection"}:
        raise ValueError("DeepX image config requires task=classification or task=detection")
    if task_name == "classification":
        # Validation is intentional even though the two arms use identical
        # locally-proven DX-COM loader opcodes.  The corrected numeric delta is
        # carried by its content-addressed ONNX adapter.
        canonical_classification_preprocessing(classification_preprocessing)
    shape = [int(x) for x in input_shape]
    if len(shape) != 4:
        raise ValueError(f"DeepX image config expects NCHW input shape with 4 dims, got {shape}")
    if int(shape[1]) in (1, 3, 4):
        height, width = int(shape[2]), int(shape[3])
    else:
        height, width = int(shape[1]), int(shape[2])
    if image_size is not None:
        height = width = int(image_size)
    if height <= 0 or width <= 0:
        raise ValueError(f"DeepX image config requires static positive spatial dimensions, got {shape}")
    if height != width:
        raise ValueError("DeepX prepared-feed runtime currently requires a square model input")
    if task_name == "classification":
        resize = {"resize": {"width": width, "height": height}}
    else:
        resize = {
            "resize": {
                "mode": "pad",
                "size": width,
                "pad_location": "edge",
                "pad_value": [114, 114, 114],
            }
        }
    return {
        "inputs": {str(input_name or "images"): shape},
        "calibration_num": int(calibration_num),
        "calibration_method": str(calibration_method or "ema"),
        "default_loader": {
            "dataset_path": str(Path(calibration_dir).expanduser()),
            "file_extensions": ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
            "preprocessings": [
                resize,
                {"div": {"x": 255.0}},
                {"convertColor": {"form": "BGR2RGB"}},
                {"transpose": {"axis": [2, 0, 1]}},
                {"expandDim": {"axis": 0}},
            ],
        },
    }


def write_dxcom_config(config: Mapping[str, Any], path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p



def tensor_activation_dxcom_config(
    *,
    inputs: Mapping[str, Sequence[int]],
    calibration_dir: str | Path,
    calibration_num: int = 100,
    calibration_method: str = "ema",
    file_extensions: Sequence[str] = ("npz", "npy", "NPZ", "NPY"),
) -> dict[str, Any]:
    """Create an experimental DX-COM config for feature-tensor calibration.

    This is used for DeepX-as-Stage2 experiments where the DeepX model starts at
    split cut tensors rather than images.  The proxy cache stores per-sample
    ``.npz`` cut tensors generated by ORT/CUDA/TensorRT Part1.  DX-COM support
    for non-image default loaders can vary by release, so callers should treat
    successful part2 DXNN builds as experimental and keep the manifest fields in
    the report.
    """
    inp: dict[str, list[int]] = {}
    for name, shape in dict(inputs or {}).items():
        dims = []
        for x in list(shape or []):
            try:
                dims.append(int(x))
            except Exception:
                pass
        if dims:
            inp[str(name)] = dims
    if not inp:
        raise ValueError("DeepX tensor activation config requires at least one input shape")
    return {
        "inputs": inp,
        "calibration_num": int(calibration_num),
        "calibration_method": str(calibration_method or "ema"),
        "default_loader": {
            "dataset_path": str(Path(calibration_dir).expanduser()),
            "file_extensions": [str(x) for x in file_extensions],
            "preprocessings": [],
        },
        "calibration_input_kind": "feature_tensor_activation_proxy",
        "note": "Experimental: feature tensor proxy calibration for split Stage2 DeepX models.",
    }
