#!/usr/bin/env python3
"""Device-side fixed16 Hailo runtime collection, using the normal HEF backend.

The controller owns bounded process supervision and setup/transport selection.
This worker accepts only already staged exact files, never invokes a compiler,
and writes only a new results directory. A runtime PASS is not a quality claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _read_json(path):
    path = Path(path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("runtime_request_over_budget")
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("runtime_duplicate_json_key:" + key)
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _file(stage, row):
    relative = PurePosixPath(row["path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in row["path"]:
        raise ValueError("runtime_input_path_must_be_inside_stage")
    path = stage.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise ValueError("runtime_input_not_regular_local_file")
    path.relative_to(stage)
    if type(row.get("size_bytes")) is not int or not 0 < row["size_bytes"] <= 2 * 1024 * 1024 * 1024:
        raise ValueError("runtime_input_size_invalid")
    if path.stat().st_size != row["size_bytes"] or _sha(path) != row.get("sha256"):
        raise ValueError("runtime_input_identity_mismatch:" + row["path"])
    return path


def validate_staged_request(stage):
    stage = Path(stage).expanduser().absolute()
    if stage.resolve(strict=True) != stage:
        raise ValueError("runtime_stage_must_not_traverse_symlinks")
    request = _read_json(stage / "request.json")
    if (request.get("schema") != "hailo_model_runtime_request_v27934" or request.get("family") != "hailo10h" or
            request.get("model") != "mobilenet_v3_large" or not isinstance(request.get("setup_id"), str) or not request["setup_id"].strip()):
        raise ValueError("runtime_fixed_hailo10_full_request_required")
    for key in ("expected_source_onnx_sha256", "expected_compiler_sha256"):
        if not isinstance(request.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", request[key]):
            raise ValueError("runtime_source_graph_binding_required:" + key)
    for key in ("input_name", "output_name"):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise ValueError("runtime_canonical_name_required:" + key)
    shape = request.get("input_shape")
    if (not isinstance(shape, list) or len(shape) != 4 or any(type(x) is not int or x <= 0 for x in shape) or
            shape[:2] != [1, 3] or 16 * 3 * shape[2] * shape[3] * 4 > 64 * 1024 * 1024):
        raise ValueError("runtime_source_NCHW_RGB_shape_required")
    if type(request.get("class_count")) is not int or not 5 <= request["class_count"] <= 100000:
        raise ValueError("runtime_explicit_class_count_required")
    from onnx_splitpoint_tool.preprocessing_contract import resolve_image_preprocessing_contract
    resolve_image_preprocessing_contract(task="classification", target_hw=shape[2:], declared=request["preprocessing_contract"])
    rows = request.get("images")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("runtime_requires_exactly_16_images")
    for row in rows:
        if (not isinstance(row.get("id"), str) or not row["id"].strip() or type(row.get("label")) is not int or
                not 0 <= row["label"] < request["class_count"]):
            raise ValueError("runtime_fixed_image_id_or_label_invalid")
        _file(stage, row)
    if len({r["id"] for r in rows}) != 16 or len({r["path"] for r in rows}) != 16:
        raise ValueError("runtime_images_must_be_distinct")
    for role in ("cpu_hef", "gpu_hef"):
        if Path(request[role]["path"]).suffix.lower() != ".hef":
            raise ValueError("runtime_direct_hef_required")
        _file(stage, request[role])
    return request


def classification_vector(output, class_count):
    """A unique class axis from the explicit compiler graph; no squeeze guess."""
    import numpy as np
    array = np.asarray(output)
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError("runtime_finite_dequantized_float_logits_required")
    axes = [i for i, size in enumerate(array.shape) if size == class_count]
    if len(axes) != 1 or any(size != 1 for i, size in enumerate(array.shape) if i != axes[0]):
        raise ValueError("runtime_class_axis_not_unique_or_nonclass_dimension_not_singleton")
    return np.ascontiguousarray(np.moveaxis(array, axes[0], -1).reshape(class_count))


def _logical_hwc(array, layout):
    import numpy as np
    if layout == "NCHW":
        array = np.transpose(array[0], (1, 2, 0))
    elif layout == "NHWC":
        array = array[0]
    elif layout == "CHW":
        array = np.transpose(array, (1, 2, 0))
    elif layout != "HWC":
        raise ValueError("runtime_image_layout_not_explicit")
    return np.ascontiguousarray(array)


def collect(stage):
    import numpy as np
    from onnx_splitpoint_tool.runners._types import RunCfg
    from onnx_splitpoint_tool.runners.backends.hailo_backend import HailoBackend
    from smoke_hailo10_hef_runner import _image_tensor, _shape_hwc
    stage = Path(stage).expanduser().absolute()
    request = validate_staged_request(stage)
    output = stage / "results"
    output.mkdir(mode=0o700, exist_ok=False)
    arrays, reports = {}, {}
    os.chdir(output)
    try:
        for role in ("cpu_hef", "gpu_hef"):
            hef = _file(stage, request[role])
            options = dict(hef_path=str(hef), hw_arch="hailo10h", runtime_api="infer_model",
                           compile_backend="local", quantized_inputs=True, quantized_outputs=False,
                           copy_outputs=True, canonical_input_slot_names=[request["input_name"]],
                           canonical_output_slot_names=[request["output_name"]])
            backend = HailoBackend(strict=True, **options)
            prepared = None
            try:
                prepared = backend.prepare(RunCfg(model_path=hef, options=options), output / role)
                if list(prepared.input_names) != [request["input_name"]] or list(prepared.output_names) != [request["output_name"]]:
                    raise ValueError("runtime_exact_canonical_io_binding_mismatch")
                handle = prepared.handle
                shape = tuple((handle.runtime_input_shapes or handle.input_shapes)[request["input_name"]])
                height, width, channels, _ = _shape_hwc(shape)
                if [height, width] != request["input_shape"][2:] or channels != 3:
                    raise ValueError("runtime_HEF_shape_differs_from_bound_compiler_graph")
                quantizer = getattr(handle.session, "quantize_input", None)
                describe = getattr(handle.session, "describe_io", None)
                if not callable(quantizer) or not callable(describe):
                    raise ValueError("runtime_exact_HEF_quantization_and_IO_required")
                io = dict(describe())
                logits, logical, actual = [], [], []
                for row in request["images"]:
                    tensor, rgb, prep = _image_tensor(_file(stage, row), shape, quantized=False,
                        task="classification", preprocess_mode="resize", letterbox_pad=0)
                    if prep["preprocessing_contract"] != request["preprocessing_contract"]:
                        raise ValueError("runtime_preprocessing_contract_mismatch")
                    feed = np.asarray(quantizer(request["input_name"], tensor))
                    if feed.dtype != np.uint8 or feed.shape != tensor.shape:
                        raise ValueError("runtime_exact_quantized_feed_shape_or_dtype_invalid")
                    feed = np.ascontiguousarray(feed)
                    result = backend.run(prepared, {request["input_name"]: feed})
                    if set(result.outputs) != {request["output_name"]}:
                        raise ValueError("runtime_output_binding_changed")
                    logits.append(classification_vector(result.outputs[request["output_name"]], request["class_count"]))
                    logical.append(_logical_hwc(tensor, prep["layout"]))
                    actual.append(feed.copy())
                arrays[role + "_logits"] = np.stack(logits)
                arrays[role + "_logical_feed"] = np.stack(logical)
                arrays[role + "_actual_feed"] = np.stack(actual)
                reports[role] = {"runtime_status": "pass", "hef_readability_status": "pass", "inference_count": len(logits),
                    "artifact_sha256": request[role]["sha256"], "ids": [r["id"] for r in request["images"]],
                    "runtime_io": io, "logical_feed_layout": "NHWC", "actual_feed_dtype": "uint8",
                    "output_dtype": str(arrays[role + "_logits"].dtype), "output_shape": list(arrays[role + "_logits"].shape),
                    "source": "normal_HailoBackend.prepare_and_run_exact_precompiled_HEF",
                    "normalization": "existing_canonical_image_tensor_imagenet_once_then_exact_HEF_QuantInfo"}
            finally:
                if prepared is not None:
                    backend.cleanup(prepared)
        logical_equal = (arrays["cpu_hef_logical_feed"].dtype == arrays["gpu_hef_logical_feed"].dtype and
                         np.array_equal(arrays["cpu_hef_logical_feed"], arrays["gpu_hef_logical_feed"]))
        if not logical_equal:
            raise ValueError("runtime_cpu_gpu_logical_feeds_differ")
        validate_staged_request(stage)
        np.savez(output / "runtime_arrays.npz", **arrays)
        result = {"schema": "hailo_model_runtime_result_v27934", "runtime_status": "pass", "setup_id": request["setup_id"],
                  "family": "hailo10h", "model": "mobilenet_v3_large", "stages": reports,
                  "expected_source_onnx_sha256": request["expected_source_onnx_sha256"],
                  "expected_compiler_sha256": request["expected_compiler_sha256"],
                  "logical_feed_equal": True,
                  "actual_quantized_feed_equal": bool(np.array_equal(arrays["cpu_hef_actual_feed"], arrays["gpu_hef_actual_feed"])),
                  "actual_quantized_feed_comparison_note": "HEF-specific QuantInfo can change integer feeds; identical logical feeds are required.",
                  "frozen_inputs_unchanged": True, "compiler_invoked": False, "quality_status": "not_evaluated",
                  "raw_energy_status": "not_run", "claim_eligible": False,
                  "arrays": {"path": "results/runtime_arrays.npz", "sha256": _sha(output / "runtime_arrays.npz"),
                             "size_bytes": (output / "runtime_arrays.npz").stat().st_size}}
        (output / "runtime_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("HAILO_FIXED16_RUNTIME=PASS", flush=True)
        return result
    except BaseException as exc:
        if arrays:
            np.savez(output / "runtime_arrays.npz", **arrays)
        (output / "runtime_result.json").write_text(json.dumps({"runtime_status": "failed", "stages": reports,
            "error": type(exc).__name__ + ": " + str(exc), "claim_eligible": False, "compiler_invoked": False}, indent=2) + "\n")
        raise


def supervised_collect(stage):
    """Standalone process-tree owner; normal remote transport need not kill jobs."""
    from deepx_full_workflow_smoke_worker_v27930 import supervised_run
    stage = Path(stage).expanduser().absolute()
    validate_staged_request(stage)
    output = stage / "results"
    if output.exists() or output.is_symlink():
        raise ValueError("runtime_results_directory_already_exists")
    process = supervised_run([sys.executable, "-B", str(Path(__file__).resolve()),
                              "--stage", str(stage), "--collect"],
                             cwd=stage, log_path=stage / "runtime.log", timeout=300, grace=5)
    output.mkdir(mode=0o700, exist_ok=True)
    result_path = output / "runtime_result.json"
    result = _read_json(result_path) if result_path.is_file() else {"runtime_status": "failed", "claim_eligible": False}
    if process["returncode"] or process["timed_out"] or process["cancelled"] or not process["cleanup_complete"]:
        result["runtime_status"] = "failed"
        result.setdefault("error", "runtime_worker_failed_or_process_cleanup_incomplete")
    result["supervision"] = process
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "supervision.json").write_text(json.dumps(process, indent=2, sort_keys=True) + "\n")
    shutil.copy2(stage / "runtime.log", output / "runtime.log")
    print(json.dumps({"runtime_status": result["runtime_status"], "cleanup_complete": process["cleanup_complete"],
                      "report": str(result_path)}, sort_keys=True), flush=True)
    return 0 if result["runtime_status"] == "pass" and process["returncode"] == 0 and process["cleanup_complete"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--collect", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.collect:
        collect(args.stage)
        return 0
    return supervised_collect(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
