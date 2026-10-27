"""One-image Quality adapter around the existing Hailo10 Native FIFO.

The producer, exact HEF quantization, boundary handling and TensorRT consumer
remain owned by the normal Native runner.  This adapter only opens that runtime
once and submits one exact dataset image at a time.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import threading
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_NATIVE_RUNTIME_LOCK = threading.Lock()


def _load_native_runtime() -> Any:
    return importlib.import_module("scripts.native_hailo10_trt_e2e_from_benchmarkset")


def _absolute_path(value: Any, *, name: str, directory: bool = False) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"native_dataset_quality_h10_absolute_path_required:{name}")
    path = path.resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise RuntimeError(f"native_dataset_quality_h10_artifact_missing:{name}:{path}")
    return path


class H10NativeQualityRuntime:
    """Persistent, no-build Hailo10 -> NativeTRT runtime for dataset Quality."""

    def __init__(self, config: dict[str, Any], work: Path):
        self.config = dict(config)
        self.work = Path(work).expanduser().resolve()
        self.output_signature: dict[str, dict[str, Any]] = {}
        self.source_paths: tuple[str, ...] = ()
        self._native: Any = None
        self._backend: Any = None
        self._prepared: Any = None
        self._trt: Any = None
        self._boundary: dict[str, Any] = {}
        self._source_sha256 = ""
        self._entered = False

    def __enter__(self) -> H10NativeQualityRuntime:
        if self._entered:
            raise RuntimeError("native_dataset_quality_h10_runtime_already_open")
        cfg = self.config
        for key in ("build", "build_missing_engine", "force_rebuild_engine", "force_build", "force_rebuild"):
            if cfg.get(key):
                raise RuntimeError("native_dataset_quality_h10_engine_build_forbidden")
        if cfg.get("engine_build_forbidden", True) is not True:
            raise RuntimeError("native_dataset_quality_h10_engine_build_forbidden")
        for key in ("quantized_inputs", "quantized_outputs", "copy_outputs"):
            if cfg.get(key, True) is not True:
                raise RuntimeError(f"native_dataset_quality_h10_required_native_option:{key}")
        if str(cfg.get("producer_impl", "async_fifo")) not in {"async_fifo", "auto"}:
            raise RuntimeError("native_dataset_quality_h10_async_fifo_required")
        if str(cfg.get("task", "detection")) != "detection":
            # The existing classification FIFO creates its own completion object.
            raise RuntimeError("native_dataset_quality_h10_detection_task_required")
        for key, default in (("inflight", 8), ("queue_depth", 3)):
            if int(cfg.get(key, default)) < 1:
                raise RuntimeError(f"native_dataset_quality_h10_positive_option_required:{key}")

        benchmark_set = _absolute_path(cfg.get("benchmark_set"), name="benchmark_set", directory=True)
        hef = _absolute_path(cfg.get("hef"), name="hef")
        engine = _absolute_path(cfg.get("engine"), name="engine")
        binding_path = _absolute_path(cfg.get("native_split_quality_binding"), name="native_split_quality_binding")
        native = self._native = _load_native_runtime()
        raw = native._load_json(binding_path)
        if not isinstance(raw, Mapping):
            raise RuntimeError("native_dataset_quality_h10_binding_invalid")
        selection = raw.get("preselection")
        if not isinstance(selection, Mapping):
            raise RuntimeError("native_dataset_quality_h10_preselection_missing")
        case = native._case_id(str(cfg.get("case_id") or ""))
        binding = native._load_split_quality_binding(
            Namespace(
                native_split_quality_binding=str(binding_path),
                model_id=str(cfg.get("model_id") or selection.get("model_id") or ""),
                setup_id=str(cfg.get("setup_id") or selection.get("setup_id") or ""),
                # This is the existing Native artifact binding's run identity.
                # The dataset evaluation has a separate run identity.
                eval_run_id=str(raw.get("eval_run_id") or ""),
                source_run_id=str(raw.get("source_run_id") or ""),
            ),
            case=case,
            task="detection",
        )
        if not isinstance(binding, Mapping):
            raise RuntimeError("native_dataset_quality_h10_binding_required")
        selection = binding["preselection"]
        for name, supplied in (("part1_runtime", hef), ("engine", engine)):
            if native._binding_path(binding, name) != supplied:
                raise RuntimeError(f"native_dataset_quality_h10_bound_artifact_mismatch:{name}")
        selected = {
            "precision": str(selection["precision"]),
            "boundary_layout": str(selection["boundary_layout"]),
            "preprocess_mode": str(selection["preprocess_mode"]),
            "letterbox_pad_value": int(selection["letterbox_pad_value"]),
        }
        defaults = {"precision": "uint8_dequant_fp16", "boundary_layout": "as_input", "preprocess_mode": "letterbox", "letterbox_pad_value": 114}
        for key, value in selected.items():
            supplied = cfg.get(key, defaults[key])
            supplied = int(supplied) if key == "letterbox_pad_value" else str(supplied)
            if supplied != value:
                raise RuntimeError(f"native_dataset_quality_h10_preselection_mismatch:{key}")
        if selected["precision"] != "uint8_dequant_fp16" or str(selection["hailo_format"]) != "uint8":
            raise RuntimeError("native_dataset_quality_h10_uint8_dequant_bridge_required")
        for key in ("dequant_scale", "dequant_zero_point"):
            if key in cfg and float(cfg[key]) != float(selection[key]):
                raise RuntimeError(f"native_dataset_quality_h10_preselection_mismatch:{key}")

        boundary_path = native._binding_path(binding, "boundary_metadata")
        if cfg.get("boundary_metadata") and _absolute_path(cfg["boundary_metadata"], name="boundary_metadata") != boundary_path:
            raise RuntimeError("native_dataset_quality_h10_bound_artifact_mismatch:boundary_metadata")
        metadata = native._load_json(boundary_path)
        tensor = metadata.get("boundary_tensor") if isinstance(metadata, Mapping) else None
        if not isinstance(tensor, Mapping):
            raise RuntimeError("native_split_quality_boundary_metadata_tensor_missing")
        self._boundary = {**tensor, "metadata_sha256": str(metadata.get("metadata_sha256") or "")}

        opts: dict[str, Any] = {
            "hef_path": str(hef), "hw_arch": str(cfg.get("hw_arch", "hailo10h")),
            "runtime_api": "infer_model", "quantized_inputs": True,
            "quantized_outputs": True, "hotloop": True, "copy_outputs": True,
        }
        part1_onnx = native._find_part1_onnx(benchmark_set / case, case)
        if part1_onnx:
            opts["onnx_model_path"] = str(part1_onnx)
        self.work.mkdir(parents=True, exist_ok=True)
        try:
            self._backend = native.HailoBackend(strict=True, **opts)
            self._prepared = self._backend.prepare(
                native.RunCfg(model_path=hef, options=opts), self.work / "hailo_artifacts",
            )
            native._selected_hailo_output_format(self._prepared, True)
            quantization = native._exact_hailo10_output_quantization(self._prepared)
            native._require_exact_hailo10_engine_quantization(
                engine, quantization, boundary_layout=selected["boundary_layout"],
            )
            self._trt = native.NativeTRT(engine)
            self.output_signature = {
                str(name): {
                    "shape": [int(dim) for dim in self._trt.shapes[name]],
                    "dtype": str(np.dtype(self._trt.dtypes[name])),
                }
                for name in self._trt.outputs
            }
            source = Path(native.__file__).resolve()
            backend_source = inspect.getsourcefile(native.HailoBackend)
            self.source_paths = tuple(dict.fromkeys(
                str(path) for path in (source, Path(backend_source).resolve() if backend_source else source)
            ))
            self._source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            self._entered = True
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def run_image(self, image_path: Path, completion_runtime: Any) -> dict[str, Any]:
        if not self._entered:
            raise RuntimeError("native_dataset_quality_h10_runtime_not_open")
        image = _absolute_path(image_path, name="image")
        if not callable(getattr(completion_runtime, "process", None)) or int(getattr(completion_runtime, "completed_count", -1)) != 0:
            raise RuntimeError("native_dataset_quality_h10_fresh_completion_runtime_required")
        timing_binding: dict[str, Any] = {}
        inputs = self._native._make_input(
            self._prepared, True, image=str(image),
            preprocess_mode=str(self.config.get("preprocess_mode", "letterbox")),
            letterbox_pad_value=int(self.config.get("letterbox_pad_value", 114)),
            task="detection", timing_input_binding=timing_binding,
            prepared_input_metadata={},
        )
        snapshot = [
            {"name": str(name), "shape": [int(dim) for dim in np.asarray(value).shape],
             "dtype": str(np.asarray(value).dtype),
             "sha256": hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()}
            for name, value in inputs.items()
        ]
        # The existing Native helpers share strict-ABI globals.  Keep the bound
        # values scoped to this single FIFO call and restore them on every exit.
        with _NATIVE_RUNTIME_LOCK:
            previous_boundary = self._native._STRICT_SPLIT_BOUNDARY
            previous_evidence = self._native._STRICT_SPLIT_BOUNDARY_EVIDENCE
            self._native._STRICT_SPLIT_BOUNDARY = dict(self._boundary)
            self._native._STRICT_SPLIT_BOUNDARY_EVIDENCE = {}
            try:
                result = self._native._hailo10_async_fifo_run(
                    self._prepared.handle.session, inputs, self._trt,
                    frames=1, warmup=0, duration_s=0.0,
                    inflight=int(self.config.get("inflight", 8)),
                    queue_depth=int(self.config.get("queue_depth", 3)),
                    task="detection", completion_runtime=completion_runtime,
                    warmup_completion_runtime=None,
                )
                boundary_evidence = dict(self._native._STRICT_SPLIT_BOUNDARY_EVIDENCE)
            finally:
                self._native._STRICT_SPLIT_BOUNDARY = previous_boundary
                self._native._STRICT_SPLIT_BOUNDARY_EVIDENCE = previous_evidence
        if int(result.get("completed_work_units", -1)) != 1 or int(getattr(completion_runtime, "completed_count", -1)) != 1:
            raise RuntimeError("native_dataset_quality_h10_exactly_one_completion_required")
        for key in ("frames", "produced_frames", "consumed_frames", "completed_frames"):
            if key in result and int(result[key]) != 1:
                raise RuntimeError("native_dataset_quality_h10_exactly_one_completion_required")
        if boundary_evidence.get("status") != "exact_runtime_boundary_verified":
            raise RuntimeError("native_dataset_quality_h10_strict_boundary_evidence_missing")
        for key in ("completion_execution_attestation", "completion_execution_contract"):
            if not isinstance(result.get(key), Mapping):
                raise RuntimeError(f"native_dataset_quality_h10_completion_evidence_missing:{key}")
        return {
            "timing_input_binding": timing_binding,
            "prepared_input_snapshot": snapshot,
            "native_completed_count": 1,
            "native_runtime_source_file_sha256": self._source_sha256,
            "runtime_boundary_evidence": boundary_evidence,
            "completion_execution_attestation": result["completion_execution_attestation"],
            "completion_execution_contract": result["completion_execution_contract"],
        }

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._entered = False
        try:
            if self._trt is not None:
                self._trt.close()
        finally:
            try:
                if self._prepared is not None:
                    self._backend.cleanup(self._prepared)
            finally:
                self._trt = None
                self._prepared = None
                self._backend = None
