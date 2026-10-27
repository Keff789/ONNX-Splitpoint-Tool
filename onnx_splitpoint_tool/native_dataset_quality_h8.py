"""Dataset Quality submissions through the existing Hailo-8 Native FIFO."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_REMOTE_SCRIPTS = Path(__file__).resolve().parent / "resources" / "remote_scripts"
_H8_SOURCE = _REMOTE_SCRIPTS / "native_hailo_trt_fifo_from_benchmarkset.py"
_TRT_SOURCE = _REMOTE_SCRIPTS / "native_hailo10_trt_e2e_from_benchmarkset.py"


def _load_file_module(name: str, source: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"native_dataset_quality_h8_source_import_failed:{source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_native_runtime() -> Any:
    # Hailo-8 imports its existing NativeTRT consumer by this short name.  Bind
    # that import to the shipped source before loading the Hailo-8 runner.
    trt_name = "native_hailo10_trt_e2e_from_benchmarkset"
    consumer = sys.modules.get(trt_name)
    if consumer is None:
        consumer = _load_file_module(trt_name, _TRT_SOURCE)
    source = Path(str(getattr(consumer, "__file__", "") or "")).resolve()
    if not source.is_file() or hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(_TRT_SOURCE.read_bytes()).digest():
        raise RuntimeError("native_dataset_quality_h8_native_trt_source_mismatch")
    name = "onnx_splitpoint_tool._native_dataset_quality_h8_existing"
    native = sys.modules.get(name)
    if native is None:
        native = _load_file_module(name, _H8_SOURCE)
    if Path(str(native.__file__)).resolve() != _H8_SOURCE.resolve() or native._native_trt_module is not consumer:
        raise RuntimeError("native_dataset_quality_h8_native_source_mismatch")
    if native.NativeTRT is None:
        raise RuntimeError("native_dataset_quality_h8_native_trt_unavailable")
    return native


def _absolute_path(value: Any, *, name: str, directory: bool = False) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"native_dataset_quality_h8_absolute_path_required:{name}")
    path = path.resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise RuntimeError(f"native_dataset_quality_h8_artifact_missing:{name}:{path}")
    return path


class H8NativeQualityRuntime:
    """Open one no-build VStreams/TRT pair and run one image per completion."""

    def __init__(self, config: dict[str, Any], work: Path):
        self.config = dict(config)
        self.work = Path(work).expanduser().resolve()
        self.output_signature: dict[str, dict[str, Any]] = {}
        self.source_paths: tuple[str, ...] = ()
        self._native: Any = None
        self._backend: Any = None
        self._prepared: Any = None
        self._trt: Any = None
        self._args: Any = None
        self._boundary: dict[str, Any] = {}
        self._source_sha256 = ""
        self._entered = False

    def __enter__(self) -> H8NativeQualityRuntime:
        if self._entered:
            raise RuntimeError("native_dataset_quality_h8_runtime_already_open")
        cfg = self.config
        for key in ("build", "build_missing_engine", "force_rebuild_engine", "force_build", "force_rebuild"):
            if cfg.get(key):
                raise RuntimeError("native_dataset_quality_h8_engine_build_forbidden")
        if cfg.get("engine_build_forbidden", True) is not True:
            raise RuntimeError("native_dataset_quality_h8_engine_build_forbidden")
        if cfg.get("copy_outputs", True) is not True:
            raise RuntimeError("native_dataset_quality_h8_copy_outputs_required")
        if str(cfg.get("task", "detection")) != "detection":
            raise RuntimeError("native_dataset_quality_h8_detection_task_required")
        if int(cfg.get("queue_depth", 3)) < 1:
            raise RuntimeError("native_dataset_quality_h8_positive_queue_depth_required")
        _absolute_path(cfg.get("benchmark_set"), name="benchmark_set", directory=True)
        hef = _absolute_path(cfg.get("hef"), name="hef")
        engine = _absolute_path(cfg.get("engine"), name="engine")
        binding_path = _absolute_path(cfg.get("native_split_quality_binding"), name="native_split_quality_binding")
        native = self._native = _load_native_runtime()
        raw = native._load_json(binding_path)
        selection = raw.get("preselection") if isinstance(raw, Mapping) else None
        if not isinstance(selection, Mapping):
            raise RuntimeError("native_dataset_quality_h8_preselection_missing")
        case_value = str(cfg.get("case_id") or "")
        case = case_value if case_value.startswith("b") else f"b{int(case_value):03d}"
        args = self._args = Namespace(
            model_id=str(cfg.get("model_id") or selection.get("model_id") or ""),
            setup_id=str(cfg.get("setup_id") or selection.get("setup_id") or ""),
            precision=str(cfg.get("precision") or selection.get("precision") or ""),
            # Preserve the existing binding's identity separately from the new
            # dataset evaluation's identity.
            eval_run_id=str(raw.get("eval_run_id") or ""),
            source_run_id=str(raw.get("source_run_id") or ""),
            hailo_format=str(cfg.get("hailo_format") or selection.get("hailo_format") or ""),
            device_id=str(cfg.get("device_id") or ""),
            build=False,
        )
        binding = native._load_native_split_quality_binding(str(binding_path), args=args, case=case, task="detection")
        if not isinstance(binding, Mapping):
            raise RuntimeError("native_dataset_quality_h8_binding_required")
        selection = binding["preselection"]
        for name, supplied in (("part1_runtime", hef), ("engine", engine)):
            if native._binding_artifact_path(binding, name) != supplied:
                raise RuntimeError(f"native_dataset_quality_h8_bound_artifact_mismatch:{name}")
        if args.hailo_format not in {"uint8", "float32"} or args.hailo_format != str(selection["hailo_format"]):
            raise RuntimeError("native_dataset_quality_h8_preselection_mismatch:hailo_format")
        for key in ("precision", "boundary_layout", "preprocess_mode", "letterbox_pad_value", "dequant_scale", "dequant_zero_point"):
            if key in cfg and cfg[key] != selection.get(key):
                raise RuntimeError(f"native_dataset_quality_h8_preselection_mismatch:{key}")
        for key in ("quantized_inputs", "quantized_outputs"):
            if key in cfg and cfg[key] is not (args.hailo_format == "uint8"):
                raise RuntimeError(f"native_dataset_quality_h8_preselection_mismatch:{key}")
        args.preprocess_mode_effective = str(selection["preprocess_mode"])
        args.letterbox_pad_value = int(selection["letterbox_pad_value"])
        boundary_path = native._binding_artifact_path(binding, "boundary_metadata")
        if cfg.get("boundary_metadata") and _absolute_path(cfg["boundary_metadata"], name="boundary_metadata") != boundary_path:
            raise RuntimeError("native_dataset_quality_h8_bound_artifact_mismatch:boundary_metadata")
        metadata = native._load_json(boundary_path)
        tensor = metadata.get("boundary_tensor") if isinstance(metadata, Mapping) else None
        if not isinstance(tensor, Mapping):
            raise RuntimeError("native_split_quality_boundary_metadata_tensor_missing")
        self._boundary = {
            "expected_boundary_name": str(tensor.get("runtime_name") or tensor.get("name") or ""),
            "expected_boundary_shape": [int(dim) for dim in list(tensor.get("shape") or [])],
            "expected_boundary_dtype": str(tensor.get("dtype") or "").lower(),
        }
        if not self._boundary["expected_boundary_name"] or not self._boundary["expected_boundary_shape"] or any(dim <= 0 for dim in self._boundary["expected_boundary_shape"]) or self._boundary["expected_boundary_dtype"] not in {"uint8", "float32"}:
            raise RuntimeError("native_split_quality_boundary_runtime_identity_invalid")
        self.work.mkdir(parents=True, exist_ok=True)
        try:
            self._backend, self._prepared, self._trt = native._open_hailo8_python_runtime(
                hef=hef, engine=engine, work=self.work, args=args, runtime_label="dataset_quality",
            )
            self.output_signature = {
                str(name): {"shape": [int(dim) for dim in self._trt.shapes[name]], "dtype": str(np.dtype(self._trt.dtypes[name]))}
                for name in self._trt.outputs
            }
            source = Path(native.__file__).resolve()
            backend_source = inspect.getsourcefile(native.HailoBackend)
            consumer_source = Path(native._native_trt_module.__file__).resolve()
            self.source_paths = tuple(dict.fromkeys(str(path) for path in (
                source, consumer_source, Path(backend_source).resolve() if backend_source else source,
            )))
            self._source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            self._entered = True
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def run_image(self, image_path: Path, completion_runtime: Any) -> dict[str, Any]:
        if not self._entered:
            raise RuntimeError("native_dataset_quality_h8_runtime_not_open")
        image = _absolute_path(image_path, name="image")
        if not callable(getattr(completion_runtime, "process", None)) or int(getattr(completion_runtime, "completed_count", -1)) != 0:
            raise RuntimeError("native_dataset_quality_h8_fresh_completion_runtime_required")
        timing_binding: dict[str, Any] = {}
        inputs = self._native._hailo8_python_input(
            self._prepared, image, quantized=self._args.hailo_format == "uint8",
            preprocess_mode=self._args.preprocess_mode_effective,
            letterbox_pad_value=self._args.letterbox_pad_value,
            timing_input_binding=timing_binding, task="detection",
        )
        snapshot = [
            {"name": str(name), "shape": [int(dim) for dim in np.asarray(value).shape], "dtype": str(np.asarray(value).dtype),
             "sha256": hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()}
            for name, value in inputs.items()
        ]
        result = self._native._hailo8_python_fifo_run(
            self._backend, self._prepared, inputs, self._trt,
            frames=1, warmup=0, duration_s=0.0, queue_depth=int(self.config.get("queue_depth", 3)),
            completion_runtime=completion_runtime, warmup_completion_runtime=None, **self._boundary,
        )
        if int(result.get("completed_work_units", -1)) != 1 or int(getattr(completion_runtime, "completed_count", -1)) != 1:
            raise RuntimeError("native_dataset_quality_h8_exactly_one_completion_required")
        for key in ("frames", "produced_frames", "completed_frames"):
            if key in result and int(result[key]) != 1:
                raise RuntimeError("native_dataset_quality_h8_exactly_one_completion_required")
        boundary_evidence = dict(result.get("strict_quality_boundary") or {})
        if boundary_evidence.get("status") != "exact_runtime_boundary_verified":
            raise RuntimeError("native_dataset_quality_h8_strict_boundary_evidence_missing")
        for key in ("completion_execution_attestation", "completion_execution_contract"):
            if not isinstance(result.get(key), Mapping):
                raise RuntimeError(f"native_dataset_quality_h8_completion_evidence_missing:{key}")
        return {
            "timing_input_binding": timing_binding, "prepared_input_snapshot": snapshot,
            "native_completed_count": 1, "native_runtime_source_file_sha256": self._source_sha256,
            "runtime_boundary_evidence": boundary_evidence,
            "completion_execution_attestation": result["completion_execution_attestation"],
            "completion_execution_contract": result["completion_execution_contract"],
        }

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._entered = False
        try:
            if self._prepared is not None and self._trt is not None:
                self._native._close_hailo8_python_runtime(self._backend, self._prepared, self._trt)
        finally:
            self._backend = self._prepared = self._trt = None
