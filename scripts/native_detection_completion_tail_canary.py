#!/usr/bin/env python3
"""Profile the frozen Detection Completion tail from saved raw model outputs.

The canary executes no Hailo or TensorRT work.  It replays the exact raw-output
payloads through the same :class:`DetectionCompletionRuntime` used by the Native
Completed-Task endpoint and reports inclusive fine-grained timings.  The raw
payload identity, completion contract and reference detections are verified
before timing.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib
import json
import pstats
import statistics
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_payload_file(
    *, row: Mapping[str, Any], manifest_path: Path, payload_root: Path,
) -> Path:
    declared = str(row.get("file") or row.get("path") or "")
    candidates: list[Path] = []
    if declared:
        value = Path(declared).expanduser()
        candidates.append(value if value.is_absolute() else manifest_path.parent / value)
        candidates.append(payload_root / value.name)
    expected_hash = str(row.get("sha256") or "").lower()
    for candidate in candidates:
        if candidate.is_file() and (
            not expected_hash or _sha256_file(candidate).lower() == expected_hash
        ):
            return candidate.resolve()
    if expected_hash:
        for candidate in payload_root.rglob("*.bin"):
            if _sha256_file(candidate).lower() == expected_hash:
                return candidate.resolve()
    raise FileNotFoundError(
        f"raw output payload missing for {row.get('name')!r}: "
        f"declared={declared!r} sha256={expected_hash!r}"
    )


def _load_outputs(
    manifest_path: Path, payload_root: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    manifest = _load_json(manifest_path)
    rows = manifest.get("outputs") if isinstance(manifest, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("output manifest has no outputs")
    outputs: dict[str, np.ndarray] = {}
    identities: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"output manifest row {index} is not an object")
        name = str(raw.get("name") or f"output_{index}")
        dtype = np.dtype(str(raw.get("dtype") or "float32"))
        shape = tuple(int(value) for value in list(raw.get("shape") or []))
        if not shape:
            raise ValueError(f"output {name!r} has no shape")
        path = _resolve_payload_file(
            row=raw, manifest_path=manifest_path, payload_root=payload_root,
        )
        expected_elements = int(np.prod(shape, dtype=np.int64))
        values = np.fromfile(path, dtype=dtype)
        if int(values.size) != expected_elements:
            raise ValueError(
                f"output {name!r} element mismatch: {values.size} != {expected_elements}"
            )
        outputs[name] = values.reshape(shape)
        actual_hash = _sha256_file(path)
        expected_hash = str(raw.get("sha256") or "")
        if expected_hash and actual_hash.lower() != expected_hash.lower():
            raise ValueError(f"output {name!r} SHA-256 mismatch")
        identities.append({
            "index": index,
            "name": name,
            "dtype": str(dtype),
            "shape": list(shape),
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": actual_hash,
        })
    return outputs, identities


class _Recorder:
    def __init__(self) -> None:
        self.iteration = -1
        self.samples: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.calls: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def add(self, stage: str, elapsed_ms: float) -> None:
        if self.iteration < 0:
            return
        self.samples[stage][self.iteration] += float(elapsed_ms)
        self.calls[stage][self.iteration] += 1

    def stage_values(self, stage: str, count: int) -> list[float]:
        return [float(self.samples[stage].get(index, 0.0)) for index in range(count)]

    def stage_calls(self, stage: str, count: int) -> list[int]:
        return [int(self.calls[stage].get(index, 0)) for index in range(count)]


def _timed_wrapper(recorder: _Recorder, stage: str, target: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return target(*args, **kwargs)
        finally:
            recorder.add(stage, (time.perf_counter_ns() - started) / 1_000_000.0)
    wrapped.__name__ = getattr(target, "__name__", stage)
    wrapped.__doc__ = getattr(target, "__doc__", None)
    return wrapped


class _Patch:
    def __init__(self, owner: Any, name: str, replacement: Any) -> None:
        self.owner = owner
        self.name = name
        self.replacement = replacement
        self.original: Any = None

    def __enter__(self) -> "_Patch":
        self.original = getattr(self.owner, self.name)
        setattr(self.owner, self.name, self.replacement)
        return self

    def __exit__(self, *_exc: Any) -> None:
        setattr(self.owner, self.name, self.original)


def _parity(
    measured: Mapping[str, Any], reference_result: Mapping[str, Any],
) -> dict[str, Any]:
    reference_artifact = reference_result.get("completed_task_result_artifact")
    reference_detections = (
        list(reference_artifact.get("detections") or [])
        if isinstance(reference_artifact, Mapping) else []
    )
    actual_detections = list(measured.get("detections") or [])
    exact_content_hash = str(measured.get("content_sha256") or "") == str(
        reference_result.get("completion_content_sha256")
        or (reference_result.get("completion_execution_attestation") or {}).get("content_sha256")
        or ""
    )
    same_count = len(actual_detections) == len(reference_detections)
    same_classes = same_count and all(
        int(actual.get("class_id", -1)) == int(reference.get("class_id", -2))
        for actual, reference in zip(actual_detections, reference_detections)
    )
    score_diffs: list[float] = []
    coordinate_diffs: list[float] = []
    if same_count:
        for actual, reference in zip(actual_detections, reference_detections):
            score_diffs.append(abs(float(actual.get("score", 0.0)) - float(reference.get("score", 0.0))))
            for key in ("x1", "y1", "x2", "y2"):
                coordinate_diffs.append(abs(float(actual.get(key, 0.0)) - float(reference.get(key, 0.0))))
    score_max = max(score_diffs, default=0.0)
    coordinate_max = max(coordinate_diffs, default=0.0)
    tolerance_pass = bool(
        same_count and same_classes and score_max <= 1.0e-5 and coordinate_max <= 1.0e-3
    )
    return {
        "exact_content_sha256": exact_content_hash,
        "reference_detection_count": len(reference_detections),
        "actual_detection_count": len(actual_detections),
        "same_detection_count": same_count,
        "same_class_sequence": same_classes,
        "max_abs_score_difference": score_max,
        "max_abs_coordinate_difference_pixels": coordinate_max,
        "tolerance": {"score": 1.0e-5, "coordinate_pixels": 1.0e-3},
        "tolerance_pass": tolerance_pass,
        "status": "exact" if exact_content_hash else "tolerance_pass" if tolerance_pass else "failed",
    }


def _stage_summary(values: list[float], calls: list[int], total_values: list[float]) -> dict[str, Any]:
    total_ms = float(sum(values))
    total_runtime_ms = float(sum(total_values))
    return {
        "measurement": "inclusive_wall_time",
        "iterations": len(values),
        "calls_total": int(sum(calls)),
        "calls_per_iteration_median": _percentile([float(v) for v in calls], 0.5),
        "total_ms": total_ms,
        "mean_ms": statistics.fmean(values) if values else None,
        "median_ms": _percentile(values, 0.5),
        "p05_ms": _percentile(values, 0.05),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "inclusive_share_of_total_percent": (
            100.0 * total_ms / total_runtime_ms if total_runtime_ms > 0 else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--result", required=True, help="Completed-Task native_fifo_results.json")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--payload-root", default="")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.profile_iterations < 0:
        parser.error("iterations/profile-iterations must be positive and warmup non-negative")

    tool_root = Path(args.tool_root).expanduser().resolve()
    result_path = Path(args.result).expanduser().resolve()
    manifest_path = Path(args.output_manifest).expanduser().resolve()
    payload_root = Path(args.payload_root).expanduser().resolve() if args.payload_root else manifest_path.parent
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(tool_root))

    ndp = importlib.import_module("onnx_splitpoint_tool.native_detection_postprocess")
    yolo = importlib.import_module("onnx_splitpoint_tool.runners.harness.yolo")
    reference = _load_json(result_path)
    contract = reference.get("completion_execution_contract") if isinstance(reference, Mapping) else None
    if not isinstance(contract, Mapping):
        raise ValueError("result has no completion_execution_contract")
    outputs, payload_identities = _load_outputs(manifest_path, payload_root)
    observation_relation = str(reference.get("completion_observation_relation") or "same_hotloop_sentinel")

    runtime = ndp.DetectionCompletionRuntime(
        contract, observation_relation=observation_relation,
    )
    for _ in range(int(args.warmup)):
        runtime.process(outputs)

    recorder = _Recorder()
    patch_specs: list[tuple[Any, str, str]] = [
        (ndp, "tensor_signature", "source_tensor_signature"),
        (ndp, "_canonicalize_yolov7_heads", "yolov7_head_mapping"),
        (ndp, "_tensor_content_sha256", "raw_output_content_hashing"),
        (ndp, "verify_frozen_postprocess_contract", "frozen_contract_verification"),
        (ndp, "_result_json", "postprocess_result_conversion"),
        (ndp, "_canonical_detection_records", "canonical_detection_records"),
        (ndp, "canonical_json_sha256", "canonical_json_hashing"),
        (ndp, "_completion_result_artifact", "completion_artifact_materialization"),
        (yolo, "_detect_yolo_format", "yolo_format_detection"),
        (yolo, "_describe_outputs", "output_shape_description"),
        (yolo, "_normalize_multiscale_outputs", "multiscale_head_normalization"),
        (yolo, "_infer_multiscale_head_activation_mode", "activation_mode_inference"),
        (yolo, "_decode_multiscale_head", "multiscale_decode"),
        (yolo, "_nms_xyxy", "class_aware_nms"),
        (yolo.YoloHarness, "postprocess", "yolo_harness_total"),
        (ndp.FrozenDetectionPostprocessor, "process", "frozen_processor_total"),
    ]

    total_values: list[float] = []
    measured_result: dict[str, Any] = {}
    with ExitStack() as stack:
        for owner, name, stage in patch_specs:
            original = getattr(owner, name)
            stack.enter_context(_Patch(owner, name, _timed_wrapper(recorder, stage, original)))
        for index in range(int(args.iterations)):
            recorder.iteration = index
            started = time.perf_counter_ns()
            measured_result = runtime.process(outputs)
            total_values.append((time.perf_counter_ns() - started) / 1_000_000.0)
        recorder.iteration = -1

    stage_names = sorted(recorder.samples)
    stages = {
        stage: _stage_summary(
            recorder.stage_values(stage, int(args.iterations)),
            recorder.stage_calls(stage, int(args.iterations)),
            total_values,
        )
        for stage in stage_names
    }
    leaf_stages = [
        "source_tensor_signature", "yolov7_head_mapping",
        "raw_output_content_hashing", "frozen_contract_verification",
        "postprocess_result_conversion", "canonical_detection_records",
        "canonical_json_hashing", "completion_artifact_materialization",
        "yolo_format_detection", "output_shape_description",
        "multiscale_head_normalization", "activation_mode_inference",
        "multiscale_decode", "class_aware_nms",
    ]
    leaf_per_iteration = [
        sum(recorder.stage_values(stage, int(args.iterations))[index] for stage in leaf_stages)
        for index in range(int(args.iterations))
    ]
    residual_values = [
        max(0.0, total_values[index] - leaf_per_iteration[index])
        for index in range(int(args.iterations))
    ]

    profile_path = out_dir / "completion_tail_profile.prof"
    profile_text = out_dir / "completion_tail_profile_top.txt"
    if int(args.profile_iterations) > 0:
        prof = cProfile.Profile()
        profile_runtime = ndp.DetectionCompletionRuntime(
            contract, observation_relation=observation_relation,
        )
        prof.enable()
        for _ in range(int(args.profile_iterations)):
            profile_runtime.process(outputs)
        prof.disable()
        prof.dump_stats(str(profile_path))
        with profile_text.open("w", encoding="utf-8") as handle:
            stats = pstats.Stats(prof, stream=handle).strip_dirs().sort_stats("cumtime")
            stats.print_stats(80)

    parity = _parity(measured_result, reference)
    report = {
        "schema": "onnx-splitpoint/detection-completion-tail-canary",
        "schema_version": 1,
        "status": "PASS" if parity["tolerance_pass"] else "FAIL_PARITY",
        "scope": "offline_exact_raw_output_replay_no_accelerator_runtime",
        "tool_root": str(tool_root),
        "result": str(result_path),
        "result_sha256": _sha256_file(result_path),
        "output_manifest": str(manifest_path),
        "output_manifest_sha256": _sha256_file(manifest_path),
        "payload_root": str(payload_root),
        "payloads": payload_identities,
        "completion_contract_sha256": str(contract.get("contract_sha256") or ""),
        "completion_mode": str(contract.get("completion_mode") or ""),
        "observation_relation": observation_relation,
        "warmup_iterations": int(args.warmup),
        "measurement_iterations": int(args.iterations),
        "reference_completed_task_completion_tail_ms": reference.get("completion_tail_ms"),
        "runtime_total": _stage_summary(
            total_values, [1] * len(total_values), total_values,
        ),
        "stages": stages,
        "leaf_stage_sum": _stage_summary(
            leaf_per_iteration, [1] * len(leaf_per_iteration), total_values,
        ),
        "unattributed_residual": _stage_summary(
            residual_values, [1] * len(residual_values), total_values,
        ),
        "inclusive_timing_note": (
            "yolo_harness_total and frozen_processor_total are inclusive parent "
            "measurements and must not be added to leaf-stage timings"
        ),
        "parity": parity,
        "measured_result": {
            "detection_count": measured_result.get("detection_count"),
            "content_sha256": measured_result.get("content_sha256"),
            "source_content_sha256": measured_result.get("source_content_sha256"),
            "execution_contract_sha256": measured_result.get("execution_contract_sha256"),
        },
        "profile": {
            "iterations": int(args.profile_iterations),
            "prof_file": str(profile_path) if profile_path.is_file() else "",
            "top_text": str(profile_text) if profile_text.is_file() else "",
        },
    }
    report_path = out_dir / "completion_tail_canary_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ordered = sorted(
        ((name, data) for name, data in stages.items()),
        key=lambda item: float(item[1].get("median_ms") or 0.0),
        reverse=True,
    )
    lines = [
        "# Detection Completion-Tail Canary",
        "",
        f"- Status: **{report['status']}**",
        f"- Completion mode: `{report['completion_mode']}`",
        f"- Iterations: {args.iterations} after {args.warmup} warm-up calls",
        f"- Runtime median: **{report['runtime_total']['median_ms']:.3f} ms**",
        f"- Reference integrated tail: **{float(reference.get('completion_tail_ms') or 0.0):.3f} ms**",
        f"- Parity: **{parity['status']}** (max coordinate delta {parity['max_abs_coordinate_difference_pixels']:.6g} px)",
        "",
        "## Inclusive stage timings",
        "",
        "Parent stages are inclusive and are not additive with their children.",
        "",
        "| Stage | Median ms | P95 ms | Calls/iteration | Inclusive share |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, data in ordered:
        lines.append(
            f"| `{name}` | {float(data.get('median_ms') or 0.0):.4f} | "
            f"{float(data.get('p95_ms') or 0.0):.4f} | "
            f"{float(data.get('calls_per_iteration_median') or 0.0):.1f} | "
            f"{float(data.get('inclusive_share_of_total_percent') or 0.0):.1f}% |"
        )
    lines += [
        "",
        "## Additive leaf approximation",
        "",
        f"- Leaf-stage median sum: {float(report['leaf_stage_sum']['median_ms'] or 0.0):.4f} ms",
        f"- Unattributed residual median: {float(report['unattributed_residual']['median_ms'] or 0.0):.4f} ms",
        "",
        "The residual contains Python control flow, NumPy indexing/materialisation, "
        "coordinate projection and operations not exposed as separate functions.",
    ]
    markdown_path = out_dir / "completion_tail_canary_report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "markdown": str(markdown_path),
        "runtime_median_ms": report["runtime_total"]["median_ms"],
        "parity": parity["status"],
    }, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
