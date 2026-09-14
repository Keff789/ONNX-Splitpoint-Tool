#!/usr/bin/env python3
"""Run one bounded DeepX output probe in an isolated remote working directory."""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_probe(request: dict, work: Path) -> dict:
    """Read original artifacts, write all probe products beneath work/results."""
    output = work / "results"
    output.mkdir(exist_ok=True)
    root = Path(request["remote_suite"]).resolve(strict=True)
    dxnn = Path(request["dxnn_path"]).resolve(strict=True)
    image = Path(request["image_path"]).resolve(strict=True)
    if not root.is_dir() or not dxnn.is_file() or not image.is_file():
        raise ValueError("original_suite_dxnn_or_image_missing")
    expected_dxnn = str(request["expected_dxnn_sha256"])
    actual_dxnn = sha256(dxnn)
    if actual_dxnn != expected_dxnn:
        raise ValueError("original_dxnn_sha256_mismatch")
    source_package = root / "splitpoint_runners"
    if not source_package.is_dir():
        raise ValueError("original_splitpoint_runners_missing")
    runtime_root = work / "runtime"
    runtime_root.mkdir(exist_ok=True)
    runtime_package = runtime_root / "splitpoint_runners"
    shutil.copytree(
        source_package, runtime_package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(work / "native_detection_postprocess.py", runtime_package / "native_detection_postprocess.py")
    # Loading from the copied package prevents bytecode/diagnostic writes to
    # the old suite. Do not put the management installation on PYTHONPATH.
    sys.path.insert(0, str(runtime_root))
    template = work / "benchmark_suite.py.txt"
    loader = importlib.machinery.SourceFileLoader("deepx_probe_suite_v27927", str(template))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    plan = read_json(root / "benchmark_plan.json")
    runs = [row for row in plan.get("runs", []) if row.get("id") == "deepx_m1_full"]
    if len(runs) != 1:
        raise ValueError("original_full_plan_not_unique")
    run = dict(runs[0])
    expected_setup = str(request["setup_id"])
    if str(run.get("setup_id") or run.get("expected_setup_id") or "") != expected_setup:
        raise ValueError("original_full_setup_mismatch")
    run["model_id"] = request["model_id"]
    args = SimpleNamespace(
        runs=1, warmup=0, energy_measurement_only=False,
        throughput_frames=0, prepared_input_manifest="",
        prepared_feed_image=str(image), benchmark_task="detection",
        quality_evidence_model_id=request["model_id"],
        quality_evidence_setup_id=expected_setup,
        validation_images="", validation_max_images=0,
    )
    result = module.run_deepx_output_value_probe(root, dxnn, run, args, output)
    result["probe_source"] = {
        "original_run_id": request["original_run_id"],
        "model_id": request["model_id"], "setup_id": expected_setup,
        "remote_suite": str(root), "dxnn_path": str(dxnn),
        "dxnn_sha256": actual_dxnn, "image_path": str(image),
        "image_sha256": sha256(image),
        "runtime_python": sys.executable,
        "source_results_modified": False, "compiler_invoked": False,
        "quality_policy_changed": False,
    }
    shutil.copy2(image, output / ("probe_input" + image.suffix.lower()))
    (output / "deepx_output_value_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args(argv)
    request_path = args.request.resolve(strict=True)
    work = request_path.parent
    request = read_json(request_path)
    if request.get("schema") != "onnx-splitpoint/deepx-full-output-probe-request/v1":
        raise ValueError("probe_request_schema_invalid")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
        MPLCONFIGDIR=str(work / "matplotlib"),
        ONNX_SPLITPOINT_ARTIFACT_POLICY="cache_verify_only",
    )
    if not args.runtime:
        python = Path(request["runtime_venv"]).expanduser() / "bin/python"
        if not python.is_file():
            raise ValueError(f"existing_deepx_runtime_python_missing:{python}")
        os.execve(str(python), [str(python), "-I", "-B", str(Path(__file__).resolve()),
                              "--request", str(request_path), "--runtime"], env)
    os.environ.update(env)
    output = work / "results"
    output.mkdir(exist_ok=True)
    try:
        result = run_probe(request, work)
        status = result.get("status", "unknown")
        print("PROBE_STATUS=" + str(status), flush=True)
        print("PROBE_ERROR=" + str(result.get("error") or ""), flush=True)
        # Capturing a model value failure is a successful diagnostic operation.
        return 0
    except Exception as exc:
        result = {"schema": "onnx-splitpoint/deepx-full-output-probe/v1",
                  "diagnostic_only": True, "status": "probe_setup_failed",
                  "error": f"{type(exc).__name__}: {exc}"}
        (output / "deepx_output_value_probe.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8",
        )
        print("PROBE_STATUS=probe_setup_failed", flush=True)
        print("PROBE_ERROR=" + result["error"], flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
