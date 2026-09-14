#!/usr/bin/env python3
"""Use only the standalone suite staged into this probe's new remote directory."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import traceback
from types import SimpleNamespace

SCHEMA = "onnx-splitpoint/deepx-full-output-probe-request/v1"
STAGING_MODE = "local_files_to_fresh_remote_directory"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_file(root: Path, relative: str) -> Path:
    token = PurePosixPath(relative)
    if not relative or token.is_absolute() or ".." in token.parts or "\\" in relative:
        raise ValueError("staged_relative_path_invalid:" + relative)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("staged_file_missing_or_outside_suite:" + relative)
    return path


def validate_stage(request: dict, work: Path) -> tuple[Path, Path, Path, dict]:
    if request.get("schema") != SCHEMA or request.get("staging_mode") != STAGING_MODE:
        raise ValueError("staged_probe_request_required; old remote-run requests are not supported")
    root = (work / "suite").resolve(strict=True)
    if not root.is_relative_to(work) or not root.is_dir():
        raise ValueError("staged_suite_outside_probe")
    dxnn = contained_file(root, request["staged_dxnn_relative"])
    image = contained_file(root, request["staged_image_relative"])
    if sha256(dxnn) != request["expected_dxnn_sha256"]:
        raise ValueError("staged_dxnn_sha256_mismatch")
    if sha256(image) != request["expected_image_sha256"]:
        raise ValueError("staged_image_sha256_mismatch")
    for name in ("splitpoint_runners/__init__.py", "splitpoint_runners/harness/base.py",
                 "splitpoint_runners/harness/yolo.py", "splitpoint_runners/native_detection_postprocess.py", "benchmark_suite.py"):
        contained_file(root, name)
    plan = read_json(contained_file(root, "benchmark_plan.json"))
    runs = [r for r in plan.get("runs", []) if r.get("id") == "deepx_m1_full"]
    if len(runs) != 1:
        raise ValueError("original_full_plan_not_unique")
    run = dict(runs[0])
    setup_id = str(run.get("setup_id") or run.get("expected_setup_id") or "")
    if setup_id != request["setup_id"]:
        raise ValueError("original_full_setup_mismatch")
    contract = read_json(contained_file(root, "deepx/deepx_m1/full/output_contract.json"))
    if contract.get("model_id") != request["model_id"] or contract.get("endpoint_mode") != "decoded_pre_nms":
        raise ValueError("original_full_contract_model_or_endpoint_mismatch")
    digest = str(contract.get("suite_artifact_sha256") or contract.get("artifact_sha256") or "").removeprefix("sha256:").lower()
    if digest != request["expected_dxnn_sha256"]:
        raise ValueError("staged_contract_dxnn_sha256_mismatch")
    # Keep model, preprocessing and endpoint semantics intact; use staged paths only.
    run["dxnn_path"] = request["staged_dxnn_relative"]
    run["contract_path"] = "deepx/deepx_m1/full/output_contract.json"
    run["model_id"] = request["model_id"]
    return root, dxnn, image, run


def run_probe(request: dict, work: Path) -> dict:
    output = work / "results"
    output.mkdir(exist_ok=True)
    root, dxnn, image, run = validate_stage(request, work)
    sys.path.insert(0, str(root))
    template = root / "benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("deepx_probe_suite_v27929", template)
    if spec is None or spec.loader is None:
        raise ValueError("probe_template_not_loadable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    function = getattr(module, "run_deepx_output_value_probe", None)
    if not callable(function):
        raise ValueError("v27929_template_probe_api_missing")
    args = SimpleNamespace(
        runs=1, warmup=0, energy_measurement_only=False,
        throughput_frames=0, prepared_input_manifest="",
        prepared_feed_image=str(image), benchmark_task="detection",
        quality_evidence_model_id=request["model_id"],
        quality_evidence_setup_id=request["setup_id"],
        validation_images="", validation_max_images=0,
    )
    write_json(output / "probe_progress.json", {"stage": "calling_single_output_probe", "diagnostic_only": True})
    print("REMOTE_PREFLIGHT=PASS; using fresh staged suite", flush=True)
    # Do not call the benchmark-suite main(), compiler, campaign or quality runner.
    result = function(root, dxnn, run, args, output)
    if not isinstance(result, dict) or not isinstance(result.get("status"), str):
        raise ValueError("probe_function_result_invalid")
    result["probe_source"] = {
        "probe_fix": "v27929", "staging_mode": STAGING_MODE,
        "original_run_id": request["original_run_id"],
        "model_id": request["model_id"], "setup_id": request["setup_id"],
        "original_remote_suite": request.get("original_remote_suite"),
        "remote_suite": str(root), "dxnn_path": str(dxnn),
        "dxnn_sha256": request["expected_dxnn_sha256"], "image_path": str(image),
        "image_sha256": request["expected_image_sha256"],
        "runtime_python": sys.executable, "source_results_modified": False,
        "compiler_invoked": False, "quality_policy_changed": False,
        "decoded_score_numeric_policy": "float32_probability_edges_abs_2pow_minus23_v1",
    }
    shutil.copyfile(image, output / ("probe_input" + image.suffix.lower()))
    write_json(output / "deepx_output_value_probe.json", result)
    write_json(output / "probe_progress.json", {"stage": "diagnostic_function_returned", "diagnostic_only": True,
                                               "probe_status": result["status"]})
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args(argv)
    request_path = args.request.resolve(strict=True)
    work = request_path.parent
    output = work / "results"
    output.mkdir(exist_ok=True)
    try:
        request = read_json(request_path)
        if request.get("schema") != SCHEMA or request.get("staging_mode") != STAGING_MODE:
            raise ValueError("staged_probe_request_required")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(work / "matplotlib"),
                   ONNX_SPLITPOINT_ARTIFACT_POLICY="cache_verify_only")
        os.chdir(work)
        if not args.runtime:
            # Preserve the venv/bin/python spelling; resolving its symlink would lose the venv.
            python = Path(request["runtime_venv"]).expanduser() / "bin/python"
            if not python.is_file() or not os.access(python, os.X_OK):
                raise ValueError(f"existing_deepx_runtime_python_missing:{python}")
            os.execve(str(python), [str(python), "-I", "-B", str(Path(__file__).resolve()),
                                   "--request", str(request_path), "--runtime"], env)
        os.environ.pop("PYTHONPATH", None)
        os.environ.pop("PYTHONHOME", None)
        os.environ.update(env)
        result = run_probe(request, work)
        print("PROBE_STATUS=" + result["status"], flush=True)
        print("PROBE_ERROR=" + str(result.get("error") or ""), flush=True)
        # A captured invalid-tensor diagnostic is useful evidence, NOT model acceptance.
        return 2 if result["status"] == "probe_setup_failed" else 0
    except Exception as exc:
        result = {"schema": "onnx-splitpoint/deepx-full-output-probe/v1", "diagnostic_only": True,
                  "status": "probe_setup_failed", "error": f"{type(exc).__name__}: {exc}"}
        write_json(output / "deepx_output_value_probe.json", result)
        (output / "probe_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        print("PROBE_STATUS=probe_setup_failed", flush=True)
        print("PROBE_ERROR=" + result["error"], flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
