#!/usr/bin/env python3
"""Create a read-only, hash-bound v2.79.7 YOLO11 R8B recovery receipt.

The source run is never modified or copied.  Only the already valid v2.79.6
Hailo-8/b067 composed invocation is terminal-eligible.  The v2.79.6 Hailo-10
Full row is retained as diagnostic provenance because it was measured by the
generic runner and must not satisfy the v2.79.7 Native-Full gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_VERSION = "2.79.6"
SOURCE_BUILD_ID = "v2.79.6-remaining-changes-yolo11-admission-closure"
TARGET_VERSION = "2.79.7"
TARGET_BUILD_ID = "v2.79.7-yolo11-six-path-runtime-identity-closure"
MODEL_ID = "yolo11l"
MODEL_SHA256 = "f0fcdf56a4ac24d87ec30c627170492ccad9db80486ec5694df6de65c1b3d147"
SCHEMA = "onnx-splitpoint/yolo11-r8b-recovery-manifest"
SCHEMA_VERSION = 1
IMPORTED_TERMINAL = "hailo8_b067_composed"
FRESH_REQUIRED = [
    "hailo8_full",
    "hailo10h_full",
    "deepx_full",
    "hailo10h_b067_composed",
    "deepx_b067_composed",
]


def _load_base():
    path = Path(__file__).resolve().with_name("verify_v2796_yolo11_r8b_gate.py")
    spec = importlib.util.spec_from_file_location(
        "_v2797_recovery_v2796_verifier", path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("v2796_verifier_import_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()


class RecoveryError(ValueError):
    pass


def _need(value: Any, reason: str) -> None:
    if not value:
        raise RecoveryError(reason)


def _mapping(value: Any, reason: str) -> dict[str, Any]:
    _need(isinstance(value, Mapping), reason)
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity(path: Path, root: Path) -> dict[str, Any]:
    info = path.stat()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": info.st_size,
    }


def _exact_row(run: Path, filename: str, run_id: str) -> dict[str, Any]:
    rows = base._rows(run, filename)
    _need(len(rows) == 1, f"source_result_row_count:{filename}:{len(rows)}")
    row = rows[0]
    _need(str(row.get("case_id") or "").strip().lower() == "b067", f"source_case_id:{filename}")
    _need(str(row.get("run_id") or "").strip() == run_id, f"source_run_id:{filename}")
    return row


def _runtime_target(
    run: Path, *, setup_id: str, expected_provider: str, run_id: str,
) -> dict[str, Any]:
    relative = (
        Path("models") / MODEL_ID / "benchmark_results"
        / f"remote_benchmark_status_{setup_id}.json"
    )
    path = base._safe_file(run, relative, f"source_remote_status_{setup_id}")
    payload = _mapping(
        base._json(run, relative, f"source_remote_status_{setup_id}"),
        f"source_remote_status_mapping:{setup_id}",
    )
    _need(str(payload.get("status") or "").lower() == "ok", f"source_remote_status:{setup_id}")
    gates = _mapping(payload.get("execution_gates"), f"source_execution_gates:{setup_id}")
    target = _mapping(gates.get("hardware_target"), f"source_hardware_target:{setup_id}")
    _need(target.get("id") == setup_id, f"source_hardware_setup:{setup_id}")
    aliases = {
        str(target.get("accelerator") or "").lower().replace("-", "_"),
        str(target.get("provider") or "").lower().replace("-", "_"),
        str(_mapping(target.get("runtime"), f"source_runtime_target:{setup_id}").get("provider") or "").lower().replace("-", "_"),
    }
    _need(expected_provider in aliases, f"source_hardware_provider:{setup_id}")
    run_ids = {str(value) for value in list(gates.get("hardware_run_ids") or [])}
    _need(run_id in run_ids, f"source_hardware_run_id:{setup_id}")
    canonical = {str(value) for value in list(payload.get("canonical_run_ids_with_rows") or [])}
    _need(run_id + "_auto" in canonical, f"source_canonical_result_missing:{setup_id}")
    return {
        "status": _identity(path, run),
        "setup_id": setup_id,
        "provider": expected_provider,
        "run_id": run_id,
    }


def _validate_hailo8_import(run: Path, part1_sha256: str) -> dict[str, Any]:
    filename = "benchmark_results_hailo8_to_trt_auto.json"
    row = _exact_row(run, filename, "hailo8_to_trt")
    _need(base._technical_success(row, "composed"), "source_hailo8_composed_not_measured")
    _need(str(row.get("provider") or "").lower() == "tensorrt", "source_hailo8_provider")
    _need(str(row.get("stage1_provider") or "").lower() == "hailo8", "source_hailo8_stage1")
    _need(str(row.get("stage2_provider") or "").lower() == "tensorrt", "source_hailo8_stage2")
    _need(str(row.get("full_provider") or "").lower() == "tensorrt", "source_hailo8_full_provider")
    tokens = _mapping(row.get("backend_tokens"), "source_hailo8_backend_tokens")
    _need(str(tokens.get("stage1") or "").lower() == "hailo8", "source_hailo8_stage1_token")
    _need(str(tokens.get("stage2") or "").lower() == "tensorrt", "source_hailo8_stage2_token")
    throughput = _mapping(row.get("throughput"), "source_hailo8_throughput")
    _need(throughput.get("mode") == "measured_streaming", "source_hailo8_throughput_mode")
    _need(throughput.get("streaming_impl") == "native_fifo", "source_hailo8_streaming_impl")
    _need(
        isinstance(throughput.get("fps_makespan"), (int, float))
        and not isinstance(throughput.get("fps_makespan"), bool)
        and float(throughput["fps_makespan"]) > 0.0,
        "source_hailo8_makespan_fps",
    )
    deployment = _mapping(row.get("deployment_contract"), "source_hailo8_deployment")
    hailo_contracts = _mapping(deployment.get("hailo_io_contracts"), "source_hailo8_contracts")
    part1_contract = _mapping(hailo_contracts.get("part1"), "source_hailo8_part1_contract")
    _need(part1_contract.get("backend") == "hailo", "source_hailo8_part1_backend")
    _need(
        str(part1_contract.get("artifact") or "").replace("\\", "/").endswith(
            "/b067/hailo/hailo8/part1/compiled.hef"
        ),
        "source_hailo8_part1_artifact",
    )
    native_trt = _mapping(row.get("native_tensorrt"), "source_hailo8_native_trt")
    _need(native_trt.get("used") is True, "source_hailo8_native_trt_unused")
    _need(int(native_trt.get("session_count") or 0) >= 1, "source_hailo8_native_trt_sessions")
    trt_contracts = _mapping(deployment.get("native_trt_io_contracts"), "source_hailo8_trt_contracts")
    part2_contract = _mapping(trt_contracts.get("part2"), "source_hailo8_part2_contract")
    _need(part2_contract.get("backend") == "native_tensorrt", "source_hailo8_part2_backend")

    receipt_rel = Path(
        "models/yolo11l/benchmark_set/legacy_suite/b067/hailo/hailo8/part1/"
        "hailo_hef_build_receipt.json"
    )
    artifact = base._hailo_success(
        run, receipt_rel, "hailo8", full=False,
        expected_source_sha256=part1_sha256,
    )
    result_rel = Path("models") / MODEL_ID / "benchmark_results" / filename
    result_path = base._safe_file(run, result_rel, "source_hailo8_result")
    log_rel = Path(
        "models/yolo11l/benchmark_results/remote_diagnostics/"
        "orin_nx_hailo8_01/logs/stdout.txt"
    )
    log_path = base._safe_file(run, log_rel, "source_hailo8_stdout")
    log = log_path.read_text(encoding="utf-8", errors="strict")
    _need("[hailo][runtime] hw_arch=hailo8 api=vstreams" in log, "source_hailo8_runtime_log")
    _need("streaming throughput: fps(makespan)=" in log, "source_hailo8_makespan_log")
    return {
        "terminal_eligible": True,
        "evidence_mode": "read_only_import",
        "source_case_id": "b067",
        "source_row_sha256": _canonical_sha256(row),
        "result": _identity(result_path, run),
        "runtime_target": _runtime_target(
            run, setup_id="orin_nx_hailo8_01",
            expected_provider="hailo8", run_id="hailo8_to_trt",
        ),
        "runtime_log": _identity(log_path, run),
        "artifact": artifact,
        "runtime_contract": {
            "run_id": "hailo8_to_trt",
            "backend": "hailo8_to_tensorrt",
            "stage1": "hailo8",
            "stage2": "native_tensorrt",
            "variant": "composed",
            "throughput_mode": "measured_streaming",
            "streaming_impl": "native_fifo",
            "fps_makespan": float(throughput["fps_makespan"]),
        },
    }


def build_manifest(source: Path) -> dict[str, Any]:
    run = base._find_run(source)
    manifest_path = base._safe_file(run, "run_manifest.json", "source_run_manifest")
    manifest = _mapping(
        base._json(run, "run_manifest.json", "source_run_manifest"),
        "source_run_manifest_mapping",
    )
    _need(manifest.get("run_id") == run.name, "source_run_id_mismatch")
    _need(manifest.get("profile_id") == "yolo11l_v2796_r8b_full_b067_gate", "source_profile_id")
    _need(manifest.get("tool_version") == SOURCE_VERSION, "source_tool_version")
    _need(manifest.get("current_tool_version") == SOURCE_VERSION, "source_current_tool_version")
    _need(manifest.get("current_workflow_version") == SOURCE_BUILD_ID, "source_build_id")

    model_rel = Path("models") / MODEL_ID / "model_manifest.json"
    model_path = base._safe_file(run, model_rel, "source_model_manifest")
    model = _mapping(base._json(run, model_rel, "source_model_manifest"), "source_model_mapping")
    observed = model.get("observed_model_sha256") or model.get("model_sha256")
    _need(base._sha(observed, "source_model") == MODEL_SHA256, "source_model_sha256")

    bindings = base._source_bindings(run)
    imported = _validate_hailo8_import(run, str(bindings["part1_sha256"]))

    diagnostic_result_rel = Path(
        "models/yolo11l/benchmark_results/benchmark_results_hailo10_auto.json"
    )
    diagnostic_result = base._safe_file(
        run, diagnostic_result_rel, "source_hailo10_generic_result",
    )
    diagnostic_row = _exact_row(run, diagnostic_result_rel.name, "hailo10")
    diagnostic = {
        "terminal_eligible": False,
        "evidence_mode": "diagnostic_only",
        "reason": "v2796_generic_full_runner_not_native_full",
        "source_case_id": "b067",
        "source_row_sha256": _canonical_sha256(diagnostic_row),
        "result": _identity(diagnostic_result, run),
        "observed_fps": (
            float(diagnostic_row["full_backend_throughput_fps"])
            if isinstance(diagnostic_row.get("full_backend_throughput_fps"), (int, float))
            and not isinstance(diagnostic_row.get("full_backend_throughput_fps"), bool)
            else None
        ),
    }

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_version": TARGET_VERSION,
        "target_build_id": TARGET_BUILD_ID,
        "source_version": SOURCE_VERSION,
        "source_build_id": SOURCE_BUILD_ID,
        "source_access_mode": "read_only_hash_bound_no_copy",
        "source_run_dir": str(run),
        "source_run_id": run.name,
        "source_run_manifest": _identity(manifest_path, run),
        "source_model_manifest": _identity(model_path, run),
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "source_bindings": bindings,
        "imported_terminal_count": 1,
        "imports": {IMPORTED_TERMINAL: imported},
        "diagnostic_only": {"hailo10h_full_generic": diagnostic},
        "fresh_required": list(FRESH_REQUIRED),
        "policy": {
            "only_terminal_import": IMPORTED_TERMINAL,
            "generic_full_rows_terminal_eligible": False,
            "fresh_native_full_required": [
                "native_full_hailo8",
                "native_full_hailo10h",
                "native_full_deepx",
            ],
            "source_artifact_index_trusted": False,
            "source_files_verified_individually": True,
        },
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RecoveryError("output_path_unsafe")
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False,
    ) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source = args.source_run.expanduser().resolve(strict=True)
        output = args.output.expanduser().resolve(strict=False)
        _need(source != output, "source_output_same_path")
        payload = build_manifest(source)
        _atomic_write(output, payload)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}:{exc}", file=sys.stderr)
        return 2
    print(f"V2797_RECOVERY_MANIFEST={output}")
    print(f"V2797_RECOVERY_MANIFEST_SHA256={_sha256(output)}")
    print("V2797_RECOVERY_IMPORT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
