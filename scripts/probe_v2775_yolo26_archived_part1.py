#!/usr/bin/env python3
from __future__ import annotations

"""Parse only the four archived v2.77.3 YOLO26s Part-1 ONNX files.

This is deliberately not a benchmark-set replay.  It reads the preserved
rejected cases, checks the managed Hailo-10 venv (including ``onnxsim``), and
retries exactly the four explicit-output-producer parser calls in a new output
directory.  It never generates candidates, builds a Full model, backfills a
case, runs calibration, or compiles a HEF.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onnx_splitpoint_tool import __version__  # noqa: E402
from onnx_splitpoint_tool.hailo_backend import (  # noqa: E402
    _managed_venv_child_env,
    _resolve_managed_venv_python,
    hailo_parse_check_auto,
)


EXPECTED_VERSION = "2.77.5"
CASE_IDS = ("b066", "b088", "b104", "b199")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return dict(value)


def _prepare_cases(run_dir: Path) -> list[dict[str, Any]]:
    archive_root = (
        run_dir
        / "models"
        / "yolo26s"
        / "benchmark_set"
        / "legacy_suite"
        / "_rejected_cases"
    )
    prepared: list[dict[str, Any]] = []
    for case_id in CASE_IDS:
        case_dir = archive_root / case_id
        manifest_path = case_dir / "split_manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"{case_id}: regular split_manifest.json missing: "
                f"{manifest_path}"
            )
        manifest = _read_mapping(manifest_path)
        hailo = manifest.get("hailo")
        if not isinstance(hailo, Mapping):
            raise ValueError(f"{case_id}: hailo manifest section missing")
        relative_model = str(hailo.get("part1_accel_model") or "").strip()
        if (
            not relative_model
            or Path(relative_model).name != relative_model
            or Path(relative_model).is_absolute()
        ):
            raise ValueError(
                f"{case_id}: invalid hailo.part1_accel_model="
                f"{relative_model!r}"
            )
        model_path = (case_dir / relative_model).resolve(strict=True)
        case_root = case_dir.resolve(strict=True)
        if not model_path.is_relative_to(case_root) or not model_path.is_file():
            raise ValueError(
                f"{case_id}: Part-1 ONNX escapes the archived case directory"
            )

        hefs = hailo.get("hefs")
        hailo10 = hefs.get("hailo10") if isinstance(hefs, Mapping) else None
        resolution = (
            hailo10.get("part1_base_conv_resolution")
            if isinstance(hailo10, Mapping)
            else None
        )
        if not isinstance(resolution, Mapping) or resolution.get("attempted") is not True:
            raise ValueError(
                f"{case_id}: archived explicit base-conv resolution missing"
            )
        end_nodes_raw = resolution.get("end_node_names")
        if not isinstance(end_nodes_raw, list):
            raise ValueError(f"{case_id}: explicit end-node list missing")
        end_nodes = [str(value).strip() for value in end_nodes_raw]
        if (
            not end_nodes
            or any(not value for value in end_nodes)
            or len(set(end_nodes)) != len(end_nodes)
        ):
            raise ValueError(f"{case_id}: explicit end-node list is invalid")
        previous_error = str(
            hailo10.get("part1_error") if isinstance(hailo10, Mapping) else ""
        )
        if "base_conv" not in previous_error:
            raise ValueError(
                f"{case_id}: archived case is not the expected base_conv failure"
            )

        prepared.append(
            {
                "case_id": case_id,
                "model_path": model_path,
                "source_onnx_sha256": _sha256(model_path),
                "end_node_names": end_nodes,
                "previous_error": previous_error.splitlines()[0][:500],
            }
        )
    return prepared


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Preserved v2.77.3 canary EvaluationRun; used read-only.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for four parse results and parse_verdict.json.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=600,
        help="Hard timeout for each direct parser call (default: 600).",
    )
    return parser


def _classify_failure(error: str) -> str:
    lowered = error.lower()
    if "base_conv" in lowered:
        return "base_conv_resolution_failed"
    if "onnxsim" in lowered:
        return "managed_venv_path_or_onnxsim_failed"
    if "timed out" in lowered:
        return "parse_timeout"
    return "parse_failed"


def _write_verdict(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    path = output_dir / "parse_verdict.json"
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout_s < 10:
        raise SystemExit("--timeout-s must be at least 10")

    run_arg = args.run_dir.expanduser()
    if run_arg.is_symlink():
        raise SystemExit("Source run must not be a symlink")
    run_dir = run_arg.resolve(strict=True)
    if not run_dir.is_dir():
        raise SystemExit(f"Source run is not a directory: {run_dir}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise SystemExit(f"Output directory already exists: {output_dir}")
    if output_dir.is_relative_to(run_dir):
        raise SystemExit("Output directory must be outside the source run")
    output_dir.mkdir(parents=True, exist_ok=False)

    preflight_errors: list[str] = []
    prepared: list[dict[str, Any]] = []
    managed_profile = ""
    managed_python = ""
    managed_python_resolved_target = ""
    managed_bin_dir = ""
    projected_path_head = ""
    expected_onnxsim_path = ""
    expected_onnxsim_exists = False
    expected_onnxsim_executable = False
    onnxsim_path = ""
    onnxsim_origin = "missing"
    warnings: list[str] = []
    if __version__ != EXPECTED_VERSION:
        preflight_errors.append(
            "tool_version_mismatch:"
            f"expected={EXPECTED_VERSION}:actual={__version__}"
        )
    try:
        prepared = _prepare_cases(run_dir)
    except Exception as exc:
        preflight_errors.append(f"archive_preflight:{type(exc).__name__}:{exc}")
    try:
        profile_id, python_path, _activate = _resolve_managed_venv_python(
            hw_arch="hailo10",
            venv_activate="auto",
        )
        managed_profile = str(profile_id)
        lexical_python = Path(
            os.path.abspath(
                os.fspath(Path(str(python_path)).expanduser())
            )
        )
        managed_python = str(lexical_python)
        managed_python_resolved_target = str(lexical_python.resolve())
        managed_bin_dir = str(lexical_python.parent)
        child_env = _managed_venv_child_env(python_path)
        projected_path_head = str(child_env.get("PATH") or "").split(
            os.pathsep,
            1,
        )[0]
        if projected_path_head != managed_bin_dir:
            raise RuntimeError(
                "managed venv PATH projection drift: "
                f"expected {managed_bin_dir}, got {projected_path_head}"
            )
        expected_onnxsim = lexical_python.parent / "onnxsim"
        expected_onnxsim_path = str(expected_onnxsim)
        expected_onnxsim_exists = expected_onnxsim.is_file()
        expected_onnxsim_executable = os.access(expected_onnxsim, os.X_OK)
        resolved_onnxsim = shutil.which(
            "onnxsim",
            path=str(child_env.get("PATH") or ""),
        )
        if resolved_onnxsim:
            onnxsim_lexical = Path(
                os.path.abspath(
                    os.fspath(Path(str(resolved_onnxsim)).expanduser())
                )
            )
            onnxsim_path = str(onnxsim_lexical)
            onnxsim_origin = (
                "managed_venv"
                if onnxsim_lexical.parent == lexical_python.parent
                else "inherited_path"
            )
        else:
            warnings.append(
                "onnxsim_not_available: optional DFC simplifier recovery "
                "is unavailable; parser result is scoped to the currently "
                "provisioned managed environment"
            )
    except Exception as exc:
        preflight_errors.append(
            f"managed_venv_preflight:{type(exc).__name__}:{exc}"
        )

    if preflight_errors or len(prepared) != len(CASE_IDS):
        payload = {
            "schema": "onnx-splitpoint/yolo26-v2775-direct-part1-parse/v1",
            "status": "PREFLIGHT_FAIL",
            "tool_version": __version__,
            "source_run_dir": str(run_dir),
            "requested_cases": list(CASE_IDS),
            "preflight_errors": preflight_errors,
            "managed_profile": managed_profile,
            "managed_python": managed_python,
            "managed_python_resolved_target": managed_python_resolved_target,
            "managed_bin_dir": managed_bin_dir,
            "projected_path_head": projected_path_head,
            "expected_onnxsim_path": expected_onnxsim_path,
            "expected_onnxsim_exists": expected_onnxsim_exists,
            "expected_onnxsim_executable": expected_onnxsim_executable,
            "onnxsim_path": onnxsim_path,
            "onnxsim_origin": onnxsim_origin,
            "warnings": warnings,
            "parser_started": False,
            "cases": [],
        }
        verdict_path = _write_verdict(output_dir, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        print(f"PARSE_VERDICT_JSON={verdict_path}")
        print("DIRECT_PARSE_RESULT=PREFLIGHT_FAIL")
        return 2

    rows: list[dict[str, Any]] = []
    source_mutation_detected = False
    for item in prepared:
        case_id = str(item["case_id"])
        model_path = Path(item["model_path"])
        result = hailo_parse_check_auto(
            model_path,
            backend="venv",
            hw_arch="hailo10",
            net_name=f"yolo26s_part1_{case_id}_v2775_parse",
            outdir=output_dir / case_id,
            fixup=True,
            add_conv_defaults=True,
            save_har=True,
            disable_rt_metadata_extraction=True,
            end_node_names=list(item["end_node_names"]),
            wsl_venv_activate="auto",
            wsl_timeout_s=int(args.timeout_s),
        )
        source_sha256_after = _sha256(model_path)
        mutated = source_sha256_after != item["source_onnx_sha256"]
        source_mutation_detected = source_mutation_detected or mutated
        error = str(result.error or "")
        rows.append(
            {
                "case_id": case_id,
                "ok": bool(result.ok),
                "failure_kind": (
                    None if result.ok else _classify_failure(error)
                ),
                "elapsed_s": float(result.elapsed_s),
                "backend": str(result.backend),
                "source_onnx": str(model_path),
                "source_onnx_sha256": item["source_onnx_sha256"],
                "source_onnx_unchanged": not mutated,
                "end_node_names": list(item["end_node_names"]),
                "previous_error": item["previous_error"],
                "parsed_har": result.har_path,
                "fixed_onnx": result.fixed_onnx_path,
                "error": error[:3000] or None,
            }
        )

    passed_cases = sum(bool(row["ok"]) for row in rows)
    base_conv_terminal_cases = sum(
        row["failure_kind"] == "base_conv_resolution_failed"
        for row in rows
    )
    expected_outcomes_only = all(
        bool(row["ok"])
        or row["failure_kind"] == "base_conv_resolution_failed"
        for row in rows
    )
    evidence_complete = (
        len(rows) == len(CASE_IDS)
        and expected_outcomes_only
        and not source_mutation_detected
    )
    if passed_cases == len(CASE_IDS):
        capability_status = "ALL_PARSE_SUPPORTED"
    elif base_conv_terminal_cases == len(CASE_IDS):
        capability_status = (
            "ALL_BASE_CONV_UNSUPPORTED_IN_PROVISIONED_ENVIRONMENT"
        )
    elif passed_cases + base_conv_terminal_cases == len(CASE_IDS):
        capability_status = "MIXED_SUPPORTED_AND_BASE_CONV_UNSUPPORTED"
    else:
        capability_status = "UNEXPECTED_PARSE_FAILURE"
    payload = {
        "schema": "onnx-splitpoint/yolo26-v2775-direct-part1-parse/v1",
        "status": "PASS" if evidence_complete else "FAIL",
        "tool_version": __version__,
        "mode": "parse_only_explicit_output_producers",
        "source_run_dir": str(run_dir),
        "requested_cases": list(CASE_IDS),
        "managed_profile": managed_profile,
        "managed_python": managed_python,
        "managed_python_resolved_target": managed_python_resolved_target,
        "managed_bin_dir": managed_bin_dir,
        "projected_path_head": projected_path_head,
        "expected_onnxsim_path": expected_onnxsim_path,
        "expected_onnxsim_exists": expected_onnxsim_exists,
        "expected_onnxsim_executable": expected_onnxsim_executable,
        "onnxsim_path": onnxsim_path,
        "onnxsim_origin": onnxsim_origin,
        "warnings": warnings,
        "parser_started": True,
        "parsed_cases": len(rows),
        "passed_cases": passed_cases,
        "base_conv_terminal_cases": base_conv_terminal_cases,
        "capability_status": capability_status,
        "evidence_complete": evidence_complete,
        "source_mutation_detected": source_mutation_detected,
        "cases": rows,
    }
    verdict_path = _write_verdict(output_dir, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"PARSE_VERDICT_JSON={verdict_path}")
    print(f"DIRECT_PARSE_RESULT={payload['status']}")
    print(f"CAPABILITY_STATUS={capability_status}")
    if source_mutation_detected:
        return 2
    return 0 if evidence_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
