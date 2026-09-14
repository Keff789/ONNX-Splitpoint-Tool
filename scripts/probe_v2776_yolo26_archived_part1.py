#!/usr/bin/env python3
from __future__ import annotations

"""Parse only the four archived v2.77.3 YOLO26s Part-1 ONNX files.

This is deliberately not a benchmark-set replay.  It reads the preserved
rejected cases, checks the managed Hailo-10 venv (including ``onnxsim``), and
replays the parser path in a new output directory.  Each case first uses the
normal automatic endpoint path.  Only an exact archived ``base_conv`` failure
may trigger one retry with fully attested DFC-visible output producers.  It
never generates candidates, builds a Full model, backfills a case, runs
calibration, or compiles a HEF.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import onnx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onnx_splitpoint_tool import __version__  # noqa: E402
from onnx_splitpoint_tool.benchmark.services import (  # noqa: E402
    BenchmarkGenerationExecutionService,
)
from onnx_splitpoint_tool.hailo_backend import (  # noqa: E402
    _managed_venv_child_env,
    _resolve_managed_venv_python,
    hailo_parse_check_auto,
)


EXPECTED_VERSION = "2.78.4"
CASE_IDS = ("b066", "b088", "b104", "b199")
EXPECTED_SOURCE_SHA256 = {
    "b066": "e4225dc1897d0da126656d229562d0ee1312a91069fdd3e63e73bacbc5aef282",
    "b088": "f828c02d144e792af6874f27b963342848873cf2158d6a0e307b83ed05243f58",
    "b104": "ecc5fe6d39809debf0f49c26a64cf8391e93a1fe81a1fd38707e06a4cb103f86",
    "b199": "3e644e2ec51e280cbf922fbedfc9c3c29d8ad222dfe35139d26066cd287133bd",
}


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


def _project_dfc_visible_end_nodes(
    model_path: Path,
    *,
    case_id: str,
    archived_end_nodes: list[str],
    identity_fix: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Attest archived endpoints and unwrap only tool-owned no-op Identities."""

    boundary = int(case_id[1:])
    model = onnx.load(str(model_path))
    direct_projection = (
        BenchmarkGenerationExecutionService
        ._graph_output_endpoint_projection(
            model,
            collapse_tool_identities=False,
        )
    )
    direct_end_nodes = list(direct_projection.get("end_node_names") or [])
    if not direct_projection:
        raise ValueError(
            f"{case_id}: archived endpoint projection is ambiguous or slot-unsafe"
        )
    if direct_end_nodes != archived_end_nodes:
        raise ValueError(
            f"{case_id}: archived endpoint attestation mismatch: "
            f"manifest={archived_end_nodes!r}:graph={direct_end_nodes!r}"
        )
    effective_projection = (
        BenchmarkGenerationExecutionService
        ._graph_output_endpoint_projection(
            model,
            collapse_tool_identities=True,
            boundary=boundary,
            identity_fix=identity_fix,
        )
    )
    if not effective_projection:
        raise ValueError(
            f"{case_id}: DFC-visible endpoint projection is ambiguous or slot-unsafe"
        )
    projection = list(effective_projection.get("outputs") or [])
    if sum(
        bool(item.get("transparent_tool_identity")) for item in projection
    ) != 2:
        raise ValueError(
            f"{case_id}: expected exactly two tool-owned Identity outputs"
        )
    effective_end_nodes = list(
        effective_projection.get("end_node_names") or []
    )
    return effective_end_nodes, projection


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
        source_onnx_sha256 = _sha256(model_path)
        if source_onnx_sha256 != EXPECTED_SOURCE_SHA256.get(case_id):
            raise ValueError(
                f"{case_id}: archived Part-1 ONNX SHA-256 mismatch: "
                f"expected={EXPECTED_SOURCE_SHA256.get(case_id)}:"
                f"actual={source_onnx_sha256}"
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
        if resolution.get("strategy") != "explicit_declared_output_producers":
            raise ValueError(
                f"{case_id}: archived base-conv resolution strategy mismatch"
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
        if not BenchmarkGenerationExecutionService._is_exact_hailo_base_conv_error(
            previous_error,
            boundary=int(case_id[1:]),
        ):
            raise ValueError(
                f"{case_id}: archived case is not the expected base_conv failure"
            )

        identity_fix = hailo.get("part1_feature_splitter_identity_fix")
        target_identity_fix = (
            hailo10.get("part1_feature_splitter_identity_fix")
            if isinstance(hailo10, Mapping)
            else None
        )
        if not isinstance(identity_fix, Mapping):
            raise ValueError(f"{case_id}: identity-fix provenance missing")
        if (
            not isinstance(target_identity_fix, Mapping)
            or dict(target_identity_fix) != dict(identity_fix)
        ):
            raise ValueError(f"{case_id}: identity-fix provenance conflict")
        effective_end_nodes, endpoint_projection = (
            _project_dfc_visible_end_nodes(
                model_path,
                case_id=case_id,
                archived_end_nodes=end_nodes,
                identity_fix=identity_fix,
            )
        )

        prepared.append(
            {
                "case_id": case_id,
                "model_path": model_path,
                "source_onnx_sha256": source_onnx_sha256,
                "archived_end_node_names": end_nodes,
                "end_node_names": effective_end_nodes,
                "endpoint_projection": endpoint_projection,
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


def _classify_failure(error: str, *, case_id: str) -> str:
    lowered = error.lower()
    if BenchmarkGenerationExecutionService._is_exact_hailo_base_conv_error(
        error,
        boundary=int(case_id[1:]),
    ):
        return "base_conv_resolution_failed"
    if "base_conv" in lowered:
        return "unexpected_base_conv_failure"
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


def _attempt_payload(
    result: Any,
    *,
    case_id: str,
    mode: str,
    end_node_names: list[str] | None,
) -> dict[str, Any]:
    error = str(result.error or "")
    return {
        "mode": mode,
        "end_node_names": (
            list(end_node_names) if end_node_names is not None else None
        ),
        "ok": bool(result.ok),
        "failure_kind": (
            None
            if result.ok
            else _classify_failure(error, case_id=case_id)
        ),
        "elapsed_s": float(result.elapsed_s),
        "backend": str(result.backend),
        "parsed_har": result.har_path,
        "fixed_onnx": result.fixed_onnx_path,
        "error": error[:3000] or None,
    }


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
            "schema": "onnx-splitpoint/yolo26-v2776-direct-part1-parse/v1",
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
    parser_attempt_count = 0
    retry_case_count = 0
    for item in prepared:
        case_id = str(item["case_id"])
        model_path = Path(item["model_path"])
        initial_result = hailo_parse_check_auto(
            model_path,
            backend="venv",
            hw_arch="hailo10",
            net_name=f"yolo26s_part1_{case_id}_v2776_auto",
            outdir=output_dir / case_id / "automatic",
            fixup=True,
            add_conv_defaults=True,
            save_har=True,
            disable_rt_metadata_extraction=True,
            wsl_venv_activate="auto",
            wsl_timeout_s=int(args.timeout_s),
        )
        parser_attempt_count += 1
        attempts = [
            _attempt_payload(
                initial_result,
                case_id=case_id,
                mode="automatic_endpoints",
                end_node_names=None,
            )
        ]
        result = initial_result
        initial_error = str(initial_result.error or "")
        source_sha256_after_initial = _sha256(model_path)
        mutated = source_sha256_after_initial != item["source_onnx_sha256"]
        source_mutation_detected = source_mutation_detected or mutated
        if (
            not initial_result.ok
            and not mutated
            and _classify_failure(initial_error, case_id=case_id)
            == "base_conv_resolution_failed"
        ):
            retry_case_count += 1
            retry_result = hailo_parse_check_auto(
                model_path,
                backend="venv",
                hw_arch="hailo10",
                net_name=f"yolo26s_part1_{case_id}_v2776_explicit",
                outdir=output_dir / case_id / "explicit_endpoints",
                fixup=True,
                add_conv_defaults=True,
                save_har=True,
                disable_rt_metadata_extraction=True,
                end_node_names=list(item["end_node_names"]),
                wsl_venv_activate="auto",
                wsl_timeout_s=int(args.timeout_s),
            )
            parser_attempt_count += 1
            attempts.append(
                _attempt_payload(
                    retry_result,
                    case_id=case_id,
                    mode="attested_dfc_visible_endpoints",
                    end_node_names=list(item["end_node_names"]),
                )
            )
            result = retry_result
        source_sha256_after = _sha256(model_path)
        mutated = mutated or source_sha256_after != item["source_onnx_sha256"]
        source_mutation_detected = source_mutation_detected or mutated
        error = str(result.error or "")
        rows.append(
            {
                "case_id": case_id,
                "ok": bool(result.ok),
                "failure_kind": (
                    None
                    if result.ok
                    else _classify_failure(error, case_id=case_id)
                ),
                "elapsed_s": float(result.elapsed_s),
                "backend": str(result.backend),
                "source_onnx": str(model_path),
                "source_onnx_sha256": item["source_onnx_sha256"],
                "source_onnx_unchanged": not mutated,
                "archived_end_node_names": list(
                    item["archived_end_node_names"]
                ),
                "end_node_names": list(item["end_node_names"]),
                "endpoint_projection": list(item["endpoint_projection"]),
                "previous_error": item["previous_error"],
                "attempts": attempts,
                "retry_attempted": len(attempts) == 2,
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
        "schema": "onnx-splitpoint/yolo26-v2776-direct-part1-parse/v1",
        "status": "PASS" if evidence_complete else "FAIL",
        "tool_version": __version__,
        "mode": "parse_only_dfc_visible_output_producers",
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
        "parser_attempt_count": parser_attempt_count,
        "retry_case_count": retry_case_count,
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
