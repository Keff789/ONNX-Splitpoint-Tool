#!/usr/bin/env python3
"""Hash-bind one Native split workload before u.RECS sampling starts.

The successful performance command contract is the only source of runtime
paths and options.  This script performs every expensive SHA-256 operation,
then emits a fresh nonce-bound attestation understood by the energy collector.
The measured runner subsequently reads only that small attestation and checks
lightweight file-stat identities; it must not hash/build/probe/dump in-window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.native_command_contract import (
    SPLIT_ENERGY_PREFLIGHT_STDOUT_MARKER,
    SPLIT_ENERGY_WORKLOAD_BINDING_SCHEMA,
    canonical_json_sha256,
    deepx_preprocess_binding,
    hailo8_preprocess_binding,
    hailo10_preprocess_binding,
    seal_split_energy_preflight_attestation,
    verify_native_energy_command_contract,
    verify_native_split_part2_input_contract,
)
from onnx_splitpoint_tool.native_detection_postprocess import (
    verify_detection_completion_execution_contract,
)


def _strict_json_text(text: str, *, label: str) -> dict[str, Any]:
    duplicate = False

    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                duplicate = True
            value[key] = item
        return value

    value = json.loads(text, object_pairs_hook=_object)
    if duplicate:
        raise ValueError(f"{label} contains duplicate JSON object keys")
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return dict(value)


def _read_json(path: Path) -> dict[str, Any]:
    return _strict_json_text(
        path.read_text(encoding="utf-8"), label="command contract JSON",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_runner(contract: Mapping[str, Any], tool_root: Path) -> Path:
    raw = Path(str(contract.get("runner") or "")).expanduser()
    return raw.resolve() if raw.is_absolute() else (tool_root / raw).resolve()


def _file_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _binding_for_contract(
    contract: Mapping[str, Any],
    *,
    runner_path: Path,
    verified_files: list[dict[str, Any]],
) -> dict[str, Any]:
    backend = str(contract.get("backend") or "").strip().lower()
    options = dict(contract.get("runtime_options") or {})
    boundary = dict(contract.get("boundary_contract") or {})
    artifacts = {
        str(name): {
            **dict(raw),
            "path": str(Path(str(raw.get("path") or "")).expanduser().resolve()),
        }
        for name, raw in dict(contract.get("artifacts") or {}).items()
        if isinstance(raw, Mapping)
    }
    resolved_input = str(Path(str(contract.get("input_image") or "")).expanduser().resolve())
    resolved_python = str(
        Path(str(contract.get("python_executable") or "")).expanduser().resolve()
    )
    source_warmup = int(options.get("warmup") or 0)
    source_dump_outputs = bool(options.get("dump_outputs"))
    source_dump_boundary = bool(options.get("dump_boundary"))
    source_build = bool(options.get("build"))
    energy_options = dict(options)
    energy_options.update({
        "warmup": 0,
        "dump_outputs": False,
        "dump_boundary": False,
        "build": False,
    })
    binding: dict[str, Any] = {
        "schema": SPLIT_ENERGY_WORKLOAD_BINDING_SCHEMA,
        "schema_version": 1,
        "workload_supported": True,
        "backend": backend,
        "model": str(contract.get("model") or ""),
        "setup_id": str(contract.get("setup_id") or ""),
        "comparison_backend": str(contract.get("comparison_backend") or ""),
        "command_contract_sha256": str(contract.get("contract_sha256") or ""),
        "runner": str(runner_path),
        "python_executable": resolved_python,
        "benchmark_set": str(contract.get("benchmark_set") or ""),
        "case": str(contract.get("case") or ""),
        "precision": str(contract.get("precision") or ""),
        "hw_arch": str(contract.get("hw_arch") or ""),
        "input_image": resolved_input,
        "input_image_sha256": str(contract.get("input_image_sha256") or ""),
        "artifacts": artifacts,
        "runtime_options": energy_options,
        "boundary_contract": boundary,
        "runtime_boundary_evidence": dict(
            contract.get("runtime_boundary_evidence") or {}
        ),
        "prepared_input_contract": dict(contract.get("prepared_input_contract") or {}),
        "verified_files": verified_files,
        "energy_overrides": {
            "source_warmup": source_warmup,
            "effective_warmup": 0,
            "source_dump_outputs": source_dump_outputs,
            "effective_dump_outputs": False,
            "source_dump_boundary": source_dump_boundary,
            "effective_dump_boundary": False,
            "source_build": source_build,
            "effective_build": False,
            "rationale": "exclude uncounted warmup/build/dump work from collector window",
        },
        "completed_work_units_contract": {
            "exact": True,
            "source": "runtime_completed_frames",
            "warmup_counted": False,
        },
    }
    if str(options.get("task") or "").strip().lower() == "detection":
        try:
            verified_completion = (
                verify_detection_completion_execution_contract(
                    options.get("completion_execution_contract")
                )
            )
            declared_completion_sha256 = str(
                options.get("completion_execution_contract_sha256") or ""
            ).strip().lower()
            if declared_completion_sha256 != str(
                verified_completion.get("contract_sha256") or ""
            ).strip().lower():
                raise ValueError(
                    "completion_execution_contract_sha256_mismatch"
                )
        except Exception as exc:
            binding.update({
                "workload_supported": False,
                "unsupported_reason": (
                    "split_energy_detection_completion_contract_"
                    f"invalid:{type(exc).__name__}:{exc}"
                ),
            })
            return binding
        energy_options["completion_execution_contract"] = (
            verified_completion
        )
        energy_options["completion_execution_contract_sha256"] = str(
            verified_completion.get("contract_sha256") or ""
        )
    if backend == "hailo8_to_trt":
        python_detection = bool(
            str(options.get("task") or "") == "detection"
            and str(options.get("producer_impl") or "")
            == "hailo8_python_vstreams_fifo"
        )
        if options.get("energy_prepared_feed_capable") is not True:
            binding.update({
                "workload_supported": False,
                "unsupported_reason": (
                    "hailo8_split_energy_legacy_executable_lacks_prepared_feed_mode"
                ),
            })
        prepared_contract = contract.get("prepared_input_contract")
        prepared_runtime_compatible = bool(
            isinstance(prepared_contract, Mapping)
            and list(prepared_contract.get("shape") or [])
            and (
                (
                    str(prepared_contract.get("dtype") or "")
                    in {"uint8", "float32"}
                    and str(prepared_contract.get("layout") or "")
                    in {"HWC", "CHW", "NHWC", "NCHW"}
                )
                if python_detection
                else (
                    str(prepared_contract.get("dtype") or "") == "uint8"
                    and str(prepared_contract.get("layout") or "") == "HWC"
                )
            )
        )
        if (
            options.get("prepared_input_bound") is not True
            or not isinstance(artifacts.get("prepared_input"), Mapping)
            or not prepared_runtime_compatible
        ):
            binding.update({
                "workload_supported": False,
                "unsupported_reason": "hailo8_split_energy_bound_prepared_input_missing",
            })
        if not isinstance(prepared_contract, Mapping) or not hailo8_preprocess_binding(
            options, prepared_contract,
        ):
            binding.update({
                "workload_supported": False,
                "unsupported_reason": "hailo8_split_energy_task_preprocess_binding_missing_or_inconsistent",
            })
        if python_detection:
            runtime_boundary = dict(
                contract.get("runtime_boundary_evidence") or {}
            )
            completion_contract = options.get(
                "completion_execution_contract"
            )
            if (
                runtime_boundary.get("status")
                != "exact_runtime_boundary_verified"
                or int(runtime_boundary.get("output_count") or 0) != 1
                or not str(runtime_boundary.get("output_name") or "")
                or not list(runtime_boundary.get("output_shape") or [])
                or not str(runtime_boundary.get("output_dtype") or "")
            ):
                binding.update({
                    "workload_supported": False,
                    "unsupported_reason": (
                        "hailo8_split_energy_exact_runtime_boundary_missing"
                    ),
                })
            if not isinstance(completion_contract, Mapping) or len(str(
                completion_contract.get("contract_sha256") or ""
            )) != 64:
                binding.update({
                    "workload_supported": False,
                    "unsupported_reason": (
                        "hailo8_split_energy_completion_contract_missing"
                    ),
                })
            binding["completed_work_units_contract"] = {
                "exact": True,
                "source": "runtime_completed_detection_counter",
                "warmup_counted": False,
                "completion_boundary": (
                    "same_hotloop_completed_task_sentinel"
                ),
            }
        binding["runtime_options"]["reuse_preprocessed_input"] = True
        binding["execution_mode"] = (
            "bound_python_vstreams_completed_detection"
            if python_detection else "bound_native_executable_direct"
        )
    elif backend == "hailo10h_to_trt":
        producer = str(options.get("producer_impl") or "").strip().lower()
        if producer not in {"auto", "async_fifo"}:
            binding.update({
                "workload_supported": False,
                "unsupported_reason": "hailo10_split_energy_requires_probe_free_async_fifo_contract",
            })
        if "part1_onnx_used" not in options:
            binding.update({
                "workload_supported": False,
                "unsupported_reason": (
                    "hailo10_split_energy_legacy_contract_missing_part1_onnx_binding"
                ),
            })
        elif options.get("part1_onnx_used") is True and "part1_onnx" not in artifacts:
            binding.update({
                "workload_supported": False,
                "unsupported_reason": "hailo10_split_energy_part1_onnx_artifact_missing",
            })
        if not list(options.get("canonical_input_slot_names") or []):
            binding.update({
                "workload_supported": False,
                "unsupported_reason": "hailo10_split_energy_canonical_input_slots_missing",
            })
        if not list(options.get("canonical_output_slot_names") or []):
            binding.update({
                "workload_supported": False,
                "unsupported_reason": "hailo10_split_energy_canonical_output_slots_missing",
            })
        prepared_contract = contract.get("prepared_input_contract")
        prepared_entries = (
            list(prepared_contract.get("entries") or [])
            if isinstance(prepared_contract, Mapping) else []
        )
        prepared_names = [
            str(entry.get("artifact_name") or "")
            for entry in prepared_entries if isinstance(entry, Mapping)
        ]
        if (
            options.get("prepared_input_bound") is not True
            or not prepared_entries
            or any(not name or name not in artifacts for name in prepared_names)
        ):
            binding.update({
                "workload_supported": False,
                "unsupported_reason": "hailo10_split_energy_bound_prepared_input_missing",
            })
        if not isinstance(prepared_contract, Mapping) or not hailo10_preprocess_binding(
            options, prepared_contract,
        ):
            binding.update({
                "workload_supported": False,
                "unsupported_reason": "hailo10_split_energy_task_preprocess_binding_missing_or_inconsistent",
            })
        binding["execution_mode"] = "bound_python_async_fifo_no_probe"
    elif backend == "deepx_to_trt":
        prepared = artifacts.get("prepared_input") or {}
        input_contract = contract.get("prepared_input_contract")
        if (
            not isinstance(prepared, Mapping)
            or not str(prepared.get("path") or "")
            or not isinstance(input_contract, Mapping)
            or not list(input_contract.get("shape") or [])
            or not str(input_contract.get("dtype") or "")
        ):
            binding.update({
                "workload_supported": False,
                "unsupported_reason": (
                    "deepx_split_energy_legacy_contract_missing_bound_prepared_input"
                ),
            })
        preprocess_ok = bool(
            isinstance(input_contract, Mapping)
            and deepx_preprocess_binding(options, input_contract)[0]
        )
        if not preprocess_ok:
            binding.update({
                "workload_supported": False,
                "unsupported_reason": "deepx_split_energy_task_preprocess_binding_missing_or_inconsistent",
            })
        binding["prepared_input_contract"] = dict(input_contract or {})
        binding["execution_mode"] = "bound_prepared_input_no_candidate_probe"
    else:
        binding.update({
            "workload_supported": False,
            "unsupported_reason": "split_energy_backend_unsupported",
        })
    return binding


def build_attestation(
    contract_path: Path | None = None,
    *,
    contract_payload: Mapping[str, Any] | None = None,
    nonce: str,
    expected_contract_sha256: str,
    expected_preflight_script_sha256: str,
    tool_root: Path,
    valid_for_s: float,
    preflight_script_path: Path | None = None,
) -> tuple[dict[str, Any], int]:
    started_ns = time.time_ns()
    raw: dict[str, Any] = {}
    artifact_rows: list[dict[str, Any]] = []
    try:
        own_path = (preflight_script_path or Path(__file__)).expanduser().resolve()
        wanted_own_sha = str(expected_preflight_script_sha256 or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", wanted_own_sha) is None:
            raise RuntimeError("expected_preflight_script_sha256_invalid")
        actual_own_sha = _sha256_file(own_path)
        if actual_own_sha != wanted_own_sha:
            raise RuntimeError("preflight_script_sha256_mismatch")
        artifact_rows.append({
            "label": "preflight_script",
            "path": str(own_path),
            "expected_sha256": wanted_own_sha,
            "actual_sha256": actual_own_sha,
            "status": "pass",
        })
        if (contract_path is None) == (contract_payload is None):
            raise ValueError("provide exactly one contract path or inline payload")
        raw = (
            _read_json(contract_path)
            if contract_path is not None else dict(contract_payload or {})
        )
        payload_sha256 = canonical_json_sha256(raw)
        contract_source = (
            "file:" + str(contract_path.resolve())
            if contract_path is not None else "inline_verified_contract_object"
        )
        contract, reason = verify_native_energy_command_contract(raw)
        if contract is None:
            raise RuntimeError(reason)
        part2_metadata, part2_reason = (
            verify_native_split_part2_input_contract(contract)
        )
        if part2_metadata is None:
            raise RuntimeError(part2_reason)
        actual_contract_sha = str(contract.get("contract_sha256") or "").lower()
        wanted = str(expected_contract_sha256 or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", wanted) is None:
            raise RuntimeError("expected_command_contract_sha256_invalid")
        if actual_contract_sha != wanted:
            raise RuntimeError("expected_command_contract_sha256_mismatch")

        runner_path = _resolve_runner(contract, tool_root)
        checks: list[tuple[str, Path, str, int | None]] = [
            ("runner", runner_path, str(contract.get("runner_sha256") or ""), None),
            (
                "input_image",
                Path(str(contract.get("input_image") or "")).expanduser(),
                str(contract.get("input_image_sha256") or ""),
                None,
            ),
        ]
        downstream_annotation_artifacts = {
            "semantic_output_manifest",
            "semantic_boundary_manifest",
        }
        for name, row in dict(contract.get("artifacts") or {}).items():
            if str(name) in downstream_annotation_artifacts:
                continue
            if isinstance(row, Mapping):
                checks.append((
                    "artifact:" + str(name),
                    Path(str(row.get("path") or "")).expanduser(),
                    str(row.get("sha256") or ""),
                    int(row.get("size_bytes"))
                    if isinstance(row.get("size_bytes"), int)
                    and not isinstance(row.get("size_bytes"), bool)
                    and int(row.get("size_bytes")) > 0 else None,
                ))
        boundary = dict(contract.get("boundary_contract") or {})
        if str(boundary.get("metadata_path") or ""):
            checks.append((
                "boundary_metadata",
                Path(str(boundary.get("metadata_path") or "")).expanduser(),
                str(boundary.get("metadata_sha256") or ""),
                None,
            ))
        # Quality/Semantics artifacts are downstream annotations.  They may
        # still be present in a successful performance contract, but their
        # availability or contents cannot veto a technically valid Energy
        # workload at preflight time.
        semantic_payload_status = "downstream_annotation_not_preflighted"

        hash_cache: dict[str, str] = {str(own_path): actual_own_sha}
        stat_cache: dict[str, dict[str, Any]] = {str(own_path): _file_stat(own_path)}
        labels_by_path: dict[str, list[str]] = {str(own_path): ["preflight_script"]}
        for label, path, expected, expected_size in checks:
            resolved = path.resolve()
            key = str(resolved)
            if not resolved.is_file():
                raise RuntimeError(f"preflight_artifact_missing:{label}:{resolved}")
            wanted_hash = str(expected or "").strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", wanted_hash) is None:
                raise RuntimeError(f"preflight_artifact_expected_sha256_invalid:{label}")
            actual = hash_cache.get(key)
            if actual is None:
                actual = _sha256_file(resolved)
                hash_cache[key] = actual
                stat_cache[key] = _file_stat(resolved)
            if actual != wanted_hash:
                raise RuntimeError(f"preflight_artifact_sha256_mismatch:{label}")
            if expected_size is not None and int(resolved.stat().st_size) != expected_size:
                raise RuntimeError(f"preflight_artifact_size_mismatch:{label}")
            labels_by_path.setdefault(key, []).append(label)
            artifact_rows.append({
                "label": label,
                "path": key,
                "expected_sha256": wanted_hash,
                "actual_sha256": actual,
                "status": "pass",
            })
        verified_files = [
            {**stat_cache[path], "labels": labels_by_path[path]}
            for path in sorted(stat_cache)
        ]
        binding = _binding_for_contract(
            contract, runner_path=runner_path, verified_files=verified_files
        )
        supported = binding.get("workload_supported") is True
        payload = {
            "ok": supported,
            "nonce": str(nonce),
            "created_at_unix_ns": started_ns,
            "expires_at_unix_ns": started_ns + max(
                1, int(float(valid_for_s) * 1_000_000_000)
            ),
            "artifact_verification_status": "pass",
            "semantic_payload_verification_status": semantic_payload_status,
            "verified_part2_input_count": len(
                list(part2_metadata.get("inputs") or [])
            ),
            "command_contract_sha256": actual_contract_sha,
            "command_contract_source": contract_source,
            "command_contract_payload_sha256": payload_sha256,
            "preflight_script_path": str(own_path),
            "preflight_script_sha256": actual_own_sha,
            "artifact_verification": artifact_rows,
            "workload_binding": binding,
            "failure_reason": str(binding.get("unsupported_reason") or ""),
        }
        return seal_split_energy_preflight_attestation(payload), (0 if supported else 4)
    except Exception as exc:
        contract_sha = str(raw.get("contract_sha256") or expected_contract_sha256 or "")
        payload = {
            "ok": False,
            "nonce": str(nonce),
            "created_at_unix_ns": started_ns,
            "expires_at_unix_ns": started_ns + max(
                1, int(float(valid_for_s) * 1_000_000_000)
            ),
            "artifact_verification_status": "fail",
            "command_contract_sha256": contract_sha,
            "command_contract_source": (
                "file:" + str(contract_path) if contract_path is not None
                else "inline_verified_contract_object"
            ),
            "command_contract_payload_sha256": (
                canonical_json_sha256(raw) if raw else ""
            ),
            "artifact_verification": artifact_rows,
            "workload_binding": {},
            "failure_reason": f"{type(exc).__name__}:{exc}",
        }
        return seal_split_energy_preflight_attestation(payload), 3


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and attest a Native split workload before energy sampling."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--contract-json", default="")
    source.add_argument(
        "--contract-json-payload", default="",
        help="Inline JSON object for the already verified successful command contract.",
    )
    parser.add_argument("--expected-command-contract-sha256", required=True)
    parser.add_argument("--expected-preflight-script-sha256", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--attestation", required=True)
    parser.add_argument("--tool-root", default=str(ROOT))
    parser.add_argument("--valid-for-s", type=float, default=120.0)
    args = parser.parse_args()

    inline_payload: dict[str, Any] | None = None
    if args.contract_json_payload:
        try:
            inline_payload = _strict_json_text(
                args.contract_json_payload, label="inline contract payload",
            )
        except Exception as exc:
            print(f"invalid_inline_contract_payload:{type(exc).__name__}:{exc}", file=sys.stderr)
            return 2
    attestation, rc = build_attestation(
        Path(args.contract_json).expanduser() if args.contract_json else None,
        contract_payload=inline_payload,
        nonce=str(args.nonce),
        expected_contract_sha256=str(args.expected_command_contract_sha256),
        expected_preflight_script_sha256=str(args.expected_preflight_script_sha256),
        tool_root=Path(args.tool_root).expanduser().resolve(),
        valid_for_s=max(1.0, float(args.valid_for_s)),
    )
    destination = Path(args.attestation).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(attestation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        SPLIT_ENERGY_PREFLIGHT_STDOUT_MARKER
        + json.dumps(attestation, sort_keys=True, separators=(",", ":"))
    )
    if rc:
        print(str(attestation.get("failure_reason") or "preflight failed"), file=sys.stderr)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
