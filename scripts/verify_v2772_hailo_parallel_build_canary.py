#!/usr/bin/env python3
"""Verify the exact v2.77.2 Hailo Part-1 pair-build canary contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


PROFILE_ID = "resnet50_v2772_hailo_parallel_build_canary"
EXPECTED_TOOL_VERSION = "2.77.2"
EXPECTED_WORKFLOW_VERSION = (
    "v2.77.2-hailo-full-scope-and-parallel-canary-closure"
)
RECEIPT_SCHEMA = "onnx-splitpoint/hailo-hef-build-receipt/v2"
VERDICT_SCHEMA = "onnx-splitpoint/v2772-hailo-parallel-canary-verdict/v1"
EVIDENCE_SCHEMA = "onnx-splitpoint/v2772-hailo-parallel-canary-evidence-manifest/v1"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_YAML_BYTES = 2 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
EXPECTED_HEF_RELATIVE = {
    "hailo8": Path(
        "models/resnet50/benchmark_set/legacy_suite/b052/"
        "hailo/hailo8/part1/compiled.hef"
    ),
    "hailo10": Path(
        "models/resnet50/benchmark_set/legacy_suite/b052/"
        "hailo/hailo10/part1/compiled.hef"
    ),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_limited(path: Path, limit: int) -> bytes:
    size = path.stat().st_size
    if size > limit:
        raise ValueError(f"file_too_large:{path.name}:{size}>{limit}")
    return path.read_bytes()


def _read_json(path: Path) -> Any:
    return json.loads(_read_limited(path, MAX_JSON_BYTES).decode("utf-8"))


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(_read_limited(path, MAX_YAML_BYTES).decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("profile_yaml_not_mapping")
    return dict(payload)


def _nested_key_values(value: Any, wanted: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) == wanted:
                found.append(item)
            found.extend(_nested_key_values(item, wanted))
    elif isinstance(value, list):
        for item in value:
            found.extend(_nested_key_values(item, wanted))
    return found


def _target_from_path(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    if "hailo8" in parts:
        return "hailo8"
    if "hailo10" in parts or "hailo10h" in parts:
        return "hailo10"
    return ""


def _normalized_target(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "")
    if "10" in token:
        return "hailo10"
    if "8" in token:
        return "hailo8"
    return token


def _arch_matches(target: str, arch: Any) -> bool:
    token = str(arch or "").strip().lower().replace("-", "")
    if target == "hailo8":
        return token == "hailo8"
    return token in {"hailo10", "hailo10h"}


def _regular_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) and not path.is_symlink()


def _cache_key(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _part1_pair_diagnostic(events: list[dict[str, Any]]) -> dict[str, Any]:
    part1 = [
        event
        for event in events
        if re.fullmatch(
            r"hailo-targets:b0?52",
            str((event.get("metadata") or {}).get("label") or ""),
        )
    ]
    diagnostic: dict[str, Any] = {
        "observed": False,
        "event_count": len(part1),
        "targets": [
            _normalized_target((event.get("metadata") or {}).get("target"))
            for event in part1
        ],
        "overlap_s": 0.0,
        "start_skew_s": None,
        "overlap_gate_pass": False,
    }
    if len(part1) != 2:
        return diagnostic
    targets = sorted(diagnostic["targets"])
    if targets != ["hailo10", "hailo8"]:
        return diagnostic
    try:
        starts = [float(event["started_at"]) for event in part1]
        ends = [float(event["ended_at"]) for event in part1]
    except Exception:
        return diagnostic
    if any(end <= start for start, end in zip(starts, ends)):
        return diagnostic
    overlap = min(ends) - max(starts)
    diagnostic.update(
        {
            "observed": True,
            "overlap_s": round(overlap, 6),
            "start_skew_s": round(abs(starts[0] - starts[1]), 6),
            "overlap_gate_pass": overlap > 1.0,
        }
    )
    return diagnostic


def _default_receipt_validator(hef: Path) -> Any:
    from onnx_splitpoint_tool.hailo_backend import _load_valid_hailo_receipt

    return _load_valid_hailo_receipt(hef)


def verify_run(
    *,
    run_dir: Path,
    scheduler_log: Path,
    workflow_exit_code: int,
    receipt_validator: Callable[[Path], Any] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    errors: list[str] = []
    warnings: list[str] = []
    artifact_files: list[Path] = []
    details: dict[str, Any] = {}
    materialized_profile: dict[str, Any] | None = None
    source_profile: dict[str, Any] | None = None

    if workflow_exit_code != 0:
        warnings.append(
            f"workflow_exit_code={workflow_exit_code}; "
            "the intentional stop-after is verified independently"
        )
    if not run_dir.is_dir():
        errors.append(f"run_dir_missing:{run_dir}")

    events: list[dict[str, Any]] = []
    if not scheduler_log.is_file():
        errors.append(f"scheduler_log_missing:{scheduler_log}")
    else:
        for line_number, raw in enumerate(
            scheduler_log.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except Exception as exc:
                errors.append(
                    f"scheduler_json_invalid:line={line_number}:"
                    f"{type(exc).__name__}"
                )
                continue
            if isinstance(event, Mapping):
                events.append(dict(event))
            else:
                errors.append(f"scheduler_event_not_object:line={line_number}")

    families = [str(event.get("family") or "") for event in events]
    targets = [
        _normalized_target((event.get("metadata") or {}).get("target"))
        for event in events
    ]
    statuses = [str(event.get("status") or "") for event in events]
    labels = [
        str((event.get("metadata") or {}).get("label") or "")
        for event in events
    ]
    if len(events) != 2:
        errors.append(f"scheduler_event_count:{len(events)}!=2")
    if sorted(families) != ["hailo10", "hailo8"]:
        errors.append(f"scheduler_families:{families}")
    if sorted(targets) != ["hailo10", "hailo8"]:
        errors.append(f"scheduler_targets:{targets}")
    if any(status != "ok" for status in statuses):
        errors.append(f"scheduler_statuses:{statuses}")
    if any(
        not re.fullmatch(r"hailo-targets:b0?52", label) for label in labels
    ):
        errors.append(f"scheduler_labels:{labels}")
    if any(int(event.get("cpu_tokens") or 0) != 4 for event in events):
        errors.append("scheduler_cpu_tokens_not_4")
    if any(int(event.get("ram_mb") or 0) != 6144 for event in events):
        errors.append("scheduler_ram_mb_not_6144")

    pair_diagnostic = _part1_pair_diagnostic(events)
    if pair_diagnostic["event_count"] != 2:
        errors.append(
            f"part1_scheduler_event_count:"
            f"{pair_diagnostic['event_count']}!=2"
        )
    elif not pair_diagnostic["observed"]:
        errors.append("part1_scheduler_interval_invalid")
    elif not pair_diagnostic["overlap_gate_pass"]:
        errors.append(
            f"process_overlap_too_small:"
            f"{pair_diagnostic['overlap_s']:.6f}s<=1.0s"
        )
    details["part1_pair_diagnostic"] = pair_diagnostic
    details["scheduler_contract"] = {
        "total_event_count": len(events),
        "unexpected_event_count": max(0, len(events) - 2),
        "pass": (
            len(events) == 2
            and pair_diagnostic["observed"]
            and pair_diagnostic["overlap_gate_pass"]
        ),
    }

    workflow_log = run_dir / "evaluation_workflow.log"
    pair_lines: list[str] = []
    if not workflow_log.is_file():
        errors.append("evaluation_workflow_log_missing")
    else:
        workflow_text = workflow_log.read_text(
            encoding="utf-8",
            errors="replace",
        )
        pair_lines = [
            line.strip()
            for line in workflow_text.splitlines()
            if "Hailo pair requested=True effective=True" in line
        ]
        if len(pair_lines) != 1:
            errors.append(f"effective_parallel_pair_log_line_count:{len(pair_lines)}!=1")
        if pair_lines:
            if "backend_effective=venv" not in pair_lines[0]:
                errors.append("pair_backend_not_venv")
            if "reason=resources_available" not in pair_lines[0]:
                errors.append("pair_reason_not_resources_available")
        if "hailo-full-targets" in workflow_text:
            errors.append("unexpected_hailo_full_scheduler_log")
        if re.search(
            r"suite: Hailo HEF generation requested .*\bfull=True\b",
            workflow_text,
        ):
            errors.append("unexpected_hailo_full_build_request")
        forbidden_stage_markers = {
            "run_benchmarks": "[workflow] start resnet50 / run_benchmarks",
            "validate_outputs": "[workflow] start resnet50 / validate_outputs",
            "hardware_smoke": "[workflow] start resnet50 / hardware_smoke",
            "evaluate_quality": "[workflow] start evaluate_quality",
            "native": "[workflow] start run_native_producers",
            "aggregate_results": "[workflow] start aggregate_results",
            "generate_report": "[workflow] start generate_report",
        }
        entered = [
            name
            for name, marker in forbidden_stage_markers.items()
            if marker in workflow_text
        ]
        if entered:
            errors.append(f"post_build_workflow_stages_entered:{entered}")

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        errors.append("run_manifest_missing")
    else:
        try:
            manifest = _read_json(manifest_path)
            if not isinstance(manifest, Mapping):
                raise TypeError("manifest_not_mapping")
            if str(manifest.get("profile_id") or "") != PROFILE_ID:
                errors.append(
                    f"run_manifest_profile_id:{manifest.get('profile_id')}"
                )
            models = manifest.get("models")
            model_rows = dict(models) if isinstance(models, Mapping) else {}
            resnet = model_rows.get("resnet50")
            resnet = dict(resnet) if isinstance(resnet, Mapping) else {}
            model_stages = resnet.get("stages")
            model_stages = (
                dict(model_stages) if isinstance(model_stages, Mapping) else {}
            )
            root_stages = manifest.get("root_stages")
            root_stages = (
                dict(root_stages) if isinstance(root_stages, Mapping) else {}
            )
            manifest_checks = {
                "tool_version": (
                    str(manifest.get("tool_version") or "")
                    == EXPECTED_TOOL_VERSION
                ),
                "workflow_version": (
                    str(manifest.get("workflow_version") or "")
                    == EXPECTED_WORKFLOW_VERSION
                ),
                "status_partial": str(manifest.get("status") or "") == "partial",
                "one_resnet_model": set(model_rows) == {"resnet50"},
                "build_stage_ok": (
                    isinstance(model_stages.get("build_backend_artifacts"), Mapping)
                    and str(
                        model_stages["build_backend_artifacts"].get("status") or ""
                    )
                    == "ok"
                ),
                "later_model_stages_absent": not any(
                    name in model_stages
                    for name in (
                        "run_benchmarks",
                        "validate_outputs",
                        "hardware_smoke",
                    )
                ),
                "later_root_stages_absent": not any(
                    name in root_stages
                    for name in (
                        "evaluate_quality",
                        "aggregate_results",
                        "run_native_producers",
                        "generate_report",
                    )
                ),
            }
            for name, passed in manifest_checks.items():
                if not passed:
                    errors.append(f"run_manifest_contract_failed:{name}")
            details["run_manifest_contract"] = manifest_checks
        except Exception as exc:
            errors.append(f"run_manifest_invalid:{type(exc).__name__}")

    profile_path = run_dir / "profile.yaml"
    if not profile_path.is_file():
        errors.append("materialized_profile_missing")
    else:
        try:
            resolved = _read_yaml_mapping(profile_path)
            materialized_profile = resolved
            workflow = dict(resolved.get("workflow") or {})
            benchmark = dict(resolved.get("benchmark_execution") or {})
            remote = dict(resolved.get("remote_execution") or {})
            preparation = dict(resolved.get("model_preparation") or {})
            hardware_smoke = dict(resolved.get("hardware_smoke") or {})
            hailo = dict(resolved.get("hailo_build") or {})
            scheduler = dict(resolved.get("build_scheduler") or {})
            artifact_store = dict(resolved.get("artifact_store") or {})
            native = dict(resolved.get("native_producers") or {})
            energy = dict(resolved.get("energy") or {})
            ranking = dict(resolved.get("ranking_validation") or {})
            validation = dict(resolved.get("validation") or {})
            contract_checks = {
                "profile_id": str(resolved.get("name") or "") == PROFILE_ID,
                "execution_preset_absent": not isinstance(
                    resolved.get("execution_preset"),
                    Mapping,
                ),
                "execution_mode": (
                    workflow.get("execution_mode") == "generate_benchmarksets"
                ),
                "runtime_skipped": (
                    workflow.get("skip_runtime_benchmarks") is True
                ),
                "stop_after_build": (
                    workflow.get("stop_after") == "build_backend_artifacts"
                ),
                "remote_parallel_disabled": (
                    workflow.get("parallel_remote_setups") is False
                    and workflow.get("max_parallel_setups") == 1
                    and workflow.get("max_parallel_uploads") == 0
                    and workflow.get("powercalc_workers") == 0
                ),
                "benchmark_minimal": (
                    benchmark.get("warmup") == 0
                    and benchmark.get("runs") == 1
                    and benchmark.get("timeout_s") == 0
                ),
                "remote_disabled": remote.get("enabled") is False,
                "model_preparation_current": (
                    preparation.get("mode") == "current"
                ),
                "hardware_smoke_disabled": (
                    hardware_smoke.get("mode") == "disabled"
                ),
                "force_build": hailo.get("force_build") is True,
                "cache_disabled": hailo.get("cache_enabled") is False,
                "full_disabled": hailo.get("build_full") is False,
                "part1_enabled": hailo.get("build_part1") is True,
                "part2_disabled": hailo.get("build_part2") is False,
                "backend_auto": hailo.get("backend") == "auto",
                "scheduler_workers": scheduler.get("max_workers") == 2,
                "scheduler_cpu": scheduler.get("cpu_tokens") == 8,
                "scheduler_ram": scheduler.get("ram_mb") == 12288,
                "artifact_store_disabled": (
                    artifact_store.get("enabled") is False
                ),
                "native_disabled": native.get("enabled") is False,
                "energy_disabled": (
                    energy.get("enabled") is False
                    and energy.get("requested_native_energy") is False
                ),
                "ranking_disabled": ranking.get("enabled") is False,
                "cpu_full_reference": (
                    validation.get("split_fidelity_reference_mode")
                    == "cpu_full"
                ),
            }
            for name, passed in contract_checks.items():
                if not passed:
                    errors.append(f"profile_contract_failed:{name}")
            details["profile_contract"] = contract_checks
        except Exception as exc:
            errors.append(
                f"materialized_profile_invalid:{type(exc).__name__}:{exc}"
            )

    source_profile_path = run_dir / "profile_source.yaml"
    if not source_profile_path.is_file():
        errors.append("profile_source_missing")
    else:
        try:
            source_profile = _read_yaml_mapping(source_profile_path)
        except Exception as exc:
            errors.append(
                f"profile_source_invalid:{type(exc).__name__}:{exc}"
            )

    snapshot_path = run_dir / "profile_start_snapshot.json"
    if not snapshot_path.is_file():
        errors.append("profile_start_snapshot_missing")
    else:
        try:
            snapshot = _read_json(snapshot_path)
            if str(snapshot.get("profile_id") or "") != PROFILE_ID:
                errors.append(
                    f"profile_snapshot_profile_id:{snapshot.get('profile_id')}"
                )
            consistency = snapshot.get("consistency")
            if isinstance(consistency, Mapping):
                if str(consistency.get("status") or "") not in {"", "ok"}:
                    errors.append(
                        f"profile_snapshot_consistency:"
                        f"{consistency.get('status')}"
                    )
            if materialized_profile is None or source_profile is None:
                errors.append("profile_snapshot_inputs_unavailable")
            else:
                from onnx_splitpoint_tool.workflow.start_snapshot import (
                    build_profile_start_snapshot,
                    public_start_snapshot_metadata,
                )

                rebuilt = build_profile_start_snapshot(
                    profile_request=str(snapshot.get("profile_request") or ""),
                    source_profile=source_profile,
                    resolved_profile=materialized_profile,
                    profile_id=str(snapshot.get("profile_id") or ""),
                    profile_path=str(snapshot.get("profile_path") or ""),
                    profile_source=str(snapshot.get("profile_source") or ""),
                    runtime_bindings=(
                        snapshot.get("runtime_bindings")
                        if isinstance(snapshot.get("runtime_bindings"), Mapping)
                        else {}
                    ),
                    schema_version=int(snapshot.get("schema_version") or 1),
                )
                rebuilt_public = public_start_snapshot_metadata(rebuilt)
                if rebuilt_public != snapshot:
                    errors.append("profile_start_snapshot_attestation_mismatch")
                details["profile_snapshot_attestation"] = {
                    "pass": rebuilt_public == snapshot,
                    "schema_version": int(snapshot.get("schema_version") or 0),
                    "resolved_profile_sha256": str(
                        snapshot.get("resolved_profile_sha256") or ""
                    ),
                }
        except Exception as exc:
            errors.append(f"profile_start_snapshot_invalid:{type(exc).__name__}")

    all_hefs = sorted(
        path for path in run_dir.rglob("compiled.hef")
        if ".hailo-generations" not in path.parts
    ) if run_dir.is_dir() else []
    actual_rel = {
        path.relative_to(run_dir).as_posix(): path for path in all_hefs
    }
    expected_rel = {
        target: relative.as_posix()
        for target, relative in EXPECTED_HEF_RELATIVE.items()
    }
    if len(all_hefs) != 2:
        errors.append(f"compiled_hef_total:{len(all_hefs)}!=2")
    if set(actual_rel) != set(expected_rel.values()):
        errors.append(
            f"compiled_hef_paths:{sorted(actual_rel)}"
        )
    if any(
        "full" in [part.lower() for part in path.parts]
        or "part2" in [part.lower() for part in path.parts]
        for path in all_hefs
    ):
        errors.append("unexpected_full_or_part2_hef")
    cache_hit_markers = (
        sorted(run_dir.rglob("hailo_cache_hit.json"))
        if run_dir.is_dir()
        else []
    )
    if cache_hit_markers:
        errors.append(
            f"hailo_cache_hit_markers_present:{len(cache_hit_markers)}"
        )

    artifact_evidence: dict[str, Any] = {}
    seen_inodes: set[tuple[int, int]] = set()
    validator = receipt_validator or _default_receipt_validator
    for target, relative in EXPECTED_HEF_RELATIVE.items():
        hef = run_dir / relative
        if not _regular_file(hef):
            errors.append(f"hef_missing_or_not_regular:{target}")
            continue
        if hef.stat().st_size <= 0:
            errors.append(f"hef_empty:{target}")
            continue
        if hef.stat().st_size > 16 * 1024 * 1024:
            errors.append(f"hef_too_large_for_evidence:{target}")
            continue
        inode = (hef.stat().st_dev, hef.stat().st_ino)
        if inode in seen_inodes:
            errors.append("target_hefs_share_inode")
        seen_inodes.add(inode)

        receipt_path = hef.parent / "hailo_hef_build_receipt.json"
        result_path = hef.parent / "hailo_hef_build_result.json"
        artifact_files.extend([hef, receipt_path, result_path])
        if not _regular_file(receipt_path):
            errors.append(f"receipt_missing:{target}")
            continue
        try:
            receipt = _read_json(receipt_path)
            if not isinstance(receipt, Mapping):
                raise TypeError("receipt_not_mapping")
            receipt = dict(receipt)
            actual_sha = _sha256(hef)
            actual_size = hef.stat().st_size
            if receipt.get("schema") != RECEIPT_SCHEMA:
                errors.append(
                    f"receipt_schema:{target}:{receipt.get('schema')}"
                )
            if not _arch_matches(target, receipt.get("hw_arch")):
                errors.append(
                    f"receipt_arch:{target}:{receipt.get('hw_arch')}"
                )
            if int(receipt.get("hef_size_bytes") or -1) != actual_size:
                errors.append(f"receipt_size_mismatch:{target}")
            if str(receipt.get("hef_sha256") or "").lower() != actual_sha:
                errors.append(f"receipt_sha256_mismatch:{target}")
            if not re.fullmatch(
                r"[0-9a-f]{64}",
                str(receipt.get("source_onnx_sha256") or "").lower(),
            ):
                errors.append(f"receipt_source_sha_missing:{target}")
            if not re.fullmatch(
                r"[0-9a-f]{64}",
                str(receipt.get("compiler_onnx_sha256") or "").lower(),
            ):
                errors.append(f"receipt_compiler_sha_missing:{target}")
            cache_payload = receipt.get("cache_payload")
            if not isinstance(cache_payload, Mapping):
                errors.append(f"receipt_cache_payload_missing:{target}")
            elif str(receipt.get("cache_key") or "") != _cache_key(cache_payload):
                errors.append(f"receipt_cache_key_mismatch:{target}")
            try:
                if validator(hef) is None:
                    errors.append(
                        f"receipt_canonical_validation_failed:{target}"
                    )
            except Exception as exc:
                errors.append(
                    f"receipt_canonical_validation_error:{target}:"
                    f"{type(exc).__name__}:{exc}"
                )
        except Exception as exc:
            errors.append(f"receipt_invalid:{target}:{type(exc).__name__}")
            continue

        if not _regular_file(result_path):
            errors.append(f"build_result_missing:{target}")
            continue
        try:
            result = _read_json(result_path)
            if not isinstance(result, Mapping):
                raise TypeError("result_not_mapping")
            result = dict(result)
            if result.get("ok") is not True:
                errors.append(f"build_result_not_ok:{target}")
            if result.get("skipped") is not False:
                errors.append(
                    f"build_result_skipped:{target}:{result.get('skipped')}"
                )
            if result.get("timed_out") is not False:
                errors.append(
                    f"build_result_timed_out:{target}:"
                    f"{result.get('timed_out')}"
                )
            if str(result.get("backend") or "") != "venv":
                errors.append(
                    f"build_result_backend:{target}:{result.get('backend')}"
                )
            if result.get("error") not in {None, ""}:
                errors.append(f"build_result_error:{target}")
            if not _arch_matches(target, result.get("hw_arch")):
                errors.append(
                    f"build_result_arch:{target}:{result.get('hw_arch')}"
                )
            cache_values = _nested_key_values(result, "cache_hit")
            if not cache_values:
                errors.append(
                    f"build_result_cache_evidence_missing:{target}"
                )
            if any(value is not False for value in cache_values):
                errors.append(
                    f"build_result_cache_hit:{target}:{cache_values}"
                )
            embedded_receipts = _nested_key_values(
                result,
                "build_receipt",
            )
            if not embedded_receipts:
                errors.append(
                    f"build_result_embedded_receipt_missing:{target}"
                )
            elif not any(item == receipt for item in embedded_receipts):
                errors.append(
                    f"build_result_embedded_receipt_mismatch:{target}"
                )
        except Exception as exc:
            errors.append(
                f"build_result_invalid:{target}:{type(exc).__name__}"
            )
            continue

        artifact_evidence[target] = {
            "hef_path": str(hef),
            "hef_relative_path": relative.as_posix(),
            "hef_size_bytes": hef.stat().st_size,
            "hef_sha256": _sha256(hef),
            "receipt_path": str(receipt_path),
            "build_result_path": str(result_path),
        }

    # Include every small result/receipt in FAIL evidence, including unexpected
    # Full metadata, while keeping large unexpected HEFs out of the bundle.
    if run_dir.is_dir():
        artifact_files.extend(
            path for name in ("hailo_hef_build_receipt.json", "hailo_hef_build_result.json")
            for path in run_dir.rglob(name)
            if ".hailo-generations" not in path.parts
        )

    verdict: dict[str, Any] = {
        "schema": VERDICT_SCHEMA,
        "status": "FAIL" if errors else "PASS",
        "technical_canary_pass": not errors,
        "not_scientific_evidence": True,
        "profile_id": PROFILE_ID,
        "run_dir": str(run_dir),
        "workflow_exit_code": workflow_exit_code,
        "scheduler_log": str(scheduler_log),
        "scheduler_events": events,
        "scheduler_families": families,
        "scheduler_targets": targets,
        "scheduler_labels": labels,
        "process_overlap_s": pair_diagnostic["overlap_s"],
        "effective_pair_log_line": pair_lines[0] if pair_lines else "",
        "artifacts": artifact_evidence,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }
    return verdict, sorted(set(artifact_files))


def _safe_member_name(name: str) -> str:
    normalized = str(name).replace("\\", "/").strip("/")
    parts = Path(normalized).parts
    if not normalized or normalized.startswith("/") or ".." in parts:
        raise ValueError(f"unsafe_evidence_member:{name}")
    return normalized


def _evidence_members(
    *,
    run_dir: Path,
    scheduler_log: Path,
    verdict_bytes: bytes,
    artifact_files: Iterable[Path],
) -> tuple[dict[str, bytes], dict[str, str]]:
    members: dict[str, bytes] = {}
    roles: dict[str, str] = {}

    def add_bytes(name: str, payload: bytes, role: str) -> None:
        member = _safe_member_name(name)
        if member in members:
            raise ValueError(f"duplicate_evidence_member:{member}")
        if len(payload) > MAX_MEMBER_BYTES:
            raise ValueError(
                f"evidence_member_too_large:{member}:{len(payload)}"
            )
        members[member] = payload
        roles[member] = role

    def add_path(name: str, path: Path, role: str) -> None:
        if not _regular_file(path):
            return
        add_bytes(name, _read_limited(path, MAX_MEMBER_BYTES), role)

    add_bytes("canary_verdict.json", verdict_bytes, "verdict")
    add_path(
        "scheduler/build_scheduler.jsonl",
        scheduler_log,
        "scheduler_events",
    )
    for name in (
        "evaluation_workflow.log",
        "run_manifest.json",
        "profile.yaml",
        "profile_source.yaml",
        "profile_start_snapshot.json",
        "profile_resolution.json",
        "effective_execution_plan.json",
        "artifact_index.json",
    ):
        add_path(f"run/{name}", run_dir / name, "run_evidence")
    for name in ("preflight.json", "workflow_command.log", "launcher.log"):
        add_path(
            f"launcher/{name}",
            run_dir.parent / name,
            "launcher_evidence",
        )

    for path in sorted(set(artifact_files)):
        if not _regular_file(path):
            continue
        try:
            relative = path.resolve().relative_to(run_dir.resolve())
        except ValueError:
            continue
        add_path(
            f"run/{relative.as_posix()}",
            path,
            "hailo_artifact_or_metadata",
        )

    script_dir = Path(__file__).resolve().parent
    source_root = script_dir.parent
    for name in (
        "preflight_v2772_hailo_parallel_build_canary.py",
        "run_v2772_hailo_parallel_build_canary.sh",
        "verify_v2772_hailo_parallel_build_canary.py",
    ):
        add_path(
            f"verification/{name}",
            script_dir / name,
            "executed_verification_source",
        )
    add_path(
        "verification/resnet50_v2772_hailo_parallel_build_canary.yaml",
        source_root
        / "profiles"
        / "resnet50_v2772_hailo_parallel_build_canary.yaml",
        "canary_profile_source",
    )
    return members, roles


def _write_evidence_zip(
    *,
    destination: Path,
    members: dict[str, bytes],
    roles: dict[str, str],
) -> None:
    manifest = {
        "schema": EVIDENCE_SCHEMA,
        "entries": [
            {
                "path": name,
                "role": roles[name],
                "size_bytes": len(members[name]),
                "sha256": _sha256_bytes(members[name]),
            }
            for name in sorted(members)
        ],
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    manifest_sha_bytes = (
        f"{_sha256_bytes(manifest_bytes)}  evidence_manifest.json\n"
    ).encode("ascii")
    payloads = dict(members)
    payloads["evidence_manifest.json"] = manifest_bytes
    payloads["evidence_manifest.sha256"] = manifest_sha_bytes
    total = sum(len(value) for value in payloads.values())
    if total > MAX_EVIDENCE_BYTES:
        raise ValueError(f"evidence_total_too_large:{total}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(payloads):
                archive.writestr(name, payloads[name])
        with zipfile.ZipFile(temporary, "r") as archive:
            if archive.testzip() is not None:
                raise ValueError("evidence_zip_crc_failure")
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("evidence_zip_duplicate_member")
            if archive.read("canary_verdict.json") != members[
                "canary_verdict.json"
            ]:
                raise ValueError("evidence_internal_verdict_mismatch")
            if _sha256_bytes(
                archive.read("evidence_manifest.json")
            ) != manifest_sha_bytes.decode("ascii").split()[0]:
                raise ValueError("evidence_manifest_hash_mismatch")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scheduler-log", required=True)
    parser.add_argument("--workflow-exit-code", type=int, default=0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--evidence-zip", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    scheduler_log = Path(args.scheduler_log).expanduser().resolve()
    verdict_path = Path(args.out_json).expanduser().resolve()
    evidence_zip = Path(args.evidence_zip).expanduser().resolve()
    verdict, artifact_files = verify_run(
        run_dir=run_dir,
        scheduler_log=scheduler_log,
        workflow_exit_code=args.workflow_exit_code,
    )
    verdict["evidence_zip"] = evidence_zip.name
    verdict_bytes = _canonical_json_bytes(verdict)
    try:
        members, roles = _evidence_members(
            run_dir=run_dir,
            scheduler_log=scheduler_log,
            verdict_bytes=verdict_bytes,
            artifact_files=artifact_files,
        )
        _write_evidence_zip(
            destination=evidence_zip,
            members=members,
            roles=roles,
        )
    except Exception as exc:
        verdict["errors"].append(
            f"evidence_zip_failed:{type(exc).__name__}:{exc}"
        )
        verdict["status"] = "FAIL"
        verdict["technical_canary_pass"] = False
        verdict_bytes = _canonical_json_bytes(verdict)
    _write_atomic(verdict_path, verdict_bytes)

    print(verdict_bytes.decode("utf-8"), end="", flush=True)
    return 0 if verdict["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
