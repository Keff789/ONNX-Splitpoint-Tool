"""Production-shaped, hardware-free Gate-A output fixtures for regressions."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from onnx_splitpoint_tool.benchmark.services import (
    _new_hailo_feasibility_state_v2783,
    normalize_hailo_feasibility_control,
)
from onnx_splitpoint_tool.build_evidence import canonical_sha256
from onnx_splitpoint_tool.workflow.contracts import StageResult


PROFILE_ID = "yolo11l_v2783_hailo8_first_b5_gate_a"
MODEL_ID = "yolo11l"
ROOT_STAGES = ("resolve_profile", "campaign_preflight")
MODEL_STAGES = (
    "resolve_model",
    "check_validation_assets",
    "prepare_model",
    "analyze_model",
    "select_split_candidates",
    "prepare_full_baselines",
    "generate_benchmark_set",
    "build_backend_artifacts",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_anchor_artifact(
    *,
    case_dir: Path,
    target: str,
    receipt_arch: str,
    builder: Path,
    boundary: int,
) -> tuple[str, Path]:
    artifact_dir = case_dir / "hailo" / target / "part1"
    artifact_dir.mkdir(parents=True)
    compiler = artifact_dir / f"{MODEL_ID}_part1_b{boundary}_hailo_fixed.onnx"
    hef = artifact_dir / "compiled.hef"
    compiler.write_bytes(f"compiler-{target}".encode("ascii"))
    hef.write_bytes(f"hef-{target}".encode("ascii"))
    preprocessing = {
        "schema": "onnx-splitpoint/image-preprocessing-contract",
        "schema_version": 2,
        "task": "detection",
        "target_hw": [640, 640],
        "image_scale": "norm",
    }
    preprocessing_sha = canonical_sha256(preprocessing)
    calibration_identity = "manifest:" + "9" * 64
    prepared_sha = canonical_sha256(
        {
            "calibration_identity": calibration_identity,
            "preprocessing_contract_sha256": preprocessing_sha,
        }
    )
    payload = {
        "schema": "onnx-splitpoint/hailo-hef-cache-key-v3",
        "model_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
        "activation_part1_sha256": "",
        "hw_arch": receipt_arch,
        "hailo_sdk_version": "hailo-dataflow-compiler:3.33.1",
        "optimization_level": 1,
        "calibration_identity": calibration_identity,
        "prepared_calibration_identity_sha256": prepared_sha,
        "calibration_count": 5,
        "requested_calibration_count": 5,
        "calibration_storage": "memory",
        "calibration_memory_cap_bytes": 1048576,
        "calibration_batch_size": 1,
        "extra_model_script": "",
        "start_nodes": [],
        "end_nodes": ["/head/Conv"],
        "integrity": "strict",
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha,
        "net_name": f"{MODEL_ID}_part1_b{boundary}",
        "net_input_shapes": [1, 3, 640, 640],
        "disable_rt_metadata_extraction": True,
    }
    cache_key = canonical_sha256(payload)
    receipt = {
        "schema": "onnx-splitpoint/hailo-hef-build-receipt/v2",
        "source_onnx_sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
        "compiler_onnx_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
        "compiler_onnx_filename": compiler.name,
        "hef_sha256": hashlib.sha256(hef.read_bytes()).hexdigest(),
        "hef_size_bytes": hef.stat().st_size,
        "hw_arch": receipt_arch,
        "net_name": f"{MODEL_ID}_part1_b{boundary}",
        "hailo_sdk_version": payload["hailo_sdk_version"],
        "calibration_identity": calibration_identity,
        "prepared_calibration_identity_sha256": prepared_sha,
        "calibration_count": 5,
        "requested_calibration_count": 5,
        "calibration_storage": "memory",
        "calibration_memory_cap_bytes": 1048576,
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha,
        "cache_key": cache_key,
        "cache_payload": payload,
    }
    _write_json(artifact_dir / "hailo_hef_build_receipt.json", receipt)
    return cache_key, hef


def write_gate_a_anchor_fixture(
    gate_output: Path,
    profile_path: Path,
    *,
    boundary: int = 67,
    workflow_status: str = "partial",
) -> Path:
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    run_id = f"{PROFILE_ID}_20260828_145933"
    run_dir = gate_output / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "profile_source.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "profile.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )

    formal = run_dir / "models" / MODEL_ID / "benchmark_set"
    suite = formal / "legacy_suite"
    case = suite / f"b{boundary:03d}"
    models = suite / "models"
    case.mkdir(parents=True)
    models.mkdir(parents=True)
    full = models / f"{MODEL_ID}.onnx"
    builder = case / f"{MODEL_ID}_part1_b{boundary}.onnx"
    full.write_bytes(b"full-yolo11l-model")
    builder.write_bytes(b"yolo11l-part1-model")
    keys: dict[str, str] = {}
    for target, arch in (("hailo8", "hailo8"), ("hailo10", "hailo10h")):
        keys[target], _ = _write_anchor_artifact(
            case_dir=case,
            target=target,
            receipt_arch=arch,
            builder=builder,
            boundary=boundary,
        )
    split_manifest = {
        "schema": "onnx-splitpoint/split-manifest",
        "boundary": boundary,
        "full_model": f"../models/{MODEL_ID}.onnx",
        "part1": builder.name,
        "hailo": {
            "hefs": {
                target: {"part1": f"hailo/{target}/part1/compiled.hef"}
                for target in keys
            }
        },
    }
    _write_json(case / "split_manifest.json", split_manifest)
    case_row = {
        "boundary": boundary,
        "case_dir": f"b{boundary:03d}",
        "folder": f"b{boundary:03d}",
        "manifest": "split_manifest.json",
    }
    _write_json(suite / "benchmark_set.json", {"cases": [case_row]})

    raw_control = profile["hailo_build"]["feasibility_control"]
    control = normalize_hailo_feasibility_control(raw_control)
    order = [boundary, 575, 581]
    state = _new_hailo_feasibility_state_v2783(
        control=control,
        candidate_order=order,
        # Production build_run_plan() sorts the physical target axis even
        # though primary_target keeps actual compiler dispatch Hailo-8-first.
        targets=["hailo10", "hailo8"],
        backend="auto",
        full_source_onnx_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),
    )
    state.update(
        {
            "outcome": "ANCHOR_FOUND",
            "anchor_boundary": boundary,
            "candidates": [
                {
                    "boundary": boundary,
                    "candidate_index": 0,
                    "target_outcomes": {
                        "hailo8": "ARTIFACT_PASS",
                        "hailo10": "ARTIFACT_PASS",
                    },
                    "phases": [
                        {
                            "phase": "exact_cache_probe",
                            "target": target,
                            "cache_only": True,
                            "ok": True,
                            "cache_hit": True,
                            "cache_key_v3": cache_key,
                        }
                        for target, cache_key in keys.items()
                    ],
                    "anchor": True,
                    "cold_builds": 0,
                    "controller_outcome": "ANCHOR_FOUND",
                }
            ],
            "attempts_by_boundary": {
                str(boundary): {
                    "cold_builds": 0,
                    "phases": [
                        {
                            "phase": "exact_cache_probe",
                            "target": target,
                            "cache_only": True,
                            "ok": True,
                            "cache_hit": True,
                            "cache_key_v3": cache_key,
                        }
                        for target, cache_key in keys.items()
                    ],
                    "target_outcomes": {
                        "hailo8": "ARTIFACT_PASS",
                        "hailo10": "ARTIFACT_PASS",
                    },
                    "anchor": True,
                }
            },
        }
    )
    generation = {
        "status": "complete",
        "completed_boundaries": [boundary],
        "accepted_boundaries": [boundary],
        "hailo_feasibility_state": state,
    }
    _write_json(suite / "generation_state.json", generation)
    _write_json(
        formal / "benchmark_set.json",
        {
            "schema": "onnx-splitpoint/benchmark-set-contract",
            "schema_version": 4,
            "mode": "legacy_benchmarkset_source_of_truth",
            "model_id": MODEL_ID,
            "profile_id": PROFILE_ID,
            "run_id": run_id,
            "materialized": True,
            "cases": [case_row],
            "status": "ok",
            "hailo_feasibility_outcome": "ANCHOR_FOUND",
            "fallback_allowed": False,
        },
    )
    receipt_rel = (
        Path("models") / MODEL_ID / "benchmark_set" / "hailo_feasibility_receipt.json"
    )
    _write_json(
        run_dir / receipt_rel,
        {
            "schema": "onnx-splitpoint/hailo8-first-feasibility-receipt/v1",
            "schema_version": 1,
            "model_id": MODEL_ID,
            "profile_id": PROFILE_ID,
            "run_id": run_id,
            "outcome": "ANCHOR_FOUND",
            "fallback_allowed": False,
            "stop_workflow": False,
            "candidate_order": order,
            "candidate_order_sha256": state["candidate_order_sha256"],
            "state": state,
            "validation_error": "",
        },
    )

    root_rows: dict[str, Any] = {}
    model_rows: dict[str, Any] = {}
    for model_id, names, target in (
        (None, ROOT_STAGES, root_rows),
        (MODEL_ID, MODEL_STAGES, model_rows),
    ):
        for stage in names:
            artifacts = []
            if stage == "generate_benchmark_set":
                artifacts = [
                    receipt_rel.as_posix(),
                    (Path("models") / MODEL_ID / "benchmark_set" / "benchmark_set.json").as_posix(),
                    (Path("models") / MODEL_ID / "benchmark_set" / "legacy_suite" / "benchmark_set.json").as_posix(),
                ]
            result = StageResult(
                stage=stage,
                model_id=model_id,
                status="ok",
                started_at="2026-08-28T14:59:33+00:00",
                finished_at="2026-08-28T20:08:41+00:00",
                artifacts=artifacts,
            ).to_dict()
            relative = (
                Path("models") / model_id / "stages" / stage / "stage_result.json"
                if model_id
                else Path("stages") / stage / "stage_result.json"
            )
            _write_json(run_dir / relative, result)
            target[stage] = {**result, "stage_result_path": relative.as_posix()}

    manifest = {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": workflow_status,
        "technical_status": workflow_status,
        "quality_decision": "not_evaluated",
        "options": {
            "stop_after": "build_backend_artifacts",
            "execution_mode": "generate_benchmarksets",
        },
        "root_stages": root_rows,
        "models": {
            MODEL_ID: {
                "model_id": MODEL_ID,
                "stages": model_rows,
            }
        },
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def write_gate_a_budget_fixture(
    gate_output: Path,
    profile_path: Path,
    *,
    boundary: int = 67,
) -> Path:
    """Materialize the runner's failed generate-stage exhaustion shape."""

    run_dir = write_gate_a_anchor_fixture(gate_output, profile_path, boundary=boundary)
    formal = run_dir / "models" / MODEL_ID / "benchmark_set"
    suite = formal / "legacy_suite"
    receipt_path = formal / "hailo_feasibility_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    state = receipt["state"]
    state["outcome"] = "CANARY_BUDGET_EXHAUSTED"
    state.pop("anchor_boundary", None)
    state["candidate_order"] = [boundary]
    state["candidate_order_sha256"] = canonical_sha256([boundary])
    state["exhaustion_reason"] = "candidate_pool_exhausted"
    state["candidates"] = [
        {
            "boundary": boundary,
            "candidate_index": 0,
            "target_outcomes": {
                "hailo8": "PARSER_UNSUPPORTED",
                "hailo10": "GATED_BY_HAILO8",
            },
            "phases": [
                {
                    "phase": "exact_cache_probe",
                    "target": "hailo8",
                    "cache_only": True,
                    "ok": False,
                    "cache_hit": False,
                    "cache_key_v3": "",
                },
                {
                    "phase": "parser_preflight",
                    "target": "hailo8",
                    "ok": False,
                    "outcome": "PARSER_UNSUPPORTED",
                },
                {
                    "phase": "exact_cache_probe",
                    "target": "hailo10",
                    "cache_only": True,
                    "ok": False,
                    "cache_hit": False,
                    "cache_key_v3": "",
                },
            ],
            "anchor": False,
            "cold_builds": 0,
            "controller_outcome": "RUNNING",
        }
    ]
    state["attempts_by_boundary"] = {
        str(boundary): {
            "cold_builds": 0,
            "phases": list(state["candidates"][0]["phases"]),
            "target_outcomes": dict(state["candidates"][0]["target_outcomes"]),
            "anchor": False,
        }
    }
    receipt.update(
        {
            "outcome": "CANARY_BUDGET_EXHAUSTED",
            "stop_workflow": True,
            "candidate_order": [boundary],
            "candidate_order_sha256": state["candidate_order_sha256"],
            "state": state,
        }
    )
    _write_json(receipt_path, receipt)

    _write_json(
        suite / "generation_state.json",
        {
            "status": "partial",
            "completed_boundaries": [boundary],
            "accepted_boundaries": [],
            "discarded_boundaries": [boundary],
            "hailo_feasibility_state": state,
        },
    )
    _write_json(
        suite / "benchmark_set.json",
        {
            "cases": [],
            "discarded_cases": [
                {
                    "boundary": boundary,
                    "reason": "CANARY_BUDGET_EXHAUSTED",
                }
            ],
        },
    )
    shutil.rmtree(suite / f"b{boundary:03d}")

    contract_path = formal / "benchmark_set.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update(
        {
            "materialized": False,
            "cases": [],
            "rejected_cases": [
                {
                    "boundary": boundary,
                    "reason": "CANARY_BUDGET_EXHAUSTED",
                }
            ],
            "status": "failed",
            "hailo_feasibility_outcome": "CANARY_BUDGET_EXHAUSTED",
        }
    )
    _write_json(contract_path, contract)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_stages = manifest["models"][MODEL_ID]["stages"]
    model_stages.pop("build_backend_artifacts")
    shutil.rmtree(
        run_dir / "models" / MODEL_ID / "stages" / "build_backend_artifacts"
    )
    generate_path = (
        run_dir
        / "models"
        / MODEL_ID
        / "stages"
        / "generate_benchmark_set"
        / "stage_result.json"
    )
    generate = json.loads(generate_path.read_text(encoding="utf-8"))
    generate.update(
        {
            "status": "failed",
            "state": "failed",
            "output_hash": "",
            "notes": [
                "CANARY_BUDGET_EXHAUSTED: fallback and unchanged B5 blocked."
            ],
        }
    )
    _write_json(generate_path, generate)
    relative = generate_path.relative_to(run_dir).as_posix()
    model_stages["generate_benchmark_set"] = {
        **generate,
        "stage_result_path": relative,
    }
    manifest.update(
        {
            "status": "failed",
            "technical_status": "failed",
            "quality_decision": "not_evaluated",
        }
    )
    _write_json(manifest_path, manifest)
    return run_dir
