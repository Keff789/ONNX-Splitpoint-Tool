from __future__ import annotations

from onnx_splitpoint_tool.native_full_quality import (
    normalise_evaluation_profile,
    resolve_native_split_plan,
)
"""Formal benchmark execution binding for the Evaluation Workflow.

v49e-v49i moves the workflow beyond the local v49d execution handoff: when the
workflow is run in ``generate_and_run`` mode, this module can dispatch a
materialized benchmark suite either through the existing local suite harness or
through the existing RemoteBenchmarkService when a remote host is configured.

The implementation is intentionally conservative:

* it never fabricates measurements;
* it runs only an explicit/generated suite runner when available;
* it records structured dispatch status, stdout/stderr and errors;
* Hailo/remote/DFC-heavy work can remain pending without failing the whole
  thesis workflow bundle.
"""

import concurrent.futures
import contextvars
import json
import os
import re
import shlex
import threading
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .artifacts import (
    now_iso,
    read_json,
    relpath,
    sha256_file,
    sha256_payload,
    write_json,
    write_text,
)
from .hardware_matrix import canon_accelerator, matrix_for_runtime, accelerator_provider
from .setup_local_trt_dispatch import (
    build_setup_local_tensorrt_quality_dispatch,
)
from onnx_splitpoint_tool.cache_verify_policy import cache_verify_guard
from onnx_splitpoint_tool.native_command_contract import (
    canonical_json_sha256 as canonical_contract_sha256,
)
from onnx_splitpoint_tool.management_reference import (
    bind_management_cpu_reference_runs,
    finalize_management_cpu_reference_plan_aliases,
    management_cpu_reference_required,
    profile_has_explicit_cpu_reference,
)
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy
from onnx_splitpoint_tool.process_control import (
    ProcessTreeRegistry,
    current_process_registry,
    terminate_process_tree,
)
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseRegistry


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _effective_str(value: Any) -> str:
    """Return a user override string; auto/none/null/default mean unset."""
    s = str(value or "").strip()
    return "" if s.lower() in {"", "auto", "none", "null", "default"} else s


def _task_gated_metrics(task: str, *, mini_coco: bool = False, mini_cls: bool = False) -> tuple[bool, bool]:
    task_l = str(task or "auto").strip().lower()
    if task_l == "classification":
        return False, bool(mini_cls)
    if task_l == "detection":
        return bool(mini_coco), False
    return False, False


def _resolve_suite_dir(run_root: Path, model_dir: Path, benchmark_set_contract: Mapping[str, Any]) -> Path:
    materialized = bool(benchmark_set_contract.get("materialized"))
    scope = str(benchmark_set_contract.get("materialization_scope") or "").lower()
    if (not materialized) and (
        "contract_only" in scope
        or benchmark_set_contract.get("source_of_truth_for_real_runs") == "legacy_benchmarkset_generator"
        or benchmark_set_contract.get("legacy_benchmarkset_required")
        or benchmark_set_contract.get("direct_suite_disabled")
    ):
        return (model_dir / "benchmark_set").resolve()
    raw = str(
        benchmark_set_contract.get("legacy_suite_dir")
        or benchmark_set_contract.get("suite_dir")
        or benchmark_set_contract.get("benchmark_suite_dir")
        or benchmark_set_contract.get("generated_suite_dir")  # compatibility with v49c-v49m runs only
        or ""
    ).strip()
    candidates: List[Path] = []
    if raw:
        p = Path(raw).expanduser()
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([run_root / p, model_dir / p, model_dir / "benchmark_set" / p])
    candidates.extend([
        model_dir / "benchmark_set" / "legacy_suite",
    ])
    # The old direct suite/generated_suite are not scanned in normal v49p
    # workflows. They are only available through the explicit debug path that
    # records a concrete suite_dir.
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    return (model_dir / "benchmark_set").resolve()


def _runner_is_contract_stub(script: Path) -> bool:
    try:
        text = script.read_text(encoding="utf-8", errors="ignore")[:4096]
    except Exception:
        return False
    needles = (
        "This v49c suite is the formal handoff",
        "does not fabricate measurements",
        "Materialize split graphs / run the existing benchmark executor",
        "benchmark suite script unavailable",
    )
    return any(n in text for n in needles)


def _has_materialized_split_cases(suite_dir: Path, suite_payload: Mapping[str, Any]) -> bool:
    cases = _as_list(suite_payload.get("cases"))
    if not cases:
        return False
    for raw in cases:
        if not isinstance(raw, Mapping):
            continue
        case_dir = str(raw.get("case_dir") or raw.get("folder") or raw.get("case_id") or "").strip()
        if not case_dir:
            continue
        cdir = suite_dir / case_dir
        if (cdir / "run_split_onnxruntime.py").is_file() and (cdir / "part1.onnx").is_file() and (cdir / "part2.onnx").is_file():
            return True
    return False


def _copy_result_files(suite_dir: Path, dst_dir: Path) -> List[Dict[str, str]]:
    copied: List[Dict[str, str]] = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        "benchmark_results_*.json",
        "benchmark_results_*.csv",
        "benchmark_results.json",
        "results.json",
        "benchmark_summary_*.md",
        "benchmark_table_*.tex",
        "benchmark_suite_status.json",
        "benchmark_suite_status_matrix.json",
        "benchmark_suite_status_matrix.csv",
        "benchmark_suite_status_matrix.md",
        "v42_pipeline_summary.json",
        "v42_pipeline_summary.csv",
        "v42_pipeline_summary.md",
        "v47_pipeline_summary.json",
        "v47_pipeline_summary.csv",
        "v47_pipeline_summary.md",
    ]
    search_dirs = [suite_dir, suite_dir / "results"]
    seen: set[str] = set()
    for root in search_dirs:
        if not root.is_dir():
            continue
        for pat in patterns:
            for src in sorted(root.glob(pat)):
                if not src.is_file():
                    continue
                key = str(src.resolve())
                if key in seen:
                    continue
                seen.add(key)
                dest = dst_dir / src.name
                try:
                    if str(src.resolve()) != str(dest.resolve()):
                        shutil.copy2(src, dest)
                    copied.append({"source": str(src), "destination": str(dest)})
                except Exception:
                    # Keep the execution status useful even if one optional file
                    # cannot be copied.
                    continue
    return copied


def _result_file_count(suite_dir: Path, dst_dir: Path) -> int:
    count = 0
    for root in (suite_dir, suite_dir / "results", dst_dir):
        if not root.is_dir():
            continue
        for pat in ("benchmark_results_*.json", "benchmark_results_*.csv", "benchmark_results.json", "results.json"):
            count += len([p for p in root.glob(pat) if p.is_file()])
    return count




def _stage_desc_from_token(token: str) -> Dict[str, Any]:
    t = str(token or "").strip().lower().replace("-", "_")
    if t in {"cpu_ort", "ort_cpu", "cpu"}:
        return {"type": "onnxruntime", "provider": "cpu"}
    if t in {"cuda_ort", "ort_cuda", "cuda", "gpu"}:
        return {"type": "onnxruntime", "provider": "cuda"}
    if t in {"trt", "tensor_rt", "tensorrt"}:
        return {"type": "onnxruntime", "provider": "tensorrt"}
    if t.startswith("hailo"):
        return {"type": "hailo", "hw_arch": t}
    if "deepx" in t or "dx_m1" in t or "dxm1" in t:
        return {"type": "deepx", "backend": "deepx_m1", "artifact_kind": "dxnn"}
    return {"type": "onnxruntime", "provider": t or "auto"}


def _split_backend_pair(backend: str) -> tuple[str, str]:
    b = str(backend or "").strip().lower().replace("-", "_")
    if "_to_" in b:
        left, right = b.split("_to_", 1)
        return left, right
    return b, b


def _native_split_quality_selection(
    profile: Mapping[str, Any],
) -> Dict[str, Any]:
    """Freeze the Native Split applicability used by the remote suite.

    Generic vendor-to-TensorRT rows and Native producer rows share the same
    generated benchmark harness.  The harness must therefore receive the
    logical Native selection explicitly; inferring applicability from the
    physical stage tokens would turn an ordinary Generic row into a Native
    Quality-FIRST producer even when Native is disabled.
    """

    split_plan = resolve_native_split_plan(profile)
    return {
        "schema": "onnx-splitpoint/native-split-quality-selection",
        "schema_version": 1,
        "applicable": bool(split_plan.enabled),
        "split_backends": list(split_plan.selected_split_backends),
        "split_selection_source": str(split_plan.source),
    }


def _prepare_suite_for_runtime(
    *,
    suite_dir: Path,
    run_root: Path,
    model_id: str,
    suite_payload: Mapping[str, Any],
    benchmark_plan: Mapping[str, Any],
    model_task: str = "",
    quality_gate_policy: Optional[Mapping[str, Any]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Repair/validate suite files before local or remote execution.

    v49j adds this explicit preflight because v49i generated valid split
    manifests, but the remote minimal bundle lacked the suite-root full ONNX and
    matrix runs used compact stage strings that the old remote template skipped.
    """
    warnings: List[str] = []
    repairs: List[Dict[str, Any]] = []
    errors: List[str] = []

    # Ensure every split_manifest.json can resolve its full/reference model.
    cases = _as_list(suite_payload.get("cases"))
    for raw in cases:
        if not isinstance(raw, Mapping):
            continue
        case_key = str(raw.get("case_dir") or raw.get("folder") or raw.get("case_id") or "").strip()
        if not case_key:
            continue
        cdir = suite_dir / case_key
        mpath = cdir / "split_manifest.json"
        manifest = read_json(mpath, default={}) if mpath.is_file() else {}
        if not isinstance(manifest, Mapping):
            continue
        full_rel = str(manifest.get("full_model") or manifest.get("full") or manifest.get("model") or "").strip()
        if not full_rel:
            full_rel = str(manifest.get("source_full_model") or "").strip()
        if not full_rel:
            errors.append(f"{case_key}: split manifest has no full_model path")
            continue
        full_path = Path(full_rel).expanduser()
        if not full_path.is_absolute():
            full_path = (cdir / full_path).resolve()
        if full_path.is_file():
            continue

        # Repair from local source if available. This works before remote
        # packaging and also for local generated-suite execution.
        source_candidates: List[Path] = []
        for key in ("full_model_source", "source_full_model"):
            src_raw = str(manifest.get(key) or "").strip()
            if src_raw:
                source_candidates.append(Path(src_raw).expanduser())
        models_obj = manifest.get("models")
        if isinstance(models_obj, Mapping):
            full_obj = models_obj.get("full")
            if isinstance(full_obj, Mapping):
                src_raw = str(full_obj.get("source") or "").strip()
                if src_raw:
                    source_candidates.append(Path(src_raw).expanduser())
        bench_full = str(suite_payload.get("full_model") or "").strip()
        if bench_full:
            bpath = Path(bench_full).expanduser()
            if not bpath.is_absolute():
                bpath = suite_dir / bpath
            source_candidates.append(bpath)

        repaired = False
        for src in source_candidates:
            try:
                if src.is_file():
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    if str(src.resolve()) != str(full_path.resolve()):
                        shutil.copy2(src, full_path)
                    repairs.append({"case_id": case_key, "repair": "copied_full_model", "source": str(src), "destination": str(full_path)})
                    repaired = True
                    break
            except Exception as exc:
                warnings.append(f"{case_key}: full_model repair from {src} failed: {type(exc).__name__}: {exc}")
        if not repaired and not full_path.is_file():
            errors.append(f"{case_key}: full_model target missing: {full_path}")

    # Normalize benchmark_plan stage descriptors for matrix runs and bind the
    # model-specific task plus the exact task-quality policy before any local
    # or remote dispatch.  v60g could regenerate a ResNet suite as detection
    # because remote execution received benchmark_task=auto and inferred from
    # stale COCO metadata.  The model entry is authoritative in v60i.
    plan_path = suite_dir / "benchmark_plan.json"
    plan_payload = read_json(plan_path, default=dict(benchmark_plan or {})) if plan_path.is_file() else dict(benchmark_plan or {})
    if not isinstance(plan_payload, dict):
        plan_payload = {}
    runs = plan_payload.get("runs") or plan_payload.get("planned_runs") or []
    changed_plan = not plan_path.is_file()
    task = str(model_task or "").strip().lower()
    if task not in {"classification", "detection"}:
        task = ""
    qpolicy = dict(quality_gate_policy or {}) if isinstance(quality_gate_policy, Mapping) else {}
    qpolicy_hash = AccuracyGatePolicy.from_mapping(qpolicy).sha256() if qpolicy else ""
    if task:
        for key in ("task", "model_task", "benchmark_task"):
            if str(plan_payload.get(key) or "").strip().lower() != task:
                plan_payload[key] = task
                changed_plan = True
    if qpolicy:
        if plan_payload.get("quality_gate") != qpolicy:
            plan_payload["quality_gate"] = qpolicy
            changed_plan = True
        if str(plan_payload.get("quality_gate_policy_sha256") or "") != qpolicy_hash:
            plan_payload["quality_gate_policy_sha256"] = qpolicy_hash
            changed_plan = True
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            rtype = str(run.get("type") or run.get("kind") or "").strip().lower()
            backend = str(run.get("backend") or run.get("id") or run.get("run_id") or "").strip().lower()
            left, right = _split_backend_pair(backend)
            if rtype == "matrix" or "_to_" in backend:
                if not run.get("stage1"):
                    run["stage1"] = _stage_desc_from_token(left)
                    changed_plan = True
                if not run.get("stage2"):
                    run["stage2"] = _stage_desc_from_token(right)
                    changed_plan = True
            # Convert compact strings to structured descriptors for old runner
            # templates while keeping string support in the new template.
            if isinstance(run.get("stage1"), str):
                run["stage1_token"] = run.get("stage1")
                run["stage1"] = _stage_desc_from_token(str(run.get("stage1")))
                changed_plan = True
            if isinstance(run.get("stage2"), str):
                run["stage2_token"] = run.get("stage2")
                run["stage2"] = _stage_desc_from_token(str(run.get("stage2")))
                changed_plan = True
            if task:
                if str(run.get("benchmark_task") or "").strip().lower() != task:
                    run["benchmark_task"] = task
                    changed_plan = True
                if str(run.get("task") or "").strip().lower() != task:
                    run["task"] = task
                    changed_plan = True
            if qpolicy:
                if run.get("task_quality_gate") != qpolicy:
                    run["task_quality_gate"] = qpolicy
                    changed_plan = True
                if str(run.get("quality_gate_policy_sha256") or "") != qpolicy_hash:
                    run["quality_gate_policy_sha256"] = qpolicy_hash
                    changed_plan = True
    if changed_plan:
        plan_payload["runs"] = runs
        plan_payload["planned_runs"] = runs
        plan_payload["stage_schema_normalized_by"] = "v60i-final-evidence-pipeline-fixes"
        write_json(plan_path, plan_payload)
        repairs.append({"repair": "normalized_benchmark_plan_stages", "path": str(plan_path), "run_count": len(runs) if isinstance(runs, list) else 0})

    # Keep benchmark_set.json self-describing as well. Remote bundles and
    # recovery tools may inspect it before benchmark_plan.json.
    suite_json = suite_dir / "benchmark_set.json"
    suite_obj = read_json(suite_json, default={}) if suite_json.is_file() else {}
    suite_changed = False
    if isinstance(suite_obj, dict):
        if task:
            for key in ("task", "model_task", "benchmark_task"):
                if str(suite_obj.get(key) or "").strip().lower() != task:
                    suite_obj[key] = task
                    suite_changed = True
        if qpolicy:
            if suite_obj.get("quality_gate") != qpolicy:
                suite_obj["quality_gate"] = qpolicy
                suite_changed = True
            if str(suite_obj.get("quality_gate_policy_sha256") or "") != qpolicy_hash:
                suite_obj["quality_gate_policy_sha256"] = qpolicy_hash
                suite_changed = True
        if suite_changed:
            write_json(suite_json, suite_obj)
            repairs.append({"repair": "bound_model_task_and_quality_policy", "path": str(suite_json), "model_task": task, "quality_gate_policy_sha256": qpolicy_hash})

    status = "ok" if not errors else "partial"
    payload = {
        "schema": "onnx-splitpoint/suite-packaging-preflight",
        "schema_version": 1,
        "model_id": model_id,
        "suite_dir": relpath(suite_dir, run_root),
        "status": status,
        "repairs": repairs,
        "warnings": warnings,
        "errors": errors,
        "model_task": task,
        "quality_gate_policy_sha256": qpolicy_hash,
        "created_at": now_iso(),
    }
    p_preflight = write_json(suite_dir / "suite_packaging_preflight.json", payload)
    payload["path"] = relpath(p_preflight, run_root)
    if callable(log):
        if repairs:
            log(f"[workflow] suite runtime preflight repaired {len(repairs)} issue(s) for {model_id}")
        if errors:
            log(f"[workflow] suite runtime preflight still has {len(errors)} error(s) for {model_id}")
    return payload


def finalize_suite_for_runtime(
    *,
    suite_dir: Path,
    run_root: Path,
    model_id: str,
    suite_payload: Mapping[str, Any],
    benchmark_plan: Mapping[str, Any],
    profile_payload: Optional[Mapping[str, Any]] = None,
    model_task: str = "",
    quality_gate_policy: Optional[Mapping[str, Any]] = None,
    log: Optional[Callable[[str], None]] = None,
    require_existing_cpu_reference: bool = False,
) -> Dict[str, Any]:
    """Normalize runtime rows, then seal and mirror their exact final form.

    The generator's formal plan is the authority before runtime preparation.
    Once stage, task and quality fields have been normalized in the executable
    suite, that executable plan becomes authoritative.  The management CPU
    reference finalizer then mirrors one canonical row list and computes its
    diagnostic hash.  Nothing may schedule CPU work or dispatch remotely
    between these two operations.
    """

    profile = dict(profile_payload or {})
    executable_plan_path = suite_dir / "benchmark_plan.json"
    formal_dir = run_root / "models" / str(model_id) / "benchmark_set"
    formal_plan_path = formal_dir / "benchmark_plan.json"
    seed_plan = read_json(
        executable_plan_path, default=benchmark_plan,
    ) or {}
    if not isinstance(seed_plan, Mapping):
        raise ValueError("runtime_benchmark_plan_invalid_before_normalization")
    seed_plan = dict(seed_plan)
    native_split_quality_selection = _native_split_quality_selection(profile)
    seed_plan_changed = not executable_plan_path.is_file()
    if (
        seed_plan.get("native_split_quality_selection")
        != native_split_quality_selection
    ):
        seed_plan["native_split_quality_selection"] = (
            native_split_quality_selection
        )
        seed_plan_changed = True
    effective_model_task = str(
        model_task
        or seed_plan.get("model_task")
        or seed_plan.get("task")
        or suite_payload.get("model_task")
        or suite_payload.get("task")
        or ""
    ).strip().lower()
    effective_quality_gate = (
        dict(quality_gate_policy)
        if isinstance(quality_gate_policy, Mapping) and quality_gate_policy
        else dict(profile.get("quality_gate") or {})
        if isinstance(profile.get("quality_gate"), Mapping)
        else dict(suite_payload.get("quality_gate") or {})
        if isinstance(suite_payload.get("quality_gate"), Mapping)
        else {}
    )
    automatic_cpu = not profile_has_explicit_cpu_reference(
        profile,
        _as_list(seed_plan.get("targets")),
    )
    cache_verify_enabled = bool(cache_verify_guard(profile))
    if (
        management_cpu_reference_required(profile)
        and not cache_verify_enabled
        and not require_existing_cpu_reference
    ):
        seed_runs = [
            dict(row)
            for row in _as_list(
                seed_plan.get("runs") or seed_plan.get("planned_runs")
            )
            if isinstance(row, Mapping)
        ]
        seed_runs = bind_management_cpu_reference_runs(
            seed_runs,
            automatic=automatic_cpu,
            require_existing=False,
        )
        seed_plan["runs"] = [dict(row) for row in seed_runs]
        seed_plan["planned_runs"] = [dict(row) for row in seed_runs]
        seed_plan_changed = True
    if seed_plan_changed:
        write_json(executable_plan_path, seed_plan)

    runtime_preflight = _prepare_suite_for_runtime(
        suite_dir=suite_dir,
        run_root=run_root,
        model_id=model_id,
        suite_payload=suite_payload,
        benchmark_plan=benchmark_plan,
        model_task=effective_model_task,
        quality_gate_policy=effective_quality_gate,
        log=log,
    )
    normalized_plan = read_json(
        executable_plan_path, default=benchmark_plan,
    ) or {}
    if not isinstance(normalized_plan, Mapping):
        raise ValueError("runtime_benchmark_plan_invalid_after_normalization")
    contract_paths = tuple(
        path
        for path in (
            suite_dir / "benchmark_set.json",
            formal_dir / "benchmark_set.json",
        )
        if path.is_file()
    )
    invariant = finalize_management_cpu_reference_plan_aliases(
        executable_plan_path=executable_plan_path,
        formal_plan_path=formal_plan_path,
        profile=profile,
        cache_verify_enabled=cache_verify_enabled,
        automatic=automatic_cpu,
        require_existing=True,
        benchmark_set_paths=contract_paths,
        authoritative_alias="executable",
    )
    authoritative_plan = read_json(
        executable_plan_path, default=normalized_plan,
    ) or {}
    if not isinstance(authoritative_plan, Mapping):
        raise ValueError("runtime_benchmark_plan_invalid_before_alias_sync")
    authoritative_plan = dict(authoritative_plan)
    management_invariant_verified = (
        str(invariant.get("status") or "") == "verified"
    )
    if not management_invariant_verified:
        authoritative_plan.pop("management_cpu_reference_invariant", None)
    authoritative_runs = [
        dict(row)
        for row in _as_list(
            authoritative_plan.get("runs")
            or authoritative_plan.get("planned_runs")
        )
        if isinstance(row, Mapping)
    ]
    runtime_keys = (
        "task",
        "model_task",
        "benchmark_task",
        "quality_gate",
        "quality_gate_policy_sha256",
        "stage_schema_normalized_by",
        "native_split_quality_selection",
    )

    def _mirror_plan(base: Any) -> Dict[str, Any]:
        mirrored = dict(base) if isinstance(base, Mapping) else {}
        mirrored["runs"] = [dict(row) for row in authoritative_runs]
        mirrored["planned_runs"] = [dict(row) for row in authoritative_runs]
        for key in runtime_keys:
            if key in authoritative_plan:
                mirrored[key] = authoritative_plan[key]
        if "management_cpu_reference_invariant" in authoritative_plan:
            mirrored["management_cpu_reference_invariant"] = dict(
                authoritative_plan["management_cpu_reference_invariant"]
            )
        else:
            mirrored.pop("management_cpu_reference_invariant", None)
        return mirrored

    for path in dict.fromkeys((executable_plan_path, formal_plan_path)):
        write_json(path, _mirror_plan(read_json(path, default={})))
    for contract_path in dict.fromkeys(contract_paths):
        contract = read_json(contract_path, default={}) or {}
        if not isinstance(contract, Mapping):
            raise ValueError(f"runtime_benchmark_contract_invalid:{contract_path}")
        contract_out = dict(contract)
        contract_out["planned_runs"] = [
            dict(row) for row in authoritative_runs
        ]
        if "runs" in contract_out:
            contract_out["runs"] = [dict(row) for row in authoritative_runs]
        for key in runtime_keys:
            if key in authoritative_plan:
                contract_out[key] = authoritative_plan[key]
        embedded = contract_out.get("plan")
        if isinstance(embedded, Mapping):
            contract_out["plan"] = _mirror_plan(embedded)
        if "management_cpu_reference_invariant" in authoritative_plan:
            contract_out["management_cpu_reference_invariant"] = dict(
                authoritative_plan["management_cpu_reference_invariant"]
            )
        else:
            contract_out.pop("management_cpu_reference_invariant", None)
        write_json(contract_path, contract_out)

    finalized_plan = read_json(executable_plan_path, default={}) or {}
    if not isinstance(finalized_plan, Mapping):
        raise ValueError("runtime_benchmark_plan_invalid_after_finalization")
    formal_final = read_json(formal_plan_path, default={}) or {}
    if (
        not isinstance(formal_final, Mapping)
        or _as_list(formal_final.get("runs")) != _as_list(finalized_plan.get("runs"))
        or _as_list(formal_final.get("planned_runs"))
        != _as_list(finalized_plan.get("runs"))
    ):
        raise ValueError("runtime_benchmark_plan_alias_sync_failed")
    return {
        "status": "verified",
        "runtime_preflight": dict(runtime_preflight),
        "cpu_reference_invariant": dict(invariant),
        "benchmark_plan": dict(finalized_plan),
        "executable_plan_path": str(executable_plan_path),
        "formal_plan_path": str(formal_plan_path),
        "authoritative_alias": "executable",
    }


def _load_remote_hosts_file(path: str | Path) -> List[Dict[str, Any]]:
    raw = str(path or "").strip()
    if not raw:
        return []
    p = Path(raw).expanduser()
    if not p.is_file():
        return []
    try:
        if p.suffix.lower() in {".yaml", ".yml"}:
            import yaml  # type: ignore
            payload = yaml.safe_load(p.read_text(encoding="utf-8"))
        else:
            payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, Mapping):
        for key in ("remote_hosts", "hosts", "items"):
            val = payload.get(key)
            if isinstance(val, list):
                return [dict(x) for x in val if isinstance(x, Mapping)]
        return [dict(payload)]
    if isinstance(payload, list):
        return [dict(x) for x in payload if isinstance(x, Mapping)]
    return []


def _profile_remote_execution(profile_payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(profile_payload, Mapping):
        return {}
    raw = profile_payload.get("remote_execution") or profile_payload.get("remote") or {}
    return dict(raw or {}) if isinstance(raw, Mapping) else {}


def _profile_parallel_remote(profile_payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(profile_payload, Mapping):
        return {}
    workflow = profile_payload.get("workflow") if isinstance(profile_payload.get("workflow"), Mapping) else {}
    bench = profile_payload.get("benchmark_execution") if isinstance(profile_payload.get("benchmark_execution"), Mapping) else {}
    remote = profile_payload.get("remote_execution") if isinstance(profile_payload.get("remote_execution"), Mapping) else {}
    candidates = [
        profile_payload.get("parallel_remote"),
        profile_payload.get("parallel_execution"),
        workflow.get("parallel_remote"),
        bench.get("parallel_remote"),
        remote.get("parallel_remote"),
    ]
    out: Dict[str, Any] = {}
    for c in candidates:
        if isinstance(c, Mapping):
            out.update(dict(c))
    for key in ("parallel_remote_setups", "max_parallel_setups", "max_parallel_uploads", "powercalc_workers"):
        if key in workflow and key not in out:
            out[key] = workflow.get(key)
        if key in bench and key not in out:
            out[key] = bench.get(key)
        if key in remote and key not in out:
            out[key] = remote.get(key)
    return out


def _parallel_remote_setups_enabled(options: Any, profile_payload: Optional[Mapping[str, Any]], *, default: bool = True) -> bool:
    raw = getattr(options, "parallel_remote_setups", None)
    cfg = _profile_parallel_remote(profile_payload)
    if raw in (None, ""):
        raw = cfg.get("enabled", cfg.get("parallel_remote_setups", None))
    if raw in (None, ""):
        env = os.environ.get("ONNX_SPLITPOINT_PARALLEL_REMOTE_SETUPS", "")
        if env != "":
            raw = env
    if raw in (None, ""):
        return bool(default)
    return _bool(raw)


def _parallel_remote_max_setups(options: Any, profile_payload: Optional[Mapping[str, Any]], *, default: int = 3) -> int:
    cfg = _profile_parallel_remote(profile_payload)
    raw = getattr(options, "max_parallel_setups", None)
    if raw in (None, "", 0, "0"):
        raw = cfg.get("max_parallel_setups", cfg.get("setup_workers", None))
    if raw in (None, "", 0, "0"):
        raw = os.environ.get("ONNX_SPLITPOINT_MAX_PARALLEL_SETUPS", "")
    return max(1, _int(raw, default))


def _parallel_remote_max_uploads(options: Any, profile_payload: Optional[Mapping[str, Any]], *, default: int = 1) -> int:
    cfg = _profile_parallel_remote(profile_payload)
    raw = getattr(options, "max_parallel_uploads", None)
    if raw in (None, "", 0, "0"):
        raw = cfg.get("max_parallel_uploads", None)
    if raw in (None, "", 0, "0"):
        raw = os.environ.get("ONNX_SPLITPOINT_MAX_PARALLEL_UPLOADS", "")
    return max(0, _int(raw, default))


def _parallel_powercalc_workers(options: Any, profile_payload: Optional[Mapping[str, Any]], *, default: int = 1) -> int:
    cfg = _profile_parallel_remote(profile_payload)
    raw = getattr(options, "powercalc_workers", None)
    if raw in (None, "", 0, "0"):
        raw = cfg.get("powercalc_workers", cfg.get("max_parallel_powercalc", None))
    if raw in (None, "", 0, "0"):
        raw = os.environ.get("ONNX_SPLITPOINT_POWER_CALC_WORKERS", "")
    return max(0, _int(raw, default))


def _hardware_targets_for_plan(profile_payload: Optional[Mapping[str, Any]], benchmark_plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(profile_payload, Mapping):
        return []
    targets = matrix_for_runtime(profile_payload)
    if not targets:
        return []
    # Treat an empty/contract-only plan as no filter.  Older code used
    # json.dumps({}) == '{}' which is truthy and accidentally filtered out all
    # selected hardware setups before the real suite benchmark_plan.json was
    # loaded.  That caused Evaluation Workflow remote dispatch to report
    # 'remote_requested_but_no_host_configured' despite hardware.selected_setups.
    if not isinstance(benchmark_plan, Mapping) or not dict(benchmark_plan or {}):
        plan_text = ""
    else:
        plan_text = json.dumps(dict(benchmark_plan or {}), ensure_ascii=False).lower()
        if plan_text.strip() in {"{}", "[]", "null"}:
            plan_text = ""
    selected: List[Dict[str, Any]] = []
    for t in targets:
        acc = canon_accelerator(t.get("accelerator"))
        if not acc:
            continue
        # Only dispatch to hardware setups that are relevant to the suite plan.
        # If the profile contains an explicit hardware_targets block but the
        # benchmark_plan is still contract-like/empty, keep the target so the
        # execution status explains the selected deployment matrix.
        if not plan_text:
            selected.append(dict(t))
        elif acc.startswith("hailo10"):
            if "hailo10" in plan_text or "hailo_10" in plan_text or "hailo-10" in plan_text:
                selected.append(dict(t))
        elif acc == "hailo8":
            # Generic "hailo" in a plan with explicit hailo10h rows is not a
            # Hailo-8 request.  This prevents hailo10h suites from being sent to
            # the Hailo-8 testbed.
            if ("hailo10" not in plan_text and "hailo_10" not in plan_text and "hailo-10" not in plan_text) and ("hailo8" in plan_text or "hailo_8" in plan_text or "hailo-8" in plan_text or '"hailo"' in plan_text):
                selected.append(dict(t))
        elif acc == "deepx_m1":
            if "deepx" in plan_text or "dx_m1" in plan_text or "dxm1" in plan_text:
                selected.append(dict(t))
        elif acc in plan_text:
            selected.append(dict(t))
    return selected



def _plan_run_entries(benchmark_plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(benchmark_plan, Mapping):
        return []
    for key in ("runs", "planned_runs", "run_profiles", "matrix_runs"):
        rows = benchmark_plan.get(key)
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
            return [dict(x) for x in rows if isinstance(x, Mapping)]
    return []


def _run_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or row.get("name") or row.get("run_id") or row.get("backend") or row.get("type") or "").strip()


def _run_mentions_accelerator(row: Mapping[str, Any], accelerator: str) -> bool:
    acc = canon_accelerator(accelerator)
    text = json.dumps(dict(row or {}), ensure_ascii=False).lower()
    if acc == "deepx_m1":
        return "deepx" in text or "dx_m1" in text or "dxm1" in text
    # v59ab: keep Hailo architecture routing strict.  A row may contain generic
    # provider text like "hailo" in addition to "hailo10h".  Generic Hailo
    # must not match the Hailo-8 setup when an explicit Hailo-10 marker exists.
    has_hailo10 = ("hailo10" in text) or ("hailo_10" in text) or ("hailo-10" in text) or ("h10" in text)
    has_hailo8 = ("hailo8" in text) or ("hailo_8" in text) or ("hailo-8" in text) or ("h8" in text)
    if acc.startswith("hailo10"):
        return has_hailo10
    if acc == "hailo8":
        return (not has_hailo10) and (has_hailo8 or '"hailo"' in text)
    return bool(acc and acc in text)


def _run_mentions_any_accelerator(row: Mapping[str, Any]) -> bool:
    text = json.dumps(dict(row or {}), ensure_ascii=False).lower()
    return (
        "deepx" in text or "dx_m1" in text or "dxm1" in text
        or "hailo8" in text or "hailo10" in text or '"hailo"' in text
    )


def _run_is_deferred(row: Mapping[str, Any]) -> bool:
    if bool(row.get("deferred")):
        return True
    status = str(row.get("status") or row.get("build_status") or "").strip().lower()
    return status in {"deferred", "deferred_cold_build", "deferred_cold_full_cache_miss"}


def _reference_run_ids_for_plan(benchmark_plan: Mapping[str, Any]) -> List[str]:
    """Return pure host-reference run ids (TensorRT/CUDA/CPU/ORT).

    These rows do not belong to Hailo/DeepX accelerators, but one configured
    NX still has to execute them so accelerator rows have a reference.
    """
    out: List[str] = []
    for row in _plan_run_entries(benchmark_plan):
        if _run_is_deferred(row):
            continue
        rid = _run_id(row)
        if not rid:
            continue
        if _run_mentions_any_accelerator(row):
            continue
        text = json.dumps(dict(row or {}), ensure_ascii=False).lower()
        if any(tok in text for tok in ("tensorrt", "trt", "cuda", "cpu", "onnxruntime", "ort_")):
            if rid not in out:
                out.append(rid)
    return out


def _is_cpu_reference_run_v263(row: Mapping[str, Any]) -> bool:
    """Return whether a plan row is the semantic ORT-CPU reference.

    CUDA/TensorRT ORT rows remain performance candidates.  Only an explicitly
    CPU-bound row is removed from accelerator execution when the central
    management reference service is active.
    """
    aliases = {"cpu", "cpu_ort", "ort_cpu", "onnxruntime_cpu", "cpu_onnxruntime"}
    rid = _run_id(row).strip().lower().replace("-", "_")
    if rid in aliases:
        return True
    for key in ("backend", "provider", "execution_provider", "device", "runtime"):
        value = str(row.get(key) or "").strip().lower().replace("-", "_")
        if value in aliases:
            return True
    return bool(row.get("semantic_reference_only") or row.get("canonical_cpu_reference"))


def _performance_run_ids_v263(benchmark_plan: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for row in _plan_run_entries(benchmark_plan):
        if _run_is_deferred(row) or _is_cpu_reference_run_v263(row):
            continue
        rid = _run_id(row)
        if rid and rid not in out:
            out.append(rid)
    return out


def _reference_target_id_for_matrix(targets: Sequence[Mapping[str, Any]]) -> str:
    """Pick the setup that should run pure TensorRT/CUDA/CPU reference rows."""
    enabled = [t for t in targets if isinstance(t, Mapping)]
    # Prefer DeepX because this project validates TensorRT in the deepx-runtime
    # venv.  Otherwise use the first selected setup so pure reference rows are
    # not silently dropped.
    for t in enabled:
        if canon_accelerator(t.get("accelerator")) == "deepx_m1":
            return str(t.get("id") or "")
    return str(enabled[0].get("id") or "") if enabled else ""


def _run_ids_for_hardware_target(hw: Mapping[str, Any], benchmark_plan: Mapping[str, Any]) -> List[str]:
    runs = _plan_run_entries(benchmark_plan)
    if not runs:
        return []
    acc = canon_accelerator(hw.get("accelerator") or hw.get("backend") or hw.get("provider"))
    ids: List[str] = []
    for row in runs:
        if _run_is_deferred(row):
            continue
        rid = _run_id(row)
        if not rid:
            continue
        # Explicit run_profile.hardware_setup_id wins when present.
        hw_ref = str(row.get("hardware_setup_id") or row.get("hardware_setup") or row.get("target_setup") or "").strip()
        if hw_ref and hw_ref == str(hw.get("id") or ""):
            ids.append(rid)
            continue
        if _run_mentions_accelerator(row, acc):
            ids.append(rid)
    # keep order, de-duplicate
    out: List[str] = []
    for rid in ids:
        if rid not in out:
            out.append(rid)
    return out


_SCHEDULER_OWNED_REMOTE_FLAGS = {
    "--plan",
    "--provider",
    "--run-id",
    "--run-ids",
    "--quality-only-run-ids",
    "--quality-evidence-eval-id",
    "--quality-evidence-model-id",
    "--quality-evidence-setup-id",
    "--quality-evidence-endpoint-id",
}


def _scheduler_owned_remote_add_args(
    raw: Any,
    *,
    run_ids: Sequence[str],
    quality_only_run_ids: Sequence[str] = (),
) -> str:
    """Bind scheduler-owned selection flags after removing stale overrides.

    ``remote.add_args`` is intentionally an advanced passthrough.  Physical
    setup identity and the sole TensorRT performance owner are not advanced
    knobs, though: allowing a second ``--run-ids`` or
    ``--quality-only-run-ids`` occurrence would let argparse's last-value rule
    silently change the sealed dispatch matrix.
    """

    try:
        tokens = shlex.split(str(raw or ""))
    except ValueError as exc:
        raise RuntimeError(f"remote_add_args_invalid:{exc}") from exc
    kept: List[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        name = token.split("=", 1)[0]
        if name in _SCHEDULER_OWNED_REMOTE_FLAGS:
            if "=" not in token and index + 1 < len(tokens):
                index += 2
            else:
                index += 1
            continue
        kept.append(token)
        index += 1
    selected = [str(value).strip() for value in run_ids if str(value).strip()]
    quality_only = [
        str(value).strip() for value in quality_only_run_ids
        if str(value).strip()
    ]
    if selected:
        kept.extend(["--run-ids", ",".join(dict.fromkeys(selected))])
    if quality_only:
        kept.extend([
            "--quality-only-run-ids",
            ",".join(dict.fromkeys(quality_only)),
        ])
    return shlex.join(kept)


def _targeted_full_quality_identity_key(
    row: Mapping[str, Any],
) -> tuple[str, str, str, str, str, str, str, bool | None]:
    """Return the sealed identity used by a targeted Full-quality dispatch."""

    source_run_id = str(
        row.get("source_run_id") or row.get("run_id") or ""
    ).strip().lower().replace("-", "_")
    dispatch_run_id = str(
        row.get("dispatch_run_id")
        or ("ort_tensorrt" if source_run_id == "native_full_tensorrt" else source_run_id)
    ).strip().lower().replace("-", "_")
    backend = str(row.get("backend") or "").strip().lower().replace("-", "_")
    backend = {
        "trt": "tensorrt",
        "ort_tensorrt": "tensorrt",
        "native_tensorrt": "tensorrt",
        "native_full_tensorrt": "tensorrt",
    }.get(backend, backend)
    return (
        str(row.get("id") or "").strip(),
        str(row.get("setup_id") or "").strip(),
        source_run_id,
        dispatch_run_id,
        backend,
        str(row.get("variant") or "").strip().lower(),
        str(row.get("execution_role") or "").strip().lower(),
        row.get("performance_claims_emitted"),
    )


def _remote_transport_run_id(
    *,
    run_root: Path,
    model_id: str,
    target_id: str,
    gates: Mapping[str, Any],
    workflow_session_id: str = "",
) -> str:
    """Keep a targeted repair out of the original remote result workspace.

    The EvaluationRun identity remains ``run_root.name`` and is sealed inside
    every quality request.  This suffix changes only the disposable transport
    workspace so stale Full-quality requests from the original dispatch cannot
    contaminate the exact missing-identity subset.
    """

    transport_id = f"{run_root.name}_{model_id}"
    if target_id:
        transport_id += f"_{target_id}"
    if gates.get("targeted_missing_full_quality_only") is True:
        if (
            not isinstance(workflow_session_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", workflow_session_id) is None
        ):
            raise ValueError(
                "targeted_missing_full_quality_workflow_session_id_invalid"
            )
        transport_id += (
            f"_missing_full_quality_resume_{workflow_session_id}"
        )
    return transport_id


def _filter_setup_local_dispatch_for_targeted_full_quality(
    contract: Mapping[str, Any],
    targeted_identities: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Reduce a Full-only setup contract to an exact caller-sealed subset.

    This helper does not select endpoints.  The caller has already derived the
    missing identities from the archived effective plan and central summary.
    Every requested identity must match one and only one identity in the normal
    setup-local dispatch contract; otherwise no remote task is admitted.
    """

    filtered = dict(contract or {})
    if filtered.get("ok") is not True:
        raise RuntimeError("targeted_full_quality_source_contract_not_ready")
    requested = [
        dict(row) for row in targeted_identities if isinstance(row, Mapping)
    ]
    requested_keys = [_targeted_full_quality_identity_key(row) for row in requested]
    if (
        not requested_keys
        or len(set(requested_keys)) != len(requested_keys)
        or any(
            not all(key[:7])
            or key[6] != "full_quality_only"
            or key[7] is not False
            for key in requested_keys
        )
    ):
        raise RuntimeError("targeted_full_quality_identity_set_invalid")

    selected_dispatches: List[Dict[str, Any]] = []
    selected_identities: List[Dict[str, Any]] = []
    matched_keys: set[
        tuple[str, str, str, str, str, str, str, bool | None]
    ] = set()
    for raw_dispatch in list(filtered.get("setup_dispatches") or []):
        if not isinstance(raw_dispatch, Mapping):
            continue
        dispatch = dict(raw_dispatch)
        expected = [
            dict(row)
            for row in list(dispatch.get("expected_full_quality_identities") or [])
            if isinstance(row, Mapping)
        ]
        expected_by_key: Dict[
            tuple[str, str, str, str, str, str, str, bool | None],
            List[Dict[str, Any]],
        ] = {}
        for row in expected:
            expected_by_key.setdefault(
                _targeted_full_quality_identity_key(row), []
            ).append(row)
        setup_selected: List[Dict[str, Any]] = []
        for key in requested_keys:
            matches = expected_by_key.get(key, [])
            if not matches:
                continue
            if len(matches) != 1 or key in matched_keys:
                raise RuntimeError(
                    "targeted_full_quality_identity_not_unique_in_dispatch"
                )
            matched_keys.add(key)
            setup_selected.append(dict(matches[0]))
        if not setup_selected:
            continue

        run_ids = [
            _targeted_full_quality_identity_key(row)[3]
            for row in setup_selected
        ]
        run_ids = list(dict.fromkeys(run_ids))
        original_run_ids = {
            str(value).strip().lower().replace("-", "_")
            for value in list(dispatch.get("run_ids") or [])
            if str(value).strip()
        }
        if not run_ids or any(run_id not in original_run_ids for run_id in run_ids):
            raise RuntimeError(
                "targeted_full_quality_dispatch_run_id_not_in_source_contract"
            )
        trt_rows = [
            row for row in setup_selected
            if _targeted_full_quality_identity_key(row)[4] == "tensorrt"
        ]
        if len(trt_rows) > 1:
            raise RuntimeError("targeted_full_quality_tensorrt_identity_not_unique")
        trt_id = str(trt_rows[0].get("id") or "") if trt_rows else ""
        quality_companion_identity = (
            dict(dispatch.get("quality_companion_identity") or {})
            if trt_rows else {}
        )
        if trt_rows and str(quality_companion_identity.get("id") or "") != trt_id:
            raise RuntimeError(
                "targeted_full_quality_companion_identity_mismatch"
            )
        dispatch.update({
            "run_ids": run_ids,
            "quality_only_run_ids": list(run_ids),
            "quality_ids": [str(row.get("id") or "") for row in setup_selected],
            "expected_full_quality_identities": setup_selected,
            "tensorrt_execution_role": "full_quality_only" if trt_rows else "",
            "quality_companion_required": bool(trt_rows),
            "quality_companion_endpoint_id": trt_id,
            "quality_companion_identity": quality_companion_identity,
            "performance_claims_emitted": False,
            "targeted_missing_full_quality_only": True,
        })
        selected_dispatches.append(dispatch)
        selected_identities.extend(setup_selected)

    if matched_keys != set(requested_keys):
        raise RuntimeError("targeted_full_quality_identity_not_in_source_contract")
    filtered.update({
        "setup_dispatches": selected_dispatches,
        "expected_full_quality_identities": selected_identities,
        "targeted_missing_full_quality_only": True,
        "targeted_identity_count": len(selected_identities),
    })
    return filtered




def _profile_energy(profile_payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(profile_payload, Mapping):
        return {}
    raw = profile_payload.get("energy") or profile_payload.get("energy_measurement") or {}
    cfg = dict(raw or {}) if isinstance(raw, Mapping) else {}

    # v60 final-campaign profiles keep scientific measurement semantics in a
    # dedicated block instead of overloading the legacy remote-run options.
    # Merge the compatible fields here so a strict campaign profile changes the
    # actual acquisition path, not only the preflight report.
    measurement = profile_payload.get("measurement_campaign")
    measurement = dict(measurement or {}) if isinstance(measurement, Mapping) else {}
    system_power = measurement.get("system_power")
    system_power = dict(system_power or {}) if isinstance(system_power, Mapping) else {}
    campaign = profile_payload.get("campaign")
    campaign = dict(campaign or {}) if isinstance(campaign, Mapping) else {}
    if system_power:
        cfg.setdefault("enabled", str(campaign.get("mode") or "").strip().lower() == "final")
        cfg.setdefault("repeat_override", system_power.get("repeats"))
        cfg.setdefault("confidence_level", system_power.get("confidence_level", 0.95))
        cfg.setdefault("physical_scope", system_power.get("scope", "FS"))
        cfg.setdefault("window_label", system_power.get("window", "command"))
        cfg.setdefault("randomize_target_order", system_power.get("randomize_run_order", False))
        cfg.setdefault("randomization_seed", system_power.get("randomization_seed", 20260710))
        cfg.setdefault("strict", str(campaign.get("enforcement") or "").strip().lower() == "strict")
        cfg.setdefault("scope", "row_variant")
    return cfg

def _energy_enabled_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> bool:
    """Enable Generic collection only for an explicit, valid Generic request.

    Native-only remains the default Evaluation Workflow contract.  This
    function deliberately does not infer Generic Energy from the legacy
    umbrella ``energy.enabled`` flag; callers must select ``generic_enabled``
    or a Generic measurement path explicitly.
    """

    if not isinstance(profile_payload, Mapping):
        return False
    top_energy = (
        profile_payload.get("energy")
        if isinstance(profile_payload.get("energy"), Mapping) else {}
    )
    explicit_generic = bool(
        _bool(top_energy.get("generic_enabled"))
        or str(top_energy.get("measurement_path") or "")
        .strip().lower() in {"generic", "native_and_generic"}
    )
    if not explicit_generic:
        return False
    from onnx_splitpoint_tool.energy.config import (
        resolve_effective_energy_config,
    )

    effective = resolve_effective_energy_config(profile_payload)
    return bool(
        effective.get("generic_energy_enabled") is True
        and effective.get("measurement_path")
        in {"generic", "native_and_generic"}
        and not effective.get("configuration_errors")
    )

def _energy_repeat_override(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> int:
    cfg = _profile_energy(profile_payload)
    # ``0`` is the documented CLI/WorkflowOptions sentinel for "use the
    # profile".  Treating it as an explicit override would silently collapse a
    # five-repeat final campaign back to the remote benchmark repeat count.
    raw = getattr(options, "energy_repeat_override", None)
    try:
        explicit = int(raw) if raw not in (None, "") else 0
    except Exception:
        explicit = 0
    if explicit > 0:
        return explicit
    return max(0, _int(cfg.get("repeat_override", cfg.get("repeats", 0)), 0))


def _energy_confidence_level_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> float:
    cfg = _profile_energy(profile_payload)
    raw = getattr(options, "energy_confidence_level", None)
    if raw in (None, ""):
        raw = cfg.get("confidence_level", 0.95)
    try:
        return min(0.999, max(0.50, float(raw)))
    except Exception:
        return 0.95


def _energy_physical_scope_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> str:
    cfg = _profile_energy(profile_payload)
    raw = getattr(options, "energy_physical_scope", None)
    if raw in (None, ""):
        raw = cfg.get("physical_scope", "")
    return str(raw or "FS").strip().upper()


def _energy_window_label_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> str:
    cfg = _profile_energy(profile_payload)
    raw = getattr(options, "energy_window_label", None)
    if raw in (None, ""):
        raw = cfg.get("window_label", "command")
    return str(raw or "command").strip().lower()


def _energy_randomize_target_order_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> bool:
    cfg = _profile_energy(profile_payload)
    raw = getattr(options, "energy_randomize_target_order", None)
    if raw in (None, ""):
        raw = cfg.get("randomize_target_order", cfg.get("randomize_run_order", False))
    return _bool(raw)


def _energy_randomization_seed_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> int:
    cfg = _profile_energy(profile_payload)
    raw = getattr(options, "energy_randomization_seed", None)
    if raw in (None, ""):
        raw = cfg.get("randomization_seed", 20260710)
    return _int(raw, 20260710)

def _energy_final_all_split_enabled(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> bool:
    """Profile-level checkbox/preset for final thesis all-split energy.

    This intentionally lives in the workflow layer so GUI-generated YAML is
    enough; users do not have to pass a fragile combination of CLI flags.
    """
    cfg = _profile_energy(profile_payload)
    for key in ("final_all_split_energy", "all_split_energy", "require_complete_split_energy", "all_split_energy_strict"):
        val = cfg.get(key)
        if _bool(val):
            return True
    return False


_CPU_ORT_ENERGY_SKIP_IDS_V59N = ("ort_cpu", "cpu_ort", "cpu")

def _energy_final_skip_cpu_ort_enabled(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> bool:
    """Whether final all-split energy should exclude CPU ORT rows.

    CPU ORT is useful as a correctness/baseline runtime, but row-level u.RECS
    windows for CPU same-backend split diagnostics are very slow and do not add
    thesis-relevant accelerator energy evidence.  v59n therefore keeps CPU ORT
    excluded by default in the GUI final-energy preset while still allowing an
    advanced opt-in via include_cpu_ort_in_final_energy=true or
    final_energy_skip_cpu_ort=false.
    """
    if not _energy_final_all_split_enabled(options, profile_payload):
        return False
    cfg = _profile_energy(profile_payload)
    # Positive opt-in wins for advanced reruns that explicitly want CPU energy.
    for key in ("include_cpu_ort_in_final_energy", "measure_cpu_ort_in_final_energy", "final_energy_include_cpu_ort"):
        if _bool(cfg.get(key)):
            return False
    # Default is True.  Accept a few explicit false aliases.
    for key in ("final_energy_skip_cpu_ort", "skip_cpu_ort_in_final_energy", "exclude_cpu_ort_in_final_energy"):
        if key in cfg:
            return _bool(cfg.get(key))
    return True

def _energy_cpu_ort_skip_ids_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> list[str]:
    return list(_CPU_ORT_ENERGY_SKIP_IDS_V59N) if _energy_final_skip_cpu_ort_enabled(options, profile_payload) else []

def _energy_scope_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> str:
    cfg = _profile_energy(profile_payload)
    if _energy_final_all_split_enabled(options, profile_payload):
        return "row_variant"
    s = str(getattr(options, "energy_scope", "") or cfg.get("scope") or "row_variant").strip().lower().replace("-", "_")
    return s if s in {"row_variant", "dispatch"} else "row_variant"

def _energy_phases_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> list[str]:
    cfg = _profile_energy(profile_payload)
    raw = getattr(options, "energy_phases", None) or cfg.get("phases") or []
    if isinstance(raw, str):
        raw = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    out = []
    for x in _as_list(raw):
        t = str(x or "").strip().lower()
        if t in {"latency", "streaming"} and t not in out:
            out.append(t)
    return out


def _energy_target_policy_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> str:
    cfg = _profile_energy(profile_payload)
    if _energy_final_all_split_enabled(options, profile_payload):
        return "all"
    raw = str(getattr(options, "energy_target_policy", "") or cfg.get("target_policy") or cfg.get("policy") or "canonical_only")
    val = raw.strip().lower().replace("-", "_")
    return val if val in {"all", "canonical_only", "deepx_only", "best_valid_only", "best_plus_predicted", "manual"} else "canonical_only"


def _energy_list_from_profile(options: Any, profile_payload: Optional[Mapping[str, Any]], attr: str, keys: tuple[str, ...]) -> list[str]:
    cfg = _profile_energy(profile_payload)
    raw = getattr(options, attr, None)
    if not raw:
        for k in keys:
            if cfg.get(k) not in (None, ""):
                raw = cfg.get(k)
                break
    if isinstance(raw, str):
        raw = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    out: list[str] = []
    for x in _as_list(raw):
        s = str(x or "").strip().lower()
        if s and s not in out:
            out.append(s)
    return out


def _energy_skip_backends_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> list[str]:
    cfg = _profile_energy(profile_payload)
    if _energy_final_all_split_enabled(options, profile_payload):
        explicit = _energy_list_from_profile(options, profile_payload, "energy_skip_backends", ("skip_backends", "skip_run_ids", "exclude_backends"))
        default_cpu_skip = _energy_cpu_ort_skip_ids_for_profile(options, profile_payload)
        cpu_ids = set(_CPU_ORT_ENERGY_SKIP_IDS_V59N)
        out: list[str] = []
        for x in list(explicit) + default_cpu_skip:
            t = str(x or "").strip().lower().replace("-", "_")
            if not t:
                continue
            # Advanced include_cpu_ort_in_final_energy=true removes stale CPU skip
            # entries from loaded profiles.  Non-CPU explicit skips still apply.
            if not default_cpu_skip and t in cpu_ids:
                continue
            if t not in out:
                out.append(t)
        return out
    # v59j: an explicit CLI ``--energy-target-policy all`` means thesis/final
    # all-split coverage.  Do not inherit old smoke-profile skip_backends
    # (ort_cpu/ort_cuda) unless the user explicitly adds --energy-skip-backend.
    cli_policy = str(getattr(options, "energy_target_policy", "") or "").strip().lower()
    cli_skip = list(getattr(options, "energy_skip_backends", []) or [])
    if cli_policy == "all" and not cli_skip:
        return []
    explicit = _energy_list_from_profile(options, profile_payload, "energy_skip_backends", ("skip_backends", "skip_run_ids", "exclude_backends"))
    # v58b default: evaluation energy should not spend hours measuring CPU/CUDA
    # references unless the user explicitly requests policy=all with an empty skip list.
    if explicit:
        return explicit
    policy = _energy_target_policy_for_profile(options, profile_payload)
    if policy in {"canonical_only", "deepx_only", "best_valid_only", "best_plus_predicted"}:
        return ["ort_cpu", "ort_cuda", "cpu", "cuda"]
    return []


def _energy_include_run_ids_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> list[str]:
    return _energy_list_from_profile(options, profile_payload, "energy_include_run_ids", ("include_run_ids", "allow_run_ids", "include_backends"))


def _energy_exclude_run_ids_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> list[str]:
    return _energy_list_from_profile(options, profile_payload, "energy_exclude_run_ids", ("exclude_run_ids", "deny_run_ids"))


def _energy_heartbeat_s_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> int:
    cfg = _profile_energy(profile_payload)
    try:
        return max(10, int(getattr(options, "energy_heartbeat_s", None) or cfg.get("heartbeat_s") or 60))
    except Exception:
        return 60


def _energy_max_targets_per_run_id_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> int:
    cfg = _profile_energy(profile_payload)
    if _energy_final_all_split_enabled(options, profile_payload):
        return 0

    # v59k: the CLI parser uses -1 as its implicit "unset" sentinel.  Treat
    # negative values as absent so a profile-level max_targets_per_run_id is not
    # accidentally ignored.  Non-negative explicit CLI values still win; 0 means
    # "no cap / all selected targets".
    raw = getattr(options, "energy_max_targets_per_run_id", None)
    if raw not in (None, ""):
        try:
            val = int(raw)
            if val >= 0:
                return val
        except Exception:
            pass

    raw = cfg.get("max_targets_per_run_id", cfg.get("max_targets", None))
    try:
        val = int(raw)
        if val >= 0:
            return val
    except Exception:
        pass

    # Safe default for evaluation campaigns: keep the selected target list
    # uncapped.  Target-policy/skip-backend settings decide which targets exist.
    policy = _energy_target_policy_for_profile(options, profile_payload)
    return 0


def _energy_max_work_units_per_window_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> int:
    cfg = _profile_energy(profile_payload)
    raw = getattr(options, "energy_max_work_units_per_window", None)
    if raw in (None, ""):
        raw = cfg.get("max_work_units_per_window", 0)
    try:
        return max(0, int(raw or 0))
    except Exception:
        return 0


def _energy_max_window_duration_s_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> int:
    cfg = _profile_energy(profile_payload)
    raw = getattr(options, "energy_max_window_duration_s", None)
    if raw in (None, ""):
        raw = cfg.get("max_window_duration_s", cfg.get("max_active_s", 0))
    try:
        return max(0, int(raw or 0))
    except Exception:
        return 0


def _energy_timeout_s_per_window_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> int:
    cfg = _profile_energy(profile_payload)
    raw = getattr(options, "energy_timeout_s_per_window", None)
    if raw in (None, ""):
        raw = cfg.get("timeout_s_per_window", cfg.get("window_timeout_s", 0))
    try:
        return max(0, int(raw or 0))
    except Exception:
        return 0


def _energy_sizing_probe_max_work_units_for_profile(options: Any, profile_payload: Optional[Mapping[str, Any]]) -> int:
    cfg = _profile_energy(profile_payload)
    raw = getattr(options, "energy_sizing_probe_max_work_units", None)
    if raw in (None, ""):
        raw = cfg.get("sizing_probe_max_work_units", 256)
    try:
        return max(1, int(raw or 256))
    except Exception:
        return 256


def _energy_should_measure_run_id(run_id: str, *, options: Any, profile_payload: Optional[Mapping[str, Any]]) -> tuple[bool, str]:
    rid = str(run_id or "").strip().lower()
    policy = _energy_target_policy_for_profile(options, profile_payload)
    include = _energy_include_run_ids_for_profile(options, profile_payload)
    exclude = set(_energy_exclude_run_ids_for_profile(options, profile_payload)) | set(_energy_skip_backends_for_profile(options, profile_payload))
    if rid in exclude:
        return False, f"energy target skipped by exclude/skip list ({rid})"
    if policy == "all":
        return True, "energy target policy=all"
    if policy == "manual":
        return (rid in set(include), "energy target policy=manual include" if rid in set(include) else f"energy target policy=manual excludes {rid}")
    if policy == "deepx_only":
        ok = "deepx" in rid or "dx_m1" in rid or "deepx_m1" in rid
        return ok, "energy target policy=deepx_only" if ok else f"energy target policy=deepx_only skips {rid}"
    if policy in {"best_valid_only", "best_plus_predicted"}:
        # Same run-id eligibility as canonical_only, but target selection later keeps only best-valid and/or predicted rows.
        pass
    # canonical_only / best_valid_only / best_plus_predicted: keep full baselines and true heterogeneous accelerator
    # splits, skip pure CPU/CUDA and diagnostic-only rows by default.
    canonical = {
        "ort_tensorrt", "tensorrt",
        "deepx_m1_full", "deepx", "deepx_full",
        "deepx_m1_to_tensorrt", "tensorrt_to_deepx_m1",
        "hailo8", "hailo8_to_trt", "trt_to_hailo8", "trt_to_hailo",
        "hailo10", "hailo10_to_tensorrt", "hailo10_to_trt",
        "trt_to_hailo10",
    }
    if rid in canonical:
        return True, "energy target policy=canonical_only"
    if ("to_tensorrt" in rid or "to_trt" in rid or rid.startswith("trt_to_") or rid.startswith("tensorrt_to_")) and not (rid.startswith("ort_") or rid in {"ort_cpu", "ort_cuda"}):
        return True, "energy target policy=canonical_only mixed-backend heuristic"
    return False, f"energy target policy=canonical_only skips {rid}"

def _profile_remote_with_override(profile_payload: Optional[Mapping[str, Any]], runtime_override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    base = _profile_remote_execution(profile_payload)
    if isinstance(runtime_override, Mapping):
        for k, v in runtime_override.items():
            if v not in (None, ""):
                base[k] = v
    return base


def _remote_host_payload_from_options(options: Any, profile_payload: Optional[Mapping[str, Any]], runtime_override: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    profile_remote = _profile_remote_with_override(profile_payload, runtime_override)
    selected_id = str(getattr(options, "remote_host_id", "") or profile_remote.get("host_id") or profile_remote.get("id") or "").strip()
    raw_json = str(getattr(options, "remote_host_json", "") or profile_remote.get("host_json") or "").strip()
    if raw_json:
        try:
            payload = json.loads(raw_json)
            if isinstance(payload, Mapping):
                out = dict(payload)
                if (not selected_id) or str(out.get("id") or out.get("label") or "").strip() == selected_id:
                    return out
        except Exception:
            pass
    direct_host = str(getattr(options, "remote_host", "") or profile_remote.get("host") or "").strip()
    direct_user = str(getattr(options, "remote_user", "") or profile_remote.get("user") or "").strip()
    if "@" in direct_host and not direct_user:
        direct_user, direct_host = direct_host.split("@", 1)
    hosts_file = str(getattr(options, "remote_hosts_file", "") or profile_remote.get("hosts_file") or "").strip()
    if hosts_file:
        for h in _load_remote_hosts_file(hosts_file):
            hid = str(h.get("id") or h.get("label") or "").strip()
            if not selected_id or hid == selected_id:
                out = dict(h)
                out.setdefault("id", hid or selected_id or "remote")
                return out
    hosts_inline = profile_remote.get("hosts") or profile_remote.get("remote_hosts")
    if isinstance(hosts_inline, list):
        for h in hosts_inline:
            if not isinstance(h, Mapping):
                continue
            hid = str(h.get("id") or h.get("label") or "").strip()
            if not selected_id or hid == selected_id:
                out = dict(h)
                out.setdefault("id", hid or selected_id or "remote")
                return out
    if direct_host:
        parsed_host = direct_host
        parsed_user = direct_user
        if "@" in direct_host and not parsed_user:
            parsed_user, parsed_host = direct_host.split("@", 1)
            parsed_user = parsed_user.strip()
            parsed_host = parsed_host.strip()
        return {
            "id": selected_id or str(getattr(options, "remote_host_id", "") or "workflow_remote") or "workflow_remote",
            "label": selected_id or direct_host,
            "user": parsed_user,
            "host": parsed_host,
            "port": int(_int(getattr(options, "remote_port", None) or profile_remote.get("port"), 22) or 22),
            "remote_base_dir": str(getattr(options, "remote_base_dir", "") or profile_remote.get("remote_base_dir") or "~/splitpoint_runs"),
            "ssh_extra_args": str(getattr(options, "remote_ssh_extra_args", "") or profile_remote.get("ssh_extra_args") or ""),
        }
    return {}


def _central_management_quality_enabled(profile_payload: Optional[Mapping[str, Any]]) -> bool:
    """Whether CPU reference/uncertainty work belongs on the management node."""
    profile = dict(profile_payload or {}) if isinstance(profile_payload, Mapping) else {}
    quality = profile.get("quality_gate") if isinstance(profile.get("quality_gate"), Mapping) else {}
    statistics = quality.get("statistics") if isinstance(quality.get("statistics"), Mapping) else {}
    execution = quality.get("execution") if isinstance(quality.get("execution"), Mapping) else {}
    value = str(
        statistics.get("execution_location")
        or quality.get("execution_location")
        or execution.get("location")
        or ""
    ).strip().lower().replace("-", "_")
    return value in {"central_management", "management", "management_node", "central", "central_cpu"}


_PARALLEL_REMOTE_CANCEL_GRACE_S = 20.0


def _cancel_requested(cancel_event: Any) -> bool:
    try:
        if cancel_event is not None and cancel_event.is_set():
            return True
    except Exception:
        pass
    registry = current_process_registry()
    return bool(
        registry is not None and getattr(registry, "cancelled", False)
    )


def _submit_with_context(
    executor: concurrent.futures.Executor,
    function: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> concurrent.futures.Future[Any]:
    """Submit one worker with an independent snapshot of ownership context."""

    submit_context = contextvars.copy_context()
    return executor.submit(submit_context.run, function, *args, **kwargs)


def _record_worker_cleanup_uncertainty(*, label: str, detail: str) -> None:
    registry = current_process_registry()
    recorder = getattr(registry, "record_cleanup_uncertainty", None)
    if callable(recorder):
        recorder(label=label, detail=detail)


def _cancel_pending_workers_and_record_uncertainty(
    pending: Sequence[concurrent.futures.Future[Any]],
    *,
    label: str,
    detail: str,
) -> list[concurrent.futures.Future[Any]]:
    unresolved: list[concurrent.futures.Future[Any]] = []
    for future in pending:
        if future.done():
            continue
        cancelled = future.cancel()
        if not cancelled and not future.done():
            unresolved.append(future)
    if unresolved:
        _record_worker_cleanup_uncertainty(label=label, detail=detail)
    return unresolved


def _terminate_local_process_group(proc: subprocess.Popen[Any]) -> None:
    """Terminate a local suite and children without leaving benchmark workers."""
    terminate_process_tree(proc, grace_s=3.0)


def _make_ssh_host_config(payload: Mapping[str, Any]) -> Any:
    from ..remote.ssh_transport import HostConfig as SSHHostConfig
    return SSHHostConfig(
        id=str(payload.get("id") or payload.get("label") or "workflow_remote"),
        label=str(payload.get("label") or payload.get("id") or payload.get("host") or "workflow remote"),
        user=str(payload.get("user") or ""),
        host=str(payload.get("host") or ""),
        port=int(_int(payload.get("port"), 22) or 22),
        remote_base_dir=str(payload.get("remote_base_dir") or "~/splitpoint_runs"),
        ssh_extra_args=str(payload.get("ssh_extra_args") or ""),
    )


def _remote_args_from_options(
    options: Any,
    profile_payload: Optional[Mapping[str, Any]],
    runtime_override: Optional[Mapping[str, Any]] = None,
    *,
    model_task: str = "",
    model_id: str = "",
) -> Any:
    from ..benchmark.remote_run import RemoteBenchmarkArgs
    from .artifact_cache_preflight import resolve_artifact_cache_preflight_policy
    profile_remote = _profile_remote_with_override(profile_payload, runtime_override)
    warmup = _int(getattr(options, "remote_warmup", None), 0) or _int(getattr(options, "benchmark_warmup", None), _int(profile_remote.get("warmup"), 10))
    iters = _int(getattr(options, "remote_iters", None), 0) or _int(getattr(options, "benchmark_runs", None), _int(profile_remote.get("iters"), 100))
    repeats = _int(getattr(options, "remote_repeats", None), _int(profile_remote.get("repeats"), 1))
    timeout_s = _int(getattr(options, "remote_timeout_s", None), 0) or _int(getattr(options, "benchmark_timeout_s", None), 0) or _int(profile_remote.get("timeout_s"), 7200)
    remote_provider = str(getattr(options, "remote_provider", "") or "").strip()
    provider = remote_provider if remote_provider and remote_provider != "auto" else str(getattr(options, "benchmark_provider", "") or profile_remote.get("provider") or "auto")
    # benchmark_suite.py uses --run-id to select hardware-specific rows; its
    # --provider flag remains a host execution provider (auto/cpu/cuda/tensorrt).
    # Do not pass accelerator tokens such as hailo8/deepx_m1 as --provider.
    if str(provider).strip().lower() in {"hailo", "hailo8", "hailo10", "hailo10n", "hailo10h", "deepx", "deepx_m1", "dx_m1", "dxm1"}:
        provider = "auto"
    add_args_list = [str(x) for x in list(getattr(options, "benchmark_extra_args", []) or []) if str(x).strip()]
    explicit_add = str(getattr(options, "remote_add_args", "") or profile_remote.get("add_args") or "").strip()
    if explicit_add:
        add_args_list.append(explicit_add)
    validation_cfg = profile_payload.get("validation") if isinstance(profile_payload, Mapping) and isinstance(profile_payload.get("validation"), Mapping) else {}
    detection_metrics = list(validation_cfg.get("detection_metrics") or []) if isinstance(validation_cfg, Mapping) else []
    classification_metrics = list(validation_cfg.get("classification_metrics") or []) if isinstance(validation_cfg, Mapping) else []
    task_override = _effective_str(getattr(options, "remote_benchmark_task", ""))
    task_profile = _effective_str(profile_remote.get("benchmark_task"))
    task_preset = _effective_str(getattr(options, "benchmark_preset", ""))
    explicit_model_task = str(model_task or "").strip().lower()
    if explicit_model_task not in {"classification", "detection"}:
        explicit_model_task = ""
    # A per-model task is authoritative. Global overrides are retained only for
    # imported/manual suites that do not carry a model entry.
    effective_task = str(explicit_model_task or task_override or task_profile or task_preset or "auto").strip().lower()
    profile_mini_coco_req = bool(profile_remote.get("mini_coco_ap50", False) or "mini_coco_ap50" in [str(x) for x in detection_metrics])
    profile_mini_cls_req = bool(profile_remote.get("mini_classification_eval", False) or bool(classification_metrics))
    mini_coco_eff, mini_cls_eff = _task_gated_metrics(
        effective_task,
        mini_coco=bool(getattr(options, "remote_mini_coco_ap50", False) or profile_mini_coco_req),
        mini_cls=bool(getattr(options, "remote_mini_classification_eval", False) or profile_mini_cls_req),
    )
    validation_execution = profile_payload.get("validation_execution") if isinstance(profile_payload, Mapping) and isinstance(profile_payload.get("validation_execution"), Mapping) else {}
    validation_limits = validation_execution.get("max_items") if isinstance(validation_execution.get("max_items"), Mapping) else {}
    mode_budget_declared = bool(
        isinstance(profile_payload, Mapping)
        and isinstance(profile_payload.get("execution_preset"), Mapping)
        and effective_task in validation_limits
    )
    mode_validation_max = _int(validation_limits.get(effective_task), 0) if effective_task in {"classification", "detection"} else 0
    explicit_validation_max = _int(getattr(options, "remote_validation_max_images", 0), 0)
    if mode_budget_declared:
        # Includes the explicit Final-mode value 0 (= complete manifest).
        effective_validation_max = mode_validation_max
    else:
        effective_validation_max = explicit_validation_max if explicit_validation_max > 0 else _int(profile_remote.get("validation_max_images"), 0)

    trt_policy = resolve_artifact_cache_preflight_policy(profile_payload)
    trt_build_guard = (
        {"policy": trt_policy, "model_id": str(model_id)}
        if trt_policy["enabled"] and trt_policy["block_on_unexpected_cold_builds"]
        else {}
    )
    return RemoteBenchmarkArgs(
        trt_build_guard=trt_build_guard,
        provider=provider or "auto",
        remote_venv=str(getattr(options, "remote_venv", "") or profile_remote.get("remote_venv") or ""),
        repeats=max(1, repeats),
        warmup=max(0, warmup),
        iters=max(1, iters),
        add_args=" ".join(add_args_list),
        timeout_s=timeout_s if timeout_s > 0 else None,
        transfer_mode=str(getattr(options, "remote_transfer_mode", "") or profile_remote.get("transfer_mode") or "bundle"),
        reuse_bundle=False if bool(getattr(options, "remote_no_reuse_bundle", False)) else bool(getattr(options, "remote_reuse_bundle", profile_remote.get("reuse_bundle", True))),
        resume=False if bool(getattr(options, "remote_no_resume", False)) else bool(getattr(options, "remote_resume", True)),
        throughput_frames=max(1, _int(getattr(options, "remote_throughput_frames", None), _int(profile_remote.get("throughput_frames"), 24))),
        throughput_warmup_frames=max(0, _int(getattr(options, "remote_throughput_warmup_frames", None), _int(profile_remote.get("throughput_warmup_frames"), 6))),
        throughput_queue_depth=max(1, _int(getattr(options, "remote_throughput_queue_depth", None), _int(profile_remote.get("throughput_queue_depth"), 2))),
        validation_images=str(_effective_str(getattr(options, "remote_validation_images", "")) or _effective_str(profile_remote.get("validation_images")) or ""),
        validation_max_images=max(0, int(effective_validation_max)),
        validation_budget_authoritative=bool(mode_budget_declared),
        validation_reference_mode=str(_effective_str(getattr(options, "remote_validation_reference_mode", "")) or _effective_str(profile_remote.get("validation_reference_mode")) or "auto"),
        mini_coco_ap50=bool(mini_coco_eff),
        benchmark_task=effective_task,
        mini_classification_eval=bool(mini_cls_eff),
        energy_enabled=False,  # set per hardware target after setup-id resolution
        energy_run_count=0,
        energy_output_root="",
    )



def _canonical_run_id_from_name(name: str) -> str:
    """Return a stable run-id hint for a canonical result filename."""
    base = Path(str(name or "")).name
    low = base.lower()
    if low.startswith("benchmark_results_"):
        return base[len("benchmark_results_"):].rsplit(".", 1)[0]
    if low in {"benchmark_results.json", "results.json"}:
        return "aggregate"
    return base.rsplit(".", 1)[0]


def _canonical_payload_row_count(payload: Any) -> int:
    """Count measured rows without treating an empty JSON container as evidence.

    Benchmark result JSON has existed in a few compatible shapes over the
    lifetime of the tool.  This helper deliberately counts only concrete list
    entries or an explicit single-row mapping carrying result fields.  A bare
    ``{}``, ``[]`` or status-only object therefore contributes zero rows.
    """
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, Mapping):
        return 0
    for key in ("rows", "results", "benchmark_results", "cases", "records", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    # Some aggregate files map run IDs to row lists.
    list_values = [value for value in payload.values() if isinstance(value, list)]
    if list_values:
        return sum(len(value) for value in list_values)
    result_markers = {
        "fps", "throughput_fps", "latency_ms", "cycle_ms", "mean_ms",
        "run_id", "case_id", "backend", "provider", "variant",
    }
    return 1 if result_markers.intersection(payload.keys()) else 0


def _inspect_canonical_result(path: Path) -> Dict[str, Any]:
    """Parse a canonical JSON/CSV result and report semantic row presence."""
    path = Path(path)
    detail: Dict[str, Any] = {
        "path": str(path),
        "run_id": _canonical_run_id_from_name(path.name),
        "parseable": False,
        "row_count": 0,
        "nonempty": False,
        "format": path.suffix.lower().lstrip("."),
        "error_sidecar": "",
    }
    error_sidecar = path.with_name(path.name + ".error.txt")
    if not error_sidecar.is_file():
        # The normal convention is benchmark_results_X.error.txt rather than
        # benchmark_results_X.json.error.txt.
        error_sidecar = path.with_suffix("").with_suffix(".error.txt")
    if error_sidecar.is_file():
        detail["error_sidecar"] = str(error_sidecar)
        try:
            detail["error_sidecar_tail"] = error_sidecar.read_text(encoding="utf-8", errors="replace")[-4000:]
        except Exception:
            pass
    try:
        if path.suffix.lower() == ".json":
            from .compact_runtime_diagnostics import source_observation, write_runtime_companion
            observed_before = source_observation(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            detail["diagnostic_summary"] = write_runtime_companion(
                path, payload, observed_before=observed_before,
            )
            detail["row_count"] = int(_canonical_payload_row_count(payload))
            detail["parseable"] = True
        elif path.suffix.lower() == ".csv":
            import csv
            # Benchmark CSV rows may carry embedded JSON evidence that is much
            # larger than Python's platform-dependent 128 KiB default.  Keep
            # the audit bounded, but raise the parser limit far enough for the
            # canonical result contract and restore the process-global setting
            # afterwards.
            previous_limit = csv.field_size_limit()
            audit_limit = min(int(sys.maxsize), (2**31) - 1)
            try:
                try:
                    csv.field_size_limit(audit_limit)
                except (OverflowError, ValueError):  # pragma: no cover - platform-specific C long width
                    csv.field_size_limit((2**31) - 1)
                with path.open("r", encoding="utf-8-sig", newline="") as fh:
                    reader = csv.DictReader(fh)
                    detail["row_count"] = sum(
                        1 for row in reader
                        if isinstance(row, Mapping) and any(str(value or "").strip() for value in row.values())
                    )
            finally:
                try:
                    csv.field_size_limit(previous_limit)
                except (OverflowError, ValueError):  # pragma: no cover - defensive restore only
                    pass
            detail["parseable"] = True
        else:
            detail["parse_error"] = "unsupported_canonical_format"
    except Exception as exc:
        detail["parse_error"] = f"{type(exc).__name__}: {exc}"
    detail["nonempty"] = bool(detail["parseable"] and int(detail["row_count"] or 0) > 0)
    if detail["error_sidecar"] and not detail["nonempty"]:
        detail["semantic_status"] = "error_without_rows"
    elif detail["nonempty"]:
        detail["semantic_status"] = "measured_rows"
    elif detail["parseable"]:
        detail["semantic_status"] = "empty_result"
    else:
        detail["semantic_status"] = "unparseable"
    return detail

def _copy_remote_result_files(remote_local_run_dir: Path, dst_dir: Path, *, flat_prefix: str = "") -> List[Dict[str, Any]]:
    """Copy canonical result files and compact diagnostics into an EvaluationRun.

    v60t separates *canonical result transport* from *diagnostic-pack slimming*.
    Historic versions applied the 2 MiB diagnostic limit to
    ``benchmark_results_*.json/csv`` as well, so large classification result
    files silently disappeared and a dispatch could still look successful
    because a small log file had been copied.  Canonical files are now copied
    without a default size cap, can be recovered from a result bundle, and are
    audited in ``result_copy_manifest.json``.
    """
    copied: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    fallback_extractions: List[Dict[str, Any]] = []
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = dst_dir / "remote_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    target_diag_dir = diagnostics_dir / str(flat_prefix).strip() if str(flat_prefix or "").strip() else diagnostics_dir
    target_diag_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_max_bytes = 2 * 1024 * 1024
    try:
        canonical_max_bytes = max(0, int(str(os.environ.get("ONNX_SPLITPOINT_CANONICAL_RESULT_MAX_BYTES", "0") or "0")))
    except Exception:
        canonical_max_bytes = 0

    def _is_canonical_name(name: str) -> bool:
        base = Path(str(name or "")).name
        low = base.lower()
        return (
            (low.startswith("benchmark_results_") and low.endswith((".json", ".csv")))
            or low in {"benchmark_results.json", "results.json"}
        )

    manifest_path = target_diag_dir / "result_copy_manifest.json"
    if not remote_local_run_dir or not Path(remote_local_run_dir).exists():
        payload = {
            "schema": "onnx-splitpoint/remote-result-copy-manifest",
            "schema_version": 1,
            "created_at": now_iso(),
            "remote_local_run_dir": str(remote_local_run_dir or ""),
            "destination": str(dst_dir),
            "target_id": str(flat_prefix or ""),
            "status": "source_missing",
            "canonical_result_count": 0,
            "copied": [],
            "skipped": [],
            "errors": [{"reason": "remote_local_run_dir_missing"}],
        }
        write_json(manifest_path, payload)
        return [{"source": "", "destination": str(manifest_path), "kind": "result_copy_manifest", "canonical": False}]

    remote_local_run_dir = Path(remote_local_run_dir)
    results_dir = remote_local_run_dir / "results"
    if not results_dir.is_dir():
        results_dir = remote_local_run_dir

    diagnostic_dirs = [diagnostics_dir]
    if target_diag_dir != diagnostics_dir:
        diagnostic_dirs.append(target_diag_dir)

    flat_patterns = [
        "benchmark_results_*.json",
        "benchmark_results_*.csv",
        "benchmark_results.json",
        "results.json",
    ]
    diagnostic_root_patterns = [
        "benchmark_suite_status_matrix.json",
        "benchmark_suite_status_matrix.csv",
        "benchmark_suite_status_matrix.md",
        "v42_pipeline_summary.json",
        "v42_pipeline_summary.csv",
        "v42_pipeline_summary.md",
        "v47_pipeline_summary.json",
        "v47_pipeline_summary.csv",
        "v47_pipeline_summary.md",
        "benchmark_plan.json",
        "benchmark_set.json",
        "preflight.json",
        "results_bundle_manifest.json",
        "benchmark_summary_*.md",
        "benchmark_table_*.tex",
        "benchmark_results_*.error.txt",
        "energy_summary.json",
        "energy_aggregate.json",
        "energy_merge_manifest.json",
        "energy_measurement.json",
        "energy_measurement_error.json",
    ]
    allowed_suffixes = {".json", ".csv", ".md", ".txt", ".tex", ".log"}
    blocked_suffixes = {".onnx", ".hef", ".har", ".npz", ".npy", ".png", ".jpg", ".jpeg", ".pdf", ".tar", ".gz", ".zip", ".engine", ".dxnn", ".bin"}
    seen_destinations: set[str] = set()

    def _copy(
        src: Path,
        dest: Path,
        *,
        kind: str,
        canonical: bool = False,
        source_label: str = "",
        unbounded: bool = False,
    ) -> bool:
        try:
            if not src.is_file():
                return False
            suffix = src.suffix.lower()
            if suffix in blocked_suffixes or suffix not in allowed_suffixes:
                skipped.append({"source": str(src), "destination": str(dest), "kind": kind, "reason": "unsupported_suffix"})
                return False
            size = int(src.stat().st_size)
            limit = 0 if unbounded else (canonical_max_bytes if canonical else diagnostic_max_bytes)
            if limit > 0 and size > limit:
                skipped.append({
                    "source": str(src), "destination": str(dest), "kind": kind,
                    "canonical": canonical, "reason": "too_large", "size_bytes": size, "limit_bytes": limit,
                })
                return False
            dkey = str(dest.resolve())
            if dkey in seen_destinations:
                return False
            seen_destinations.add(dkey)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if str(src.resolve()) != str(dest.resolve()):
                shutil.copy2(src, dest)
            copied.append({
                "source": source_label or str(src),
                "destination": str(dest),
                "kind": kind,
                "canonical": bool(canonical),
                "size_bytes": size,
            })
            return True
        except Exception as exc:
            errors.append({
                "source": source_label or str(src), "destination": str(dest), "kind": kind,
                "canonical": canonical, "reason": f"{type(exc).__name__}: {exc}",
            })
            return False

    # Canonical benchmark result files are copied flat for the normal ingestion
    # path. They are deliberately not subject to the diagnostic 2 MiB limit.
    for pat in flat_patterns:
        for src in sorted(results_dir.glob(pat)):
            _copy(src, dst_dir / src.name, kind="benchmark_result", canonical=True)

    # Compact root diagnostics.
    for base in (remote_local_run_dir, results_dir):
        if not base.is_dir():
            continue
        for pat in diagnostic_root_patterns:
            for src in sorted(base.glob(pat)):
                for dd in diagnostic_dirs:
                    _copy(src, dd / src.name, kind="remote_diagnostic")
        logs_dir = base / "logs"
        if logs_dir.is_dir():
            for src in sorted(logs_dir.glob("*")):
                if src.is_file() and src.suffix.lower() in {".txt", ".log"}:
                    for dd in diagnostic_dirs:
                        _copy(src, dd / "logs" / src.name, kind="remote_log")

    # Per-case validation reports and compact energy summaries.
    for src in sorted(remote_local_run_dir.rglob("validation_report.json")):
        try:
            rel = src.relative_to(remote_local_run_dir)
        except Exception:
            rel = Path(src.name)
        for dd in diagnostic_dirs:
            _copy(src, dd / "case_reports" / rel, kind="case_validation_report")

    # Central quality inputs are scientific evidence, not optional diagnostics.
    # Preserve their relative case/results directory and never apply the 2 MiB
    # diagnostic limit; detection predictions for 500 images can legitimately
    # be larger.  Request descriptors remain portable because their referenced
    # files are copied beside them.
    quality_root = dst_dir / "quality_inputs"
    if str(flat_prefix or "").strip():
        quality_root = quality_root / str(flat_prefix).strip()
    quality_names = ("*_request.json", "*_candidate.json", "canonical_*_reference.json")
    for pattern in quality_names:
        for src in sorted(remote_local_run_dir.rglob(f"task_quality_inputs/{pattern}")):
            try:
                rel = src.relative_to(remote_local_run_dir)
            except Exception:
                rel = Path(src.parent.name) / src.name
            _copy(src, quality_root / rel, kind="central_quality_input", unbounded=True)
    for src in (
        sorted(remote_local_run_dir.rglob("energy_summary.json"))
        + sorted(remote_local_run_dir.rglob("energy_aggregate.json"))
        + sorted(remote_local_run_dir.rglob("energy_merge_manifest.json"))
    ):
        try:
            rel = src.relative_to(remote_local_run_dir)
        except Exception:
            rel = Path(src.name)
        for dd in diagnostic_dirs:
            _copy(src, dd / "energy" / rel, kind="energy_summary")

    # Bundle fallback: canonical files are recovered even when they are larger
    # than the diagnostic limit or were omitted from an older flat-copy path.
    tar_candidates: List[Path] = []
    for base in (remote_local_run_dir, results_dir):
        if not base.is_dir():
            continue
        for name in ("results_bundle_lean.tar.gz", "results_bundle.tar.gz"):
            path = base / name
            if path.is_file() and path not in tar_candidates:
                tar_candidates.append(path)
    for tar_path in tar_candidates:
        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    name = str(member.name or "").lstrip("/")
                    if not name or ".." in Path(name).parts:
                        continue
                    base_name = Path(name).name
                    canonical = _is_canonical_name(base_name)
                    if canonical:
                        dest = dst_dir / base_name
                        if dest.is_file() and int(dest.stat().st_size) > 0:
                            continue
                        size = int(getattr(member, "size", 0) or 0)
                        if canonical_max_bytes > 0 and size > canonical_max_bytes:
                            skipped.append({
                                "source": f"{tar_path}!{name}", "destination": str(dest),
                                "kind": "benchmark_result_fallback", "canonical": True,
                                "reason": "too_large", "size_bytes": size, "limit_bytes": canonical_max_bytes,
                            })
                            continue
                        try:
                            extracted = tf.extractfile(member)
                            if extracted is None:
                                raise RuntimeError("tar member could not be opened")
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            tmp = dest.with_name(dest.name + ".v60t.tmp")
                            with tmp.open("wb") as fh:
                                shutil.copyfileobj(extracted, fh, length=1024 * 1024)
                            os.replace(tmp, dest)
                            copied.append({
                                "source": f"{tar_path}!{name}", "destination": str(dest),
                                "kind": "benchmark_result_fallback", "canonical": True,
                                "size_bytes": int(dest.stat().st_size),
                            })
                            fallback_extractions.append({"bundle": str(tar_path), "member": name, "destination": str(dest)})
                        except Exception as exc:
                            errors.append({
                                "source": f"{tar_path}!{name}", "destination": str(dest),
                                "kind": "benchmark_result_fallback", "canonical": True,
                                "reason": f"{type(exc).__name__}: {exc}",
                            })
                        continue

                    suffix = Path(name).suffix.lower()
                    if suffix in blocked_suffixes or suffix not in allowed_suffixes:
                        continue
                    size = int(getattr(member, "size", 0) or 0)
                    if size > diagnostic_max_bytes:
                        continue
                    if not (
                        base_name.startswith("benchmark_suite_status_matrix")
                        or base_name.startswith("v42_pipeline_summary")
                        or base_name.startswith("v47_pipeline_summary")
                        or base_name in {"preflight.json", "results_bundle_manifest.json", "benchmark_plan.json", "benchmark_set.json", "validation_report.json", "stdout.txt", "stderr.txt"}
                        or base_name.startswith("benchmark_summary_")
                        or base_name.startswith("benchmark_table_")
                        or (base_name.startswith("benchmark_results_") and base_name.endswith(".error.txt"))
                    ):
                        continue
                    for dd in diagnostic_dirs:
                        dest = dd / "lean_bundle" / name
                        dkey = str(dest.resolve())
                        if dkey in seen_destinations:
                            continue
                        try:
                            extracted = tf.extractfile(member)
                            if extracted is None:
                                continue
                            data = extracted.read(diagnostic_max_bytes + 1)
                            if len(data) > diagnostic_max_bytes:
                                continue
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(data)
                            seen_destinations.add(dkey)
                            copied.append({
                                "source": f"{tar_path}!{name}", "destination": str(dest),
                                "kind": "lean_bundle_diagnostic", "canonical": False,
                                "size_bytes": len(data),
                            })
                        except Exception as exc:
                            errors.append({
                                "source": f"{tar_path}!{name}", "destination": str(dest),
                                "kind": "lean_bundle_diagnostic", "canonical": False,
                                "reason": f"{type(exc).__name__}: {exc}",
                            })
        except Exception as exc:
            errors.append({"source": str(tar_path), "kind": "result_bundle", "reason": f"{type(exc).__name__}: {exc}"})

    canonical_entries = [row for row in copied if bool(row.get("canonical"))]
    canonical_details: List[Dict[str, Any]] = []
    seen_canonical_destinations: set[str] = set()
    for entry in canonical_entries:
        dest = Path(str(entry.get("destination") or ""))
        dkey = str(dest.resolve()) if dest.exists() else str(dest)
        if dkey in seen_canonical_destinations:
            continue
        seen_canonical_destinations.add(dkey)
        detail = _inspect_canonical_result(dest)
        canonical_details.append(detail)
        entry["canonical_parseable"] = bool(detail.get("parseable"))
        entry["canonical_row_count"] = int(detail.get("row_count") or 0)
        entry["canonical_nonempty"] = bool(detail.get("nonempty"))
        entry["canonical_run_id"] = str(detail.get("run_id") or "")
        entry["canonical_semantic_status"] = str(detail.get("semantic_status") or "")

    parseable_count = sum(1 for row in canonical_details if row.get("parseable"))
    nonempty_files = [row for row in canonical_details if row.get("nonempty")]
    nonempty_row_count = sum(int(row.get("row_count") or 0) for row in nonempty_files)
    run_ids_with_rows = sorted({str(row.get("run_id") or "") for row in nonempty_files if str(row.get("run_id") or "")})
    all_canonical_run_ids = {str(row.get("run_id") or "") for row in canonical_details if str(row.get("run_id") or "")}
    run_ids_without_rows = sorted(all_canonical_run_ids.difference(run_ids_with_rows))
    if nonempty_row_count <= 0:
        semantic_status = "missing_canonical_results"
    elif errors or run_ids_without_rows:
        semantic_status = "partial"
    else:
        semantic_status = "ok"

    manifest_payload = {
        "schema": "onnx-splitpoint/remote-result-copy-manifest",
        "schema_version": 2,
        "created_at": now_iso(),
        "remote_local_run_dir": str(remote_local_run_dir),
        "destination": str(dst_dir),
        "target_id": str(flat_prefix or ""),
        "diagnostic_max_bytes": diagnostic_max_bytes,
        "canonical_max_bytes": canonical_max_bytes or None,
        "status": semantic_status,
        "canonical_result_count": len(canonical_details),
        "canonical_file_count": len(canonical_details),
        "canonical_parseable_file_count": parseable_count,
        "canonical_nonempty_file_count": len(nonempty_files),
        "canonical_nonempty_row_count": nonempty_row_count,
        "canonical_run_ids_with_rows": run_ids_with_rows,
        "canonical_run_ids_without_rows": run_ids_without_rows,
        "canonical_files": canonical_details,
        "fallback_extraction_count": len(fallback_extractions),
        "copied": copied,
        "fallback_extractions": fallback_extractions,
        "skipped": skipped,
        "errors": errors,
    }
    write_json(manifest_path, manifest_payload)
    copied.append({
        "source": str(remote_local_run_dir), "destination": str(manifest_path),
        "kind": "result_copy_manifest", "canonical": False,
        "size_bytes": int(manifest_path.stat().st_size) if manifest_path.is_file() else 0,
    })
    return copied


_DEEPX_PREPARED_INPUT_ROLES = (
    "native_full_input_manifest.json",
    "runtime_input.bin",
    "input_rgb_uint8.bin",
)


def _bare_sha256(value: Any) -> str:
    token = str(value or "").strip().lower()
    return token[7:] if token.startswith("sha256:") else token


def _canonical_json_rows(path: Path) -> List[Mapping[str, Any]]:
    """Return concrete rows from one canonical result JSON."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("rows", "results", "benchmark_results", "records", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
    if _canonical_payload_row_count(payload) == 1:
        return [payload]
    return []


def _materialize_deepx_original_full_failure(
    *, remote_local_run_dir: Path, suite_dir: Path,
    model_id: str, target_id: str = "",
) -> Dict[str, Any]:
    """Carry the exact failed Full row across the lean Native handoff.

    Generic result files are intentionally excluded by the Native inventory.
    This compact diagnostic is scoped by the current model/setup dispatch and
    never supplies runtime, quality, or prepared-input success evidence.
    """
    source = (Path(remote_local_run_dir) / "results"
              / "benchmark_results_deepx_m1_full_auto.json")
    destination = (Path(suite_dir) / "results" / "deepx_m1_full"
                   / "original_full_failure.json")
    if source.is_symlink() or not source.is_file():
        return {"status": "not_available"}
    candidates: List[Mapping[str, Any]] = []
    for row in _canonical_json_rows(source):
        prepared = row.get("deepx_prepared_feed_benchmark")
        if (
            str(row.get("run_id") or "") != "deepx_m1_full"
            or str(row.get("backend") or row.get("provider") or "") != "deepx_m1"
            or str(row.get("variant") or row.get("primary_variant") or "") != "full"
            or not isinstance(prepared, Mapping)
        ):
            continue
        # Older rows omit duplicate model/setup fields. The originating
        # dispatch supplies those identities; any explicit conflict rejects it.
        identities = [(row.get("model_id") or row.get("model"), model_id),
                      (row.get("setup_id") or row.get("hardware_target_id"), target_id)]
        contract = prepared.get("input_contract")
        if isinstance(contract, Mapping):
            identities.append((contract.get("model_id"), model_id))
        if any(str(value) != str(expected) for value, expected in identities
               if value and expected):
            continue
        candidates.append(row)
    if len(candidates) != 1:
        return {"status": "not_available", "reason": "canonical_full_row_not_unique"}
    row = candidates[0]
    prepared = row["deepx_prepared_feed_benchmark"]
    if (
        row.get("runtime_ok") is True
        or str(prepared.get("status") or "") not in {"runtime_failed", "failed", "error"}
        or not str(prepared.get("error") or "").strip()
    ):
        # A later successful dispatch must not retain earlier failure context.
        if destination.is_file() and not destination.is_symlink():
            destination.unlink()
        return {"status": "not_applicable"}
    payload = {
        "schema": "onnx-splitpoint/deepx-full-failure-context",
        "schema_version": 1,
        "model": str(model_id), "setup_id": str(target_id),
        "run_id": "deepx_m1_full", "backend": "deepx_m1", "variant": "full",
        "identity_source": "current_remote_dispatch",
        "source_file": str(source),
        "status": str(prepared["status"]), "error": str(prepared["error"]),
        "diagnostic_only": True,
    }
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise RuntimeError("deepx_full_failure_destination_symlink_forbidden")
    write_json(destination, payload)
    return {"status": "materialized", "source": str(source),
            "destination": str(destination), "original_full_error": str(prepared["error"])}


def _materialize_deepx_prepared_input(
    *,
    remote_local_run_dir: Path,
    suite_dir: Path,
    result_dir: Path,
    model_id: str,
    target_id: str = "",
) -> Dict[str, Any]:
    """Validate and restore the sealed DeepX Full input into the suite.

    The generic remote collector deliberately blocks arbitrary binary files.
    This narrow handoff admits exactly the three files generated by
    ``native_full_input`` and binds them to the canonical DeepX Full result row
    before the later Native transfer inventory can see them.
    """

    original_failure = _materialize_deepx_original_full_failure(
        remote_local_run_dir=remote_local_run_dir, suite_dir=suite_dir,
        model_id=model_id, target_id=target_id,
    )
    source_root = (
        Path(remote_local_run_dir)
        / "results"
        / "deepx_m1_full"
        / "prepared_input"
    )
    if original_failure.get("status") == "materialized":
        # v27 keeps the failed call's sealed input for diagnosis. It must not
        # become accepted performance input merely because those files exist.
        # Return the known runtime block instead of turning it into an unrelated
        # local result-processing exception (row_status_not_ok) below.
        return {
            "status": "blocked_by_original_full_failure",
            "reason": "deepx_full_runtime_failed",
            "source": str(source_root), "prepared_input_admitted": False,
            "original_full_error": original_failure["original_full_error"],
            "original_full_failure_context_file": original_failure["destination"],
        }
    receipt_dir = Path(result_dir) / "remote_diagnostics"
    if str(target_id or "").strip():
        receipt_dir = receipt_dir / str(target_id).strip()
    receipt_path = receipt_dir / "deepx_prepared_input_materialization.json"
    if not source_root.is_dir():
        return {
            "status": "not_available",
            "reason": "deepx_prepared_input_not_collected",
            "source": str(source_root),
        }
    if source_root.is_symlink():
        raise RuntimeError("deepx_prepared_input_source_symlink_forbidden")

    observed = sorted(path.name for path in source_root.iterdir())
    if observed != sorted(_DEEPX_PREPARED_INPUT_ROLES):
        raise RuntimeError(
            "deepx_prepared_input_role_set_invalid:"
            + ",".join(observed)
        )
    sources = {name: source_root / name for name in _DEEPX_PREPARED_INPUT_ROLES}
    if any(path.is_symlink() or not path.is_file() for path in sources.values()):
        raise RuntimeError("deepx_prepared_input_role_not_regular_file")

    manifest_path = sources["native_full_input_manifest.json"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("deepx_prepared_input_manifest_invalid_json") from exc
    if not isinstance(manifest, Mapping):
        raise RuntimeError("deepx_prepared_input_manifest_invalid")
    manifest = dict(manifest)
    if (
        manifest.get("schema") != "onnx-splitpoint/native-full-input-dump"
        or manifest.get("schema_version") != 2
        or manifest.get("backend") != "native_full_deepx"
        or str(manifest.get("model") or "").strip() != str(model_id)
        or str(manifest.get("task") or "").strip().lower()
        not in {"classification", "detection"}
        or str(manifest.get("runtime_input_file") or "") != "runtime_input.bin"
        or str(manifest.get("input_dump") or "") != "input_rgb_uint8.bin"
        or str(manifest.get("comparison_backend") or "").strip().lower()
        not in {"deepx", "deepx_m1", "deepx_m1_full"}
    ):
        raise RuntimeError("deepx_prepared_input_manifest_identity_invalid")
    expected_setup = str(target_id or "").strip()
    if expected_setup and str(manifest.get("setup_id") or "").strip() != expected_setup:
        raise RuntimeError("deepx_prepared_input_manifest_setup_mismatch")

    runtime_file = sources["runtime_input.bin"]
    input_file = sources["input_rgb_uint8.bin"]
    runtime_sha = _bare_sha256(sha256_file(runtime_file))
    input_sha = _bare_sha256(sha256_file(input_file))
    manifest_sha = _bare_sha256(sha256_file(manifest_path))
    if (
        int(manifest.get("runtime_input_bytes") or -1) != runtime_file.stat().st_size
        or str(manifest.get("runtime_input_sha256") or "").strip().lower()
        != runtime_sha
        or int(manifest.get("input_dump_bytes") or -1) != input_file.stat().st_size
        or str(manifest.get("input_dump_sha256") or "").strip().lower()
        != input_sha
    ):
        raise RuntimeError("deepx_prepared_input_manifest_hash_or_size_mismatch")

    result_root = Path(remote_local_run_dir) / "results"
    matching_rows: List[Mapping[str, Any]] = []
    for path in sorted(result_root.glob("benchmark_results_*.json")):
        for row in _canonical_json_rows(path):
            if str(row.get("run_id") or "").strip() != "deepx_m1_full":
                continue
            if (
                str(row.get("backend") or row.get("provider") or "")
                .strip().lower() != "deepx_m1"
                or str(row.get("variant") or "").strip().lower() != "full"
                or str(row.get("task") or "").strip().lower()
                != str(manifest.get("task") or "").strip().lower()
            ):
                continue
            prepared = row.get("deepx_prepared_feed_benchmark")
            if not isinstance(prepared, Mapping):
                continue
            matching_rows.append(prepared)
    if len(matching_rows) != 1:
        raise RuntimeError(
            f"deepx_prepared_input_canonical_row_count:{len(matching_rows)}"
        )
    row = dict(matching_rows[0])
    input_contract = row.get("input_contract")
    input_contract = dict(input_contract) if isinstance(input_contract, Mapping) else {}
    contract_model_id = str(input_contract.get("model_id") or "").strip()
    binding_failure_reasons: List[str] = []
    if row.get("status") != "ok":
        binding_failure_reasons.append("row_status_not_ok")
    if row.get("prepared_input_binding_verified") is not True:
        binding_failure_reasons.append("prepared_input_binding_unverified")
    if (
        str(row.get("task") or "").strip().lower()
        != str(manifest.get("task") or "").strip().lower()
    ):
        binding_failure_reasons.append("task_mismatch")
    # A missing duplicate is not legacy-compatible by implication: this field
    # binds the benchmark result row itself to the requested model identity.
    if contract_model_id != str(model_id):
        binding_failure_reasons.append("input_contract_model_id_mismatch")
    if (
        str(row.get("prepared_input_manifest_sha256") or "").strip().lower()
        != manifest_sha
    ):
        binding_failure_reasons.append("manifest_sha256_mismatch")
    if (
        str(row.get("prepared_input_sha256") or "").strip().lower()
        != runtime_sha
    ):
        binding_failure_reasons.append("runtime_sha256_mismatch")
    if (
        str(row.get("prepared_input_file_sha256") or "").strip().lower()
        != runtime_sha
    ):
        binding_failure_reasons.append("runtime_file_sha256_mismatch")
    try:
        prepared_input_bytes = int(row.get("prepared_input_bytes") or -1)
    except (TypeError, ValueError):
        prepared_input_bytes = -1
    if prepared_input_bytes != runtime_file.stat().st_size:
        binding_failure_reasons.append("runtime_size_mismatch")
    if (
        str(row.get("prepared_input_source_image_sha256") or "").strip().lower()
        != str(manifest.get("input_image_sha256") or "").strip().lower()
    ):
        binding_failure_reasons.append("source_image_sha256_mismatch")
    if binding_failure_reasons:
        raise RuntimeError(
            "deepx_prepared_input_result_binding_invalid:"
            + ",".join(binding_failure_reasons)
        )

    destination = (
        Path(suite_dir) / "results" / "deepx_m1_full" / "prepared_input"
    )
    if destination.is_symlink():
        raise RuntimeError("deepx_prepared_input_destination_symlink_forbidden")
    destination.mkdir(parents=True, exist_ok=True)
    unexpected_destination_roles = sorted(
        path.name for path in destination.iterdir()
        if path.name not in _DEEPX_PREPARED_INPUT_ROLES
    )
    if unexpected_destination_roles:
        raise RuntimeError(
            "deepx_prepared_input_destination_role_set_invalid:"
            + ",".join(unexpected_destination_roles)
        )
    # Validate every existing destination before creating any missing role, so
    # a conflict cannot leave a partially updated scientific input contract.
    for name in _DEEPX_PREPARED_INPUT_ROLES:
        dest = destination / name
        if not dest.exists():
            continue
        if (
            dest.is_symlink()
            or not dest.is_file()
            or _bare_sha256(sha256_file(dest))
            != _bare_sha256(sha256_file(sources[name]))
        ):
            raise RuntimeError(
                f"deepx_prepared_input_destination_conflict:{name}"
            )
    materialized: List[Dict[str, Any]] = []
    for name in _DEEPX_PREPARED_INPUT_ROLES:
        source = sources[name]
        dest = destination / name
        source_sha = _bare_sha256(sha256_file(source))
        if dest.exists():
            status = "reused_identical"
        else:
            with tempfile.NamedTemporaryFile(
                prefix=f".{name}.", suffix=".tmp", dir=str(destination),
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                with source.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, handle)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                if _bare_sha256(sha256_file(temporary)) != source_sha:
                    raise RuntimeError(
                        f"deepx_prepared_input_temporary_hash_mismatch:{name}"
                    )
                os.replace(temporary, dest)
            finally:
                if temporary.exists():
                    temporary.unlink()
            status = "materialized"
        materialized.append({
            "role": name,
            "path": str(dest),
            "sha256": source_sha,
            "size_bytes": int(dest.stat().st_size),
            "status": status,
        })

    receipt = {
        "schema": "onnx-splitpoint/deepx-prepared-input-materialization",
        "schema_version": 1,
        "created_at": now_iso(),
        "status": "verified_exact",
        "model_id": str(model_id),
        "setup_id": expected_setup,
        "source": str(source_root),
        "destination": str(destination),
        "manifest_sha256": manifest_sha,
        "files": materialized,
    }
    write_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def _deepx_prepared_input_full_only_quality_na(
    *, gates: Mapping[str, Any], model_id: str, target_id: str,
) -> Optional[Dict[str, Any]]:
    """Return an explicit N/A decision for the sealed DeepX quality canary.

    A normal DeepX Full dispatch binds the portable prepared input to exactly
    one canonical performance row in :func:`_materialize_deepx_prepared_input`.
    The Full-only quality canary intentionally emits no performance rows, so
    that binding is not applicable.  Keep this exception fail-closed: it is
    admitted only when the setup-local dispatch matrix and both expected
    quality identities describe the exact DeepX-Full + TensorRT-Full pair.
    """

    if str(gates.get("execution_scope") or "").strip().lower() != "full_only":
        return None

    selected = [
        str(value).strip() for value in list(
            gates.get("hardware_run_ids") or []
        ) if str(value).strip()
    ]
    if not selected:
        selected = [
            token.strip() for token in str(
                gates.get("hardware_run_id") or ""
            ).split(",") if token.strip()
        ]
    quality_only = [
        str(value).strip() for value in list(
            gates.get("quality_only_run_ids") or []
        ) if str(value).strip()
    ]
    selected = list(dict.fromkeys(selected))
    quality_only = list(dict.fromkeys(quality_only))
    if (
        not selected
        or set(selected) != set(quality_only)
        or len(selected) != len(quality_only)
    ):
        return None

    setup_id = str(target_id or "").strip()
    expected = [
        dict(row) for row in list(
            gates.get("expected_full_quality_identities") or []
        )
        if isinstance(row, Mapping)
        and str(row.get("setup_id") or "").strip() == setup_id
    ]
    if len(expected) != 2:
        return None

    def _backend(value: Any) -> str:
        token = str(value or "").strip().lower().replace("-", "_")
        return {
            "trt": "tensorrt",
            "ort_tensorrt": "tensorrt",
            "native_tensorrt": "tensorrt",
            "native_full_tensorrt": "tensorrt",
        }.get(token, token)

    dispatch_ids: List[str] = []
    identity_keys: List[tuple[str, str, str]] = []
    for row in expected:
        source_run_id = str(
            row.get("source_run_id") or row.get("run_id") or ""
        ).strip()
        dispatch_run_id = str(
            row.get("dispatch_run_id") or source_run_id
        ).strip()
        if (
            not source_run_id
            or not dispatch_run_id
            or str(row.get("variant") or "").strip().lower() != "full"
            or str(row.get("execution_role") or "").strip().lower()
            != "full_quality_only"
            or row.get("performance_claims_emitted") is not False
        ):
            return None
        dispatch_ids.append(dispatch_run_id)
        identity_keys.append(
            (source_run_id, dispatch_run_id, _backend(row.get("backend")))
        )
    if (
        len(set(identity_keys)) != 2
        or set(dispatch_ids) != set(selected)
        or set(identity_keys) != {
            ("deepx_m1_full", "deepx_m1_full", "deepx_m1"),
            ("native_full_tensorrt", "ort_tensorrt", "tensorrt"),
        }
    ):
        return None

    return {
        "schema": "onnx-splitpoint/deepx-prepared-input-materialization",
        "schema_version": 1,
        "status": "not_applicable",
        "reason": "full_only_quality_dispatch_has_no_performance_rows",
        "model_id": str(model_id),
        "setup_id": setup_id,
        "execution_scope": "full_only",
        "quality_only_run_ids": quality_only,
        "expected_full_quality_identity_count": len(expected),
        "performance_claims_emitted": False,
    }


def _canonical_copy_summary(copied: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return semantic canonical-result counts from a copy operation."""
    manifest_path = next((Path(str(row.get("destination") or "")) for row in copied if row.get("kind") == "result_copy_manifest"), None)
    if manifest_path and manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            return {
                "manifest": str(manifest_path),
                "status": str(payload.get("status") or ""),
                "file_count": int(payload.get("canonical_file_count", payload.get("canonical_result_count", 0)) or 0),
                "parseable_file_count": int(payload.get("canonical_parseable_file_count", 0) or 0),
                "nonempty_file_count": int(payload.get("canonical_nonempty_file_count", 0) or 0),
                "nonempty_row_count": int(payload.get("canonical_nonempty_row_count", 0) or 0),
                "run_ids_with_rows": list(payload.get("canonical_run_ids_with_rows") or []),
                "run_ids_without_rows": list(payload.get("canonical_run_ids_without_rows") or []),
            }
        except Exception:
            pass
    canonical = [row for row in copied if bool(row.get("canonical"))]
    return {
        "manifest": str(manifest_path or ""),
        "status": "ok" if any(bool(row.get("canonical_nonempty")) for row in canonical) else "missing_canonical_results",
        "file_count": len(canonical),
        "parseable_file_count": sum(1 for row in canonical if row.get("canonical_parseable")),
        "nonempty_file_count": sum(1 for row in canonical if row.get("canonical_nonempty")),
        "nonempty_row_count": sum(int(row.get("canonical_row_count") or 0) for row in canonical),
        "run_ids_with_rows": sorted({str(row.get("canonical_run_id") or "") for row in canonical if row.get("canonical_nonempty")}),
        "run_ids_without_rows": sorted({str(row.get("canonical_run_id") or "") for row in canonical if not row.get("canonical_nonempty")}),
    }


def _full_only_quality_evidence_summary(
    copied: Sequence[Mapping[str, Any]],
    *,
    expected_identities: Sequence[Mapping[str, Any]],
    setup_id: str,
    model_id: str,
    eval_run_id: str,
) -> Dict[str, Any]:
    """Verify the exact setup-local request/candidate evidence set.

    Full-only quality dispatches intentionally have no canonical benchmark
    rows.  Their success authority is instead the pair of portable central
    quality inputs named by the preflight: vendor Full and setup-local Native
    Full TensorRT.  Every candidate is checked against its request descriptor
    before the remote dispatch can be considered successful.
    """

    expected = [
        dict(row) for row in expected_identities
        if isinstance(row, Mapping)
        and str(row.get("setup_id") or "") == str(setup_id or "")
    ]
    if not expected:
        return {
            "requested": False,
            "status": "not_requested",
            "expected_count": 0,
            "quality_evidence_count": 0,
            "errors": [],
            "evidence": [],
        }
    errors: List[str] = []
    evidence: List[Dict[str, Any]] = []
    expected_eval_run_id = str(eval_run_id or "").strip()
    if not expected_eval_run_id:
        errors.append("full_only_expected_eval_run_id_missing")

    def backend_token(value: Any) -> str:
        token = str(value or "").strip().lower().replace("-", "_")
        return {
            "trt": "tensorrt",
            "ort_tensorrt": "tensorrt",
            "native_tensorrt": "tensorrt",
            "native_full_tensorrt": "tensorrt",
            "hailo10": "hailo10h",
            "hailo10n": "hailo10h",
        }.get(token, token)

    expected_keys: set[tuple[str, str, str, str, str, str, bool]] = set()
    expected_plan_identities: Dict[
        tuple[str, str, str, str, str, str, bool], Dict[str, Any]
    ] = {}
    expected_canary_ids: set[str] = set()
    for row in expected:
        quality_canary_id = str(row.get("id") or "").strip()
        expected_key = (
            str(model_id or "").strip(),
            str(row.get("source_run_id") or row.get("run_id") or "").strip(),
            str(row.get("setup_id") or "").strip(),
            backend_token(row.get("backend")),
            str(row.get("variant") or "").strip().lower(),
            str(row.get("execution_role") or "").strip().lower(),
            row.get("performance_claims_emitted"),
        )
        if (
            not quality_canary_id
            or not all(expected_key[:6])
            or expected_key[4] != "full"
            or expected_key[5] != "full_quality_only"
            or expected_key[6] is not False
        ):
            errors.append(
                "full_only_expected_identity_invalid:"
                f"{expected_key[1]}@{expected_key[2]}/{expected_key[4]}"
            )
            continue
        if quality_canary_id in expected_canary_ids:
            errors.append(
                "full_only_expected_quality_canary_id_duplicate:"
                f"{quality_canary_id}"
            )
            continue
        if expected_key in expected_keys:
            errors.append(
                "full_only_expected_identity_duplicate:"
                f"{expected_key[1]}@{expected_key[2]}/{expected_key[4]}"
            )
            continue
        expected_canary_ids.add(quality_canary_id)
        expected_keys.add(expected_key)
        expected_plan_identities[expected_key] = {
            "schema": "onnx-splitpoint/full-only-quality-request-identity",
            "schema_version": 1,
            "quality_canary_id": quality_canary_id,
            "eval_run_id": expected_eval_run_id,
            "model_id": expected_key[0],
            "setup_id": expected_key[2],
            "source_run_id": expected_key[1],
            "backend": expected_key[3],
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        }
    central_destinations = {
        str(Path(str(row.get("destination") or "")).resolve())
        for row in copied
        if row.get("kind") == "central_quality_input"
        and str(row.get("destination") or "")
    }
    request_paths = sorted({
        Path(str(row.get("destination") or ""))
        for row in copied
        if row.get("kind") == "central_quality_input"
        and str(row.get("destination") or "").endswith("_request.json")
    })
    candidate_destinations = {
        str(Path(str(row.get("destination") or "")).resolve())
        for row in copied
        if row.get("kind") == "central_quality_input"
        and str(row.get("destination") or "").endswith("_candidate.json")
    }
    referenced_candidate_destinations: set[str] = set()

    def identity_from_payload(
        payload: Mapping[str, Any], *, label: str, path: Path,
    ) -> tuple[tuple[str, str, str, str, str, str, bool] | None, List[str]]:
        producer = (
            payload.get("producer_identity")
            if isinstance(payload.get("producer_identity"), Mapping) else {}
        )

        def value(field_name: str) -> Any:
            if field_name in payload:
                return payload.get(field_name)
            return producer.get(field_name)

        observed_model = str(value("model_id") or "").strip()
        observed_source = str(value("source_run_id") or "").strip()
        observed_setup = str(value("setup_id") or "").strip()
        observed_backend = backend_token(value("backend"))
        observed_variant = str(value("variant") or "").strip().lower()
        observed_role = str(value("execution_role") or "").strip().lower()
        observed_performance = value("performance_claims_emitted")
        local_errors: List[str] = []
        observed_text = {
            "model_id": observed_model,
            "source_run_id": observed_source,
            "setup_id": observed_setup,
            "backend": observed_backend,
            "variant": observed_variant,
            "execution_role": observed_role,
        }
        for field_name, field_value in observed_text.items():
            if not field_value:
                local_errors.append(
                    f"quality_{label}_{field_name}_missing:{path}"
                )
        if observed_performance is not False:
            local_errors.append(
                f"quality_{label}_performance_claims_emitted_not_false:{path}"
            )
        if observed_model and observed_model != str(model_id or "").strip():
            local_errors.append(
                f"quality_{label}_model_mismatch:"
                f"{observed_model}:{model_id}:{path}"
            )
        if observed_setup and observed_setup != str(setup_id or "").strip():
            local_errors.append(
                f"quality_{label}_setup_mismatch:"
                f"{observed_setup}:{setup_id}:{path}"
            )
        if observed_variant and observed_variant != "full":
            local_errors.append(
                f"quality_{label}_variant_not_full:{observed_variant}:{path}"
            )
        if observed_role and observed_role != "full_quality_only":
            local_errors.append(
                f"quality_{label}_execution_role_invalid:{observed_role}:{path}"
            )
        if local_errors:
            return None, local_errors
        return (
            observed_model,
            observed_source,
            observed_setup,
            observed_backend,
            observed_variant,
            observed_role,
            False,
        ), []

    def full_only_seal_errors(
        payload: Mapping[str, Any], *,
        expected_identity: Mapping[str, Any],
        label: str,
        path: Path,
    ) -> List[str]:
        """Validate the duplicated immutable canary identity on one artefact."""

        local_errors: List[str] = []
        if payload.get("full_only_plan_identity_required") is not True:
            local_errors.append(
                f"quality_{label}_full_only_plan_identity_not_required:{path}"
            )
        observed_identity = payload.get("full_only_plan_identity")
        if (
            not isinstance(observed_identity, Mapping)
            or dict(observed_identity) != dict(expected_identity)
        ):
            local_errors.append(
                f"quality_{label}_full_only_plan_identity_mismatch:{path}"
            )
        expected_identity_sha = canonical_contract_sha256(
            dict(expected_identity)
        )
        observed_identity_sha = _bare_sha256(
            payload.get("full_only_plan_identity_sha256")
        )
        if observed_identity_sha != expected_identity_sha:
            local_errors.append(
                f"quality_{label}_full_only_plan_identity_sha256_mismatch:{path}"
            )
        expected_canary_id = str(
            expected_identity.get("quality_canary_id") or ""
        )
        if str(payload.get("quality_canary_id") or "").strip() != expected_canary_id:
            local_errors.append(
                f"quality_{label}_quality_canary_id_mismatch:{path}"
            )
        if str(payload.get("eval_run_id") or "").strip() != expected_eval_run_id:
            local_errors.append(
                f"quality_{label}_eval_run_id_mismatch:{path}"
            )
        return local_errors

    for request_path in request_paths:
        request_parts = request_path.parts
        setup_markers = [
            request_parts[index + 1]
            for index, part in enumerate(request_parts[:-1])
            if part == "quality_inputs"
        ]
        if setup_markers and setup_markers[-1] != str(setup_id):
            errors.append(
                "quality_request_setup_path_mismatch:"
                f"{setup_markers[-1]}:{setup_id}:{request_path}"
            )
            continue
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(
                f"quality_request_invalid:{request_path}:"
                f"{type(exc).__name__}:{exc}"
            )
            continue
        if not isinstance(request, Mapping):
            errors.append(f"quality_request_not_mapping:{request_path}")
            continue
        if str(request.get("schema") or "") != (
            "onnx-splitpoint/central-quality-evaluation-request"
        ):
            errors.append(f"quality_request_schema_invalid:{request_path}")
            continue
        identity, identity_errors = identity_from_payload(
            request, label="request", path=request_path,
        )
        if identity_errors or identity is None:
            errors.extend(identity_errors)
            continue
        (
            observed_model, source_run_id, observed_setup, observed_backend,
            variant, observed_role, observed_performance,
        ) = identity
        try:
            record_count = int(request.get("record_count") or 0)
        except (TypeError, ValueError):
            record_count = 0
        if record_count <= 0:
            errors.append(f"quality_request_empty:{request_path}")
            continue
        if identity not in expected_keys:
            errors.append(
                "quality_request_identity_unexpected:"
                f"{source_run_id}@{observed_setup}/{variant}:"
                f"{observed_backend}:{observed_role}:"
                f"{observed_performance}"
            )
            continue
        expected_plan_identity = expected_plan_identities.get(identity)
        if expected_plan_identity is None:
            errors.append(
                "quality_request_full_only_plan_identity_missing:"
                f"{source_run_id}@{observed_setup}/{variant}"
            )
            continue
        request_seal_errors = full_only_seal_errors(
            request,
            expected_identity=expected_plan_identity,
            label="request",
            path=request_path,
        )
        if request_seal_errors:
            errors.extend(request_seal_errors)
            continue
        candidate = request.get("candidate")
        if not isinstance(candidate, Mapping):
            errors.append(f"quality_candidate_descriptor_missing:{request_path}")
            continue
        candidate_raw = str(
            candidate.get("path") or candidate.get("file") or ""
        ).strip()
        candidate_rel = Path(candidate_raw)
        if not candidate_raw or candidate_rel.is_absolute() or ".." in candidate_rel.parts:
            errors.append(f"quality_candidate_path_not_portable:{request_path}")
            continue
        candidate_path = (request_path.parent / candidate_rel).resolve()
        if str(candidate_path) not in central_destinations or not candidate_path.is_file():
            errors.append(f"quality_candidate_not_collected:{candidate_path}")
            continue
        referenced_candidate_destinations.add(str(candidate_path))
        expected_sha = _bare_sha256(
            candidate.get("sha256") or candidate.get("file_sha256")
        )
        actual_sha = _bare_sha256(sha256_file(candidate_path))
        try:
            expected_size = int(candidate["size_bytes"])
        except (TypeError, ValueError):
            expected_size = -1
        except KeyError:
            expected_size = -1
        actual_size = int(candidate_path.stat().st_size)
        if (
            len(expected_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha)
            or expected_sha != actual_sha
            or expected_size < 1
            or expected_size != actual_size
        ):
            errors.append(f"quality_candidate_hash_or_size_mismatch:{candidate_path}")
            continue
        try:
            candidate_payload = json.loads(
                candidate_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            errors.append(
                f"quality_candidate_invalid:{candidate_path}:"
                f"{type(exc).__name__}:{exc}"
            )
            continue
        if not isinstance(candidate_payload, Mapping):
            errors.append(f"quality_candidate_not_mapping:{candidate_path}")
            continue
        if str(candidate_payload.get("schema") or "") != (
            "onnx-splitpoint/task-quality-candidate-input"
        ):
            errors.append(f"quality_candidate_schema_invalid:{candidate_path}")
            continue
        candidate_records = candidate_payload.get("records")
        try:
            candidate_record_count = int(candidate_payload["record_count"])
        except (KeyError, TypeError, ValueError):
            candidate_record_count = -1
        if (
            not isinstance(candidate_records, list)
            or not candidate_records
            or candidate_record_count < 1
            or candidate_record_count != len(candidate_records)
            or candidate_record_count != record_count
        ):
            errors.append(
                f"quality_candidate_record_count_invalid:{candidate_path}:"
                f"{candidate_record_count}:{len(candidate_records) if isinstance(candidate_records, list) else -1}:"
                f"{record_count}"
            )
            continue
        candidate_identity, candidate_identity_errors = identity_from_payload(
            candidate_payload, label="candidate", path=candidate_path,
        )
        if candidate_identity_errors or candidate_identity is None:
            errors.extend(candidate_identity_errors)
            continue
        if candidate_identity != identity:
            errors.append(
                "quality_candidate_request_identity_mismatch:"
                f"{candidate_path}:{request_path}"
            )
            continue
        candidate_seal_errors = full_only_seal_errors(
            candidate_payload,
            expected_identity=expected_plan_identity,
            label="candidate",
            path=candidate_path,
        )
        if candidate_seal_errors:
            errors.extend(candidate_seal_errors)
            continue
        evidence.append({
            "model_id": observed_model,
            "source_run_id": source_run_id,
            "setup_id": observed_setup,
            "backend": observed_backend,
            "variant": variant,
            "execution_role": observed_role,
            "performance_claims_emitted": observed_performance,
            "quality_canary_id": str(
                expected_plan_identity.get("quality_canary_id") or ""
            ),
            "eval_run_id": expected_eval_run_id,
            "full_only_plan_identity": dict(expected_plan_identity),
            "full_only_plan_identity_sha256": canonical_contract_sha256(
                dict(expected_plan_identity)
            ),
            "request_path": str(request_path),
            "request_sha256": sha256_file(request_path),
            "candidate_path": str(candidate_path),
            "candidate_sha256": sha256_file(candidate_path),
            "candidate_size_bytes": actual_size,
        })
    observed_keys = [
        (
            str(row.get("model_id") or ""),
            str(row.get("source_run_id") or ""),
            str(row.get("setup_id") or ""),
            backend_token(row.get("backend")),
            str(row.get("variant") or ""),
            str(row.get("execution_role") or ""),
            row.get("performance_claims_emitted"),
        )
        for row in evidence
    ]
    for identity in sorted(expected_keys):
        count = observed_keys.count(identity)
        if count != 1:
            errors.append(
                "quality_request_identity_count_not_one:"
                f"{identity[1]}@{identity[2]}/{identity[4]}:{count}"
            )
    if len(request_paths) != len(expected_keys):
        errors.append(
            f"quality_request_count_mismatch:{len(request_paths)}:"
            f"{len(expected_keys)}"
        )
    for unexpected_candidate in sorted(
        candidate_destinations - referenced_candidate_destinations
    ):
        errors.append(
            f"quality_candidate_unreferenced:{unexpected_candidate}"
        )
    unique_errors = list(dict.fromkeys(errors))
    return {
        "requested": True,
        "status": "verified_exact" if not unique_errors else "blocked",
        "expected_count": len(expected_keys),
        "quality_evidence_count": len(evidence),
        "errors": unique_errors,
        "evidence": evidence,
    }


def _pre_mutation_remote_dispatch_cancellation(
    output: Any,
) -> Dict[str, Any]:
    """Normalize an explicit cancellation before any remote side effect."""

    if not isinstance(output, Mapping):
        return {}
    if str(output.get("status") or "").strip().lower() != "cancelled":
        return {}
    if (
        str(output.get("dispatch_status") or "").strip().lower()
        != "cancelled_before_dispatch"
    ):
        return {}
    if output.get("remote_dispatch_failed") is not False:
        return {}
    if output.get("remote_dispatched") is not False:
        return {}
    failure_kind = str(output.get("failure_kind") or "").strip()
    if not failure_kind:
        return {}
    nested_primary = (
        dict(output.get("primary_failure") or {})
        if isinstance(output.get("primary_failure"), Mapping)
        else {}
    )
    primary_error = str(
        output.get("primary_error")
        or nested_primary.get("primary_error")
        or output.get("error")
        or "Remote dispatch cancelled before start"
    ).strip()
    return {
        "status": "cancelled",
        "dispatch_status": "cancelled_before_dispatch",
        "reason": failure_kind,
        "failure_kind": failure_kind,
        "primary_error": primary_error,
        "error": str(output.get("error") or primary_error).strip(),
        "remote_rc": output.get("remote_rc", 130),
        "remote_dispatch_failed": False,
        "remote_dispatched": False,
        "pre_remote_mutation": True,
    }


def _pre_mutation_remote_dispatch_failure(
    output: Any,
) -> Dict[str, Any]:
    """Normalize the service's explicit pre-mutation dispatch contract.

    A normal remote execution failure is deliberately *not* classified here:
    once either a remote mutation or leased operation may have started, the
    existing partial/failed collection and cleanup path remains authoritative.
    All three fields are required so an older or incomplete service response
    cannot accidentally weaken that fail-closed behaviour.
    """

    if not isinstance(output, Mapping):
        return {}
    if output.get("remote_dispatch_failed") is not True:
        return {}
    if output.get("remote_dispatched") is not False:
        return {}
    failure_kind = str(output.get("failure_kind") or "").strip()
    if not failure_kind:
        return {}

    nested_primary = (
        dict(output.get("primary_failure") or {})
        if isinstance(output.get("primary_failure"), Mapping)
        else {}
    )
    primary_error = str(
        output.get("primary_error")
        or nested_primary.get("primary_error")
        or output.get("error")
        or failure_kind
    ).strip()
    return {
        "status": "failed_to_dispatch",
        "reason": failure_kind,
        "failure_kind": failure_kind,
        "primary_error": primary_error,
        "error": str(output.get("error") or primary_error).strip(),
        "remote_rc": output.get("remote_rc"),
        "remote_dispatch_failed": True,
        "remote_dispatched": False,
        "pre_remote_mutation": bool(
            output.get("pre_remote_mutation", True)
        ),
        "service_status": str(output.get("status") or "").strip(),
        "dispatch_status": str(
            output.get("dispatch_status") or "failed_to_dispatch"
        ).strip(),
    }

def _run_remote_dispatch_once(
    *,
    run_root: Path,
    model_id: str,
    options: Any,
    profile_payload: Optional[Mapping[str, Any]],
    suite_dir: Path,
    benchmark_set_json: Path,
    result_dir: Path,
    gates: Mapping[str, Any],
    log: Optional[Callable[[str], None]],
    runtime_override: Optional[Mapping[str, Any]] = None,
    target_id: str = "",
    model_task: str = "",
    cancel_event: Any = None,
    remote_process_registry: RemoteProcessLeaseRegistry | None = None,
    workflow_session_id: str = "",
) -> ExecutionBindingResult:
    status_suffix = f"_{target_id}" if target_id else ""
    status_path = result_dir / f"remote_benchmark_status{status_suffix}.json"
    dispatch_path = result_dir / f"remote_benchmark_dispatch{status_suffix}.json"
    stdout_path = result_dir / f"remote_benchmark_stdout{status_suffix}.txt"
    stderr_path = result_dir / f"remote_benchmark_stderr{status_suffix}.txt"

    def _write_status(payload: Dict[str, Any]) -> Path:
        payload.setdefault("schema", "onnx-splitpoint/remote-benchmark-status")
        payload.setdefault("schema_version", 1)
        payload.setdefault("created_at", now_iso())
        payload.setdefault("model_id", model_id)
        payload.setdefault("suite_dir", relpath(suite_dir, run_root))
        if target_id:
            payload.setdefault("hardware_target_id", target_id)
        return write_json(status_path, payload)

    quality_companion_required = (
        gates.get("quality_companion_required") is True
    )
    quality_companion_endpoint_id = str(
        gates.get("quality_companion_endpoint_id") or ""
    ).strip()
    if quality_companion_required and not quality_companion_endpoint_id:
        # This check deliberately precedes host construction and the remote
        # service call.  A setup-local TRT quality producer without a concrete
        # endpoint identity can never emit admissible evidence; dispatching it
        # would only fail after upload (or, worse, after earlier plan rows).
        failure_kind = "setup_local_tensorrt_quality_identity_missing"
        primary_error = (
            f"{failure_kind}:model={model_id}:setup={target_id or 'missing'}"
        )
        p_status = _write_status({
            "status": "failed_to_dispatch",
            "reason": failure_kind,
            "failure_kind": failure_kind,
            "primary_error": primary_error,
            "remote_dispatch_failed": True,
            "remote_dispatched": False,
            "pre_remote_mutation": True,
            "execution_gates": dict(gates),
            "runtime_override": dict(runtime_override or {}),
        })
        p_dispatch = write_json(dispatch_path, {
            "schema": "onnx-splitpoint/remote-benchmark-dispatch",
            "schema_version": 1,
            "created_at": now_iso(),
            "model_id": model_id,
            "status": "failed_to_dispatch",
            "reason": failure_kind,
            "failure_kind": failure_kind,
            "primary_error": primary_error,
            "remote_dispatch_failed": True,
            "remote_dispatched": False,
            "pre_remote_mutation": True,
            "suite_dir": relpath(suite_dir, run_root),
            "benchmark_set_json": relpath(benchmark_set_json, run_root),
            "execution_gates": dict(gates),
            "runtime_override": dict(runtime_override or {}),
        })
        stderr_path.write_text(primary_error + "\n", encoding="utf-8")
        return ExecutionBindingResult(
            artifacts={
                f"remote_benchmark_status{status_suffix}_json": p_status,
                f"remote_benchmark_dispatch{status_suffix}_json": p_dispatch,
                f"remote_benchmark_stderr{status_suffix}_txt": stderr_path,
            },
            metrics={
                "remote_requested": True,
                "remote_dispatched": False,
                "remote_dispatch_failed": True,
                "remote_status": "failed_to_dispatch",
                "failure_kind": failure_kind,
                "remote_reason": failure_kind,
                "primary_error": primary_error,
                "pre_remote_mutation": True,
                "hardware_target_id": target_id,
            },
            status="failed_to_dispatch",
            message=primary_error,
        )

    host_payload = _remote_host_payload_from_options(options, profile_payload, runtime_override=runtime_override)
    if not host_payload or not str(host_payload.get("host") or "").strip():
        p_status = _write_status({
            "status": "not_dispatched",
            "reason": "remote_requested_but_no_host_configured",
            "execution_gates": dict(gates),
            "runtime_override": dict(runtime_override or {}),
        })
        p_dispatch = write_json(dispatch_path, {
            "schema": "onnx-splitpoint/remote-benchmark-dispatch",
            "schema_version": 1,
            "created_at": now_iso(),
            "model_id": model_id,
            "status": "not_dispatched",
            "reason": "remote_requested_but_no_host_configured",
            "suite_dir": relpath(suite_dir, run_root),
            "benchmark_set_json": relpath(benchmark_set_json, run_root),
            "execution_gates": dict(gates),
            "runtime_override": dict(runtime_override or {}),
        })
        return ExecutionBindingResult(
            artifacts={f"remote_benchmark_status{status_suffix}_json": p_status, f"remote_benchmark_dispatch{status_suffix}_json": p_dispatch},
            metrics={"remote_requested": True, "remote_dispatched": False, "remote_pending": True, "remote_reason": "remote_requested_but_no_host_configured", "hardware_target_id": target_id},
            status="partial",
            message="Remote benchmark execution was requested, but no host is configured.",
        )

    host = _make_ssh_host_config(host_payload)
    args = _remote_args_from_options(options, profile_payload, runtime_override=runtime_override, model_task=model_task, model_id=model_id)
    # Keep the remote signed producer and the management-side collector scope
    # on the same physical setup identity.  Logical run IDs are deliberately
    # not accepted as a fallback here.
    quality_setup_id = str(target_id or "").strip()
    if quality_setup_id:
        setattr(args, "quality_evidence_eval_id", str(run_root.name))
        setattr(args, "quality_evidence_model_id", str(model_id))
        setattr(args, "quality_evidence_setup_id", quality_setup_id)
    if quality_companion_endpoint_id:
        setattr(
            args, "quality_evidence_endpoint_id",
            quality_companion_endpoint_id,
        )
    local_working_dir = Path(str(getattr(options, "remote_working_dir", "") or _profile_remote_with_override(profile_payload, runtime_override).get("local_working_dir") or result_dir / "remote_runs")).expanduser()

    # v58b: Evaluation Workflow energy integration with target policy.
    # Energy is expensive; by default we measure canonical full baselines and
    # real heterogeneous accelerator splits, not every CPU/CUDA diagnostic run.
    energy_cfg = _profile_energy(profile_payload)
    energy_ab_cfg = (
        dict(energy_cfg.get("window_method_ab") or {})
        if isinstance(energy_cfg.get("window_method_ab"), Mapping)
        else {}
    )
    energy_enabled = _energy_enabled_for_profile(options, profile_payload)
    energy_run_id = str(gates.get("hardware_run_id") or "").strip()
    if not energy_run_id and isinstance(runtime_override, Mapping):
        energy_run_id = str(runtime_override.get("run_id") or runtime_override.get("hardware_run_id") or "").strip()
    if not energy_run_id:
        try:
            import shlex as _shlex
            toks = _shlex.split(str(getattr(args, "add_args", "") or ""))
            for i, t in enumerate(toks):
                if t == "--run-id" and i + 1 < len(toks):
                    energy_run_id = toks[i + 1]
                    break
        except Exception:
            pass
    energy_allowed = False
    energy_reason = "generic_energy_disabled_native_only"
    if energy_enabled:
        energy_allowed, energy_reason = _energy_should_measure_run_id(energy_run_id, options=options, profile_payload=profile_payload)
    if energy_enabled and energy_allowed:
        setup_id = ""
        energy_registry_path = ""
        if isinstance(runtime_override, Mapping):
            setup_id = str(runtime_override.get("setup_id") or runtime_override.get("id") or "").strip()
        if not setup_id and isinstance(gates.get("hardware_target"), Mapping):
            setup_id = str(gates.get("hardware_target", {}).get("id") or "").strip()
        if isinstance(gates.get("hardware_target"), Mapping):
            energy_registry_path = str(
                gates.get("hardware_target", {}).get("setup_source") or ""
            ).strip()
        if not energy_registry_path:
            energy_registry_path = str(
                getattr(options, "hardware_setups_file", "") or ""
            ).strip()
        if not setup_id:
            setup_id = str(target_id or host.id or "").strip()
        try:
            setattr(args, "energy_enabled", True)
            setattr(args, "energy_setup_id", setup_id)
            setattr(args, "energy_registry_path", energy_registry_path)
            setattr(args, "energy_run_count", _energy_repeat_override(options, profile_payload))
            setattr(args, "energy_output_root", str(result_dir / "energy" / (target_id or setup_id or "remote")))
            setattr(args, "energy_scope", _energy_scope_for_profile(options, profile_payload))
            setattr(args, "energy_phases", _energy_phases_for_profile(options, profile_payload))
            setattr(args, "energy_strict", bool(getattr(options, "energy_strict", False) or _bool(energy_cfg.get("strict", False)) or _energy_final_all_split_enabled(options, profile_payload)))
            setattr(args, "energy_heartbeat_s", _energy_heartbeat_s_for_profile(options, profile_payload))
            setattr(args, "energy_target_policy", _energy_target_policy_for_profile(options, profile_payload))
            setattr(args, "energy_max_targets_per_run_id", _energy_max_targets_per_run_id_for_profile(options, profile_payload))
            setattr(args, "energy_max_work_units_per_window", _energy_max_work_units_per_window_for_profile(options, profile_payload))
            setattr(args, "energy_max_window_duration_s", _energy_max_window_duration_s_for_profile(options, profile_payload))
            setattr(args, "energy_timeout_s_per_window", _energy_timeout_s_per_window_for_profile(options, profile_payload))
            setattr(args, "energy_sizing_probe_max_work_units", _energy_sizing_probe_max_work_units_for_profile(options, profile_payload))
            setattr(args, "energy_confidence_level", _energy_confidence_level_for_profile(options, profile_payload))
            setattr(args, "energy_physical_scope", _energy_physical_scope_for_profile(options, profile_payload))
            setattr(args, "energy_window_label", _energy_window_label_for_profile(options, profile_payload))
            setattr(args, "energy_window_ab_enabled", bool(energy_ab_cfg.get("enabled", False)))
            setattr(args, "energy_window_ab_baseline_method", str(energy_ab_cfg.get("baseline_method") or "chapter4_baseline"))
            setattr(args, "energy_window_ab_candidate_method", str(energy_ab_cfg.get("candidate_method") or "candidate_v263"))
            setattr(args, "energy_window_ab_same_raw_capture", bool(energy_ab_cfg.get("same_raw_capture", True)))
            setattr(args, "energy_window_ab_mode", str(energy_ab_cfg.get("mode") or "shadow"))
            setattr(args, "energy_window_ab_auto_switch", bool(energy_ab_cfg.get("auto_switch", False)))
            setattr(args, "energy_window_ab_smoke_repeats", max(1, int(energy_ab_cfg.get("smoke_repeats") or 3)))
            setattr(args, "energy_window_ab_include_raw_parquet", bool(energy_ab_cfg.get("include_raw_parquet", True)))
            setattr(args, "energy_window_ab_requires_picoscope", bool(energy_ab_cfg.get("requires_picoscope", False)))
            setattr(args, "energy_randomize_target_order", _energy_randomize_target_order_for_profile(options, profile_payload))
            setattr(args, "energy_randomization_seed", _energy_randomization_seed_for_profile(options, profile_payload))
            setattr(args, "reuse_bundle", False)
            setattr(args, "resume", False)
        except Exception:
            pass
    elif energy_enabled:
        try:
            setattr(args, "energy_enabled", False)
        except Exception:
            pass
    remote_run_id = _remote_transport_run_id(
        run_root=run_root,
        model_id=model_id,
        target_id=target_id,
        gates=gates,
        workflow_session_id=workflow_session_id,
    )
    dispatch_payload = {
        "schema": "onnx-splitpoint/remote-benchmark-dispatch",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "status": "dispatching",
        "suite_dir": relpath(suite_dir, run_root),
        "benchmark_set_json": relpath(benchmark_set_json, run_root),
        "local_working_dir": str(local_working_dir),
        "run_id": remote_run_id,
        "hardware_target_id": target_id,
        "host": {"id": host.id, "label": host.label, "user": host.user, "host": host.host, "port": host.port, "remote_base_dir": host.remote_base_dir},
        "energy_policy": {
            "profile_energy_enabled": bool(energy_enabled),
            "final_all_split_energy": _energy_final_all_split_enabled(options, profile_payload),
            "final_energy_skip_cpu_ort": _energy_final_skip_cpu_ort_enabled(options, profile_payload),
            "run_id": energy_run_id,
            "enabled_for_this_dispatch": bool(energy_enabled and energy_allowed),
            "reason": energy_reason,
            "target_policy": _energy_target_policy_for_profile(options, profile_payload),
            "skip_backends": _energy_skip_backends_for_profile(options, profile_payload),
            "include_run_ids": _energy_include_run_ids_for_profile(options, profile_payload),
            "exclude_run_ids": _energy_exclude_run_ids_for_profile(options, profile_payload),
            "heartbeat_s": _energy_heartbeat_s_for_profile(options, profile_payload),
            "max_targets_per_run_id": _energy_max_targets_per_run_id_for_profile(options, profile_payload),
            "max_work_units_per_window": _energy_max_work_units_per_window_for_profile(options, profile_payload),
            "max_window_duration_s": _energy_max_window_duration_s_for_profile(options, profile_payload),
            "timeout_s_per_window": _energy_timeout_s_per_window_for_profile(options, profile_payload),
            "sizing_probe_max_work_units": _energy_sizing_probe_max_work_units_for_profile(options, profile_payload),
            "confidence_level": _energy_confidence_level_for_profile(options, profile_payload),
            "physical_scope": _energy_physical_scope_for_profile(options, profile_payload),
            "window_label": _energy_window_label_for_profile(options, profile_payload),
            "randomize_target_order": _energy_randomize_target_order_for_profile(options, profile_payload),
            "randomization_seed": _energy_randomization_seed_for_profile(options, profile_payload),
        },
        "args": getattr(args, "__dict__", {}),
        "execution_gates": dict(gates),
        "runtime_override": dict(runtime_override or {}),
    }
    p_dispatch = write_json(dispatch_path, dispatch_payload)
    if callable(log):
        label = f" ({target_id})" if target_id else ""
        log(f"[workflow][remote] dispatching {model_id}{label} to {host.user}@{host.host}:{host.port}")
        if energy_enabled:
            log(f"[workflow][energy] run_id={energy_run_id or '?'} target_policy={_energy_target_policy_for_profile(options, profile_payload)} enabled_for_dispatch={bool(energy_enabled and energy_allowed)} reason={energy_reason}")
            if energy_enabled and energy_allowed:
                phases = _energy_phases_for_profile(options, profile_payload) or ["latency", "streaming"]
                reps = _energy_repeat_override(options, profile_payload) or int(getattr(args, "repeats", 1) or 1)
                max_targets = _energy_max_targets_per_run_id_for_profile(options, profile_payload)
                max_wu = _energy_max_work_units_per_window_for_profile(options, profile_payload)
                max_dur = _energy_max_window_duration_s_for_profile(options, profile_payload)
                log(f"[workflow][energy] planned windows≈{len(phases) * max(1, int(reps))} per target phases={','.join(phases)} repeats={reps}; max_targets_per_run_id={max_targets}; max_work_units={max_wu or 'off'}; max_window_s={max_dur or 'off'}; long u.RECS windows write energy_current_status.json every {_energy_heartbeat_s_for_profile(options, profile_payload)}s")

    remote_stdout: List[str] = []
    remote_was_dispatched = False
    remote_output: Dict[str, Any] = {}
    local_run_dir = Path("")
    copied: List[Dict[str, Any]] = []
    # Both streams are formal artifacts on every launched-dispatch result.
    # Materialize them before the service call so a successful run with no
    # stderr output is represented by a real, indexable zero-byte file rather
    # than a dangling stage artifact.
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    try:
        from ..benchmark.services import RemoteBenchmarkService
        service = RemoteBenchmarkService()
        out = service.run(
            host=host,
            benchmark_set_json=benchmark_set_json,
            local_working_dir=local_working_dir,
            run_id=remote_run_id,
            args=args,
            log=lambda msg: (remote_stdout.append(str(msg)), callable(log) and log(f"[workflow][remote] {msg}")),
            progress=lambda pct, msg: callable(log) and log(f"[workflow][remote] {int(round(float(pct) * 100.0))}% {msg}"),
            cancel_event=cancel_event,
            remote_process_registry=remote_process_registry,
            workflow_session_id=workflow_session_id,
        )
        remote_output = dict(out) if isinstance(out, Mapping) else {}
        remote_was_dispatched = bool(
            remote_output.get("remote_dispatched") is True
        )
        stdout_path.write_text("\n".join(remote_stdout) + ("\n" if remote_stdout else ""), encoding="utf-8")
        local_run_dir = Path(str(out.get("local_run_dir") or "")) if isinstance(out, Mapping) else Path("")
        pre_dispatch_cancel = _pre_mutation_remote_dispatch_cancellation(out)
        if pre_dispatch_cancel:
            primary_error = str(pre_dispatch_cancel["primary_error"])
            stderr_path.write_text(primary_error + "\n", encoding="utf-8")
            p_dispatch = write_json(dispatch_path, {
                **dispatch_payload,
                "status": "cancelled_before_dispatch",
                "reason": pre_dispatch_cancel["reason"],
                "failure_kind": pre_dispatch_cancel["failure_kind"],
                "primary_error": primary_error,
                "remote_dispatch_failed": False,
                "remote_dispatched": False,
            })
            p_status = _write_status({
                "status": "cancelled",
                "dispatch_status": "cancelled_before_dispatch",
                "reason": pre_dispatch_cancel["reason"],
                "failure_kind": pre_dispatch_cancel["failure_kind"],
                "primary_error": primary_error,
                "primary_failure": dict(pre_dispatch_cancel),
                "remote_rc": pre_dispatch_cancel.get("remote_rc"),
                "remote_dispatch_failed": False,
                "remote_dispatched": False,
                "pre_remote_mutation": True,
                "host": dispatch_payload["host"],
                "remote_output": dict(out),
                "local_run_dir": str(local_run_dir),
                "stdout_path": relpath(stdout_path, run_root),
                "stderr_path": relpath(stderr_path, run_root),
                "execution_gates": dict(gates),
                "runtime_override": dict(runtime_override or {}),
            })
            return ExecutionBindingResult(
                artifacts={
                    f"remote_benchmark_status{status_suffix}_json": p_status,
                    f"remote_benchmark_dispatch{status_suffix}_json": p_dispatch,
                    f"remote_benchmark_stdout{status_suffix}_txt": stdout_path,
                    f"remote_benchmark_stderr{status_suffix}_txt": stderr_path,
                },
                metrics={
                    "remote_requested": True,
                    "remote_dispatched": False,
                    "remote_dispatch_failed": False,
                    "remote_status": "cancelled",
                    "cancelled": True,
                    "cancelled_before_dispatch": True,
                    "failure_kind": pre_dispatch_cancel["failure_kind"],
                    "remote_reason": pre_dispatch_cancel["reason"],
                    "primary_error": primary_error,
                    "remote_rc": pre_dispatch_cancel.get("remote_rc"),
                    "pre_remote_mutation": True,
                    "hardware_target_id": target_id,
                },
                status="cancelled",
                message=primary_error,
            )
        pre_dispatch_failure = _pre_mutation_remote_dispatch_failure(out)
        if pre_dispatch_failure:
            primary_error = str(
                pre_dispatch_failure.get("primary_error")
                or pre_dispatch_failure["failure_kind"]
            )
            stderr_path.write_text(primary_error + "\n", encoding="utf-8")
            p_dispatch = write_json(dispatch_path, {
                **dispatch_payload,
                "status": "failed_to_dispatch",
                "reason": pre_dispatch_failure["reason"],
                "failure_kind": pre_dispatch_failure["failure_kind"],
                "primary_error": primary_error,
                "remote_dispatch_failed": True,
                "remote_dispatched": False,
            })
            p_status = _write_status({
                "status": "failed_to_dispatch",
                "reason": pre_dispatch_failure["reason"],
                "failure_kind": pre_dispatch_failure["failure_kind"],
                "error_class": "remote_dispatch_failed",
                "error_detail": pre_dispatch_failure["error"],
                "primary_error": primary_error,
                "primary_failure": dict(pre_dispatch_failure),
                "remote_rc": pre_dispatch_failure.get("remote_rc"),
                "remote_dispatch_failed": True,
                "remote_dispatched": False,
                "pre_remote_mutation": bool(
                    pre_dispatch_failure.get("pre_remote_mutation")
                ),
                "host": dispatch_payload["host"],
                "remote_output": dict(out),
                "local_run_dir": str(local_run_dir),
                "stdout_path": relpath(stdout_path, run_root),
                "stderr_path": relpath(stderr_path, run_root),
                "execution_gates": dict(gates),
                "runtime_override": dict(runtime_override or {}),
            })
            return ExecutionBindingResult(
                artifacts={
                    f"remote_benchmark_status{status_suffix}_json": p_status,
                    f"remote_benchmark_dispatch{status_suffix}_json": p_dispatch,
                    f"remote_benchmark_stdout{status_suffix}_txt": stdout_path,
                    f"remote_benchmark_stderr{status_suffix}_txt": stderr_path,
                },
                metrics={
                    "remote_requested": True,
                    "remote_dispatched": False,
                    "remote_dispatch_failed": True,
                    "remote_status": "failed_to_dispatch",
                    "failure_kind": pre_dispatch_failure["failure_kind"],
                    "remote_reason": pre_dispatch_failure["reason"],
                    "remote_error": primary_error,
                    "primary_error": primary_error,
                    "remote_rc": pre_dispatch_failure.get("remote_rc"),
                    "pre_remote_mutation": bool(
                        pre_dispatch_failure.get("pre_remote_mutation")
                    ),
                    "hardware_target_id": target_id,
                },
                status="failed_to_dispatch",
                message=primary_error,
            )
        copied = _copy_remote_result_files(local_run_dir, result_dir, flat_prefix=target_id) if local_run_dir else []
        deepx_prepared_input = _deepx_prepared_input_full_only_quality_na(
            gates=gates, model_id=model_id, target_id=target_id,
        )
        if deepx_prepared_input is None:
            deepx_prepared_input = (
                _materialize_deepx_prepared_input(
                    remote_local_run_dir=local_run_dir,
                    suite_dir=suite_dir,
                    result_dir=result_dir,
                    model_id=model_id,
                    target_id=target_id,
                )
                if local_run_dir else {"status": "not_available"}
            )
        if str(deepx_prepared_input.get("receipt_path") or ""):
            copied.append({
                "source": str(deepx_prepared_input.get("source") or ""),
                "destination": str(deepx_prepared_input["receipt_path"]),
                "kind": "deepx_prepared_input_materialization_receipt",
                "canonical": False,
            })
        canonical_copied = [row for row in copied if bool(row.get("canonical"))]
        copy_summary = _canonical_copy_summary(copied)
        copy_manifest = str(copy_summary.get("manifest") or "")
        status_name = str(out.get("status") or ("ok" if out.get("ok") else "failed")).strip().lower() if isinstance(out, Mapping) else "failed"
        semantic_rows = int(copy_summary.get("nonempty_row_count") or 0)
        quality_evidence = _full_only_quality_evidence_summary(
            copied,
            expected_identities=[
                dict(row) for row in list(
                    gates.get("expected_full_quality_identities") or []
                ) if isinstance(row, Mapping)
            ],
            setup_id=target_id,
            model_id=model_id,
            eval_run_id=run_root.name,
        )
        full_only_quality = quality_evidence.get("requested") is True
        if full_only_quality:
            ok = bool(
                status_name == "ok"
                and semantic_rows == 0
                and quality_evidence.get("status") == "verified_exact"
            )
            if status_name == "ok" and not ok:
                status_name = "partial"
        else:
            ok = status_name == "ok" and semantic_rows > 0
        if (
            not full_only_quality
            and status_name == "ok" and semantic_rows <= 0
        ):
            status_name = "partial"
        elif status_name == "ok" and str(copy_summary.get("status") or "") == "partial":
            status_name = "partial"
            ok = False
        p_status = _write_status({
            "status": status_name,
            "reason": (
                "remote_full_only_quality_evidence_verified"
                if ok and full_only_quality
                else "remote_service_completed" if ok
                else str(out.get("error") or "remote_service_partial_or_failed")
                if isinstance(out, Mapping) else "remote_service_failed"
            ),
            "host": dispatch_payload["host"],
            "remote_output": dict(out or {}) if isinstance(out, Mapping) else {},
            "local_run_dir": str(local_run_dir) if local_run_dir else "",
            "copied_result_files": copied,
            "copied_result_count": len(copied),
            "canonical_result_count": int(copy_summary.get("file_count") or 0),
            "canonical_nonempty_row_count": semantic_rows,
            "canonical_run_ids_with_rows": list(copy_summary.get("run_ids_with_rows") or []),
            "canonical_run_ids_without_rows": list(copy_summary.get("run_ids_without_rows") or []),
            "quality_evidence_count": int(
                quality_evidence.get("quality_evidence_count") or 0
            ),
            "full_only_quality_evidence": quality_evidence,
            "result_copy_manifest": str(copy_manifest or ""),
            "deepx_prepared_input_materialization": deepx_prepared_input,
            "stdout_path": relpath(stdout_path, run_root),
            "stderr_path": relpath(stderr_path, run_root),
            "execution_gates": dict(gates),
            "runtime_override": dict(runtime_override or {}),
        })
        return ExecutionBindingResult(
            artifacts={f"remote_benchmark_status{status_suffix}_json": p_status, f"remote_benchmark_dispatch{status_suffix}_json": p_dispatch, f"remote_benchmark_stdout{status_suffix}_txt": stdout_path, f"remote_benchmark_stderr{status_suffix}_txt": stderr_path},
            metrics={"remote_requested": True, "remote_dispatched": True, "remote_status": status_name, "remote_result_files_copied": len(copied), "canonical_result_files_copied": int(copy_summary.get("file_count") or 0), "canonical_result_rows_copied": semantic_rows, "quality_evidence_count": int(quality_evidence.get("quality_evidence_count") or 0), "quality_evidence_status": str(quality_evidence.get("status") or ""), "quality_evidence_errors": list(quality_evidence.get("errors") or []), "hardware_target_id": target_id},
            status="ok" if ok else "partial",
            message=(
                "Remote Full-only quality evidence was verified with no performance rows."
                if ok and full_only_quality
                else "Remote benchmark service completed and canonical result files were collected."
                if ok else "Remote benchmark service was dispatched; required result evidence was incomplete or missing."
            ),
        )
    except Exception as exc:
        stdout_path.write_text(
            "\n".join(remote_stdout) + ("\n" if remote_stdout else ""),
            encoding="utf-8",
        )
        stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        post_dispatch_processing_failure = bool(remote_was_dispatched)
        p_status = _write_status({
            "status": (
                "partial" if post_dispatch_processing_failure
                else "failed_to_dispatch"
            ),
            "reason": (
                "remote_result_processing_failed"
                if post_dispatch_processing_failure
                else "remote_service_dispatch_failed"
            ),
            "error_class": (
                "result_processing_failed"
                if post_dispatch_processing_failure else "runtime_failed"
            ),
            "error_detail": f"{type(exc).__name__}: {exc}",
            "host": dispatch_payload["host"],
            "remote_output": remote_output,
            "local_run_dir": (
                str(local_run_dir) if str(local_run_dir) not in {"", "."}
                else ""
            ),
            "copied_result_files": copied,
            "copied_result_count": len(copied),
            "remote_dispatch_failed": not post_dispatch_processing_failure,
            "remote_dispatched": post_dispatch_processing_failure,
            "stdout_path": relpath(stdout_path, run_root),
            "stderr_path": relpath(stderr_path, run_root),
            "execution_gates": dict(gates),
            "runtime_override": dict(runtime_override or {}),
        })
        return ExecutionBindingResult(
            artifacts={f"remote_benchmark_status{status_suffix}_json": p_status, f"remote_benchmark_dispatch{status_suffix}_json": p_dispatch, f"remote_benchmark_stdout{status_suffix}_txt": stdout_path, f"remote_benchmark_stderr{status_suffix}_txt": stderr_path},
            metrics={
                "remote_requested": True,
                "remote_dispatched": post_dispatch_processing_failure,
                "remote_dispatch_failed": not post_dispatch_processing_failure,
                "remote_status": (
                    "partial" if post_dispatch_processing_failure
                    else "failed_to_dispatch"
                ),
                "remote_result_processing_failed": (
                    post_dispatch_processing_failure
                ),
                "remote_result_files_copied": len(copied),
                "hardware_target_id": target_id,
            },
            status="partial",
            message=(
                "Remote benchmark completed, but local result processing "
                "failed; collected work was retained."
                if post_dispatch_processing_failure
                else "Remote benchmark dispatch failed; status recorded."
            ),
        )


def _remote_execution_if_requested(
    *,
    run_root: Path,
    model_id: str,
    options: Any,
    profile_payload: Optional[Mapping[str, Any]],
    suite_dir: Path,
    benchmark_set_json: Path,
    result_dir: Path,
    contains_hailo: bool,
    gates: Mapping[str, Any],
    log: Optional[Callable[[str], None]],
    model_task: str = "",
    cancel_event: Any = None,
    remote_process_registry: RemoteProcessLeaseRegistry | None = None,
    workflow_session_id: str = "",
    targeted_full_quality_identities: Optional[
        Sequence[Mapping[str, Any]]
    ] = None,
) -> Optional[ExecutionBindingResult]:
    no_remote = bool(getattr(options, "no_remote", False))
    profile_remote = _profile_remote_execution(profile_payload)
    hardware_targets_for_request = _hardware_targets_for_plan(profile_payload, {})
    hardware_remote_requested = any(
        bool((t.get("remote") or t.get("runtime") or {}).get("enabled")) or bool((t.get("remote") or t.get("runtime") or {}).get("host"))
        for t in hardware_targets_for_request if isinstance(t, Mapping)
    )
    # v55f: Profile editor no longer writes remote_execution.enabled; selected
    # hardware setups already carry their remote runtime config.  Older GUI
    # paths still set options.no_remote=True.  Treat that as stale unless the
    # user explicitly selected benchmark_execution_backend=local.
    if no_remote and hardware_remote_requested:
        try:
            if callable(log):
                log("[workflow][remote] selected hardware setups contain remote runtimes; ignoring stale no_remote=True")
        except Exception:
            pass
        no_remote = False
    hardware_accelerator_requested = any(
        canon_accelerator(t.get("accelerator")) in {"hailo8", "hailo10", "hailo10n", "hailo10h", "deepx_m1"}
        for t in hardware_targets_for_request if isinstance(t, Mapping)
    )
    execution_backend = str(getattr(options, "benchmark_execution_backend", "auto") or "auto").strip().lower()
    if execution_backend == "local":
        remote_requested = False
    elif execution_backend == "remote":
        remote_requested = not no_remote
    else:
        remote_requested = (not no_remote) and (
            contains_hailo
            or hardware_remote_requested
            or hardware_accelerator_requested
            or bool(profile_remote.get("enabled"))
            or bool(getattr(options, "remote_host", ""))
            or bool(getattr(options, "remote_host_json", ""))
            or bool(getattr(options, "remote_host_id", ""))
            or bool(getattr(options, "remote_hosts_file", ""))
        )
    if not remote_requested:
        return None

    status_path = result_dir / "remote_benchmark_status.json"
    dispatch_path = result_dir / "remote_benchmark_dispatch.json"
    stdout_path = result_dir / "remote_benchmark_stdout.txt"
    stderr_path = result_dir / "remote_benchmark_stderr.txt"

    def _write_status(payload: Dict[str, Any]) -> Path:
        payload.setdefault("schema", "onnx-splitpoint/remote-benchmark-status")
        payload.setdefault("schema_version", 1)
        payload.setdefault("created_at", now_iso())
        payload.setdefault("model_id", model_id)
        payload.setdefault("suite_dir", relpath(suite_dir, run_root))
        return write_json(status_path, payload)

    # v59a: If the profile defines hardware_targets/hardware_setups, dispatch the
    # authoritative benchmark suite per selected accelerator setup.  Different
    # Orin/u.RECS setups can run concurrently, while each setup serializes its
    # own run-ids/energy windows.  This keeps measurements independent: one
    # remote host + one u.RECS collector per setup.
    hardware_targets = hardware_targets_for_request
    if hardware_targets:
        all_artifacts: Dict[str, Path] = {}
        statuses: List[str] = []
        total_copied = 0
        dispatch_records: List[Dict[str, Any]] = []
        benchmark_plan_for_runs = read_json(suite_dir / "benchmark_plan.json", default={}) if (suite_dir / "benchmark_plan.json").is_file() else {}
        plan_has_runs = bool(_plan_run_entries(benchmark_plan_for_runs))
        reference_target_id = _reference_target_id_for_matrix(hardware_targets)
        reference_run_ids = _reference_run_ids_for_plan(benchmark_plan_for_runs)
        task_groups: List[Dict[str, Any]] = []

        # v2.75.21: central quality is keyed by physical setup.  Build and
        # validate all setup-local TensorRT companion dispatches before a
        # worker, upload, or SSH session can start.  The two Hailo setups emit
        # quality-only evidence; DeepX remains the sole TensorRT performance
        # owner.  ORT-CPU never enters this remote contract.
        plan_rows_for_dispatch = _plan_run_entries(benchmark_plan_for_runs)
        full_only_quality_plan = bool(
            plan_rows_for_dispatch
            and all(
                bool(
                    row.get("semantic_reference_only")
                    or row.get("canonical_cpu_reference")
                    or (
                        str(row.get("execution_scope") or "") == "full_only"
                        and str(row.get("execution_role") or "")
                        == "full_quality_only"
                        and row.get("performance_claims_emitted") is False
                    )
                )
                for row in plan_rows_for_dispatch
            )
        )
        setup_local_trt_contract: Dict[str, Any] = {}
        setup_local_by_id: Dict[str, Dict[str, Any]] = {}
        if (
            (full_only_quality_plan or _central_management_quality_enabled(profile_payload))
            and any(
                _run_id(row).strip().lower().replace("-", "_")
                == "ort_tensorrt"
                for row in plan_rows_for_dispatch
            )
            and any(
                canon_accelerator(target.get("accelerator"))
                in {"hailo8", "hailo10", "hailo10h", "hailo10n", "deepx_m1"}
                for target in hardware_targets if isinstance(target, Mapping)
            )
        ):
            setup_local_trt_contract = (
                build_setup_local_tensorrt_quality_dispatch(
                    dict(profile_payload or {}),
                    hardware_targets=hardware_targets,
                    plan_rows=plan_rows_for_dispatch,
                )
            )
            if targeted_full_quality_identities is not None:
                setup_local_trt_contract = (
                    _filter_setup_local_dispatch_for_targeted_full_quality(
                        setup_local_trt_contract,
                        targeted_full_quality_identities,
                    )
                )
            preflight_path = write_json(
                result_dir / "setup_local_tensorrt_dispatch_preflight.json",
                setup_local_trt_contract,
            )
            all_artifacts[
                "setup_local_tensorrt_dispatch_preflight_json"
            ] = preflight_path
            if setup_local_trt_contract.get("ok") is not True:
                errors = ",".join(
                    str(value)
                    for value in list(
                        setup_local_trt_contract.get("errors") or []
                    )
                )
                raise RuntimeError(
                    "setup_local_tensorrt_dispatch_preflight_failed:"
                    + (errors or "unknown_contract_error")
                )
            setup_local_by_id = {
                str(row.get("setup_id") or ""): dict(row)
                for row in list(
                    setup_local_trt_contract.get("setup_dispatches") or []
                )
                if isinstance(row, Mapping)
                and str(row.get("setup_id") or "").strip()
            }

        for hw in hardware_targets:
            runtime_base = dict(hw.get("remote") or hw.get("runtime") or {}) if isinstance(hw, Mapping) else {}
            runtime_base.setdefault("enabled", True)
            runtime_base.setdefault("provider", str(hw.get("provider") or accelerator_provider(hw.get("accelerator"))))
            target_id = str(hw.get("id") or hw.get("accelerator") or "hardware")
            runtime_base.setdefault("setup_id", target_id)
            setup_local_dispatch = setup_local_by_id.get(target_id, {})
            run_ids = (
                list(setup_local_dispatch.get("run_ids") or [])
                if setup_local_dispatch
                else []
                if full_only_quality_plan
                else _run_ids_for_hardware_target(hw, benchmark_plan_for_runs)
            )
            if (
                not full_only_quality_plan
                and not setup_local_dispatch
                and target_id == reference_target_id
            ):
                for _rid in reference_run_ids:
                    if _rid not in run_ids:
                        run_ids.append(_rid)
            if _central_management_quality_enabled(profile_payload):
                # ORT-CPU is the semantic reference service, not a remote
                # latency/FPS baseline.  It is generated once on the management
                # node and must not be repeated on an arbitrary accelerator
                # host merely because that host owns the reference run group.
                cpu_reference_ids = {
                    _run_id(row).strip().lower().replace("-", "_")
                    for row in _plan_run_entries(benchmark_plan_for_runs)
                    if _is_cpu_reference_run_v263(row)
                }
                run_ids = [
                    rid for rid in run_ids
                    if str(rid).strip().lower().replace("-", "_") not in cpu_reference_ids
                ]
            if not run_ids and plan_has_runs:
                p_skip = write_json(result_dir / f"remote_benchmark_status_{target_id}.json", {
                    "schema": "onnx-splitpoint/remote-benchmark-status",
                    "schema_version": 1,
                    "created_at": now_iso(),
                    "model_id": model_id,
                    "hardware_target_id": target_id,
                    "status": "skipped",
                    "reason": "no_matching_run_ids_for_hardware_target",
                    "accelerator": hw.get("accelerator"),
                    "suite_dir": relpath(suite_dir, run_root),
                })
                all_artifacts[f"remote_benchmark_status_{target_id}_json"] = p_skip
                statuses.append("skipped")
                dispatch_records.append({"hardware_target_id": target_id, "status": "skipped", "message": "No benchmark_plan run id matched this hardware target.", "run_ids": []})
                continue
            if not run_ids:
                run_ids = [""]
            # v60o: one remote suite upload/process per model and physical setup.
            # The suite executes all setup-matching run ids sequentially in the
            # same remote workspace, sharing TensorRT engines and validation assets.
            runtime = dict(runtime_base)
            selected_ids = [str(x).strip() for x in run_ids if str(x).strip()]
            if selected_ids:
                raw_add_args = runtime.get("add_args")
                if targeted_full_quality_identities is not None:
                    try:
                        targeted_tokens = shlex.split(str(raw_add_args or ""))
                    except ValueError as exc:
                        raise RuntimeError(
                            f"remote_add_args_invalid:{exc}"
                        ) from exc
                    targeted_tokens = [
                        token for token in targeted_tokens
                        if token.split("=", 1)[0] not in {
                            "--native-trt-build", "--no-native-trt-build",
                        }
                    ]
                    targeted_tokens.append("--no-native-trt-build")
                    raw_add_args = shlex.join(targeted_tokens)
                runtime["add_args"] = _scheduler_owned_remote_add_args(
                    raw_add_args,
                    run_ids=selected_ids,
                    quality_only_run_ids=list(
                        setup_local_dispatch.get("quality_only_run_ids") or []
                    ),
                )
            group_tasks = [{
                "hw": dict(hw),
                "rid": ",".join(selected_ids),
                "run_ids": selected_ids,
                "runtime": runtime,
                "target_id": target_id,
                "base_target_id": target_id,
                "tensorrt_execution_role": str(
                    setup_local_dispatch.get("tensorrt_execution_role") or ""
                ),
                "quality_only_run_ids": list(
                    setup_local_dispatch.get("quality_only_run_ids") or []
                ),
                "quality_companion_required": (
                    setup_local_dispatch.get("quality_companion_required")
                    is True
                ),
                "quality_companion_endpoint_id": str(
                    setup_local_dispatch.get(
                        "quality_companion_endpoint_id"
                    ) or ""
                ),
                "execution_scope": str(
                    setup_local_trt_contract.get("execution_scope") or ""
                ),
                "expected_full_quality_identities": [
                    dict(row) for row in list(
                        setup_local_dispatch.get(
                            "expected_full_quality_identities"
                        ) or []
                    ) if isinstance(row, Mapping)
                ],
            }]
            task_groups.append({
                "target_id": target_id,
                "tasks": group_tasks,
                "run_ids": selected_ids,
                "tensorrt_execution_role": str(
                    setup_local_dispatch.get("tensorrt_execution_role") or ""
                ),
                "quality_only_run_ids": list(
                    setup_local_dispatch.get("quality_only_run_ids") or []
                ),
                "quality_companion_required": (
                    setup_local_dispatch.get("quality_companion_required")
                    is True
                ),
                "quality_companion_endpoint_id": str(
                    setup_local_dispatch.get(
                        "quality_companion_endpoint_id"
                    ) or ""
                ),
                "expected_full_quality_identities": [
                    dict(row) for row in list(
                        setup_local_dispatch.get(
                            "expected_full_quality_identities"
                        ) or []
                    ) if isinstance(row, Mapping)
                ],
            })

        unique_setup_count = len(task_groups)
        parallel_enabled = _parallel_remote_setups_enabled(options, profile_payload, default=True) and unique_setup_count > 1
        max_workers = min(unique_setup_count, _parallel_remote_max_setups(options, profile_payload, default=3)) if parallel_enabled else 1
        max_uploads = _parallel_remote_max_uploads(options, profile_payload, default=1)
        power_workers = _parallel_powercalc_workers(options, profile_payload, default=1)
        old_upload_env = os.environ.get("ONNX_SPLITPOINT_MAX_PARALLEL_UPLOADS")
        old_power_env = os.environ.get("ONNX_SPLITPOINT_POWER_CALC_WORKERS")
        os.environ["ONNX_SPLITPOINT_MAX_PARALLEL_UPLOADS"] = str(max_uploads)
        os.environ["ONNX_SPLITPOINT_POWER_CALC_WORKERS"] = str(power_workers)
        log_lock = threading.Lock()

        def _safe_log(msg: str) -> None:
            if callable(log):
                with log_lock:
                    try:
                        log(str(msg))
                    except Exception:
                        pass

        if callable(log):
            _safe_log(
                f"[workflow][remote] parallel setup dispatch={'enabled' if parallel_enabled else 'disabled'} "
                f"setup_groups={unique_setup_count} max_workers={max_workers} max_parallel_uploads={max_uploads} powercalc_workers={power_workers}"
            )

        def _run_group(group: Dict[str, Any]) -> Dict[str, Any]:
            group_artifacts: Dict[str, Path] = {}
            group_statuses: List[str] = []
            group_copied = 0
            group_records: List[Dict[str, Any]] = []
            gid = str(group.get("target_id") or "hardware")
            tasks = list(group.get("tasks") or [])
            for task in tasks:
                rid = str(task.get("rid") or "")
                if _cancel_requested(cancel_event):
                    group_statuses.append("cancelled")
                    group_records.append({
                        "hardware_target_id": gid,
                        "run_id": rid,
                        "status": "cancelled",
                        "message": "Remote setup dispatch cancelled before task start.",
                        "metrics": {"cancelled": True, "returncode": 130},
                    })
                    break
                runtime = dict(task.get("runtime") or {})
                hw = dict(task.get("hw") or {})
                child_id = str(task.get("target_id") or gid)
                try:
                    _safe_log(f"[workflow][remote][{gid}] start run_id={rid or '(all)'}")
                    res = _run_remote_dispatch_once(
                        run_root=run_root,
                        model_id=model_id,
                        options=options,
                        profile_payload=profile_payload,
                        suite_dir=suite_dir,
                        benchmark_set_json=benchmark_set_json,
                        result_dir=result_dir,
                        gates={
                            **dict(gates),
                            "hardware_target": dict(hw),
                            "hardware_run_id": rid,
                            "hardware_run_ids": list(task.get("run_ids") or []),
                            "tensorrt_execution_role": str(
                                task.get("tensorrt_execution_role") or ""
                            ),
                            "quality_only_run_ids": list(
                                task.get("quality_only_run_ids") or []
                            ),
                            "quality_companion_required": (
                                task.get("quality_companion_required") is True
                            ),
                            "quality_companion_endpoint_id": str(
                                task.get("quality_companion_endpoint_id") or ""
                            ),
                            "execution_scope": str(
                                task.get("execution_scope") or ""
                            ),
                            "expected_full_quality_identities": [
                                dict(row) for row in list(
                                    task.get(
                                        "expected_full_quality_identities"
                                    ) or []
                                ) if isinstance(row, Mapping)
                            ],
                        },
                        log=lambda m, _gid=gid: _safe_log(f"[{_gid}] {m}"),
                        runtime_override=runtime,
                        target_id=child_id,
                        model_task=model_task,
                        cancel_event=cancel_event,
                        remote_process_registry=remote_process_registry,
                        workflow_session_id=workflow_session_id,
                    )
                    group_artifacts.update(res.artifacts)
                    group_statuses.append(res.status)
                    group_copied += int(res.metrics.get("remote_result_files_copied") or 0)
                    group_records.append({
                        "hardware_target_id": gid,
                        "run_id": rid,
                        "run_ids": list(task.get("run_ids") or []),
                        "tensorrt_execution_role": str(
                            task.get("tensorrt_execution_role") or ""
                        ),
                        "quality_only_run_ids": list(
                            task.get("quality_only_run_ids") or []
                        ),
                        "quality_companion_required": (
                            task.get("quality_companion_required") is True
                        ),
                        "quality_companion_endpoint_id": str(
                            task.get("quality_companion_endpoint_id") or ""
                        ),
                        "expected_full_quality_identities": [
                            dict(row) for row in list(
                                task.get(
                                    "expected_full_quality_identities"
                                ) or []
                            ) if isinstance(row, Mapping)
                        ],
                        "status": res.status,
                        "message": res.message,
                        "metrics": dict(res.metrics),
                    })
                    _safe_log(f"[workflow][remote][{gid}] done run_id={rid or '(all)'} status={res.status}")
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    was_cancelled = _cancel_requested(cancel_event)
                    status = "cancelled" if was_cancelled else "partial"
                    _safe_log(f"[workflow][remote][{gid}] {status} run_id={rid or '(all)'}: {err}")
                    group_statuses.append(status)
                    group_records.append({"hardware_target_id": gid, "run_id": rid, "status": status, "message": err, "metrics": {"remote_exception": err, "cancelled": was_cancelled}})
            return {"target_id": gid, "artifacts": group_artifacts, "statuses": group_statuses, "copied": group_copied, "records": group_records}

        try:
            if parallel_enabled:
                pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="osp-eval-remote-setup",
                )
                futures: Dict[concurrent.futures.Future[Any], str] = {}
                pending: set[concurrent.futures.Future[Any]] = set()
                cancellation_seen = False
                abnormal_exit = False
                try:
                    for group in task_groups:
                        # ThreadPoolExecutor does not propagate ContextVars.
                        # Use a distinct snapshot per worker so the workflow's
                        # bound local/remote ownership registries remain active.
                        future = _submit_with_context(
                            pool,
                            _run_group,
                            group,
                        )
                        futures[future] = str(
                            group.get("target_id") or "hardware"
                        )
                        pending.add(future)
                    cancel_deadline: float | None = None
                    while pending:
                        if _cancel_requested(cancel_event):
                            cancellation_seen = True
                            if cancel_deadline is None:
                                cancel_deadline = (
                                    time.monotonic()
                                    + _PARALLEL_REMOTE_CANCEL_GRACE_S
                                )
                            for future in pending:
                                future.cancel()
                            if time.monotonic() >= cancel_deadline:
                                break
                        done, pending = concurrent.futures.wait(
                            pending,
                            timeout=0.1,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            target = futures[future]
                            try:
                                out = future.result(timeout=0.0)
                            except concurrent.futures.CancelledError:
                                statuses.append("cancelled")
                                dispatch_records.append({
                                    "hardware_target_id": target,
                                    "status": "cancelled",
                                    "message": "Remote setup dispatch cancelled before worker start.",
                                    "metrics": {"cancelled": True, "returncode": 130},
                                })
                                continue
                            all_artifacts.update(out.get("artifacts") or {})
                            statuses.extend(list(out.get("statuses") or []))
                            total_copied += int(out.get("copied") or 0)
                            dispatch_records.extend(list(out.get("records") or []))

                    unresolved_pending = set(
                        _cancel_pending_workers_and_record_uncertainty(
                            list(pending),
                            label="parallel-remote-setup-workers",
                            detail=(
                                f"{len(pending)} remote setup worker(s) remained "
                                "after bounded cancellation grace"
                            ),
                        )
                    )
                    for future in pending:
                        target = futures[future]
                        statuses.append("cancelled")
                        dispatch_records.append({
                            "hardware_target_id": target,
                            "status": (
                                "cancelled_cleanup_unresolved"
                                if future in unresolved_pending
                                else "cancelled"
                            ),
                            "message": (
                                "Remote setup worker did not exit within the "
                                "bounded cancellation grace period."
                                if future in unresolved_pending
                                else "Remote setup dispatch cancelled before worker start."
                            ),
                            "metrics": {"cancelled": True, "returncode": 130},
                        })
                    pending = unresolved_pending
                except BaseException:
                    abnormal_exit = True
                    if pending:
                        _cancel_pending_workers_and_record_uncertainty(
                            list(pending),
                            label="parallel-remote-setup-workers",
                            detail=(
                                f"{len(pending)} remote setup worker(s) remained "
                                "during exceptional pool exit"
                            ),
                        )
                    raise
                finally:
                    pool.shutdown(
                        wait=not (cancellation_seen or abnormal_exit or pending),
                        cancel_futures=True,
                    )
            else:
                for g in task_groups:
                    out = _run_group(g)
                    all_artifacts.update(out.get("artifacts") or {})
                    statuses.extend(list(out.get("statuses") or []))
                    total_copied += int(out.get("copied") or 0)
                    dispatch_records.extend(list(out.get("records") or []))
        finally:
            if old_upload_env is None:
                os.environ.pop("ONNX_SPLITPOINT_MAX_PARALLEL_UPLOADS", None)
            else:
                os.environ["ONNX_SPLITPOINT_MAX_PARALLEL_UPLOADS"] = old_upload_env
            if old_power_env is None:
                os.environ.pop("ONNX_SPLITPOINT_POWER_CALC_WORKERS", None)
            else:
                os.environ["ONNX_SPLITPOINT_POWER_CALC_WORKERS"] = old_power_env

        quality_evidence_count = sum(
            int((row.get("metrics") or {}).get("quality_evidence_count") or 0)
            for row in dispatch_records if isinstance(row, Mapping)
        )
        expected_full_quality_count = sum(
            len(list(row.get("expected_full_quality_identities") or []))
            for row in dispatch_records if isinstance(row, Mapping)
        )
        pre_dispatch_failures = [
            dict(row.get("metrics") or {})
            for row in dispatch_records
            if isinstance(row, Mapping)
            and isinstance(row.get("metrics"), Mapping)
            and (row.get("metrics") or {}).get("remote_dispatch_failed")
            is True
            and (row.get("metrics") or {}).get("remote_dispatched")
            is False
            and str(
                (row.get("metrics") or {}).get("failure_kind") or ""
            ).strip()
        ]
        pre_dispatch_cancellations = [
            dict(row.get("metrics") or {})
            for row in dispatch_records
            if isinstance(row, Mapping)
            and isinstance(row.get("metrics"), Mapping)
            and (row.get("metrics") or {}).get("cancelled_before_dispatch")
            is True
            and (row.get("metrics") or {}).get("remote_dispatch_failed")
            is False
            and (row.get("metrics") or {}).get("remote_dispatched")
            is False
        ]
        any_remote_dispatched = any(
            isinstance(row, Mapping)
            and isinstance(row.get("metrics"), Mapping)
            and (
                (row.get("metrics") or {}).get("remote_dispatched") is True
                or (
                    "remote_dispatched" not in (row.get("metrics") or {})
                    and str(row.get("status") or "") in {"ok", "partial"}
                )
            )
            for row in dispatch_records
        )
        all_pre_dispatch_terminal = bool(
            (pre_dispatch_failures or pre_dispatch_cancellations)
            and not any_remote_dispatched
            and all(
                status in {"failed_to_dispatch", "cancelled", "skipped"}
                for status in statuses
            )
        )
        all_pre_dispatch_failed = bool(
            all_pre_dispatch_terminal and pre_dispatch_failures
        )
        all_pre_dispatch_cancelled = bool(
            all_pre_dispatch_terminal
            and pre_dispatch_cancellations
            and not pre_dispatch_failures
        )
        failure_kinds = list(dict.fromkeys(
            str(row.get("failure_kind") or "").strip()
            for row in pre_dispatch_failures
            if str(row.get("failure_kind") or "").strip()
        ))
        cancellation_kinds = list(dict.fromkeys(
            str(row.get("failure_kind") or "").strip()
            for row in pre_dispatch_cancellations
            if str(row.get("failure_kind") or "").strip()
        ))
        event_kinds = failure_kinds or cancellation_kinds
        primary_errors = list(dict.fromkeys(
            str(row.get("primary_error") or row.get("remote_error") or "").strip()
            for row in pre_dispatch_failures
            if str(
                row.get("primary_error") or row.get("remote_error") or ""
            ).strip()
        ))
        cancellation_errors = list(dict.fromkeys(
            str(row.get("primary_error") or "").strip()
            for row in pre_dispatch_cancellations
            if str(row.get("primary_error") or "").strip()
        ))
        ok_status = bool(
            statuses
            and all(x in {"ok", "skipped"} for x in statuses)
            and any(x == "ok" for x in statuses)
            and total_copied > 0
        )
        matrix_status = (
            "ok" if ok_status
            else "failed_to_dispatch" if all_pre_dispatch_failed
            else "cancelled" if all_pre_dispatch_cancelled
            else "partial"
        )
        p_multi = write_json(result_dir / "remote_hardware_matrix_status.json", {
            "schema": "onnx-splitpoint/remote-hardware-matrix-status",
            "schema_version": 2,
            "created_at": now_iso(),
            "model_id": model_id,
            "status": matrix_status,
            "remote_dispatched": any_remote_dispatched,
            "remote_dispatch_failed": bool(pre_dispatch_failures),
            "cancelled": all_pre_dispatch_cancelled,
            "cancelled_before_dispatch": bool(pre_dispatch_cancellations),
            "failure_kind": event_kinds[0] if len(event_kinds) == 1 else "",
            "failure_kinds": event_kinds,
            "primary_error": (
                primary_errors[0] if primary_errors
                else cancellation_errors[0] if cancellation_errors else ""
            ),
            "primary_errors": primary_errors or cancellation_errors,
            "hardware_target_count": len(hardware_targets),
            "setup_group_count": unique_setup_count,
            "parallel_setup_dispatch": bool(parallel_enabled),
            "max_parallel_setups": max_workers,
            "max_parallel_uploads": max_uploads,
            "powercalc_workers": power_workers,
            "setup_local_tensorrt_dispatch": setup_local_trt_contract,
            "quality_evidence_count": quality_evidence_count,
            "expected_full_quality_count": expected_full_quality_count,
            "dispatches": dispatch_records,
        })
        all_artifacts["remote_hardware_matrix_status_json"] = p_multi
        primary_error = (
            primary_errors[0] if primary_errors
            else cancellation_errors[0] if cancellation_errors else ""
        )
        return ExecutionBindingResult(
            artifacts=all_artifacts,
            metrics={
                "remote_requested": True,
                "remote_dispatched": any_remote_dispatched,
                "remote_dispatch_failed": bool(pre_dispatch_failures),
                "cancelled": all_pre_dispatch_cancelled,
                "cancelled_before_dispatch": bool(
                    pre_dispatch_cancellations
                ),
                "remote_status": matrix_status,
                "failure_kind": (
                    event_kinds[0] if len(event_kinds) == 1 else ""
                ),
                "failure_kinds": event_kinds,
                "remote_reason": (
                    event_kinds[0] if len(event_kinds) == 1 else ""
                ),
                "primary_error": primary_error,
                "primary_errors": primary_errors,
                "remote_result_files_copied": total_copied,
                "hardware_target_count": len(hardware_targets),
                "hardware_matrix_dispatch": True,
                "parallel_setup_dispatch": bool(parallel_enabled),
                "max_parallel_setups": max_workers,
                "max_parallel_uploads": max_uploads,
                "powercalc_workers": power_workers,
                "quality_evidence_count": quality_evidence_count,
                "expected_full_quality_count": expected_full_quality_count,
            },
            status=matrix_status,
            message=(
                primary_error
                if (all_pre_dispatch_failed or all_pre_dispatch_cancelled)
                and primary_error
                else "Remote benchmark suite dispatched per hardware target/run-profile."
                if total_copied > 0
                else "Remote hardware target matrix dispatched; complete measured results were not collected from all targets."
            ),
        )

    host_payload = _remote_host_payload_from_options(options, profile_payload)
    if not host_payload or not str(host_payload.get("host") or "").strip():
        p_status = _write_status({
            "status": "not_dispatched",
            "reason": "remote_requested_but_no_host_configured",
            "execution_gates": dict(gates),
            "contains_hailo": bool(contains_hailo),
            "profile_remote_execution": profile_remote,
        })
        p_dispatch = write_json(dispatch_path, {
            "schema": "onnx-splitpoint/remote-benchmark-dispatch",
            "schema_version": 1,
            "created_at": now_iso(),
            "model_id": model_id,
            "status": "not_dispatched",
            "reason": "remote_requested_but_no_host_configured",
            "suite_dir": relpath(suite_dir, run_root),
            "benchmark_set_json": relpath(benchmark_set_json, run_root),
            "execution_gates": dict(gates),
        })
        return ExecutionBindingResult(
            artifacts={"remote_benchmark_status_json": p_status, "remote_benchmark_dispatch_json": p_dispatch},
            metrics={"remote_requested": True, "remote_dispatched": False, "remote_pending": True, "remote_reason": "remote_requested_but_no_host_configured"},
            status="partial",
            message="Remote benchmark execution was requested, but no host is configured.",
        )

    host = _make_ssh_host_config(host_payload)
    args = _remote_args_from_options(options, profile_payload, model_task=model_task)
    if _central_management_quality_enabled(profile_payload):
        single_host_plan = read_json(suite_dir / "benchmark_plan.json", default={}) or {}
        performance_run_ids = _performance_run_ids_v263(single_host_plan)
        if performance_run_ids:
            existing_args = str(getattr(args, "add_args", "") or "").strip()
            run_filter = "--run-ids " + ",".join(performance_run_ids)
            args.add_args = (existing_args + " " + run_filter).strip()
    local_working_dir = Path(str(getattr(options, "remote_working_dir", "") or profile_remote.get("local_working_dir") or result_dir / "remote_runs")).expanduser()
    remote_run_id = _remote_transport_run_id(
        run_root=run_root,
        model_id=model_id,
        target_id="",
        gates=gates,
        workflow_session_id=workflow_session_id,
    )
    dispatch_payload = {
        "schema": "onnx-splitpoint/remote-benchmark-dispatch",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "status": "dispatching",
        "suite_dir": relpath(suite_dir, run_root),
        "benchmark_set_json": relpath(benchmark_set_json, run_root),
        "local_working_dir": str(local_working_dir),
        "run_id": remote_run_id,
        "host": {"id": host.id, "label": host.label, "user": host.user, "host": host.host, "port": host.port, "remote_base_dir": host.remote_base_dir},
        "energy_policy": {
            "profile_energy_enabled": bool(energy_enabled),
            "final_all_split_energy": _energy_final_all_split_enabled(options, profile_payload),
            "final_energy_skip_cpu_ort": _energy_final_skip_cpu_ort_enabled(options, profile_payload),
            "run_id": energy_run_id,
            "enabled_for_this_dispatch": bool(energy_enabled and energy_allowed),
            "reason": energy_reason,
            "target_policy": _energy_target_policy_for_profile(options, profile_payload),
            "skip_backends": _energy_skip_backends_for_profile(options, profile_payload),
            "include_run_ids": _energy_include_run_ids_for_profile(options, profile_payload),
            "exclude_run_ids": _energy_exclude_run_ids_for_profile(options, profile_payload),
            "heartbeat_s": _energy_heartbeat_s_for_profile(options, profile_payload),
            "max_targets_per_run_id": _energy_max_targets_per_run_id_for_profile(options, profile_payload),
            "max_work_units_per_window": _energy_max_work_units_per_window_for_profile(options, profile_payload),
            "max_window_duration_s": _energy_max_window_duration_s_for_profile(options, profile_payload),
            "timeout_s_per_window": _energy_timeout_s_per_window_for_profile(options, profile_payload),
            "sizing_probe_max_work_units": _energy_sizing_probe_max_work_units_for_profile(options, profile_payload),
        },
        "args": getattr(args, "__dict__", {}),
        "execution_gates": dict(gates),
    }
    p_dispatch = write_json(dispatch_path, dispatch_payload)

    if callable(log):
        log(f"[workflow][remote] dispatching {model_id} to {host.user}@{host.host}:{host.port}")

    remote_stdout: List[str] = []
    remote_stderr: List[str] = []
    # Keep the legacy single-host path on the same artifact contract as the
    # setup-local dispatcher: successful silence is an empty regular file.
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    try:
        from ..benchmark.services import RemoteBenchmarkService
        service = RemoteBenchmarkService()
        out = service.run(
            host=host,
            benchmark_set_json=benchmark_set_json,
            local_working_dir=local_working_dir,
            run_id=remote_run_id,
            args=args,
            log=lambda msg: (remote_stdout.append(str(msg)), callable(log) and log(f"[workflow][remote] {msg}")),
            progress=lambda pct, msg: callable(log) and log(f"[workflow][remote] {int(round(float(pct) * 100.0))}% {msg}"),
            cancel_event=cancel_event,
            remote_process_registry=remote_process_registry,
            workflow_session_id=workflow_session_id,
        )
        stdout_path.write_text("\n".join(remote_stdout) + ("\n" if remote_stdout else ""), encoding="utf-8")
        local_run_dir = Path(str(out.get("local_run_dir") or "")) if isinstance(out, Mapping) else Path("")
        pre_dispatch_cancel = _pre_mutation_remote_dispatch_cancellation(out)
        if pre_dispatch_cancel:
            primary_error = str(pre_dispatch_cancel["primary_error"])
            stderr_path.write_text(primary_error + "\n", encoding="utf-8")
            p_dispatch = write_json(dispatch_path, {
                **dispatch_payload,
                "status": "cancelled_before_dispatch",
                "reason": pre_dispatch_cancel["reason"],
                "failure_kind": pre_dispatch_cancel["failure_kind"],
                "primary_error": primary_error,
                "remote_dispatch_failed": False,
                "remote_dispatched": False,
            })
            p_status = _write_status({
                "status": "cancelled",
                "dispatch_status": "cancelled_before_dispatch",
                "reason": pre_dispatch_cancel["reason"],
                "failure_kind": pre_dispatch_cancel["failure_kind"],
                "primary_error": primary_error,
                "primary_failure": dict(pre_dispatch_cancel),
                "remote_rc": pre_dispatch_cancel.get("remote_rc"),
                "remote_dispatch_failed": False,
                "remote_dispatched": False,
                "pre_remote_mutation": True,
                "host": dispatch_payload["host"],
                "remote_output": dict(out),
                "local_run_dir": str(local_run_dir),
                "stdout_path": relpath(stdout_path, run_root),
                "stderr_path": relpath(stderr_path, run_root),
                "execution_gates": dict(gates),
            })
            return ExecutionBindingResult(
                artifacts={
                    "remote_benchmark_status_json": p_status,
                    "remote_benchmark_dispatch_json": p_dispatch,
                    "remote_benchmark_stdout_txt": stdout_path,
                    "remote_benchmark_stderr_txt": stderr_path,
                },
                metrics={
                    "remote_requested": True,
                    "remote_dispatched": False,
                    "remote_dispatch_failed": False,
                    "remote_status": "cancelled",
                    "cancelled": True,
                    "cancelled_before_dispatch": True,
                    "failure_kind": pre_dispatch_cancel["failure_kind"],
                    "remote_reason": pre_dispatch_cancel["reason"],
                    "primary_error": primary_error,
                    "remote_rc": pre_dispatch_cancel.get("remote_rc"),
                    "pre_remote_mutation": True,
                    "contains_hailo": bool(contains_hailo),
                },
                status="cancelled",
                message=primary_error,
            )
        pre_dispatch_failure = _pre_mutation_remote_dispatch_failure(out)
        if pre_dispatch_failure:
            primary_error = str(
                pre_dispatch_failure.get("primary_error")
                or pre_dispatch_failure["failure_kind"]
            )
            stderr_path.write_text(primary_error + "\n", encoding="utf-8")
            p_dispatch = write_json(dispatch_path, {
                **dispatch_payload,
                "status": "failed_to_dispatch",
                "reason": pre_dispatch_failure["reason"],
                "failure_kind": pre_dispatch_failure["failure_kind"],
                "primary_error": primary_error,
                "remote_dispatch_failed": True,
                "remote_dispatched": False,
            })
            p_status = _write_status({
                "status": "failed_to_dispatch",
                "reason": pre_dispatch_failure["reason"],
                "failure_kind": pre_dispatch_failure["failure_kind"],
                "error_class": "remote_dispatch_failed",
                "error_detail": pre_dispatch_failure["error"],
                "primary_error": primary_error,
                "primary_failure": dict(pre_dispatch_failure),
                "remote_rc": pre_dispatch_failure.get("remote_rc"),
                "remote_dispatch_failed": True,
                "remote_dispatched": False,
                "pre_remote_mutation": bool(
                    pre_dispatch_failure.get("pre_remote_mutation")
                ),
                "host": dispatch_payload["host"],
                "remote_output": dict(out),
                "local_run_dir": str(local_run_dir),
                "stdout_path": relpath(stdout_path, run_root),
                "stderr_path": relpath(stderr_path, run_root),
                "execution_gates": dict(gates),
            })
            return ExecutionBindingResult(
                artifacts={
                    "remote_benchmark_status_json": p_status,
                    "remote_benchmark_dispatch_json": p_dispatch,
                    "remote_benchmark_stdout_txt": stdout_path,
                    "remote_benchmark_stderr_txt": stderr_path,
                },
                metrics={
                    "remote_requested": True,
                    "remote_dispatched": False,
                    "remote_dispatch_failed": True,
                    "remote_status": "failed_to_dispatch",
                    "failure_kind": pre_dispatch_failure["failure_kind"],
                    "remote_reason": pre_dispatch_failure["reason"],
                    "remote_error": primary_error,
                    "primary_error": primary_error,
                    "remote_rc": pre_dispatch_failure.get("remote_rc"),
                    "pre_remote_mutation": bool(
                        pre_dispatch_failure.get("pre_remote_mutation")
                    ),
                    "contains_hailo": bool(contains_hailo),
                },
                status="failed_to_dispatch",
                message=primary_error,
            )
        copied = _copy_remote_result_files(local_run_dir, result_dir) if local_run_dir else []
        canonical_copied = [row for row in copied if bool(row.get("canonical"))]
        copy_summary = _canonical_copy_summary(copied)
        copy_manifest = str(copy_summary.get("manifest") or "")
        status_name = str(out.get("status") or ("ok" if out.get("ok") else "failed")).strip().lower() if isinstance(out, Mapping) else "failed"
        semantic_rows = int(copy_summary.get("nonempty_row_count") or 0)
        ok = status_name == "ok" and semantic_rows > 0
        if status_name == "ok" and semantic_rows <= 0:
            status_name = "partial"
        elif status_name == "ok" and str(copy_summary.get("status") or "") == "partial":
            status_name = "partial"
            ok = False
        p_status = _write_status({
            "status": status_name,
            "reason": "remote_service_completed" if ok else str(out.get("error") or "remote_service_partial_or_failed") if isinstance(out, Mapping) else "remote_service_failed",
            "host": dispatch_payload["host"],
            "remote_output": dict(out or {}) if isinstance(out, Mapping) else {},
            "local_run_dir": str(local_run_dir) if local_run_dir else "",
            "copied_result_files": copied,
            "copied_result_count": len(copied),
            "canonical_result_count": int(copy_summary.get("file_count") or 0),
            "canonical_nonempty_row_count": semantic_rows,
            "canonical_run_ids_with_rows": list(copy_summary.get("run_ids_with_rows") or []),
            "canonical_run_ids_without_rows": list(copy_summary.get("run_ids_without_rows") or []),
            "result_copy_manifest": str(copy_manifest or ""),
            "stdout_path": relpath(stdout_path, run_root),
            "stderr_path": relpath(stderr_path, run_root),
            "execution_gates": dict(gates),
        })
        return ExecutionBindingResult(
            artifacts={"remote_benchmark_status_json": p_status, "remote_benchmark_dispatch_json": p_dispatch, "remote_benchmark_stdout_txt": stdout_path, "remote_benchmark_stderr_txt": stderr_path},
            metrics={"remote_requested": True, "remote_dispatched": True, "remote_status": status_name, "remote_result_files_copied": len(copied), "canonical_result_files_copied": int(copy_summary.get("file_count") or 0), "canonical_result_rows_copied": semantic_rows, "contains_hailo": bool(contains_hailo)},
            status="ok" if ok else "partial",
            message="Remote benchmark service completed and canonical result files were collected." if ok else "Remote benchmark service was dispatched; canonical measured result files were incomplete or missing.",
        )
    except Exception as exc:
        stdout_path.write_text(
            "\n".join(remote_stdout) + ("\n" if remote_stdout else ""),
            encoding="utf-8",
        )
        stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        p_status = _write_status({
            "status": "failed_to_dispatch",
            "reason": "remote_service_dispatch_failed",
            "error_class": "runtime_failed",
            "error_detail": f"{type(exc).__name__}: {exc}",
            "host": dispatch_payload["host"],
            "stdout_path": relpath(stdout_path, run_root),
            "stderr_path": relpath(stderr_path, run_root),
            "execution_gates": dict(gates),
        })
        return ExecutionBindingResult(
            artifacts={"remote_benchmark_status_json": p_status, "remote_benchmark_dispatch_json": p_dispatch, "remote_benchmark_stdout_txt": stdout_path, "remote_benchmark_stderr_txt": stderr_path},
            metrics={"remote_requested": True, "remote_dispatched": False, "remote_dispatch_failed": True, "contains_hailo": bool(contains_hailo)},
            status="partial",
            message="Remote benchmark dispatch failed; status recorded.",
        )

@dataclass
class ExecutionBindingResult:
    artifacts: Dict[str, Path] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "skipped"
    message: str = ""
    command: List[str] = field(default_factory=list)


def execute_benchmark_suite_if_requested(
    *,
    run_dir: str | Path,
    model_id: str,
    options: Any,
    benchmark_set_contract: Mapping[str, Any],
    benchmark_plan: Mapping[str, Any],
    profile_payload: Optional[Mapping[str, Any]] = None,
    model_entry: Optional[Mapping[str, Any]] = None,
    log: Optional[Callable[[str], None]] = None,
    cancel_event: Any = None,
    process_registry: ProcessTreeRegistry | None = None,
    remote_process_registry: RemoteProcessLeaseRegistry | None = None,
    workflow_session_id: str = "",
    runtime_plan_finalization: Optional[Mapping[str, Any]] = None,
    targeted_full_quality_identities: Optional[
        Sequence[Mapping[str, Any]]
    ] = None,
) -> ExecutionBindingResult:
    """Run a materialized benchmark suite in ``generate_and_run`` mode.

    The caller should normalize results afterwards. This function only dispatches
    the existing harness and records what happened.
    """

    run_root = Path(run_dir)
    model_dir = run_root / "models" / str(model_id)
    result_dir = model_dir / "benchmark_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    suite_dir = _resolve_suite_dir(run_root, model_dir, benchmark_set_contract)
    suite_payload = read_json(suite_dir / "benchmark_set.json", default={}) or {}
    if not isinstance(suite_payload, Mapping):
        suite_payload = {}

    mode = str(getattr(options, "execution_mode", "") or "").strip().lower().replace("-", "_")
    skip_benchmarks = bool(getattr(options, "skip_benchmarks", False))
    dry_run = bool(getattr(options, "dry_run", False))
    no_remote = bool(getattr(options, "no_remote", False))
    script = suite_dir / "benchmark_suite.py"
    plan_path = suite_dir / "benchmark_plan.json"
    status_path = result_dir / "benchmark_executor_status.json"
    dispatch_path = result_dir / "benchmark_execution_dispatch.json"
    stdout_path = result_dir / "benchmark_executor_stdout.txt"
    stderr_path = result_dir / "benchmark_executor_stderr.txt"

    def _write_status(payload: Dict[str, Any]) -> Path:
        payload.setdefault("schema", "onnx-splitpoint/benchmark-executor-status")
        payload.setdefault("schema_version", 1)
        payload.setdefault("created_at", now_iso())
        payload.setdefault("model_id", model_id)
        payload.setdefault("suite_dir", relpath(suite_dir, run_root))
        return write_json(status_path, payload)

    gates = {
        "execution_mode": mode or "contracts_only",
        "skip_benchmarks": skip_benchmarks,
        "dry_run": dry_run,
        "no_remote": no_remote,
        "benchmark_execution_backend": str(getattr(options, "benchmark_execution_backend", "auto") or "auto"),
        "targeted_missing_full_quality_only": (
            targeted_full_quality_identities is not None
        ),
    }
    model_task = str((model_entry or {}).get("task") or benchmark_plan.get("model_task") or benchmark_plan.get("task") or "").strip().lower()
    if model_task not in {"classification", "detection"}:
        model_task = ""
    quality_gate_policy = (
        dict(profile_payload.get("quality_gate") or {})
        if isinstance(profile_payload, Mapping) and isinstance(profile_payload.get("quality_gate"), Mapping)
        else {}
    )
    if suite_dir.is_dir():
        finalization = (
            dict(runtime_plan_finalization)
            if isinstance(runtime_plan_finalization, Mapping)
            else finalize_suite_for_runtime(
                suite_dir=suite_dir,
                run_root=run_root,
                model_id=str(model_id),
                suite_payload=suite_payload,
                benchmark_plan=benchmark_plan,
                profile_payload=profile_payload,
                model_task=model_task,
                quality_gate_policy=quality_gate_policy,
                log=log,
            )
        )
        finalized_plan = finalization.get("benchmark_plan")
        if isinstance(finalized_plan, Mapping):
            benchmark_plan = dict(finalized_plan)
        suite_payload = read_json(
            suite_dir / "benchmark_set.json", default=suite_payload,
        ) or suite_payload
        runtime_preflight = (
            dict(finalization.get("runtime_preflight") or {})
            if isinstance(finalization.get("runtime_preflight"), Mapping)
            else {}
        )
    else:
        finalization = {"status": "suite_dir_missing"}
        runtime_preflight = {"status": "suite_dir_missing"}
    planned_runs = (
        _as_list(benchmark_plan.get("runs"))
        or _as_list(benchmark_plan.get("planned_runs"))
    )
    central_quality = _central_management_quality_enabled(profile_payload)
    performance_run_ids = (
        _performance_run_ids_v263(benchmark_plan) if central_quality else []
    )
    if central_quality:
        gates["cpu_reference_execution"] = "central_management_semantic_only"
        gates["performance_run_ids"] = list(performance_run_ids)
    contains_hailo = (
        any("hailo" in str(row).lower() for row in planned_runs)
        or any(
            "hailo" in str(target).lower()
            for target in _as_list(benchmark_plan.get("targets"))
        )
    )
    has_splits = _has_materialized_split_cases(suite_dir, suite_payload)
    gates["runtime_plan_finalization_status"] = str(
        finalization.get("status") or ""
    )
    gates["runtime_preflight_status"] = str(runtime_preflight.get("status") or "") if isinstance(runtime_preflight, Mapping) else ""
    gates["runtime_preflight_path"] = str(runtime_preflight.get("path") or "") if isinstance(runtime_preflight, Mapping) else ""
    gates["model_task"] = model_task
    gates["quality_gate_policy_sha256"] = str(runtime_preflight.get("quality_gate_policy_sha256") or "") if isinstance(runtime_preflight, Mapping) else ""

    if dry_run or skip_benchmarks or mode != "generate_and_run":
        gate_reason = "runtime benchmark execution disabled by workflow options"
        gate_hint = (
            "Set workflow.execution_mode=generate_and_run and "
            "workflow.skip_runtime_benchmarks=false in the Evaluation Profile. "
            "For Hailo runtime verification, also enable remote_execution and set "
            "remote_venv, e.g. ~/hailo_py/bin/activate, or rely on auto-discovery."
        )
        p = _write_status({
            "status": "skipped_by_gate",
            "reason": gate_reason,
            "error_class": "runtime_benchmarks_disabled_by_profile",
            "severity": "blocking_if_runtime_expected",
            "diagnostic_hint": gate_hint,
            "execution_gates": gates,
            "contains_hailo": contains_hailo,
            "planned_run_count": len(planned_runs),
            "runtime_preflight": runtime_preflight,
        })
        p2 = write_json(dispatch_path, {"schema": "onnx-splitpoint/benchmark-execution-dispatch", "schema_version": 1, "status": "not_dispatched", "execution_gates": gates, "suite_dir": relpath(suite_dir, run_root)})
        return ExecutionBindingResult(
            artifacts={"benchmark_executor_status_json": p, "benchmark_execution_dispatch_json": p2},
            metrics={"dispatched": False, "reason": "skipped_by_gate", "error_class": "runtime_benchmarks_disabled_by_profile", "planned_runs": len(planned_runs), "contains_hailo": contains_hailo},
            status="skipped",
            message="Runtime benchmark execution skipped by workflow gate. Profile must use generate_and_run with skip_runtime_benchmarks=false.",
        )

    if not suite_dir.is_dir():
        p = _write_status({"status": "pending_service_execution", "reason": "suite_dir_missing", "execution_gates": gates})
        p2 = write_json(dispatch_path, {"schema": "onnx-splitpoint/benchmark-execution-dispatch", "schema_version": 1, "status": "not_dispatched", "reason": "suite_dir_missing", "suite_dir": str(suite_dir)})
        return ExecutionBindingResult({"benchmark_executor_status_json": p, "benchmark_execution_dispatch_json": p2}, {"dispatched": False, "reason": "suite_dir_missing"}, "skipped", "No generated benchmark suite directory found.")

    benchmark_set_json = suite_dir / "benchmark_set.json"

    # Do not dispatch an authoritative BenchmarkSet that contains zero
    # accepted cases. Older remote runners return rc=1 for this situation,
    # which made an already-clear generation problem look like a remote/runtime
    # failure. Keep imported/custom suites executable, but for the tool-owned
    # benchmark-set schema an empty cases list means there is nothing to run.
    suite_schema = str(suite_payload.get("schema") or "") if isinstance(suite_payload, Mapping) else ""
    suite_tool = suite_payload.get("tool") if isinstance(suite_payload, Mapping) else {}
    suite_tool_gui = ""
    if isinstance(suite_tool, Mapping):
        suite_tool_gui = str(suite_tool.get("gui") or "")
    suite_cases = _as_list(suite_payload.get("cases")) if isinstance(suite_payload, Mapping) else []
    tool_owned_empty_suite = (
        suite_schema == "onnx-splitpoint/benchmark-set"
        and "workflow" in suite_tool_gui.lower()
        and not suite_cases
        and not (suite_dir / "benchmark_results.json").is_file()
    )
    if tool_owned_empty_suite:
        p = _write_status({
            "status": "pending_materialization",
            "reason": "legacy_benchmark_set_has_no_accepted_cases",
            "execution_gates": gates,
            "has_materialized_split_cases": has_splits,
            "case_count": 0,
            "runtime_preflight": runtime_preflight,
        })
        p2 = write_json(dispatch_path, {
            "schema": "onnx-splitpoint/benchmark-execution-dispatch",
            "schema_version": 1,
            "status": "not_dispatched",
            "reason": "legacy_benchmark_set_has_no_accepted_cases",
            "suite_dir": relpath(suite_dir, run_root),
            "benchmark_set_json": relpath(benchmark_set_json, run_root),
        })
        if log:
            log(f"[benchmark execution:{model_id}] not dispatching: BenchmarkSet has no accepted cases")
        return ExecutionBindingResult(
            {"benchmark_executor_status_json": p, "benchmark_execution_dispatch_json": p2},
            {"dispatched": False, "reason": "legacy_benchmark_set_has_no_accepted_cases", "case_count": 0},
            "skipped",
            "BenchmarkSet has no accepted cases; benchmark execution not dispatched.",
        )

    if central_quality and _plan_run_entries(benchmark_plan) and not performance_run_ids:
        p = _write_status({
            "status": "skipped",
            "reason": "central_cpu_reference_only_no_performance_runs",
            "execution_gates": gates,
            "planned_run_count": 0,
        })
        p2 = write_json(dispatch_path, {
            "schema": "onnx-splitpoint/benchmark-execution-dispatch",
            "schema_version": 1,
            "status": "not_dispatched",
            "reason": "central_cpu_reference_only_no_performance_runs",
            "suite_dir": relpath(suite_dir, run_root),
        })
        return ExecutionBindingResult(
            {"benchmark_executor_status_json": p, "benchmark_execution_dispatch_json": p2},
            {"dispatched": False, "semantic_reference_delegated": True, "planned_runs": 0},
            "skipped",
            "Only the semantic ORT-CPU reference was planned; it is handled once on the management node.",
        )

    remote_result = _remote_execution_if_requested(
        run_root=run_root,
        model_id=model_id,
        options=options,
        profile_payload=profile_payload,
        suite_dir=suite_dir,
        benchmark_set_json=benchmark_set_json,
        result_dir=result_dir,
        contains_hailo=contains_hailo,
        gates=gates,
        log=log,
        model_task=model_task,
        cancel_event=cancel_event,
        remote_process_registry=remote_process_registry,
        workflow_session_id=workflow_session_id,
        targeted_full_quality_identities=targeted_full_quality_identities,
    )
    if remote_result is not None:
        return remote_result

    if not script.is_file():
        p = _write_status({"status": "pending_service_execution", "reason": "benchmark_suite_py_missing", "execution_gates": gates, "has_materialized_split_cases": has_splits})
        p2 = write_json(dispatch_path, {"schema": "onnx-splitpoint/benchmark-execution-dispatch", "schema_version": 1, "status": "not_dispatched", "reason": "benchmark_suite_py_missing", "suite_dir": relpath(suite_dir, run_root)})
        return ExecutionBindingResult({"benchmark_executor_status_json": p, "benchmark_execution_dispatch_json": p2}, {"dispatched": False, "reason": "benchmark_suite_py_missing"}, "skipped", "Benchmark suite runner is missing; execution remains delegated.")

    if _runner_is_contract_stub(script):
        p = _write_status({"status": "pending_materialization", "reason": "contract_stub_runner_only", "execution_gates": gates, "has_materialized_split_cases": has_splits})
        p2 = write_json(dispatch_path, {"schema": "onnx-splitpoint/benchmark-execution-dispatch", "schema_version": 1, "status": "not_dispatched", "reason": "contract_stub_runner_only", "suite_dir": relpath(suite_dir, run_root)})
        return ExecutionBindingResult({"benchmark_executor_status_json": p, "benchmark_execution_dispatch_json": p2}, {"dispatched": False, "reason": "contract_stub_runner_only"}, "skipped", "Only the v49c contract stub runner is present; no runtime measurements executed.")

    if not has_splits and not (suite_dir / "benchmark_results.json").is_file():
        # Imported suites may still contain their own runnable logic. Do not block
        # them solely because generated split artifacts are not visible, but record
        # the risk in the dispatch metadata. A direct-generated suite with no
        # accepted cases, however, has nothing useful to execute.
        split_note = "no generated part1/part2 ONNX cases detected"
        direct_empty = str(suite_payload.get("source") or "").strip() == "formal_workflow_direct_generator_binding" and not _as_list(suite_payload.get("cases"))
        if direct_empty:
            p = _write_status({"status": "pending_materialization", "reason": "direct_generated_suite_has_no_accepted_cases", "execution_gates": gates, "has_materialized_split_cases": has_splits})
            p2 = write_json(dispatch_path, {"schema": "onnx-splitpoint/benchmark-execution-dispatch", "schema_version": 1, "status": "not_dispatched", "reason": "direct_generated_suite_has_no_accepted_cases", "suite_dir": relpath(suite_dir, run_root)})
            return ExecutionBindingResult({"benchmark_executor_status_json": p, "benchmark_execution_dispatch_json": p2}, {"dispatched": False, "reason": "direct_generated_suite_has_no_accepted_cases"}, "skipped", "Generated suite has no accepted split cases; execution remains pending.")
    else:
        split_note = "materialized split cases detected"

    command = [sys.executable, str(script), "--plan", str(plan_path.name), "--resume", "--no-plot"]
    if central_quality and performance_run_ids:
        command.extend(["--run-ids", ",".join(performance_run_ids)])
    provider = str(getattr(options, "benchmark_provider", "") or "").strip()
    if provider:
        command.extend(["--provider", provider])
    preset = str(getattr(options, "benchmark_preset", "") or "").strip()
    if preset:
        command.extend(["--preset", preset])
    image = str(getattr(options, "benchmark_image", "") or "").strip()
    if image:
        command.extend(["--image", image])
    warmup = _int(getattr(options, "benchmark_warmup", 1), 1)
    runs = _int(getattr(options, "benchmark_runs", 3), 3)
    timeout_s = _int(getattr(options, "benchmark_timeout_s", 0), 0)
    command.extend(["--warmup", str(max(0, warmup)), "--runs", str(max(1, runs))])
    if timeout_s > 0:
        command.extend(["--timeout", str(timeout_s)])
    extra = list(getattr(options, "benchmark_extra_args", []) or [])
    command.extend([str(x) for x in extra if str(x).strip()])

    dispatch_payload = {
        "schema": "onnx-splitpoint/benchmark-execution-dispatch",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "suite_dir": relpath(suite_dir, run_root),
        "benchmark_set_json": relpath(suite_dir / "benchmark_set.json", run_root),
        "benchmark_plan_json": relpath(plan_path, run_root),
        "command": command,
        "cwd": str(suite_dir),
        "execution_gates": gates,
        "planned_run_count": len(planned_runs),
        "contains_hailo": contains_hailo,
        "has_materialized_split_cases": has_splits,
        "split_note": split_note,
        "runtime_preflight": runtime_preflight,
        "stdout_path": relpath(stdout_path, run_root),
        "stderr_path": relpath(stderr_path, run_root),
    }
    p_dispatch = write_json(dispatch_path, dispatch_payload)

    if callable(log):
        log(f"[workflow] executing benchmark suite for {model_id}: {' '.join(command)}")

    env = os.environ.copy()
    # Keep the package importable when running from a source checkout.
    pkg_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(pkg_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    if _cancel_requested(cancel_event):
        p_status = _write_status({
            "status": "cancelled",
            "reason": "user_cancelled_before_local_benchmark_start",
            "returncode": 130,
            "command": command,
            "execution_gates": gates,
        })
        return ExecutionBindingResult(
            {
                "benchmark_executor_status_json": p_status,
                "benchmark_execution_dispatch_json": p_dispatch,
            },
            {"dispatched": False, "cancelled": True, "returncode": 130},
            "cancelled",
            "Local benchmark execution was cancelled before process start.",
            command,
        )

    try:
        started = time.monotonic()
        popen_kwargs: Dict[str, Any] = {
            "cwd": str(suite_dir),
            "env": env,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):  # pragma: no cover
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_stream, stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_stream:
            proc = subprocess.Popen(command, stdout=stdout_stream, stderr=stderr_stream, **popen_kwargs)
            if process_registry is not None:
                process_registry.register(proc, label=f"benchmark-suite:{model_id}")
            try:
                cancelled = False
                timed_out = False
                while proc.poll() is None:
                    if _cancel_requested(cancel_event):
                        cancelled = True
                        if process_registry is not None:
                            process_registry.terminate_registered(
                                proc, grace_s=3.0
                            )
                        else:
                            _terminate_local_process_group(proc)
                        break
                    if timeout_s > 0 and (time.monotonic() - started) >= float(timeout_s):
                        timed_out = True
                        _terminate_local_process_group(proc)
                        break
                    time.sleep(0.1)
                if _cancel_requested(cancel_event):
                    cancelled = True
                if proc.poll() is None:
                    try:
                        returncode = int(proc.wait(timeout=2.0))
                    except subprocess.TimeoutExpired:
                        if process_registry is not None:
                            process_registry.terminate_registered(
                                proc, grace_s=0.0
                            )
                        else:
                            _terminate_local_process_group(proc)
                        try:
                            returncode = int(proc.wait(timeout=0.5))
                        except subprocess.TimeoutExpired:
                            returncode = 130 if cancelled else 124 if timed_out else 1
                else:
                    returncode = int(proc.returncode or 0)
            except BaseException:
                if process_registry is not None:
                    process_registry.terminate_registered(proc, grace_s=0.5)
                else:
                    _terminate_local_process_group(proc)
                raise
            finally:
                if process_registry is not None:
                    process_registry.unregister(proc)
        if cancelled:
            p_status = _write_status({
                "status": "cancelled",
                "reason": "user_cancelled_local_benchmark",
                "returncode": returncode,
                "command": command,
                "execution_gates": gates,
            })
            return ExecutionBindingResult(
                {"benchmark_executor_status_json": p_status, "benchmark_execution_dispatch_json": p_dispatch, "benchmark_executor_stdout_txt": stdout_path, "benchmark_executor_stderr_txt": stderr_path},
                {"dispatched": True, "cancelled": True, "returncode": returncode},
                "cancelled",
                "Local benchmark execution was cancelled and its process group was terminated.",
                command,
            )
        if timed_out:
            with stderr_path.open("a", encoding="utf-8") as stream:
                stream.write(f"\nTIMEOUT after {timeout_s}s\n")
            p_status = _write_status({
                "status": "timeout",
                "reason": "benchmark_executor_timeout",
                "error_class": "timeout",
                "command": command,
                "timeout_s": timeout_s,
                "stdout_path": relpath(stdout_path, run_root),
                "stderr_path": relpath(stderr_path, run_root),
                "execution_gates": gates,
            })
            return ExecutionBindingResult(
                {"benchmark_executor_status_json": p_status, "benchmark_execution_dispatch_json": p_dispatch, "benchmark_executor_stdout_txt": stdout_path, "benchmark_executor_stderr_txt": stderr_path},
                {"dispatched": True, "timeout": True, "timeout_s": timeout_s},
                "partial",
                "Benchmark executor timed out; its process group was terminated and status recorded.",
                command,
            )
        copied = _copy_result_files(suite_dir, result_dir)
        result_count = _result_file_count(suite_dir, result_dir)
        ok = returncode == 0 and result_count > 0
        status = "ok" if ok else ("partial" if result_count > 0 else "pending_or_failed")
        reason = "executor_completed" if ok else ("executor_returned_results_with_nonzero_rc" if result_count > 0 else "executor_produced_no_results")
        p_status = _write_status({
            "status": status,
            "reason": reason,
            "returncode": returncode,
            "command": command,
            "cwd": str(suite_dir),
            "stdout_path": relpath(stdout_path, run_root),
            "stderr_path": relpath(stderr_path, run_root),
            "copied_result_files": copied,
            "result_file_count": result_count,
            "execution_gates": gates,
            "contains_hailo": contains_hailo,
            "has_materialized_split_cases": has_splits,
            "runtime_preflight": runtime_preflight,
        })
        return ExecutionBindingResult(
            artifacts={"benchmark_executor_status_json": p_status, "benchmark_execution_dispatch_json": p_dispatch, "benchmark_executor_stdout_txt": stdout_path, "benchmark_executor_stderr_txt": stderr_path},
            metrics={"dispatched": True, "returncode": returncode, "result_file_count": result_count, "copied_result_files": len(copied), "contains_hailo": contains_hailo},
            status="ok" if ok else "partial",
            message="Benchmark suite executed and result files were collected." if ok else "Benchmark suite was dispatched, but no complete measured result set was collected.",
            command=command,
        )
    except Exception as exc:
        stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        p_status = _write_status({
            "status": "failed_to_dispatch",
            "reason": "benchmark_executor_dispatch_failed",
            "error_class": "runtime_failed",
            "error_detail": f"{type(exc).__name__}: {exc}",
            "command": command,
            "stdout_path": relpath(stdout_path, run_root),
            "stderr_path": relpath(stderr_path, run_root),
            "execution_gates": gates,
        })
        return ExecutionBindingResult({"benchmark_executor_status_json": p_status, "benchmark_execution_dispatch_json": p_dispatch, "benchmark_executor_stdout_txt": stdout_path, "benchmark_executor_stderr_txt": stderr_path}, {"dispatched": False, "dispatch_failed": True}, "partial", "Benchmark executor dispatch failed; status recorded.", command)
