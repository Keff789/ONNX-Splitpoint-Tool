from __future__ import annotations

"""Formal Evaluation Workflow -> benchmark-set generator binding.

This module keeps the v49 workflow runner independent from Tk/GUI state while
still producing a concrete benchmark-suite directory from the mandatory
``prediction.json`` -> ``final_candidate_plan.json`` chain.

The heavy GUI generator remains available for Hailo/remote builds, but the
formal runner can now materialize real split artifacts for ordinary ONNX models
and record structured generator/build decisions.  When an already-generated
suite is supplied in the profile, it is imported and audited against the formal
candidate plan instead of silently trusting the old Benchmark tab state.
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..cache_verify_policy import cache_verify_guard
from ..management_reference import (
    finalize_management_cpu_reference_plan_aliases,
    profile_has_explicit_cpu_reference,
)
from ..native_full_quality import enabled_run_profiles
from .artifacts import now_iso, relpath, write_csv, write_json, write_text


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value).strip()))
    except Exception:
        return None


def _case_id_for(candidate: Mapping[str, Any], idx: int) -> str:
    raw = str(candidate.get("case_id") or candidate.get("case_dir") or candidate.get("folder") or "").strip()
    if raw:
        return raw
    split = _safe_int(candidate.get("split_index") or candidate.get("boundary") or candidate.get("boundary_index"))
    if split is not None:
        return f"b{split:03d}"
    return f"case_{idx:03d}"


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _copy_if_file(src: Any, dst: Path) -> bool:
    raw = str(src or "").strip()
    if not raw:
        return False
    p = Path(raw).expanduser()
    if not p.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if str(p.resolve()) != str(dst.resolve()):
            shutil.copy2(p, dst)
        return dst.is_file()
    except Exception:
        return False




def _copy_scientific_freeze_artifacts(model_dir: Path, suite_dir: Path) -> Dict[str, str]:
    """Copy prospective prediction/quality artefacts into a standalone suite.

    The generated benchmark script runs on a remote host where the parent
    EvaluationRun tree is not present.  Keeping these files beside
    benchmark_plan.json allows the in-run reporter to verify the exact
    predictions that existed before measurements were opened.
    """
    analysis_dir = Path(model_dir) / "analysis"
    copied: Dict[str, str] = {}
    for name in (
        "prediction.json",
        "predictions_frozen.csv",
        "holdout_predictions_frozen.csv",
        "ranking_predictions_frozen.csv",
        "holdout_ranking_predictions_frozen.csv",
        "prediction_freeze_manifest.json",
        "prediction_freeze_conflict.json",
        "candidate_universe_manifest.json",
        "candidate_universe.csv",
        "prediction_freeze_approval.json",
        "prediction_freeze_approval.json.sha256",
        "prediction_freeze_approval.json.sig",
        "prediction_freeze_public_key.pem",
        "prediction_freeze_approval_verification.json",
    ):
        src = analysis_dir / name
        if not src.is_file():
            continue
        dst = Path(suite_dir) / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied[name] = name
    return copied

def _copy_full_model_artifacts(src_model: Path, suite_dir: Path, *, model_id: str) -> Tuple[str, Dict[str, Any], List[str]]:
    """Copy the full/reference model into the suite root and record sidecars.

    The remote benchmark runner executes every case from ``suite/bXXX`` and the
    split manifest references the full model via ``../<model>.onnx``.  v49i
    created that manifest key, but the remote minimal bundle did not include
    the suite-root ONNX file, so full_* runs failed before measuring.  v49j makes
    the suite self-contained and writes a small manifest for debugging.
    """
    errors: List[str] = []
    copied: List[Dict[str, Any]] = []
    src_model = Path(src_model).expanduser()
    suite_dir.mkdir(parents=True, exist_ok=True)
    if not src_model.is_file():
        return "", {
            "schema": "onnx-splitpoint/suite-model-artifacts",
            "schema_version": 1,
            "model_id": model_id,
            "source_model": str(src_model),
            "suite_full_model": "",
            "copied": [],
            "status": "source_model_missing",
            "created_at": now_iso(),
        }, ["source_model_missing"]

    def _copy_one(src: Path, rel: Optional[Path] = None, role: str = "sidecar") -> Optional[str]:
        try:
            src = Path(src).expanduser()
            if not src.is_file():
                return None
            if rel is None:
                rel = Path(src.name)
            # Keep external-data locations relative and safe inside suite root.
            rel = Path(*[part for part in rel.parts if part not in ("", ".", "..")])
            dst = suite_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if str(src.resolve()) != str(dst.resolve()):
                shutil.copy2(src, dst)
            try:
                size = int(dst.stat().st_size)
            except Exception:
                size = 0
            copied.append({"role": role, "source": str(src), "relative_path": rel.as_posix(), "size_bytes": size})
            return rel.as_posix()
        except Exception as exc:
            errors.append(f"copy_{role}_failed:{src}:{type(exc).__name__}:{exc}")
            return None

    full_rel = _copy_one(src_model, Path(src_model.name), role="full_model") or ""

    # Common external-data sidecars used by ONNX exporters.
    sidecar_candidates: List[Tuple[Path, Path]] = []
    for cand in [
        Path(str(src_model) + ".data"),
        src_model.with_suffix(".data"),
        src_model.with_suffix(src_model.suffix + ".data"),
        src_model.parent / (src_model.name + ".data"),
        src_model.parent / (src_model.stem + ".data"),
        src_model.parent / (src_model.name + ".bin"),
        src_model.parent / (src_model.stem + ".bin"),
    ]:
        if cand.is_file():
            sidecar_candidates.append((cand, Path(cand.name)))

    # External-data locations stored in the ONNX protobuf.  Keep the original
    # relative path where possible so ONNX can resolve tensors after extraction.
    try:
        import onnx  # type: ignore
        model = onnx.load(str(src_model), load_external_data=False)
        tensors = list(getattr(model.graph, "initializer", []) or [])
        for graph in list(getattr(model, "functions", []) or []):
            tensors.extend(list(getattr(graph, "initializer", []) or []))
        seen_locations: set[str] = set()
        for tensor in tensors:
            for entry in getattr(tensor, "external_data", []) or []:
                if getattr(entry, "key", "") != "location":
                    continue
                loc = str(getattr(entry, "value", "") or "").strip()
                if not loc or loc in seen_locations:
                    continue
                seen_locations.add(loc)
                cand = (src_model.parent / loc).resolve() if not os.path.isabs(loc) else Path(loc)
                if cand.is_file():
                    sidecar_candidates.append((cand, Path(loc)))
    except Exception as exc:
        # This is diagnostic only; normal single-file ONNX does not need it.
        errors.append(f"external_data_scan_skipped:{type(exc).__name__}:{exc}")

    seen_dst: set[str] = set()
    for cand, rel in sidecar_candidates:
        key = rel.as_posix()
        if key in seen_dst:
            continue
        seen_dst.add(key)
        _copy_one(cand, rel, role="external_data")

    payload = {
        "schema": "onnx-splitpoint/suite-model-artifacts",
        "schema_version": 1,
        "model_id": model_id,
        "source_model": str(src_model),
        "suite_full_model": full_rel,
        "copied": copied,
        "errors": errors,
        "status": "ok" if full_rel else "partial",
        "created_at": now_iso(),
    }
    return full_rel, payload, errors


def _selected_candidates(candidate_plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in list(candidate_plan.get("selected_candidates") or []):
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def _candidate_boundaries(candidate_plan: Mapping[str, Any]) -> List[int]:
    out: List[int] = []
    for item in _selected_candidates(candidate_plan):
        split = _safe_int(item.get("split_index") or item.get("boundary") or item.get("boundary_index"))
        if split is not None and split not in out:
            out.append(int(split))
    return out


def _suite_candidate_ids(path: Path) -> Tuple[List[str], List[int]]:
    payload = _read_json(path, default={}) or {}
    cases = payload.get("cases") if isinstance(payload, Mapping) else []
    ids: List[str] = []
    boundaries: List[int] = []
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, Mapping):
                continue
            cid = str(case.get("case_id") or case.get("case_dir") or case.get("folder") or "").strip()
            if cid:
                ids.append(cid)
            split = _safe_int(case.get("split_index") or case.get("boundary") or case.get("boundary_index"))
            if split is not None:
                boundaries.append(int(split))
    return ids, boundaries


def _relative_or_empty(path: Any, root: Path) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    return relpath(raw, root)


@dataclass
class BoundGeneratorResult:
    artifacts: Dict[str, Path] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    status: str = "ok"
    suite_dir: Optional[Path] = None
    accepted_cases: List[Dict[str, Any]] = field(default_factory=list)
    rejected_cases: List[Dict[str, Any]] = field(default_factory=list)
    generator_overrides: List[Dict[str, Any]] = field(default_factory=list)


def resolve_existing_suite(row: Mapping[str, Any]) -> Optional[Path]:
    """Resolve a profile-supplied or row-supplied benchmark suite directory.

    Accepted keys intentionally cover older profile/campaign vocabulary.
    """

    keys = (
        "benchmark_set_dir",
        "benchmark_suite_dir",
        "suite_dir",
        "generated_suite_dir",
        "benchmarkset_dir",
        "benchmark_set",
    )
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            values = value
        else:
            values = [value]
        for raw in values:
            s = str(raw or "").strip()
            if not s:
                continue
            p = Path(s).expanduser()
            if p.is_file() and p.name == "benchmark_set.json":
                return p.parent.resolve()
            if p.is_dir() and (p / "benchmark_set.json").is_file():
                return p.resolve()
    return None


def import_existing_suite(
    *,
    model_id: str,
    row: Mapping[str, Any],
    model_dir: Path,
    run_dir: Path,
    profile_id: str,
    run_id: str,
    prediction: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
    targets: Sequence[str],
    profile_payload: Optional[Mapping[str, Any]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[BoundGeneratorResult]:
    existing = resolve_existing_suite(row)
    if existing is None:
        return None
    dst = model_dir / "benchmark_set" / "legacy_suite"
    stale_direct = model_dir / "benchmark_set" / "generated_suite"
    if stale_direct.exists() and stale_direct.is_dir():
        shutil.rmtree(stale_direct)
    if dst.exists() and dst.is_dir():
        shutil.rmtree(dst)
    shutil.copytree(existing, dst)

    plan_boundaries = _candidate_boundaries(candidate_plan)
    suite_ids, suite_boundaries = _suite_candidate_ids(dst / "benchmark_set.json")
    plan_set = set(plan_boundaries)
    suite_set = set(suite_boundaries)
    missing_from_suite = sorted(plan_set - suite_set)
    extra_in_suite = sorted(suite_set - plan_set)
    overrides: List[Dict[str, Any]] = []
    if missing_from_suite:
        overrides.append({"kind": "candidate_plan_boundary_missing_from_imported_suite", "boundaries": missing_from_suite})
    if extra_in_suite:
        overrides.append({"kind": "imported_suite_contains_extra_boundaries", "boundaries": extra_in_suite})

    accepted: List[Dict[str, Any]] = []
    for idx, cand in enumerate(_selected_candidates(candidate_plan), start=1):
        split = _safe_int(cand.get("split_index") or cand.get("boundary") or cand.get("boundary_index"))
        cid = _case_id_for(cand, idx)
        status = "accepted_imported_suite" if split in suite_set or cid in suite_ids else "candidate_plan_missing_from_imported_suite"
        accepted.append({**dict(cand), "case_id": cid, "boundary": split, "generation_status": status})

    planned_runs = _logical_profile_or_target_runs(
        profile_payload, targets, accepted, {},
    )
    p_plan_import = write_json(model_dir / "benchmark_set" / "benchmark_plan.json", {
        "schema": "onnx-splitpoint/benchmark-generation-plan",
        "schema_version": 3,
        "model_id": model_id,
        "profile_id": profile_id,
        "run_id": run_id,
        "source": "imported_existing_suite",
        "candidate_plan_is_authoritative": True,
        "cases": accepted,
        "rejected_cases": [],
        "planned_runs": planned_runs,
        "runs": planned_runs,
        "imported_suite_dir": str(existing),
        "legacy_suite_dir": relpath(dst, run_dir),
        "status": "imported_with_overrides" if overrides else "imported_matches_candidate_plan",
        "created_at": now_iso(),
    })

    decisions = {
        "schema": "onnx-splitpoint/benchmark-generation-decisions",
        "schema_version": 3,
        "model_id": model_id,
        "profile_id": profile_id,
        "run_id": run_id,
        "source": "imported_existing_suite",
        "imported_suite_dir": str(existing),
        "legacy_suite_dir": relpath(dst, run_dir),
        "source_prediction_artifact_id": prediction.get("artifact_id", ""),
        "source_candidate_plan_artifact_id": candidate_plan.get("artifact_id", ""),
        "candidate_plan_is_authoritative": True,
        "accepted_cases": accepted,
        "rejected_cases": [],
        "policy_promotions": list(candidate_plan.get("policy_promotions") or []),
        "policy_backfills": list(candidate_plan.get("policy_backfills") or []),
        "generator_overrides": overrides,
        "targets": list(targets),
        "status": "imported_with_overrides" if overrides else "imported_matches_candidate_plan",
        "created_at": now_iso(),
    }
    p_dec = write_json(model_dir / "benchmark_set" / "generation_decisions.json", decisions)
    p_input = write_json(model_dir / "benchmark_set" / "generator_input.json", {
        "schema": "onnx-splitpoint/benchmark-generator-input",
        "schema_version": 2,
        "model_id": model_id,
        "profile_id": profile_id,
        "run_id": run_id,
        "mode": "import_existing_suite",
        "source_candidate_plan_path": "../analysis/final_candidate_plan.json",
        "candidate_plan_boundaries": plan_boundaries,
        "imported_suite_dir": str(existing),
        "legacy_suite_dir": relpath(dst, run_dir),
        "force_candidate_plan_first": True,
        "targets": list(targets),
    })
    p_cases = write_csv(
        model_dir / "benchmark_set" / "benchmark_cases.csv",
        [
            {
                "model_id": model_id,
                "case_id": c.get("case_id", ""),
                "split_index": c.get("split_index", c.get("boundary", "")),
                "prediction_rank": c.get("prediction_rank", c.get("source_rank", c.get("rank", ""))),
                "predicted_total_latency_ms": c.get("predicted_total_latency_ms", ""),
                "predicted_transfer_latency_ms": c.get("predicted_transfer_latency_ms", ""),
                "predicted_hailo_feasible": c.get("predicted_hailo_feasible", ""),
                "generation_status": c.get("generation_status", ""),
            }
            for c in accepted
        ],
        ["model_id", "case_id", "split_index", "prediction_rank", "predicted_total_latency_ms", "predicted_transfer_latency_ms", "predicted_hailo_feasible", "generation_status"],
    )
    p_contract = write_json(model_dir / "benchmark_set" / "benchmark_set.json", {
        "schema": "onnx-splitpoint/benchmark-set-contract",
        "schema_version": 3,
        "model_id": model_id,
        "profile_id": profile_id,
        "cases": accepted,
        "planned_runs": planned_runs,
        "suite_dir": relpath(dst, run_dir),
        "legacy_suite_dir": relpath(dst, run_dir),
        "imported_suite_dir": str(existing),
        "materialized": True,
        "materialization_scope": "imported_existing_suite",
        "source_of_truth": "existing_benchmarkset_pipeline",
    })
    cpu_invariant = finalize_management_cpu_reference_plan_aliases(
        executable_plan_path=dst / "benchmark_plan.json",
        formal_plan_path=p_plan_import,
        profile=profile_payload,
        cache_verify_enabled=bool(cache_verify_guard(profile_payload or {})),
        automatic=not profile_has_explicit_cpu_reference(
            profile_payload, targets,
        ),
        require_existing=False,
        benchmark_set_paths=(dst / "benchmark_set.json", p_contract),
    )
    if callable(log):
        log(f"[workflow] imported existing benchmark suite for {model_id}: {existing}")
    return BoundGeneratorResult(
        artifacts={
            "benchmark_plan_json": p_plan_import,
            "generator_input_json": p_input,
            "generation_decisions_json": p_dec,
            "benchmark_cases_csv": p_cases,
            "benchmark_set_json": p_contract,
            "legacy_suite_benchmark_set_json": dst / "benchmark_set.json",
            "legacy_suite_benchmark_plan_json": dst / "benchmark_plan.json",
        },
        metrics={
            "planned_cases": len(plan_boundaries),
            "accepted_cases": len(accepted),
            "rejected_cases": 0,
            "generator_overrides": len(overrides),
            "suite_dir": relpath(dst, run_dir),
            "binding_mode": "import_existing_suite",
            "management_cpu_reference_invariant": cpu_invariant,
        },
        message="Existing benchmark suite imported and audited against final_candidate_plan.json.",
        status="ok" if not overrides else "partial",
        suite_dir=dst,
        accepted_cases=accepted,
        generator_overrides=overrides,
    )


def _write_suite_readme(path: Path, *, model_id: str, run_id: str, accepted: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]]) -> Path:
    lines = [
        f"# Benchmark suite for `{model_id}`",
        "",
        "Generated by the formal EvaluationWorkflowRunner.",
        "",
        f"Run ID: `{run_id}`",
        f"Accepted cases: {len(accepted)}",
        f"Rejected cases: {len(rejected)}",
        "",
        "This suite was generated from `analysis/prediction.json` and `analysis/final_candidate_plan.json`.",
        "The runner did not invent additional split points silently; changes are recorded in `generation_decisions.json`.",
    ]
    return write_text(path, "\n".join(lines) + "\n")


def _prepare_hailo_full_reuse(
    *,
    suite_dir: Path,
    model_id: str,
    full_baseline_plan: Mapping[str, Any],
    output_contracts: Mapping[str, Any],
    targets: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    decisions: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    contracts = [dict(x) for x in list(output_contracts.get("contracts") or []) if isinstance(x, Mapping)]
    contract_by_backend = {str(c.get("backend") or ""): c for c in contracts}
    for baseline in list(full_baseline_plan.get("baselines") or []):
        if not isinstance(baseline, Mapping):
            continue
        backend = str(baseline.get("backend") or "").strip()
        if "hailo" not in backend.lower():
            decisions.append({
                "backend": backend,
                "variant": "full",
                "decision": "not_hailo_no_hef_required",
                "artifact_path": baseline.get("artifact_path", ""),
            })
            continue
        contract = contract_by_backend.get(backend, {})
        if (
            baseline.get("requested") is False
            or contract.get("requested") is False
        ):
            decisions.append({
                "backend": backend,
                "variant": "full",
                "requested": False,
                "decision": "not_requested_by_profile",
                "source_artifact_path": baseline.get("artifact_path", ""),
                "output_contract": contract,
            })
            continue
        hw = "hailo8" if "8" in backend else backend
        out_full = suite_dir / "hailo" / hw / "full"
        hef_dst = out_full / "compiled.hef"
        copied = _copy_if_file(baseline.get("artifact_path"), hef_dst)
        decision = "hef_reused_prepared_full_baseline" if copied else "hef_build_pending"
        decisions.append({
            "backend": backend,
            "hw_arch": hw,
            "variant": "full",
            "endpoint_mode": baseline.get("endpoint_mode") or contract.get("endpoint_mode") or "decoded",
            "decision": decision,
            "source_artifact_path": baseline.get("artifact_path", ""),
            "suite_artifact_path": str(hef_dst) if copied else "",
            "suite_artifact_rel": str(Path("hailo") / hw / "full" / "compiled.hef") if copied else "",
            "output_contract": contract,
        })
        if copied:
            artifacts.append({
                "kind": "hailo_full_hef",
                "backend": backend,
                "path": str(hef_dst),
                "path_rel": str(Path("hailo") / hw / "full" / "compiled.hef"),
                "endpoint_mode": baseline.get("endpoint_mode") or contract.get("endpoint_mode") or "decoded",
            })
            write_json(out_full / "output_contract.json", contract or {
                "schema": "onnx-splitpoint/output-contract",
                "schema_version": 1,
                "model_id": model_id,
                "backend": backend,
                "variant": "full",
                "endpoint_mode": baseline.get("endpoint_mode") or "decoded",
            })
    return decisions, artifacts




def _canon_target(value: Any) -> str:
    s = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if s in {"cpu", "cpu_ort", "ort_cpu"}:
        return "cpu_ort"
    if s in {"cuda", "cuda_ort", "ort_cuda", "gpu"}:
        return "cuda_ort"
    if "deepx" in s or "dx_m1" in s or "dxm1" in s:
        if "trt" in s or "tensorrt" in s:
            return "deepx_m1_to_tensorrt"
        return "deepx_m1"
    if "trt" in s or "tensorrt" in s:
        return "tensorrt"
    if "hailo10h" in s:
        return "hailo10h"
    if "hailo10" in s:
        return "hailo10"
    if "hailo8r" in s:
        return "hailo8r"
    if "hailo8l" in s:
        return "hailo8l"
    if "hailo8" in s or "hailo" in s:
        return "hailo8"
    return s or "cpu_ort"



def _provider_for_target(target: str) -> str:
    t = str(target or "").strip().lower()
    if t in {"cpu", "cpu_ort", "ort_cpu"}:
        return "cpu"
    if t in {"cuda", "cuda_ort", "ort_cuda"}:
        return "cuda"
    if "trt" in t or "tensorrt" in t:
        return "tensorrt"
    if t.startswith("deepx") or "dx_m1" in t:
        return "deepx_m1" if "to" not in t else t
    if t.startswith("hailo"):
        return t
    return t or "auto"

def _run_type_for_target(target: str) -> str:
    t = str(target or "").lower()
    if t.startswith("hailo"):
        return "hailo"
    if t.startswith("deepx") or "dx_m1" in t:
        return "deepx"
    return "onnxruntime"

def _stage_descriptor_for_target(target: str) -> Dict[str, Any]:
    """Return the structured stage object expected by benchmark_suite.py.

    v49i wrote compact strings (for example ``"hailo8"``) into stage1/stage2.
    The remote benchmark template primarily consumed dictionaries and therefore
    skipped matrix runs as "missing stage1/stage2".  v49j writes the structured
    form and the template remains backward-compatible with compact strings.
    """
    t = _canon_target(target)
    if str(t).startswith("deepx") or "dx_m1" in str(t):
        return {"type": "deepx", "target": "deepx_m1", "artifact_kind": "dxnn"}
    if str(t).startswith("hailo"):
        return {"type": "hailo", "hw_arch": t}
    return {"type": "onnxruntime", "provider": _provider_for_target(t)}


def _stage_token_for_target(target: str) -> str:
    t = _canon_target(target)
    return t if str(t).startswith("hailo") else _provider_for_target(t)


def _planned_runs_for(targets: Sequence[str], cases: Sequence[Mapping[str, Any]], output_contracts: Mapping[str, Any]) -> List[Dict[str, Any]]:
    canonical: List[str] = []
    for raw in targets or ["cpu_ort"]:
        t = _canon_target(raw)
        if t and t not in canonical:
            canonical.append(t)
    if not canonical:
        canonical = ["cpu_ort"]
    contracts = [dict(x) for x in list(output_contracts.get("contracts") or []) if isinstance(x, Mapping)]
    contract_by_backend = {str(c.get("backend") or ""): c for c in contracts}
    runs: List[Dict[str, Any]] = []
    for t in canonical:
        contract = contract_by_backend.get(t) or {}
        rid = f"full_{t}"
        runs.append({
            "id": rid,
            "run_id": rid,
            "type": _run_type_for_target(t),
            "provider": _provider_for_target(t),
            "case_id": "full",
            "backend": t,
            "variant": "full",
            "variants": ["full"],
            "stage1": _stage_descriptor_for_target(t),
            "stage2": _stage_descriptor_for_target(t),
            "stage1_token": _stage_token_for_target(t),
            "stage2_token": _stage_token_for_target(t),
            "hw_arch": t if str(t).startswith("hailo") else "",
            "endpoint_mode": contract.get("endpoint_mode", "decoded"),
        })
    host = "tensorrt" if "tensorrt" in canonical else ("cuda_ort" if "cuda_ort" in canonical else ("cpu_ort" if "cpu_ort" in canonical else canonical[0]))
    split_profiles: List[tuple[str, str]] = []
    for t in canonical:
        if (t.startswith("hailo") or t.startswith("deepx") or "dx_m1" in t) and host != t:
            split_profiles.append((t, host))
    if not split_profiles and len(canonical) >= 2:
        split_profiles.append((canonical[0], canonical[1]))
    if not split_profiles:
        split_profiles.append((canonical[0], canonical[0]))
    for case in cases:
        cid = str(case.get("case_id") or case.get("case_dir") or case.get("folder") or "")
        for left, right in split_profiles:
            backend = left if left == right else f"{left}_to_{right}"
            rid = f"{cid}_{backend}"
            runs.append({
                "id": rid,
                "run_id": rid,
                "type": "matrix" if left != right else _run_type_for_target(left),
                "provider": _provider_for_target(left),
                "case_id": cid,
                "backend": backend,
                "variant": "split",
                "variants": ["part1", "part2", "composed"],
                "stage1": _stage_descriptor_for_target(left),
                "stage2": _stage_descriptor_for_target(right),
                "stage1_token": _stage_token_for_target(left),
                "stage2_token": _stage_token_for_target(right),
                "split_index": case.get("split_index", case.get("boundary", "")),
                "prediction_rank": case.get("prediction_rank", case.get("rank", "")),
                "endpoint_mode": case.get("endpoint_mode", "decoded"),
            })
    return runs


def _logical_profile_or_target_runs(
    profile_payload: Optional[Mapping[str, Any]],
    targets: Sequence[str],
    cases: Sequence[Mapping[str, Any]],
    output_contracts: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Prefer selected logical profiles over target-Cartesian synthesis."""

    profile = dict(profile_payload or {})
    raw_profiles = profile.get("run_profiles")
    if isinstance(raw_profiles, list) and raw_profiles:
        enabled_profiles = enabled_run_profiles(raw_profiles)
        if not enabled_profiles:
            return []
        profile["run_profiles"] = list(enabled_profiles)
        # This is the same projection used by the current formal contract.
        # In particular, four Full profiles remain four Full rows; merely
        # selecting vendor and TensorRT targets does not invent split runs.
        from .benchmark_binding import _benchmark_runs_from_profile

        logical_rows = _benchmark_runs_from_profile(profile, targets)
        executable_rows: List[Dict[str, Any]] = []

        def _profile_target(value: Any) -> str:
            if isinstance(value, Mapping):
                value = (
                    value.get("target") or value.get("backend")
                    or value.get("provider") or value.get("hw_arch")
                    or value.get("type") or ""
                )
            return _canon_target(value) if str(value or "").strip() else ""

        for raw in logical_rows:
            row = dict(raw)
            if row.get("semantic_reference_only") is True:
                executable_rows.append(row)
                continue
            profile_type = str(row.get("type") or "").strip().lower()
            stage1 = _profile_target(row.get("stage1"))
            stage2 = _profile_target(row.get("stage2"))
            full = _profile_target(
                row.get("full") or row.get("full_reference") or ""
            )
            mixed = bool(
                profile_type in {"mixed", "mixed_backend", "matrix", "split"}
                or (stage1 and stage2 and stage1 != stage2)
            )
            row.setdefault("profile_type", profile_type)
            if mixed:
                left = stage1 or full
                right = stage2 or full or left
                row.update({
                    "type": "matrix",
                    "provider": _provider_for_target(left),
                    "backend": f"{left}_to_{right}" if left != right else left,
                    "variant": "split",
                    "variants": ["part1", "part2", "composed"],
                    "stage1": _stage_descriptor_for_target(left),
                    "stage2": _stage_descriptor_for_target(right),
                    "stage1_token": _stage_token_for_target(left),
                    "stage2_token": _stage_token_for_target(right),
                })
            else:
                backend = full or stage1 or stage2
                row.update({
                    "type": _run_type_for_target(backend),
                    "provider": _provider_for_target(backend),
                    "backend": backend,
                    "variant": "full",
                    "variants": ["full"],
                    "stage1": _stage_descriptor_for_target(backend),
                    "stage2": _stage_descriptor_for_target(backend),
                    "stage1_token": _stage_token_for_target(backend),
                    "stage2_token": _stage_token_for_target(backend),
                })
                if str(backend).startswith("hailo"):
                    row["hw_arch"] = backend
            executable_rows.append(row)
        return executable_rows
    return _planned_runs_for(targets, cases, output_contracts)

def materialize_suite_from_candidate_plan(
    *,
    model_id: str,
    model_path: str,
    model_dir: Path,
    run_dir: Path,
    profile_id: str,
    run_id: str,
    prediction: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
    targets: Sequence[str],
    full_baseline_plan: Mapping[str, Any],
    output_contracts: Mapping[str, Any],
    profile_payload: Optional[Mapping[str, Any]] = None,
    model_entry: Optional[Mapping[str, Any]] = None,
    dry_run: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> BoundGeneratorResult:
    """Create a concrete benchmark suite from final_candidate_plan.json.

    For valid ONNX models this exports part1/part2 ONNX files per accepted case.
    If ONNX parsing or splitting is unavailable, the function still writes a
    complete generator contract and structured rejection reasons.
    """

    suite_dir = model_dir / "benchmark_set" / "generated_suite"
    if suite_dir.exists() and suite_dir.is_dir():
        shutil.rmtree(suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)
    profile_payload = dict(profile_payload or {})
    model_entry = dict(model_entry or {"id": model_id})
    scientific_freeze_artifacts = _copy_scientific_freeze_artifacts(model_dir, suite_dir)
    p_model = Path(str(model_path or "")).expanduser()
    selected = _selected_candidates(candidate_plan)
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    split_errors: List[str] = []
    full_model_rel = ""
    suite_model_artifacts: Dict[str, Any] = {}

    full_model_rel, suite_model_artifacts, model_artifact_errors = _copy_full_model_artifacts(p_model, suite_dir, model_id=model_id)
    split_errors.extend([e for e in model_artifact_errors if not str(e).startswith("external_data_scan_skipped")])

    analysis_payload: Optional[Mapping[str, Any]] = None
    if not dry_run and p_model.is_file():
        try:
            from ..core_analysis import analyze_model
            analysis_payload = analyze_model(str(p_model), min_gap=0)
        except Exception as exc:
            split_errors.append(f"analysis_for_split_failed: {type(exc).__name__}: {exc}")
            analysis_payload = None
    elif dry_run:
        split_errors.append("dry_run: split ONNX export skipped")
    else:
        split_errors.append("model file not resolved; split ONNX export skipped")

    model = analysis_payload.get("model") if isinstance(analysis_payload, Mapping) else None
    nodes = analysis_payload.get("nodes") if isinstance(analysis_payload, Mapping) else None
    order = analysis_payload.get("order") if isinstance(analysis_payload, Mapping) else None
    can_split = model is not None and isinstance(nodes, list) and isinstance(order, list)

    for idx, cand in enumerate(selected, start=1):
        candidate = dict(cand)
        split = _safe_int(candidate.get("split_index") or candidate.get("boundary") or candidate.get("boundary_index"))
        cid = _case_id_for(candidate, idx)
        case_dir = suite_dir / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        base_case = {
            **candidate,
            "case_id": cid,
            "case_dir": cid,
            "folder": cid,
            "boundary": split,
            "boundary_index": split,
            "prediction_rank": candidate.get("prediction_rank", candidate.get("source_rank", candidate.get("rank", idx))),
            "source_prediction_artifact_id": prediction.get("artifact_id", ""),
            "source_candidate_plan_artifact_id": candidate_plan.get("artifact_id", ""),
        }
        if not can_split:
            reason = split_errors[-1] if split_errors else "analysis_for_split_unavailable"
            rejected.append({**base_case, "generation_status": "rejected", "reject_reason": reason, "error_class": "parse_failed"})
            continue
        if split is None or split < 0 or split >= len(order) - 1:
            rejected.append({**base_case, "generation_status": "rejected", "reject_reason": "boundary_out_of_range", "error_class": "resource_infeasible"})
            continue
        try:
            from ..split_export import cut_tensors_for_boundary, save_model, split_model_on_cut_tensors, write_runner_skeleton_onnxruntime
            cut_tensors = cut_tensors_for_boundary(order, nodes, int(split))
            if not cut_tensors:
                rejected.append({**base_case, "generation_status": "rejected", "reject_reason": "no_crossing_tensors", "error_class": "resource_infeasible"})
                continue
            part1, part2, manifest = split_model_on_cut_tensors(model, list(cut_tensors), strict_boundary=False)
            part1_path = case_dir / "part1.onnx"
            part2_path = case_dir / "part2.onnx"
            save_model(part1, str(part1_path), external_data=False)
            save_model(part2, str(part2_path), external_data=False)
            full_model_for_case = str(Path("..") / full_model_rel).replace("\\", "/") if full_model_rel else str(p_model).replace("\\", "/")
            # v49i: generated suites must keep the Benchmark runner-compatible
            # manifest keys.  The remote runner expects full_model plus part1/part2
            # (or *_model aliases); earlier v49 direct suites only wrote
            # source_full_model, which made real remote execution fail before any
            # measurement could start.
            manifest_payload = {
                "tool": {"workflow": "v51a-profile-persistence-runtime-gates", "generator": "formal_workflow_direct_generator_binding"},
                "schema": "onnx-splitpoint/split-manifest",
                "schema_version": 2,
                "model_id": model_id,
                "case_id": cid,
                "boundary": split,
                "boundary_index": split,
                "strict_boundary": False,
                "full_model": full_model_for_case,
                "full_model_source": str(p_model).replace("\\", "/"),
                "source_full_model": full_model_for_case,
                "part1": "part1.onnx",
                "part1_model": "part1.onnx",
                "part2": "part2.onnx",
                "part2_model": "part2.onnx",
                "source_prediction_artifact_id": prediction.get("artifact_id", ""),
                "source_candidate_plan_artifact_id": candidate_plan.get("artifact_id", ""),
                "cut_tensors": list(cut_tensors),
                "split_export": manifest,
                "prediction": {k: candidate.get(k) for k in ("predicted_total_latency_ms", "predicted_transfer_latency_ms", "predicted_hailo_feasible", "rank", "source_rank")},
                "models": {
                    "full": {"path": full_model_for_case, "source": str(p_model).replace("\\", "/")},
                    "part1": {"path": "part1.onnx"},
                    "part2": {"path": "part2.onnx"},
                },
                "created_at": now_iso(),
            }
            if isinstance(manifest, Mapping):
                # Preserve low-level split metadata, but never let it remove the
                # compatibility keys above.
                merged = dict(manifest)
                merged.update(manifest_payload)
                manifest_payload = merged
            p_manifest = write_json(case_dir / "split_manifest.json", manifest_payload)
            try:
                write_runner_skeleton_onnxruntime(str(case_dir), manifest_filename="split_manifest.json", target="auto")
            except Exception as runner_exc:
                manifest_payload.setdefault("warnings", []).append(f"runner_skeleton_failed: {type(runner_exc).__name__}: {runner_exc}")
                write_json(p_manifest, manifest_payload)
            accepted.append({
                **base_case,
                "generation_status": "accepted_split_exported",
                "part1_model": str(Path(cid) / "part1.onnx"),
                "part2_model": str(Path(cid) / "part2.onnx"),
                "split_manifest": str(Path(cid) / "split_manifest.json"),
                "cut_tensor_count": len(cut_tensors),
                "compile_ok": None,
                "runtime_ok": None,
                "validation_ok": None,
            })
        except Exception as exc:
            rejected.append({**base_case, "generation_status": "rejected", "reject_reason": f"split_export_failed: {type(exc).__name__}: {exc}", "error_class": "parse_failed"})

    hef_decisions, hef_artifacts = _prepare_hailo_full_reuse(
        suite_dir=suite_dir,
        model_id=model_id,
        full_baseline_plan=full_baseline_plan,
        output_contracts=output_contracts,
        targets=targets,
    )

    p_suite_model_artifacts = write_json(suite_dir / "suite_model_artifacts.json", suite_model_artifacts)
    p_model_artifacts = write_json(model_dir / "benchmark_set" / "suite_model_artifacts.json", suite_model_artifacts)

    planned_runs = _logical_profile_or_target_runs(
        profile_payload, targets, accepted, output_contracts,
    )
    benchmark_plan_payload = {
        "schema": "onnx-splitpoint/benchmark-generation-plan",
        "schema_version": 3,
        "model_id": model_id,
        "profile_id": profile_id,
        "run_id": run_id,
        "created_at": now_iso(),
        "quality_gate": dict(profile_payload.get("quality_gate") or {}) if isinstance(profile_payload.get("quality_gate"), Mapping) else {},
        "ranking_validation": dict(profile_payload.get("ranking_validation") or {}) if isinstance(profile_payload.get("ranking_validation"), Mapping) else {},
        "campaign": dict(profile_payload.get("campaign") or {}) if isinstance(profile_payload.get("campaign"), Mapping) else {},
        "model_suite": {"primary": [model_entry], "reserve": []},
        "prediction_freeze_artifacts": scientific_freeze_artifacts,
        "source_prediction_artifact_id": prediction.get("artifact_id", ""),
        "source_candidate_plan_artifact_id": candidate_plan.get("artifact_id", ""),
        "candidate_plan_is_authoritative": True,
        "model_path": str(p_model),
        "suite_full_model": full_model_rel,
        "suite_model_artifacts": "suite_model_artifacts.json",
        "suite_dir": relpath(suite_dir, run_dir),
        "targets": list(targets),
        "planned_runs": planned_runs,
        "cases": accepted,
        "rejected_cases": rejected,
        "planned_runs": planned_runs,
        "runs": planned_runs,
        "status": "ready_for_runtime_execution" if accepted else "no_materialized_cases",
    }

    bench_payload = {
        "schema": "onnx-splitpoint/benchmark-set",
        "schema_version": 3,
        "model_id": model_id,
        "profile_id": profile_id,
        "run_id": run_id,
        "created_at": now_iso(),
        "source": "formal_workflow_direct_generator_binding",
        "source_prediction_artifact_id": prediction.get("artifact_id", ""),
        "source_candidate_plan_artifact_id": candidate_plan.get("artifact_id", ""),
        "candidate_plan_is_authoritative": True,
        "full_model": full_model_rel or str(p_model),
        "suite_model_artifacts": "suite_model_artifacts.json",
        "targets": list(targets),
        "planned_runs": planned_runs,
        "cases": accepted,
        "rejected_cases": rejected,
        "hailo_full_baseline_decisions": hef_decisions,
        "hailo_full_baseline_artifacts": hef_artifacts,
        "status": "ok" if accepted else "partial",
    }
    p_plan = write_json(model_dir / "benchmark_set" / "benchmark_plan.json", benchmark_plan_payload)
    p_suite_plan = write_json(suite_dir / "benchmark_plan.json", benchmark_plan_payload)
    p_suite_bench = write_json(suite_dir / "benchmark_set.json", bench_payload)
    p_dec = write_json(model_dir / "benchmark_set" / "generation_decisions.json", {
        "schema": "onnx-splitpoint/benchmark-generation-decisions",
        "schema_version": 3,
        "model_id": model_id,
        "profile_id": profile_id,
        "run_id": run_id,
        "source": "formal_workflow_direct_generator_binding",
        "source_prediction_artifact_id": prediction.get("artifact_id", ""),
        "source_candidate_plan_artifact_id": candidate_plan.get("artifact_id", ""),
        "candidate_plan_is_authoritative": True,
        "accepted_cases": accepted,
        "rejected_cases": rejected,
        "policy_promotions": list(candidate_plan.get("policy_promotions") or []),
        "policy_backfills": list(candidate_plan.get("policy_backfills") or []),
        "generator_overrides": [],
        "expected_unsupported": [r for r in rejected if str(r.get("reject_reason") or "").startswith("no_crossing")],
        "hef_decisions": hef_decisions,
        "status": "ok" if accepted else "partial",
        "split_errors": split_errors,
        "created_at": now_iso(),
    })
    p_input = write_json(model_dir / "benchmark_set" / "generator_input.json", {
        "schema": "onnx-splitpoint/benchmark-generator-input",
        "schema_version": 2,
        "model_id": model_id,
        "profile_id": profile_id,
        "run_id": run_id,
        "mode": "formal_workflow_direct_generator_binding",
        "model_path": str(p_model),
        "source_candidate_plan_path": "../analysis/final_candidate_plan.json",
        "candidate_plan_boundaries": _candidate_boundaries(candidate_plan),
        "force_candidate_plan_first": True,
        "targets": list(targets),
        "suite_dir": relpath(suite_dir, run_dir),
    })
    p_contract = write_json(model_dir / "benchmark_set" / "benchmark_set.json", {
        "schema": "onnx-splitpoint/benchmark-set-contract",
        "schema_version": 3,
        "model_id": model_id,
        "profile_id": profile_id,
        "cases": accepted,
        "rejected_cases": rejected,
        "planned_runs": planned_runs,
        "suite_dir": relpath(suite_dir, run_dir),
        "generated_suite_dir": relpath(suite_dir, run_dir),
        "generated_suite_benchmark_set": relpath(p_suite_bench, run_dir),
        "materialized": True,
        "materialization_scope": "direct_split_export" if accepted else "contract_with_structured_rejections",
    })
    cpu_invariant = finalize_management_cpu_reference_plan_aliases(
        executable_plan_path=p_suite_plan,
        formal_plan_path=p_plan,
        profile=profile_payload,
        cache_verify_enabled=bool(cache_verify_guard(profile_payload)),
        automatic=not profile_has_explicit_cpu_reference(
            profile_payload, targets,
        ),
        require_existing=False,
        benchmark_set_paths=(p_suite_bench, p_contract),
    )
    p_cases = write_csv(
        model_dir / "benchmark_set" / "benchmark_cases.csv",
        [
            {
                "model_id": model_id,
                "case_id": c.get("case_id", ""),
                "split_index": c.get("split_index", c.get("boundary", "")),
                "prediction_rank": c.get("prediction_rank", ""),
                "predicted_total_latency_ms": c.get("predicted_total_latency_ms", ""),
                "predicted_transfer_latency_ms": c.get("predicted_transfer_latency_ms", ""),
                "predicted_hailo_feasible": c.get("predicted_hailo_feasible", ""),
                "generation_status": c.get("generation_status", ""),
            }
            for c in accepted
        ],
        ["model_id", "case_id", "split_index", "prediction_rank", "predicted_total_latency_ms", "predicted_transfer_latency_ms", "predicted_hailo_feasible", "generation_status"],
    )
    p_rej = write_csv(
        model_dir / "benchmark_set" / "rejected_cases.csv",
        [
            {
                "model_id": model_id,
                "case_id": c.get("case_id", ""),
                "split_index": c.get("split_index", c.get("boundary", "")),
                "reject_reason": c.get("reject_reason", ""),
                "error_class": c.get("error_class", ""),
            }
            for c in rejected
        ],
        ["model_id", "case_id", "split_index", "reject_reason", "error_class"],
    )
    p_readme = _write_suite_readme(suite_dir / "README.md", model_id=model_id, run_id=run_id, accepted=accepted, rejected=rejected)
    try:
        from ..gui.controller import write_benchmark_suite_script
        p_suite_script = Path(write_benchmark_suite_script(suite_dir, bench_json_name="benchmark_set.json"))
        suite_script_status = "written"
    except Exception as suite_exc:
        p_suite_script = write_text(suite_dir / "benchmark_suite.py", "#!/usr/bin/env python3\nprint('benchmark suite script unavailable')\n")
        suite_script_status = f"fallback_stub: {type(suite_exc).__name__}: {suite_exc}"
    p_backend = write_json(model_dir / "benchmark_set" / "backend_artifacts_manifest.json", {
        "schema": "onnx-splitpoint/backend-artifacts-manifest",
        "schema_version": 1,
        "model_id": model_id,
        "suite_dir": relpath(suite_dir, run_dir),
        "split_artifacts_materialized": bool(accepted),
        "accepted_cases": [
            {
                "case_id": c.get("case_id"),
                "part1_model": _relative_or_empty(suite_dir / str(c.get("part1_model") or ""), run_dir),
                "part2_model": _relative_or_empty(suite_dir / str(c.get("part2_model") or ""), run_dir),
                "split_manifest": _relative_or_empty(suite_dir / str(c.get("split_manifest") or ""), run_dir),
            }
            for c in accepted
        ],
        "hailo_full_baseline_artifacts": hef_artifacts,
        "hailo_full_baseline_decisions": hef_decisions,
        "backend_build_status": "split_artifacts_ready_backend_builds_pending" if accepted else "no_split_artifacts_ready",
        "benchmark_suite_script_status": suite_script_status,
    })

    if callable(log):
        log(f"[workflow] materialized benchmark suite for {model_id}: accepted={len(accepted)} rejected={len(rejected)} dir={suite_dir}")

    scientific_artifact_paths = {
        f"suite_{name.replace('.', '_')}": suite_dir / relative
        for name, relative in scientific_freeze_artifacts.items()
    }
    status = "ok" if accepted else "partial"
    message = "Benchmark suite materialized from final_candidate_plan.json."
    if not accepted:
        message = "Benchmark suite contract written, but no split cases could be materialized in this environment."
    return BoundGeneratorResult(
        artifacts={
            "benchmark_plan_json": p_plan,
            "generated_suite_benchmark_plan_json": p_suite_plan,
            "generator_input_json": p_input,
            "generation_decisions_json": p_dec,
            "benchmark_set_json": p_contract,
            "benchmark_cases_csv": p_cases,
            "rejected_cases_csv": p_rej,
            "generated_suite_benchmark_set_json": p_suite_bench,
            "generated_suite_readme": p_readme,
            "benchmark_suite_py": p_suite_script,
            "backend_artifacts_manifest_json": p_backend,
            "suite_model_artifacts_json": p_model_artifacts,
            "generated_suite_model_artifacts_json": p_suite_model_artifacts,
            **scientific_artifact_paths,
        },
        metrics={
            "planned_cases": len(selected),
            "accepted_cases": len(accepted),
            "rejected_cases": len(rejected),
            "suite_dir": relpath(suite_dir, run_dir),
            "binding_mode": "direct_split_export",
            "benchmark_suite_script_status": suite_script_status,
            "suite_full_model_materialized": bool(full_model_rel),
            "suite_model_artifact_count": len(suite_model_artifacts.get("copied") or []),
            "hef_decisions": len(hef_decisions),
            "hef_reused": sum(1 for d in hef_decisions if str(d.get("decision") or "") == "hef_reused_prepared_full_baseline"),
            "management_cpu_reference_invariant": cpu_invariant,
        },
        message=message,
        status=status,
        suite_dir=suite_dir,
        accepted_cases=accepted,
        rejected_cases=rejected,
    )
