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
import re
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


def _register_imported_generic_hailo_hint(
    suite: Path, case: str, destination: Path, runs: List[Mapping[str, Any]],
) -> None:
    """Project the verified current artifact through ordinary availability."""
    from ..benchmark.services import (
        BenchmarkGenerationService, build_case_hailo_variant_availability,
        case_hailo_backend_terminal_states,
    )
    from .hailo_artifact_scope import hailo_scope_backend

    manifest_path = suite / case / "split_manifest.json"
    manifest = _read_json(manifest_path)
    catalogue = _read_json(suite / "benchmark_set.json")
    if not isinstance(manifest, dict) or not isinstance(catalogue, dict):
        raise ValueError("generic_quality_hailo_import_hint_current_manifest_invalid")
    hailo = manifest.setdefault("hailo", {})
    if not isinstance(hailo, dict):
        raise ValueError("generic_quality_hailo_import_hint_current_manifest_invalid")
    hefs = hailo.setdefault("hefs", {})
    if not isinstance(hefs, dict) or any(not isinstance(meta, dict) for meta in hefs.values()):
        raise ValueError("generic_quality_hailo_import_hint_current_manifest_invalid")
    if any(hailo_scope_backend(arch) == "hailo10"
           and any(str(key).startswith("part1") for key in meta)
           for arch, meta in hefs.items()):
        raise ValueError("generic_quality_hailo_import_hint_current_locator_conflict")
    hefs.setdefault("hailo10h", {}).update(
        part1=relpath(destination, manifest_path.parent),
        part1_original_reuse={
            "receipt": relpath(destination.with_name("hailo_hef_build_receipt.json"), manifest_path.parent),
            "provenance": relpath(destination.with_name("generic_quality_original_hailo_reuse.json"), manifest_path.parent),
        },
    )
    availability = build_case_hailo_variant_availability(
        catalogue.get("hailo", {}).get("hefs", {}), hefs)
    terminal = case_hailo_backend_terminal_states(runs, availability)
    hailo["case_variant_availability"] = availability
    hailo["backend_terminal_states"] = terminal
    projected = {
        "hailo_case_variant_availability": availability,
        "hailo_backend_terminal_states": terminal,
        "hailo_compile": BenchmarkGenerationService().case_hailo_compile_from_hefs(hefs),
    }
    mirrors = []
    for path, key in ((suite / "benchmark_set.json", "cases"),
                      (suite.parent / "benchmark_set.json", "cases"),
                      (suite.parent / "generation_decisions.json", "accepted_cases")):
        if not path.is_file():
            continue
        payload = _read_json(path)
        rows = payload.get(key, []) if isinstance(payload, dict) else []
        matches = [row for index, row in enumerate(rows, 1) if _case_id_for(row, index) == case]
        if len(matches) != 1:
            raise ValueError("generic_quality_hailo_import_hint_current_case_scope_changed")
        matches[0].update(projected)
        mirrors.append((path, payload))
    write_json(manifest_path, manifest)
    for path, payload in mirrors:
        write_json(path, payload)


def _materialize_imported_generic_hailo_hints(
    source: Path, suite: Path, *, model_id: str,
    profile: Mapping[str, Any], runs: List[Mapping[str, Any]],
    skip_requests: set[tuple[str, str, str]] | None = None,
) -> List[Dict[str, Any]]:
    """Expose an original, verified H10 HEF to ordinary import preflight.

    An imported H8 suite has no H10 deferred call to rebind. Only an explicit
    current Generic config can supply that missing artifact; its original
    receipt is copied unchanged, never projected into a successful build call.
    """
    from ..artifact_store import materialize_file
    from ..build_dispatch_policy import require_productive_force_off
    from ..hailo_backend import _hailo_receipt_path, _load_valid_hailo_receipt
    from ..preprocessing_contract import canonical_runtime_dtype, resolve_image_preprocessing_contract
    from ..runners.generic_quality_accounting import authorized_runs
    from .artifacts import sha256_file
    from .generic_quality_sidecars import _contained, _producer_source_path
    from .hailo_artifact_scope import hailo_scope_backend

    def require(ok: bool, reason: str) -> None:
        if not ok:
            raise ValueError("generic_quality_hailo_import_hint_" + reason)

    def matches(path: Path, descriptor: Mapping[str, Any]) -> bool:
        return (not path.is_symlink() and path.is_file()
                and path.stat().st_size == descriptor.get("size_bytes")
                and str(sha256_file(path)).removeprefix("sha256:") == descriptor.get("sha256"))

    records = []
    for run in authorized_runs({"runs": runs}, profile):
        if hailo_scope_backend(run.get("stage1")) != "hailo10":
            continue
        for case in run["case_ids"]:
            if ("hailo10", case, "part1") in (skip_requests or set()):
                continue
            destination = _contained(suite, f"{case}/hailo/hailo10/part1/compiled.hef")
            if destination.exists():
                continue  # Existing artifacts retain their ordinary verifier.
            declared = run["generic_quality_completion_sources"][case]
            origin = _producer_source_path(declared["path"])
            require(matches(origin, declared), "config_changed")
            config = _read_json(origin)
            artifacts = config.get("expected_runtime_artifacts") or {}
            hef = artifacts.get("part1_runtime") or {}
            receipt = artifacts.get("part1_build_receipt") or {}
            if not hef.get("local_path") or not receipt.get("local_path"):
                continue  # No original hint: preserve the normal UNKNOWN result.
            identity = config.get("identity") or {}
            require(config.get("schema") == "onnx-splitpoint/generic-composed-quality-completion-config"
                    and identity.get("model_id") == model_id and identity.get("case_id") == case
                    and identity.get("setup_id") == run["hardware_setup_id"]
                    and identity.get("comparison_backend") == "hailo10h", "identity_changed")
            binding = (config.get("generic_artifact_binding") or {}).get("artifacts") or {}
            require(all(binding.get(role) == artifacts.get(role) for role in
                        ("part1_runtime", "part1_build_receipt", "source_part1_onnx")), "artifact_sections_differ")
            require_productive_force_off(profile)
            build = profile.get("hailo_build") or {}
            require(build.get("mode") in {"reuse_only", "cache_verify_only"}, "requires_restrictive_policy")
            source_graph = artifacts.get("source_part1_onnx") or {}
            graph = _contained(suite, source_graph.get("path", ""))
            require(graph.is_relative_to(suite / case) and matches(graph, source_graph), "source_onnx_changed")
            require((source / case / "split_manifest.json").read_bytes()
                    == (suite / case / "split_manifest.json").read_bytes(), "split_manifest_changed")
            original_hef = _producer_source_path(hef["local_path"])
            original_receipt = _producer_source_path(receipt["local_path"])
            require(original_hef.is_absolute() and original_receipt.is_absolute()
                    and matches(original_hef, hef) and matches(original_receipt, receipt), "original_artifact_changed")
            require(_hailo_receipt_path(original_hef) == original_receipt, "original_receipt_not_adjacent")
            saved = _read_json(original_receipt)
            require(hailo_scope_backend(saved.get("hw_arch")) == "hailo10", "receipt_arch_changed")
            payload = saved.get("cache_payload") or {}
            for key, field, multiplier in (
                ("calib_count", "calibration_count", 1), ("calib_count", "requested_calibration_count", 1),
                ("calib_batch_size", "calibration_batch_size", 1), ("optimization_level", "optimization_level", 1),
                ("calibration_memory_cap_mb", "calibration_memory_cap_bytes", 1024**2),
            ):
                require(type(build.get(key)) is int and build[key] * multiplier == payload.get(field),
                        "compiler_contract_changed:" + key)
            require(build.get("calibration_storage") == payload.get("calibration_storage")
                    and build.get("cache_integrity") == payload.get("integrity"), "calibration_contract_changed")
            manifests = (profile.get("campaign") or {}).get("dataset_manifests") or {}
            calibration = manifests.get("detection") or {}
            manifest = Path(str(calibration.get("calibration") or "")).expanduser()
            require(manifest.is_absolute() and manifest.is_file()
                    and payload.get("calibration_identity") == "manifest:" + str(sha256_file(manifest)).removeprefix("sha256:"),
                    "calibration_manifest_changed")
            hw = ((config.get("completion_contract") or {}).get("processor_contract") or {}).get("input_hw")
            preprocess = config.get("preprocess") or {}
            sealed_preprocess = saved.get("preprocessing_contract") or {}
            require(isinstance(hw, list) and len(hw) == 2 and all(type(n) is int and n > 0 for n in hw),
                    "preprocessing_changed")
            current_preprocess, current_preprocess_sha = resolve_image_preprocessing_contract(
                task="detection", target_hw=hw)
            prepared = ((config.get("representative_input") or {}).get("prepared_input") or {})
            require(sealed_preprocess == current_preprocess
                    and saved.get("preprocessing_contract_sha256") == current_preprocess_sha
                    and run.get("image_scale") == current_preprocess["image_scale"]
                    and preprocess.get("mode") == current_preprocess["preprocess_mode"]
                    and preprocess.get("pad_value") == current_preprocess["pad_value"]
                    and preprocess.get("rgb") is True
                    and canonical_runtime_dtype(prepared.get("dtype")) == "uint8"
                    and prepared.get("shape") == [*hw, 3],
                    "preprocessing_changed")
            verified = _load_valid_hailo_receipt(original_hef,
                preprocessing_sha256=current_preprocess_sha,
                source_onnx_sha256=source_graph["sha256"], cache_key=saved.get("cache_key"),
                cache_payload=payload, expected_net_name=f"{model_id}_part1_b{int(case[1:])}",
                expected_net_input_shapes=[1, 3, *hw], expected_disable_rt_metadata_extraction=True)
            require(verified is not None and verified == saved, "original_receipt_invalid")
            destinations = ((original_hef, destination, hef),
                            (original_receipt, destination.with_name("hailo_hef_build_receipt.json"), receipt))
            for _origin, target, descriptor in destinations:
                require(not target.exists() or matches(target, descriptor), "destination_conflict")
            for old, target, descriptor in destinations:
                if not target.exists():
                    materialize_file(old, target, allow_hardlink=False)
                require(matches(target, descriptor), "materialized_artifact_changed")
            record = {"config": dict(declared), "source_part1_onnx": dict(source_graph),
                "original_HEF": dict(hef), "original_receipt": dict(receipt),
                "current_suite_HEF": str(destination), "current_suite_receipt": str(destinations[1][1]),
                "reuse": "verified_original_bytes_no_compiler_invocation",
                "original_receipt_bytes_preserved": True, "compiler_dispatch_count": 0}
            write_json(destination.with_name("generic_quality_original_hailo_reuse.json"), record)
            _register_imported_generic_hailo_hint(suite, case, destination, runs)
            records.append(record)
    return records


def _rebind_imported_generic_hailo_requests(
    source: Path, suite: Path, *, model_id: str,
    profile: Mapping[str, Any], runs: List[Mapping[str, Any]],
) -> None:
    """Keep historical Hailo calls and stage current cache-only import calls.

    No saved success is carried into the new request. The existing finalizer
    still performs the ordinary receipt-aware cache lookup before execution.
    """
    from ..build_dispatch_policy import profile_hailo_compute_mapping, require_productive_force_off
    from ..hailo_compiler_context import normalize_compute_by_family
    from ..runners.generic_quality_accounting import authorized_runs
    from .artifacts import sha256_file
    from .deferred_hailo_builds import (
        REQUEST_NAME, _generic_deferred_artifact_scope, _generic_hailo_request_identity,
    )
    from .hailo_artifact_scope import hailo_scope_backend

    def require(condition: bool, reason: str) -> None:
        if not condition:
            raise ValueError("generic_quality_hailo_import_" + reason)

    def digest(value: Any) -> str:
        return str(value or "").removeprefix("sha256:")

    def descriptor(path: Path) -> Dict[str, Any]:
        return {"path": str(path.resolve()), "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size}

    catalogue = _read_json(suite / "benchmark_set.json")
    accepted = {str(row.get("case_dir") or row.get("folder") or "")
                for row in catalogue.get("cases", [])}
    scope = _generic_deferred_artifact_scope(suite, profile, accepted)
    requests = [path for path in sorted(source.rglob(REQUEST_NAME))
                if scope is not None and _generic_hailo_request_identity(source, path) in scope]
    _materialize_imported_generic_hailo_hints(source, suite, model_id=model_id,
        profile=profile, runs=runs,
        skip_requests={_generic_hailo_request_identity(source, path) for path in requests})
    if not requests:
        return
    require_productive_force_off(profile)
    build = profile.get("hailo_build") or {}
    require(build.get("mode") in {"reuse_only", "cache_verify_only"}, "requires_restrictive_build_policy")
    authorized = authorized_runs({"runs": runs}, profile)
    for original in requests:
        backend, case, stage = _generic_hailo_request_identity(source, original)
        require(stage in {"part1", "part2"}, "full_request_not_admitted")
        relative, target = original.relative_to(source), suite / original.relative_to(source)
        require(original.read_bytes() == target.read_bytes(), "request_changed_during_copy")
        request = _read_json(original)
        require(isinstance(request, Mapping)
                and request.get("schema") == "onnx-splitpoint/deferred-hailo-build/v1"
                and isinstance(request.get("kwargs"), Mapping), "request_invalid")
        args = dict(request["kwargs"])
        require_productive_force_off({}, hailo_force_build=args.get("force", False))
        require(hailo_scope_backend(args.get("hw_arch")) == backend, "backend_changed")
        old_output = Path(str(args.get("outdir") or "")).expanduser()
        old_source = Path(str(request.get("source_onnx") or "")).expanduser()
        require(old_output.is_absolute() and old_source.is_absolute()
                and old_output.parts[-4:] == relative.parts[:-1], "original_output_scope_invalid")
        before, after = source / case / "split_manifest.json", suite / case / "split_manifest.json"
        require(before.read_bytes() == after.read_bytes(), "split_manifest_changed")
        manifest = _read_json(after)
        part = str(manifest.get(stage + "_model") or manifest.get(stage) or manifest.get(stage + "_path") or "")
        require(part and not Path(part).is_absolute()
                and manifest.get("boundary", manifest.get("boundary_index")) == int(case[1:]), "source_boundary_changed")
        source_relative = Path(case) / part
        require(".." not in source_relative.parts
                and old_source == old_output.parents[3] / source_relative, "original_source_scope_invalid")
        actual_source, imported_source = source / source_relative, suite / source_relative
        expected_hash = digest(request.get("source_onnx_sha256"))
        require(re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None
                and actual_source.is_file() and imported_source.is_file()
                and digest(sha256_file(actual_source)) == expected_hash
                and digest(sha256_file(imported_source)) == expected_hash, "source_onnx_changed:" + case)
        context = args.get("build_evidence_context") or {}
        require(context.get("model_id") == model_id and context.get("stage") == stage
                and _safe_int(context.get("boundary")) == int(case[1:]), "request_identity_changed")
        saved_manifest = context.get("split_manifest") or {}
        for key in ("part1", "part2", "cut_tensors", "cut_tensors_full",
                    "part1_cut_names", "part2_cut_names", "orig_inputs", "orig_outputs"):
            require(key not in saved_manifest or saved_manifest[key] == manifest.get(key), "cut_contract_changed:" + key)
        for current, saved in (("calib_count", "calib_count"), ("calib_batch_size", "calib_batch_size"),
                               ("optimization_level", "opt_level")):
            require(current not in build or build[current] == args.get(saved), "compiler_contract_changed:" + current)
        require(normalize_compute_by_family(args.get("compute_by_family"))
                == profile_hailo_compute_mapping(profile), "compute_profile_changed")
        bindings = []
        for run in authorized:
            if case not in run["case_ids"] or hailo_scope_backend(run.get("stage" + stage[-1])) != backend:
                continue
            declared = run["generic_quality_completion_sources"][case]
            config_path = Path(declared["path"])
            require(config_path.is_file() and config_path.stat().st_size == declared["size_bytes"]
                    and digest(sha256_file(config_path)) == digest(declared["sha256"]), "config_changed:" + case)
            config = _read_json(config_path)
            identity = config.get("identity") or {}
            require(identity.get("model_id") == model_id and identity.get("case_id") == case
                    and identity.get("setup_id") == run["hardware_setup_id"], "config_identity_changed")
            artifact = (config.get("expected_runtime_artifacts") or {}).get(stage + "_runtime") or {}
            source_hef, imported_hef = original.parent / "compiled.hef", target.parent / "compiled.hef"
            require(artifact.get("sha256") and source_hef.is_file() and imported_hef.is_file()
                    and source_hef.stat().st_size == imported_hef.stat().st_size == artifact.get("size_bytes")
                    and digest(sha256_file(source_hef)) == digest(artifact["sha256"])
                    and digest(sha256_file(imported_hef)) == digest(artifact["sha256"]), "runtime_artifact_changed:" + case)
            bindings.append({"config": descriptor(config_path), "source_runtime_artifact": descriptor(source_hef),
                             "imported_runtime_artifact": descriptor(imported_hef)})
        require(bool(bindings), "current_config_binding_missing:" + case)
        history = target.with_name(target.stem + "." + digest(sha256_file(original)) + ".json")
        require(not history.exists() or history.read_bytes() == original.read_bytes(), "history_conflict")
        if not history.exists():
            history.write_bytes(original.read_bytes())
        args.update(outdir=str(target.parent.resolve()), cache_only=True, force=False)
        write_json(target, {"schema": request["schema"], "status": "pending",
            "source_onnx": str(imported_source.resolve()), "source_onnx_sha256": request["source_onnx_sha256"],
            "kwargs": args, "environment": dict(request.get("environment") or {}),
            "import_source": {"request": descriptor(original), "history": descriptor(history),
                "suite_dir": str(source.resolve()), "original_out_dir": str(old_output),
                "split_manifest": descriptor(before), "runtime_bindings": bindings,
                "profile_payload": dict(profile), "bench_plan_runs": list(runs)}})


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


def _bind_imported_generic_validation_budget(
    runs: List[Dict[str, Any]], *, profile: Mapping[str, Any],
    row: Mapping[str, Any], model_id: str,
) -> List[Dict[str, Any]]:
    management = (profile.get("quality_gate") or {}).get("management_reference") or {}
    if not isinstance(management, Mapping) or "existing_reference" not in management:
        return runs
    from ..runners.generic_quality_accounting import authorized_runs
    from .legacy_benchmarkset_binding import _validation_defaults

    selected = authorized_runs({"runs": runs}, profile)
    if (not selected or any(run.get("id") not in {item["id"] for item in selected}
                           and run.get("semantic_reference_only") is not True for run in runs)):
        raise ValueError("generic_quality_suite_import_validation_scope_invalid")
    defaults = _validation_defaults(profile, row, model_id)
    budget = defaults.get("validation_items_requested")
    limits = (profile.get("validation_execution") or {}).get("max_items") or {}
    reference = management.get("existing_reference") or {}
    dataset = ((reference.get("quality_contract") or {}).get("dataset") or {}) if isinstance(reference, Mapping) else {}
    if (defaults.get("validation_budget_authoritative") is not True
            or type(budget) is not int or budget <= 0
            or limits.get(defaults["benchmark_task"]) != budget
            or defaults.get("validation_max_images") != budget
            or type(dataset.get("image_count")) is not int or dataset["image_count"] != budget):
        raise ValueError("generic_quality_suite_import_validation_budget_invalid")
    fields = ("validation_images", "validation_max_images", "validation_manifest",
              "validation_source_kind", "image_scale", "letterbox", "preprocess_mode",
              "validation_reference_mode", "mini_coco_ap50", "mini_classification_eval")
    template = {key: selected[0][key] for key in fields if key in selected[0]}
    if (not str(template.get("validation_images") or "").strip()
            or type(template.get("validation_max_images")) is not int
            or template["validation_max_images"] != budget
            or any({key: run[key] for key in fields if key in run} != template for run in selected)):
        raise ValueError("generic_quality_suite_import_validation_source_ambiguous")
    result = [dict(run) for run in runs]
    cpu = [run for run in result if run.get("semantic_reference_only") is True]
    if len(cpu) != 1:
        raise ValueError("generic_quality_suite_import_cpu_recipe_ambiguous")
    if any(key in cpu[0] and cpu[0][key] != value for key, value in template.items()):
        raise ValueError("generic_quality_suite_import_cpu_validation_changed")
    cpu[0].update(template)
    for run in result:
        run.setdefault("benchmark_task", defaults["benchmark_task"])
        run["validation_items_requested"] = budget
        run["validation_budget_authoritative"] = True
    return result


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
    profile = dict(profile_payload or {})
    declared_runs = list(enabled_run_profiles(profile.get("run_profiles") or []))
    generic_import = any(
        "generic_quality_completion_configs" in run for run in declared_runs
    )
    existing = resolve_existing_suite(row)
    if existing is None:
        if generic_import and any(row.get(key) for key in (
            "benchmark_set_dir", "benchmark_suite_dir", "suite_dir",
            "generated_suite_dir", "benchmarkset_dir", "benchmark_set",
        )):
            raise ValueError("generic_quality_suite_import_source_missing")
        return None
    if generic_import:
        # Import the complete scientific catalogue, then project only the
        # current explicitly authorized runtime work. Never turn a source-only
        # import into partial acceptance or silently keep an old run's scope.
        if (run_dir.name != run_id
                or candidate_plan.get("model_id", model_id) != model_id):
            raise ValueError("generic_quality_suite_import_current_run_binding_invalid")
        source_set = _read_json(existing / "benchmark_set.json")
        if (not isinstance(source_set, Mapping)
                or source_set.get("model_id", model_id) != model_id):
            raise ValueError("generic_quality_suite_import_source_model_invalid")

        def identities(cases):
            if not isinstance(cases, list) or not cases:
                raise ValueError("generic_quality_suite_import_catalogue_invalid")
            result = []
            for index, case in enumerate(cases, 1):
                if not isinstance(case, Mapping):
                    raise ValueError("generic_quality_suite_import_catalogue_invalid")
                boundary = _safe_int(case.get("split_index") or case.get("boundary")
                                     or case.get("boundary_index"))
                identity = (_case_id_for(case, index), boundary)
                if (boundary is None or identity in result
                        or re.fullmatch(r"b[0-9]+", identity[0]) is None):
                    raise ValueError("generic_quality_suite_import_catalogue_invalid")
                result.append(identity)
            return result

        source_cases = identities(source_set.get("cases"))
        current_cases = identities(candidate_plan.get("selected_candidates"))
        if source_cases != current_cases:
            raise ValueError("generic_quality_suite_import_scientific_scope_changed")
        catalogue = {case for case, _boundary in source_cases}
        for run in declared_runs:
            if run.get("semantic_reference_only") is True:
                continue
            cases = run.get("case_ids")
            configs = run.get("generic_quality_completion_configs")
            hashes = run.get("generic_quality_completion_config_sha256")
            sources = run.get("generic_quality_completion_sources")
            if (run.get("quality_source_only") is not True
                    or run.get("variants") != ["composed"]
                    or not isinstance(cases, list) or not cases
                    or len(cases) != len(set(cases))
                    or not set(cases) <= catalogue
                    or not all(isinstance(value, Mapping)
                               and set(value) == set(cases)
                               for value in (configs, hashes, sources))):
                raise ValueError("generic_quality_suite_import_runtime_scope_invalid")
        from ..runners.generic_quality_accounting import authorized_runs
        authorized = authorized_runs({"runs": declared_runs}, profile)
        if not authorized or any(not str(run.get("id") or "").strip() for run in authorized):
            raise ValueError("generic_quality_suite_import_authority_unresolved")

    local_binary_excludes = ()
    if generic_import:
        from ..benchmark.generic_quality_transport import selected_source_import_excludes
        local_binary_excludes = selected_source_import_excludes(
            existing, runs=declared_runs, model_id=model_id, log=log)

    def ignore_transport_archive(directory, names):
        # The known old transport output is regenerated normally. Omit only
        # additionally proved unselected regular binaries; retain all case
        # directories, scientific metadata and selected/shared dependencies.
        archive = Path(directory) / "suite_bundle.tar.gz"
        ignored = [archive.name] if (
            Path(directory) == existing / "dist" and archive.name in names
            and archive.is_file() and not archive.is_symlink()
        ) else []
        if local_binary_excludes:
            import fnmatch
            for name in names:
                member = Path(directory) / name
                relative = member.relative_to(existing).as_posix()
                if (member.is_file() and not member.is_symlink()
                        and any(fnmatch.fnmatch(relative, pattern)
                                for pattern in local_binary_excludes)):
                    ignored.append(name)
        return ignored

    dst = model_dir / "benchmark_set" / "legacy_suite"
    stale_direct = model_dir / "benchmark_set" / "generated_suite"
    if stale_direct.exists() and stale_direct.is_dir():
        shutil.rmtree(stale_direct)
    if dst.exists() and dst.is_dir():
        shutil.rmtree(dst)
    shutil.copytree(existing, dst, ignore=ignore_transport_archive if generic_import else None)
    if generic_import:
        # Reuse exported models/artifacts, but emit the ordinary current
        # runtime in this new target. Old suites can predate the exact Generic
        # completion protocol declared by the current source-bound configs.
        from ..gui.controller import _copy_runner_lib, write_benchmark_suite_script
        from ..split_export_runners import write_runner_skeleton_onnxruntime

        _copy_runner_lib(dst)
        for case_id in sorted({case for run in declared_runs
                               for case in run.get("case_ids", [])}):
            write_runner_skeleton_onnxruntime(str(dst / case_id))
        write_benchmark_suite_script(dst, bench_json_name="benchmark_set.json")

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
    if generic_import:
        planned_runs = _bind_imported_generic_validation_budget(
            planned_runs, profile=profile, row=row, model_id=model_id,
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
        "local_runtime_binary_excludes": list(local_binary_excludes),
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
        "local_runtime_binary_excludes": list(local_binary_excludes),
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
    if generic_import:
        # The executable plan must not retain the imported suite's old run ID,
        # setup matrix, CPU-reference target or Generic completion bindings.
        # Retain all original case artifacts; ordinary sidecar/source/reference
        # admission below runtime preparation still validates their exact bytes.
        write_json(dst / "benchmark_plan.json", _read_json(p_plan_import))
        suite_contract = dict(source_set)
        suite_contract.update(model_id=model_id, profile_id=profile_id,
                              run_id=run_id, planned_runs=planned_runs)
        if "runs" in suite_contract:
            suite_contract["runs"] = planned_runs
        if isinstance(suite_contract.get("plan"), Mapping):
            suite_contract["plan"] = _read_json(p_plan_import)
        write_json(dst / "benchmark_set.json", suite_contract)
        from .deferred_hailo_builds import rebind_imported_deepx_part1_request
        rebind_imported_deepx_part1_request(
            existing, dst, profile_payload=profile, bench_plan_runs=planned_runs,
        )
        _rebind_imported_generic_hailo_requests(
            existing, dst, model_id=model_id, profile=profile, runs=planned_runs,
        )
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
        return {"type": "deepx", "target": "deepx_m1", "backend": "deepx_m1", "artifact_kind": "dxnn"}
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
                if (raw.get("quality_source_only") is True
                        and "generic_quality_completion_configs" in raw):
                    row["variants"] = list(raw.get("variants") or [])
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
