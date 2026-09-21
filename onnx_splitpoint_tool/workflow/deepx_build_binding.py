from __future__ import annotations

"""DeepX DXNN artifact binding for the formal Evaluation Workflow.

MVP scope:
* build/reuse full-model DXNN artifacts from ONNX via dxcom;
* copy ready DXNNs into the authoritative BenchmarkSet suite;
* write structured status/queue/contracts so eval runs fail visibly when a
  selected deepx_m1 target cannot be built or benchmarked.

DeepX split part1/part2 support remains staged: part1 can be added later using
existing split ONNX exports; part2 needs feature-tensor calibration.
"""

import json
import hashlib
import shutil
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .artifacts import now_iso, read_json, relpath, write_json
from ..config_values import parse_config_bool, validate_profile_config_booleans
from .hardware_matrix import normalize_hardware_targets
from .dataset_binding import calibration_manifest_for_task, manifest_dataset_root
from ..campaign import verify_dataset_manifest
from ..deepx.artifacts import (
    cache_dxnn_artifact,
    deepx_cache_key,
    deepx_cached_artifact_compatible,
    deepx_cached_artifact_identity_compatible,
    strict_artifact_sha256,
)
from ..deepx.compiler import compile_dxnn
from ..deepx.config import (
    CLASSIFICATION_PREPROCESSING_CURRENT,
    CLASSIFICATION_PREPROCESSING_IMAGENET,
    canonical_classification_preprocessing,
    resolve_profile_classification_preprocessing,
    classification_profile_admission,
    deepx_classification_preprocessing_contract,
    image_model_dxcom_config,
    materialize_imagenet_normalized_build_onnx,
    write_dxcom_config,
)
from ..deepx.env_status import compiler_cuda_preflight_message, inspect_deepx_environment, profile_compiler_configuration


def _has_deepx_target(targets: list[str], profile_payload: Mapping[str, Any]) -> bool:
    if any("deepx" in str(t).lower() or "dx_m1" in str(t).lower() or "dxm1" in str(t).lower() for t in targets):
        return True
    try:
        for hw in normalize_hardware_targets(profile_payload):
            if str(hw.get("accelerator") or "") == "deepx_m1" and bool(hw.get("enabled", True)):
                return True
    except Exception:
        pass
    return False


def _as_int(value: Any, default: int) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(str(value)))
    except Exception:
        return int(default)


def _as_shape(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        out=[]
        for x in value:
            try: out.append(int(x))
            except Exception: pass
        if len(out) >= 4:
            return out[:4]
    s=str(value or "").strip().replace("x", ",").replace("X", ",")
    if s:
        out=[]
        for part in s.replace(" ", ",").split(","):
            if part.strip():
                try: out.append(int(part.strip()))
                except Exception: pass
        if len(out) >= 4:
            return out[:4]
    return [1,3,640,640]


def _mirror_deepx_cache_receipt(
    receipt_path: str | Path,
    *,
    local_dir: str | Path,
) -> Path:
    """Copy an external cache receipt into claim-local workflow evidence.

    Backend caches are intentionally outside an evaluation run.  Their receipt
    can be used to admit a cache hit, but registering that external path in the
    run's artifact index breaks the run-local terminal-closure contract.  Keep
    the verified bytes unchanged and register only this local mirror.
    """

    source = Path(receipt_path).expanduser()
    if not source.is_file() or source.is_symlink():
        raise RuntimeError("deepx_cache_receipt_source_invalid")
    destination_dir = Path(local_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "deepx_cache_receipt.json"
    source_sha256 = strict_artifact_sha256(source)
    source_size = source.stat().st_size
    if source.resolve(strict=True) != destination.resolve(strict=False):
        shutil.copy2(source, destination)
    if (
        not destination.is_file()
        or destination.is_symlink()
        or destination.stat().st_size != source_size
        or strict_artifact_sha256(destination) != source_sha256
    ):
        raise RuntimeError("deepx_cache_receipt_mirror_identity_mismatch")
    return destination


def _explicit_classification_v2_contract_requested(
    *, task: str, cfg: Mapping[str, Any],
) -> bool:
    """Keep the diagnostic cache migration classification-only and opt-in."""
    return bool(
        str(task or "").strip().lower() == "classification"
        and "classification_preprocessing" in cfg
    )


def task_bound_deepx_runtime_input_contract(
    *,
    task: str,
    input_name: str,
    source_shape: list[int],
    classification_preprocessing: str = CLASSIFICATION_PREPROCESSING_CURRENT,
) -> dict[str, Any]:
    """Describe the raw HWC image accepted by a task-bound DXNN runtime."""
    task_name = str(task or "").strip().lower()
    if task_name not in {"classification", "detection"}:
        raise ValueError("deepx runtime input contract requires an explicit task")
    shape = _as_shape(source_shape)
    if int(shape[1]) in (1, 3, 4):
        runtime_h, runtime_w = int(shape[2]), int(shape[3])
    else:
        runtime_h, runtime_w = int(shape[1]), int(shape[2])
    preprocess_mode = "resize" if task_name == "classification" else "letterbox"
    effective_pad = 0 if preprocess_mode == "resize" else 114
    classification_mode = (
        canonical_classification_preprocessing(classification_preprocessing)
        if task_name == "classification" else "not_applicable"
    )
    return {
        "name": str(input_name or "images"),
        "shape": [runtime_h, runtime_w, 3],
        "dtype": "uint8",
        "layout": "HWC",
        "color_space": "RGB",
        "task": task_name,
        "preprocess_mode_requested": "auto",
        "preprocess_mode": preprocess_mode,
        "preprocess_mode_effective": preprocess_mode,
        "letterbox_pad_value_requested": 114,
        "letterbox_pad_value_effective": effective_pad,
        "letterbox_pad_value": effective_pad,
        "normalization": "embedded_dxcom_preprocessing",
        "classification_preprocessing": classification_mode,
        "embedded_numeric_path": (
            "dxcom_div255_then_build_onnx_imagenet_mean_std"
            if classification_mode == CLASSIFICATION_PREPROCESSING_IMAGENET
            else "dxcom_div255_only"
            if task_name == "classification"
            else "dxcom_div255_detection_default"
        ),
        "contract_source": "task_bound_dxcom_image_loader_preprocessing",
    }


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_if_regular_file(value: Any) -> str:
    try:
        path = Path(str(value or "")).expanduser()
        return strict_artifact_sha256(path) if path.is_file() else ""
    except Exception:
        return ""


def _deepx_compiler_identity(
    *, cfg: Mapping[str, Any], environment_status: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a stable DX-COM identity before any cache lookup or build."""
    override = cfg.get("compiler_identity") or cfg.get("dxcom_identity")
    if isinstance(override, Mapping) and override:
        evidence: dict[str, Any] = {
            "source": "profile_override",
            "value": json.loads(json.dumps(dict(override), sort_keys=True, default=str)),
        }
    elif str(override or "").strip():
        evidence = {
            "source": "profile_override",
            "value": str(override).strip(),
        }
    else:
        dx_com_probe = next((
            dict(record)
            for record in list(environment_status.get("compiler_imports") or [])
            if isinstance(record, Mapping)
            and str(record.get("module") or "") == "dx_com"
            and bool(record.get("ok"))
        ), {})
        compiler_cli = str(environment_status.get("compiler_cli") or "").strip()
        module_file = str(dx_com_probe.get("module_file") or "").strip()
        evidence = {
            "source": "active_environment",
            "python_tag": str(
                environment_status.get("compiler_python_tag") or ""
            ).strip(),
            "dx_com_distribution": str(
                dx_com_probe.get("distribution") or ""
            ).strip(),
            "dx_com_version": str(
                dx_com_probe.get("package_version") or ""
            ).strip(),
            "dx_com_module_file_name": Path(module_file).name if module_file else "",
            "dx_com_module_file_sha256": str(
                dx_com_probe.get("module_file_sha256")
                or _sha256_if_regular_file(module_file)
            ).strip().lower(),
            "compiler_cli_name": Path(compiler_cli).name if compiler_cli else "",
            "compiler_cli_sha256": _sha256_if_regular_file(compiler_cli),
        }
    strong_identity = bool(
        evidence.get("value")
        or evidence.get("dx_com_version")
        or evidence.get("dx_com_module_file_sha256")
        or evidence.get("compiler_cli_sha256")
    )
    payload = {
        "schema": "onnx-splitpoint/deepx-compiler-identity",
        "schema_version": 1,
        "status": "resolved" if strong_identity else "unresolved",
        "evidence": evidence,
    }
    payload["identity_sha256"] = _canonical_json_sha256(payload)
    return payload


def _deepx_calibration_manifest_contract(
    *,
    profile_payload: Mapping[str, Any],
    task: str,
    calibration_dir: str,
    effective_count: int,
) -> dict[str, Any]:
    """Bind Full-DXNN quantization to manifest bytes and item identities."""
    manifest_ref = calibration_manifest_for_task(profile_payload, task)
    path = Path(str(manifest_ref or "")).expanduser()
    base: dict[str, Any] = {
        "schema": "onnx-splitpoint/deepx-calibration-manifest-contract",
        "schema_version": 1,
        "task": str(task),
        "effective_count": int(effective_count),
    }
    if not path.is_file():
        return {**base, "status": "missing", "reason": "calibration_manifest_missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            **base,
            "status": "invalid",
            "reason": f"calibration_manifest_unreadable:{type(exc).__name__}",
        }
    if not isinstance(payload, Mapping):
        return {**base, "status": "invalid", "reason": "calibration_manifest_not_object"}
    manifest_task = str(payload.get("task") or "").strip().lower()
    manifest_role = str(payload.get("role") or "").strip().lower()
    hash_mode = str(payload.get("hash_mode") or "").strip().lower()
    try:
        item_count = int(payload.get("item_count") or len(payload.get("items") or []))
    except Exception:
        item_count = -1
    items_identity = str(payload.get("items_identity_sha256") or "").strip().lower()
    resolved_root = manifest_dataset_root(path)
    try:
        requested_root = str(Path(calibration_dir).expanduser().resolve())
    except Exception:
        requested_root = str(Path(calibration_dir).expanduser())
    errors: list[str] = []
    if manifest_task != str(task):
        errors.append("task_mismatch")
    if manifest_role != "calibration":
        errors.append("role_mismatch")
    if hash_mode != "content":
        errors.append("hash_mode_not_content")
    if not items_identity.startswith("sha256:") or len(items_identity) != 71:
        errors.append("items_identity_sha256_missing")
    if item_count != int(effective_count) or int(effective_count) <= 0:
        # DX-COM receives a directory plus calibration_num, not a manifest
        # item list.  Exact cardinality is therefore required: a larger tree
        # would leave the compiler's selected subset unspecified.
        errors.append("effective_count_not_exact")
    if not resolved_root:
        errors.append("manifest_dataset_root_unavailable")
    elif requested_root != str(Path(resolved_root).resolve()):
        errors.append("calibration_dir_manifest_root_mismatch")
    verification: dict[str, Any] = {}
    try:
        raw_verification = verify_dataset_manifest(
            payload,
            verify_files=True,
            verification_mode="full",
        )
        # Do not bind the verifier's created_at timestamp into a cache key.
        verification = {
            key: raw_verification.get(key)
            for key in (
                "ok",
                "schema_ok",
                "payload_hash_ok",
                "identity_hash_ok",
                "item_count_ok",
                "verification_mode",
                "manifest_item_count",
                "checked_item_count",
                "missing_count",
                "mismatch_count",
            )
        }
        if not bool(raw_verification.get("ok")):
            errors.append("manifest_full_verification_failed")
    except Exception as exc:
        verification = {
            "ok": False,
            "verification_mode": "full",
            "error": f"{type(exc).__name__}:{exc}",
        }
        errors.append("manifest_full_verification_failed")

    # Prove that the directory scanned by DX-COM contains exactly the files in
    # the verified manifest.  Non-image sidecars are irrelevant because the
    # generated loader limits its extensions; image symlinks and path escapes
    # are rejected so the receipt binds ordinary files under one root.
    allowed_extensions = {"jpg", "jpeg", "png"}
    manifest_records: list[dict[str, Any]] = []
    manifest_paths: list[str] = []
    root_inventory_paths: list[str] = []
    root_path = Path(resolved_root).resolve() if resolved_root else None
    for item in list(payload.get("items") or []):
        if not isinstance(item, Mapping):
            errors.append("manifest_item_not_object")
            continue
        relative = str(item.get("relative_path") or "").strip().replace("\\", "/")
        rel_path = Path(relative)
        if (
            not relative
            or rel_path.is_absolute()
            or ".." in rel_path.parts
            or rel_path.suffix.lstrip(".").lower() not in allowed_extensions
        ):
            errors.append("manifest_item_path_invalid")
            continue
        expected_sha = str(item.get("sha256") or "").strip().lower()
        if expected_sha.startswith("sha256:"):
            expected_sha = expected_sha.split(":", 1)[1]
        try:
            expected_size = int(item.get("size_bytes"))
        except Exception:
            expected_size = -1
        if (
            len(expected_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha)
            or expected_size <= 0
        ):
            errors.append("manifest_item_identity_incomplete")
            continue
        manifest_paths.append(relative)
        manifest_records.append({
            "relative_path": relative,
            "size_bytes": expected_size,
            "sha256": expected_sha,
        })
        if root_path is not None:
            candidate = root_path / rel_path
            try:
                if candidate.is_symlink() or not candidate.is_file():
                    errors.append("manifest_item_not_regular_file")
                elif root_path not in candidate.resolve().parents:
                    errors.append("manifest_item_path_escape")
            except Exception:
                errors.append("manifest_item_not_regular_file")
    if len(set(manifest_paths)) != len(manifest_paths):
        errors.append("manifest_relative_paths_not_unique")
    if root_path is not None:
        try:
            for candidate in root_path.rglob("*"):
                if (
                    candidate.is_file()
                    and candidate.suffix.lstrip(".").lower() in allowed_extensions
                ):
                    if candidate.is_symlink():
                        errors.append("calibration_inventory_contains_symlink")
                    root_inventory_paths.append(
                        candidate.relative_to(root_path).as_posix()
                    )
        except Exception:
            errors.append("calibration_inventory_unreadable")
    root_inventory_paths.sort()
    manifest_paths.sort()
    manifest_records.sort(key=lambda row: str(row["relative_path"]))
    if root_inventory_paths != manifest_paths:
        errors.append("calibration_inventory_manifest_mismatch")

    if errors:
        return {
            **base,
            "status": "invalid",
            "reason": ",".join(dict.fromkeys(errors)),
            "manifest_file_sha256": strict_artifact_sha256(path),
            "manifest_verification": verification,
            "root_inventory_count": len(root_inventory_paths),
            "root_inventory_sha256": _canonical_json_sha256(
                root_inventory_paths
            ),
        }
    contract = {
        **base,
        "status": "resolved",
        "dataset_id": str(payload.get("dataset_id") or ""),
        "split": str(payload.get("split") or ""),
        "role": manifest_role,
        "hash_mode": hash_mode,
        "item_count": item_count,
        "items_identity_sha256": items_identity,
        "manifest_payload_sha256": str(
            payload.get("manifest_payload_sha256") or ""
        ).strip().lower(),
        "manifest_file_sha256": strict_artifact_sha256(path),
        "manifest_file_name": path.name,
        "dataset_root_name": Path(resolved_root).name,
        "manifest_verification": verification,
        "root_inventory_count": len(root_inventory_paths),
        "root_inventory_sha256": _canonical_json_sha256(
            manifest_records
        ),
        "dataset_registry_binding_sha256": str(
            (
                profile_payload.get("campaign")
                if isinstance(profile_payload.get("campaign"), Mapping)
                else {}
            ).get("dataset_registry_binding_sha256")
            or ""
        ).strip().lower(),
    }
    contract["identity_sha256"] = _canonical_json_sha256(contract)
    return contract


def _deepx_full_preprocessing_contract(
    *, task: str, classification_mode: str, runtime_input: Mapping[str, Any],
) -> dict[str, Any]:
    if str(task) == "classification":
        numeric = deepx_classification_preprocessing_contract(classification_mode)
    else:
        numeric = {
            "schema": "onnx-splitpoint/deepx-detection-preprocessing",
            "schema_version": 1,
            "mode": "task_default_letterbox_scale_only",
            "dxcom_loader": {
                "scale_divisor": 255.0,
                "output_layout": "NCHW",
                "output_dtype": "float32",
                "color_space": "RGB",
                "letterbox_pad_value": 114,
                "operations_contract": "letterbox_div255_bgr2rgb_transpose_expanddim",
            },
            "build_onnx_adapter": {"kind": "identity"},
        }
    payload = {
        "schema": "onnx-splitpoint/deepx-full-preprocessing-contract",
        "schema_version": 1,
        "task": str(task),
        "runtime_input": dict(runtime_input),
        "numeric_preprocessing": numeric,
    }
    payload["contract_sha256"] = _canonical_json_sha256(payload)
    return payload


def build_full_deepx_cache_contract(
    *,
    task: str,
    classification_mode: str,
    source_onnx_path: str | Path,
    build_onnx_path: str | Path,
    config_path: str | Path,
    preprocessing_contract: Mapping[str, Any],
    calibration_contract: Mapping[str, Any],
    compiler_identity: Mapping[str, Any],
    calibration_method: str,
    calibration_count: int,
    opt_level: int,
) -> dict[str, Any]:
    """Create the exact v2 Full-DeepX cache/receipt contract."""
    if str(calibration_contract.get("status") or "") != "resolved":
        raise ValueError("DeepX Full cache requires a resolved calibration manifest")
    if str(compiler_identity.get("status") or "") != "resolved":
        raise ValueError("DeepX Full cache requires a resolved compiler identity")
    source_path = Path(source_onnx_path).expanduser()
    build_path = Path(build_onnx_path).expanduser()
    cfg_path = Path(config_path).expanduser()
    payload = {
        "schema": "onnx-splitpoint/deepx-full-cache-contract",
        "schema_version": 2,
        "target": "deepx_m1",
        "variant": "full",
        "task": str(task),
        "classification_preprocessing": (
            str(classification_mode)
            if str(task) == "classification" else "not_applicable"
        ),
        "source_onnx_sha256": strict_artifact_sha256(source_path),
        "build_onnx_sha256": strict_artifact_sha256(build_path),
        "dxcom_config_sha256": strict_artifact_sha256(cfg_path),
        "preprocessing_contract": dict(preprocessing_contract),
        "calibration_manifest_contract": dict(calibration_contract),
        "compiler_identity": dict(compiler_identity),
        "build_options": {
            "calibration_method": str(calibration_method),
            "calibration_count": int(calibration_count),
            "opt_level": int(opt_level),
        },
    }
    payload["contract_sha256"] = _canonical_json_sha256(payload)
    return payload


def recorded_deepx_full_endpoint_attestation(
    *, model_dir: Path, suite_dir: Path, model_id: str,
) -> dict[str, Any]:
    """Resolve the authoritative recorded DeepX Full endpoint contract.

    ``output_contracts.json`` is the workflow's endpoint decision.  The DXNN
    contract adds runtime input details, but must not independently invent a
    contradictory host decoder/NMS tail.  All available workflow copies are
    cross-checked and only one internally consistent semantic signature is
    propagated into the per-artifact contract.
    """
    paths = [
        suite_dir / "output_contracts.json",
        model_dir / "benchmark_set" / "output_contracts.json",
        model_dir / "full_baselines" / "output_contracts.json",
    ]
    sources: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen_paths or not path.is_file():
            continue
        seen_paths.add(key)
        payload = read_json(path, default={}) or {}
        source = {
            "path": str(path),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema": str(payload.get("schema") or "") if isinstance(payload, Mapping) else "",
        }
        sources.append(source)
        contracts = list(payload.get("contracts") or []) if isinstance(payload, Mapping) else []
        for raw in contracts:
            if not isinstance(raw, Mapping):
                continue
            backend = str(raw.get("backend") or "").strip().lower().replace("-", "_")
            variant = str(raw.get("variant") or "full").strip().lower()
            declared_model = str(raw.get("model_id") or payload.get("model_id") or "").strip()
            if backend not in {"deepx", "deepx_m1", "dx_m1", "dxm1"} or variant != "full":
                continue
            if declared_model and declared_model != str(model_id):
                continue
            matches.append(dict(raw))
    if not matches:
        return {
            "status": "unavailable",
            "pass": False,
            "reason": "recorded_deepx_full_endpoint_contract_missing",
            "sources": sources,
        }

    normalized: list[dict[str, Any]] = []
    for contract in matches:
        status = str(contract.get("contract_status") or "").strip().lower()
        endpoint_mode = str(contract.get("endpoint_mode") or "").strip().lower()
        host_tail = contract.get("host_tail_required")
        postprocessing = contract.get("postprocessing_required")
        valid = bool(
            status == "recorded"
            and endpoint_mode in {"decoded", "decoded_pre_nms", "raw_detection_head"}
            and isinstance(host_tail, bool)
            and isinstance(postprocessing, bool)
            and (
                (endpoint_mode == "decoded" and not host_tail and not postprocessing)
                or (endpoint_mode in {"decoded_pre_nms", "raw_detection_head"} and host_tail and postprocessing)
            )
        )
        if not valid:
            return {
                "status": "conflict",
                "pass": False,
                "reason": "recorded_deepx_full_endpoint_contract_invalid",
                "sources": sources,
                "invalid_contract": contract,
            }
        normalized.append({
            "endpoint_mode": endpoint_mode,
            "host_tail_required": host_tail,
            "postprocessing_required": postprocessing,
        })
    signatures = {
        (item["endpoint_mode"], item["host_tail_required"], item["postprocessing_required"])
        for item in normalized
    }
    if len(signatures) != 1:
        return {
            "status": "conflict",
            "pass": False,
            "reason": "recorded_deepx_full_endpoint_contract_copies_disagree",
            "sources": sources,
            "signatures": [list(item) for item in sorted(signatures)],
        }
    endpoint_mode, host_tail, postprocessing = next(iter(signatures))
    authoritative = dict(matches[0])
    from ..native_output_endpoint import bn6_candidate_selection
    candidate_selection = bn6_candidate_selection(authoritative)
    authoritative_sha256 = _canonical_json_sha256(authoritative)
    return {
        "schema": "onnx-splitpoint/deepx-endpoint-semantic-attestation",
        "schema_version": 1,
        "status": "attested",
        "pass": True,
        "reason": "authoritative_recorded_output_contract_bound",
        "model_id": str(model_id),
        "backend": "deepx_m1",
        "variant": "full",
        "endpoint_mode": endpoint_mode,
        "host_tail_required": bool(host_tail),
        "postprocessing_required": bool(postprocessing),
        "source_endpoint_semantics": (
            "fixed_topk_xyxy_score_class_candidates" if candidate_selection else
            "decoded_final_output" if endpoint_mode == "decoded"
            else endpoint_mode
        ),
        "source_endpoint_has_integrated_nms": bool(endpoint_mode == "decoded" and not candidate_selection),
        "authoritative_contract": authoritative,
        "authoritative_contract_sha256": authoritative_sha256,
        "sources": sources,
    }


def _onnx_first_input_info(path: Path) -> tuple[str, list[int]]:
    """Return first real ONNX input name and shape.

    Torch/Ultralytics exports used by the project often name the classifier input
    `images`; some older DeepX code hard-coded `input`, which makes DX-COM fail
    with ConfigInputError.  We inspect the ONNX graph and prefer the actual model
    input whenever available.
    """
    try:
        import onnx  # type: ignore
        m = onnx.load(str(path))
        initializers = {str(x.name) for x in getattr(m.graph, 'initializer', [])}
        for vi in list(getattr(m.graph, 'input', []) or []):
            name = str(getattr(vi, 'name', '') or '')
            if not name or name in initializers:
                continue
            dims=[]
            try:
                for d in vi.type.tensor_type.shape.dim:
                    val = int(d.dim_value) if int(getattr(d, 'dim_value', 0) or 0) else 0
                    dims.append(val)
            except Exception:
                dims=[]
            if len(dims) >= 4:
                return name, [int(x) if int(x or 0)>0 else 1 for x in dims[:4]]
            return name, []
    except Exception:
        pass
    return "", []


def _find_cached_dxnn_legacy(cache_root: Path, key: str) -> Path | None:
    for name in ("model.dxnn",):
        p = cache_root.expanduser() / key / name
        if p.is_file():
            return p
    return None


def _lookup_full_deepx_cache_candidate(
    *, cache_root: Path, cache_key: str, use_v2: bool, force_build: bool,
) -> Path | None:
    """Lookup policy for Full DXNNs with no force/ArtifactStore ambiguity."""
    if not str(cache_key or "") or parse_config_bool(force_build, field="deepx_build.force_build"):
        return None
    if use_v2:
        # v2 validity requires a colocated receipt.  Do not call the wrapped
        # finder, which may materialize only model.dxnn from ArtifactStore.
        return _find_cached_dxnn_legacy(cache_root, cache_key)
    return _find_cached_dxnn(cache_root, cache_key)


def _looks_like_image_dir(path: Path) -> bool:
    try:
        if not path.is_dir():
            return False
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        # Avoid expensive recursion over large datasets; a few files are enough.
        count = 0
        for p in path.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                count += 1
                if count >= 1:
                    return True
        return False
    except Exception:
        return False


def _task_hint(row: Mapping[str, Any], model_id: str) -> str:
    explicit = str(row.get("task") or "").strip().lower()
    if explicit in {"classification", "detection"}:
        return explicit
    raw = " ".join(str(x or "") for x in (
        row.get("family"), row.get("id"), row.get("model_id"), model_id, row.get("onnx"), row.get("resolved_path")
    )).lower()
    if any(tok in raw for tok in ("yolo", "detect", "coco")):
        return "detection"
    if any(tok in raw for tok in ("resnet", "mobilenet", "regnet", "efficientnet", "convnext", "densenet", "classification", "imagenet", "imagenette")):
        return "classification"
    return ""


def _preset_images_dir(name: str) -> Optional[Path]:
    raw = str(name or "").strip().lower().replace("-", "_")
    root = Path(os.environ.get("ONNX_SPLITPOINT_TOOL_VALIDATION_DATASETS") or (Path.home() / ".onnx_splitpoint_tool" / "validation_datasets")).expanduser()
    candidates = []
    if raw in {"imagenette_val_mini_500", "imagenette500", "imagenet_val_mini_500", "imagenet500", "classification_calib", ""}:
        candidates += [root / "classification" / "imagenette_val_mini_500" / "images", root / "classification" / "imagenette_val_mini_500"]
    if raw in {"imagenette_val_mini_200", "imagenette200", "imagenet_val_mini_200", "imagenet200"}:
        candidates += [root / "classification" / "imagenette_val_mini_200" / "images", root / "classification" / "imagenette_val_mini_200"]
    if raw in {"coco_200", "coco200", "coco_200_data", "detection_calib", ""}:
        candidates += [root / "detection" / "coco_200_data"]
    if raw in {"coco_50", "coco50", "coco_50_data"}:
        candidates += [root / "detection" / "coco_50_data"]
    # If caller passed a path, accept it too.
    if name:
        candidates.insert(0, Path(os.path.expanduser(str(name))))
    for cand in candidates:
        if _looks_like_image_dir(cand):
            return cand.resolve()
    return None


def _infer_deepx_calibration_dir(*, row: Mapping[str, Any], model_id: str, cfg: Mapping[str, Any], profile_payload: Mapping[str, Any]) -> str:
    task = _task_hint(row, model_id)

    # Task-specific explicit values are unambiguous and therefore win.  A single
    # legacy ``deepx_build.calib_dir`` is not safe for a heterogeneous
    # ResNet+YOLO campaign: the v60p hardware log showed the ResNet full DXNN
    # resolving to the COCO calibration tree.
    if task == "detection":
        task_specific = (cfg.get("detection_calib_dir"), cfg.get("detection_calibration_dir"))
    else:
        task_specific = (cfg.get("classification_calib_dir"), cfg.get("classification_calibration_dir"))
    for val in task_specific:
        if str(val or "").strip():
            p = _preset_images_dir(str(val))
            return str(p or Path(os.path.expanduser(str(val))).resolve())

    # Registered task-specific calibration manifests are authoritative for
    # Smoke/Standard/Final and must take precedence over a generic legacy path.
    manifest = calibration_manifest_for_task(profile_payload, task)
    if manifest:
        root = manifest_dataset_root(manifest)
        if root:
            p = _preset_images_dir(root)
            return str(p or Path(root).resolve())

    # Backwards-compatible generic override for profiles without a bound
    # task-specific registry.
    for val in (cfg.get("calib_dir"), cfg.get("calibration_dir"), profile_payload.get("calib_dir")):
        if str(val or "").strip():
            p = _preset_images_dir(str(val))
            return str(p or Path(os.path.expanduser(str(val))).resolve())

    # Profile/tool defaults if present; otherwise use the Tool Config defaults.
    tc = profile_payload.get("tool_config") if isinstance(profile_payload.get("tool_config"), Mapping) else {}
    val_cfg = profile_payload.get("validation") if isinstance(profile_payload.get("validation"), Mapping) else {}
    if task == "detection":
        preset = str(cfg.get("detection_calib_preset") or tc.get("detection_calib") or val_cfg.get("detection_calib") or "coco_200")
    else:
        preset = str(cfg.get("classification_calib_preset") or tc.get("classification_calib") or val_cfg.get("classification_calib") or "imagenette_val_mini_500")
    p = _preset_images_dir(preset)
    return str(p) if p else ""


def _legacy_suite_dir(model_dir: Path, benchmark_set_contract: Mapping[str, Any]) -> Path:
    raw = str(benchmark_set_contract.get("legacy_suite_dir") or benchmark_set_contract.get("suite_dir") or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = model_dir.parent.parent / p
        if p.is_dir():
            return p
    return model_dir / "benchmark_set" / "legacy_suite"


def materialize_deepx_build_binding(
    *,
    run_dir: str | Path,
    model_id: str,
    model_path: str,
    row: Mapping[str, Any],
    profile_payload: Mapping[str, Any],
    targets: list[str],
    benchmark_set_contract: Mapping[str, Any],
    log: Optional[Callable[[str], None]] = None,
    process_registry: Any = None,
    cancel_event: Any = None,
) -> Dict[str, Any]:
    validate_profile_config_booleans(profile_payload)
    run_root = Path(run_dir)
    model_dir = run_root / "models" / str(model_id)
    bdir = model_dir / "benchmark_set"
    out_dir = bdir / "deepx"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Path] = {}
    metrics: Dict[str, Any] = {}
    cfg = profile_compiler_configuration(profile_payload) if isinstance(profile_payload, Mapping) else {}
    status_payload: Dict[str, Any] = {
        "schema": "onnx-splitpoint/deepx-artifact-status",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "backend": "deepx_m1",
        "selected": bool(_has_deepx_target(targets, profile_payload)),
        "mode": str(cfg.get("mode") or "reuse_only"),
        "artifacts": [],
        "queue": [],
        "contracts": [],
        "warnings": [],
    }

    def _record_cache_lookup(
        outcome: str,
        *,
        reason: str,
        cache_key: str,
        candidate: Optional[Path] = None,
    ) -> None:
        """Expose the existing persistent-cache decision without changing it."""

        row = {
            "schema": "onnx-splitpoint/deepx-cache-lookup",
            "schema_version": 1,
            "outcome": str(outcome).upper(),
            "role": "full",
            "model_id": str(model_id),
            "identity": str(cache_key or ""),
            "reason": str(reason or "unknown"),
            "artifact": str(candidate) if candidate is not None else "",
        }
        status_payload["cache_lookup"] = row
        if callable(log):
            try:
                line = (
                    f"[deepx-cache] {row['outcome']} role=full "
                    f"model={model_id} identity={row['identity'] or '<unresolved>'} "
                    f"reason={row['reason']}"
                )
                if row["artifact"]:
                    line += f" artifact={row['artifact']}"
                log(line)
            except Exception:
                pass
    if not status_payload["selected"]:
        status_payload["status"] = "not_selected"
        p = write_json(out_dir / "deepx_artifact_status.json", status_payload)
        return {"artifacts": {"deepx_artifact_status_json": p}, "metrics": {"deepx_selected": False}, "status": "ok", "message": "DeepX not selected."}

    mode = str(cfg.get("mode") or "reuse_only").strip().lower().replace("-", "_")
    env_status = inspect_deepx_environment(
        # cache_verify_only may inspect paths/receipts, but must not start or
        # import the DX-COM compiler environment merely to report a miss.
        probe_import=False, path_only=True,
        config=cfg,
        process_registry=process_registry,
        cancel_event=cancel_event,
    )
    p_env = write_json(out_dir / "deepx_environment_status.json", env_status)
    artifacts["deepx_environment_status_json"] = p_env
    metrics.update({"deepx_selected": True, "deepx_compiler_ready": bool(env_status.get("compiler_ready")), "deepx_runtime_ready": bool(env_status.get("runtime_ready"))})
    status_payload["environment_status"] = env_status

    if mode in {"disabled", "off", "none"}:
        status_payload.update({"status":"disabled", "message":"deepx_build.mode disabled"})
        p = write_json(out_dir / "deepx_artifact_status.json", status_payload)
        return {"artifacts": {**artifacts, "deepx_artifact_status_json": p}, "metrics": metrics, "status": "skipped", "message": "DeepX build disabled."}

    task_name = _task_hint(row, model_id)
    if task_name not in {"classification", "detection"}:
        status_payload.update({
            "status": "task_contract_missing",
            "message": "DeepX build requires an explicit or unambiguously inferred classification/detection task.",
        })
        p = write_json(out_dir / "deepx_artifact_status.json", status_payload)
        return {
            "artifacts": {**artifacts, "deepx_artifact_status_json": p},
            "metrics": metrics,
            "status": "failed",
            "message": status_payload["message"],
        }
    status_payload["task"] = task_name
    if task_name == "classification":
        admission = classification_profile_admission(profile_payload, {**dict(row), "task": task_name})
        status_payload["classification_admission"] = admission
        metrics.update(admission)
        cfg["classification_preprocessing"] = admission["classification_preprocessing"]
        if not admission["allowed"]:
            status_payload.update(status="deepx_legacy_classification_preprocessing", message=admission["required_setting"])
            p = write_json(out_dir / "deepx_artifact_status.json", status_payload)
            return {"artifacts": {**artifacts, "deepx_artifact_status_json": p}, "metrics": {**metrics, **admission}, "status": "failed", "message": admission["reason"] + "; " + admission["required_setting"]}
    onnx_path = Path(str(model_path or "")).expanduser()
    classification_mode_explicit = _explicit_classification_v2_contract_requested(
        task=task_name, cfg=cfg,
    )
    try:
        classification_mode = (
            canonical_classification_preprocessing(
                cfg.get("classification_preprocessing")
            )
            if task_name == "classification"
            else CLASSIFICATION_PREPROCESSING_CURRENT
        )
    except ValueError as exc:
        status_payload.update({
            "status": "classification_preprocessing_invalid",
            "message": str(exc),
        })
        p = write_json(out_dir / "deepx_artifact_status.json", status_payload)
        return {
            "artifacts": {**artifacts, "deepx_artifact_status_json": p},
            "metrics": metrics,
            "status": "failed",
            "message": status_payload["message"],
        }
    status_payload["classification_preprocessing"] = (
        classification_mode if task_name == "classification" else "not_applicable"
    )
    source_identity_error = ""
    try:
        status_payload["source_onnx_sha256"] = (
            strict_artifact_sha256(onnx_path) if onnx_path.is_file() else ""
        )
    except Exception as exc:
        status_payload["source_onnx_sha256"] = ""
        source_identity_error = f"{type(exc).__name__}: {exc}"
        status_payload["warnings"].append(
            "could not hash DeepX source ONNX strictly: " + source_identity_error
        )
    # Existing profiles retain the historical cache/build behaviour.  The
    # stronger v2 receipt is activated only by an explicit classification A/B
    # arm so ordinary Standard and detection/YOLO runs are not forced into a
    # cold rebuild by this diagnostic feature.
    use_full_v2_contract = bool(classification_mode_explicit)
    status_payload["full_cache_contract_mode"] = (
        "v2_exact_explicit" if use_full_v2_contract else "legacy_implicit"
    )
    calib_dir = _infer_deepx_calibration_dir(row=row, model_id=model_id, cfg=cfg, profile_payload=profile_payload).strip()
    if calib_dir and callable(log):
        try:
            log(f"[deepx:{model_id}] calibration dir resolved: {calib_dir}")
        except Exception:
            pass
    cache_root = Path(str(cfg.get("cache_dir") or env_status.get("cache_dir") or "~/Models/BackendArtifacts/deepx")).expanduser()
    _onnx_input_name, _onnx_input_shape = _onnx_first_input_info(onnx_path) if onnx_path.is_file() else ("", [])
    shape = _as_shape(row.get("input_shape") or cfg.get("input_shape") or _onnx_input_shape)
    if _onnx_input_shape and (not row.get("input_shape") and not cfg.get("input_shape")):
        shape = _as_shape(_onnx_input_shape)
    input_name = str(cfg.get("input_name") or _onnx_input_name or "images")
    try:
        if callable(log):
            log(f"[deepx:{model_id}] model input resolved: {input_name} shape={shape}")
    except Exception:
        pass
    calibration_count = _as_int(
        cfg.get("calib_count") or cfg.get("calibration_num"), 500,
    )
    calibration_method = str(cfg.get("calibration_method") or "ema")
    opt_level = _as_int(cfg.get("opt_level"), 0)
    cache_variant = (
        f"full_{classification_mode}" if use_full_v2_contract else "full"
    )
    # Explicit A/B arms get independent workspaces and a v2 key namespace.
    # Legacy classification and all detection runs keep their historical Full
    # path and key, avoiding an unrelated cold-build migration.
    config_dir = (
        out_dir / "full" / classification_mode
        if use_full_v2_contract else out_dir / "full"
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = config_dir / "config_deepx.json"
    build_status = "pending"
    dxnn_ready = None
    cache_key = ""
    cache_contract: dict[str, Any] = {}
    cache_receipt: dict[str, Any] = {}
    build_onnx_path = onnx_path
    build_onnx_receipt: dict[str, Any] = {}
    preparation_error = ""
    compiler_identity = (
        _deepx_compiler_identity(cfg=cfg, environment_status=env_status)
        if use_full_v2_contract
        else {"status": "not_applicable", "reason": "legacy_implicit"}
    )
    calibration_contract = (
        _deepx_calibration_manifest_contract(
            profile_payload=profile_payload,
            task=task_name,
            calibration_dir=calib_dir,
            effective_count=calibration_count,
        )
        if use_full_v2_contract
        else {"status": "not_applicable", "reason": "legacy_implicit"}
    )
    runtime_input_contract = task_bound_deepx_runtime_input_contract(
        task=task_name,
        input_name=input_name,
        source_shape=shape,
        classification_preprocessing=classification_mode,
    )
    preprocessing_contract = _deepx_full_preprocessing_contract(
        task=task_name,
        classification_mode=classification_mode,
        runtime_input=runtime_input_contract,
    )
    status_payload["compiler_identity"] = compiler_identity
    status_payload["calibration_manifest_contract"] = calibration_contract
    status_payload["preprocessing_contract"] = preprocessing_contract
    if onnx_path.is_file() and not source_identity_error:
        try:
            if (
                use_full_v2_contract
                and classification_mode == CLASSIFICATION_PREPROCESSING_IMAGENET
            ):
                build_onnx_path, build_onnx_receipt = (
                    materialize_imagenet_normalized_build_onnx(
                        source_onnx=onnx_path,
                        output_dir=config_dir / "build_onnx",
                        input_name=input_name,
                    )
                )
                adapter_receipt_path = write_json(
                    config_dir / "build_onnx_adapter_receipt.json",
                    build_onnx_receipt,
                )
                artifacts["deepx_build_onnx_adapter"] = build_onnx_path
                artifacts["deepx_build_onnx_adapter_receipt_json"] = (
                    adapter_receipt_path
                )
            else:
                build_onnx_receipt = {
                    "schema": "onnx-splitpoint/deepx-build-onnx-adapter",
                    "schema_version": 1,
                    "kind": "identity",
                    "source_onnx_sha256": strict_artifact_sha256(onnx_path),
                    "build_onnx_sha256": strict_artifact_sha256(onnx_path),
                }
            if calib_dir:
                dx_cfg = image_model_dxcom_config(
                    task=task_name,
                    input_name=input_name,
                    input_shape=shape,
                    calibration_dir=calib_dir,
                    calibration_num=calibration_count,
                    calibration_method=calibration_method,
                    classification_preprocessing=classification_mode,
                )
            else:
                dx_cfg = {"inputs": {input_name: shape}, "calibration_num": 0, "calibration_method": calibration_method, "task": task_name, "note": "No calibration_dir configured; build may fail for quantized DeepX compile."}
            write_dxcom_config(dx_cfg, cfg_path)
            if use_full_v2_contract:
                cache_contract = build_full_deepx_cache_contract(
                    task=task_name,
                    classification_mode=classification_mode,
                    source_onnx_path=onnx_path,
                    build_onnx_path=build_onnx_path,
                    config_path=cfg_path,
                    preprocessing_contract=preprocessing_contract,
                    calibration_contract=calibration_contract,
                    compiler_identity=compiler_identity,
                    calibration_method=calibration_method,
                    calibration_count=calibration_count,
                    opt_level=opt_level,
                )
                cache_contract_path = write_json(
                    config_dir / "full_cache_contract.json", cache_contract,
                )
                artifacts["deepx_full_cache_contract_json"] = cache_contract_path
                status_payload["cache_contract"] = cache_contract
                cache_key = deepx_cache_key(
                    onnx_path=build_onnx_path,
                    config_path=cfg_path,
                    target="deepx_m1",
                    variant=cache_variant,
                    cache_contract=cache_contract,
                )
            else:
                cache_key = deepx_cache_key(
                    onnx_path=onnx_path,
                    config_path=cfg_path,
                    target="deepx_m1",
                    variant="full",
                )
        except Exception as exc:
            preparation_error = f"{type(exc).__name__}: {exc}"
            status_payload["warnings"].append(f"could not prepare DeepX config/cache key: {type(exc).__name__}: {exc}")
    elif source_identity_error:
        preparation_error = "source ONNX strict identity unavailable: " + source_identity_error
        status_payload["warnings"].append(preparation_error)
    else:
        preparation_error = f"model ONNX not found: {onnx_path}"
        status_payload["warnings"].append(f"model ONNX not found: {onnx_path}")

    # A forced arm is a real compiler dispatch, never a hidden cache lookup.
    # Content-addressed publication below still refuses to overwrite different
    # bytes at the same v2 key.
    force_build_requested = parse_config_bool(cfg.get("force_build", False), field="deepx_build.force_build")
    candidate = _lookup_full_deepx_cache_candidate(
        cache_root=cache_root,
        cache_key=cache_key,
        use_v2=use_full_v2_contract,
        force_build=force_build_requested,
    )
    if candidate:
        if use_full_v2_contract:
            compatible, cache_reason, cache_receipt = (
                deepx_cached_artifact_compatible(
                    cache_dir=Path(candidate).parent,
                    expected_contract=cache_contract,
                    require_artifact_identity=True,
                )
            )
        else:
            compatible, cache_reason, cache_receipt = (
                deepx_cached_artifact_identity_compatible(
                    cache_dir=Path(candidate).parent,
                )
            )
        if compatible:
            dxnn_ready = candidate
            _record_cache_lookup(
                "HIT",
                reason=cache_reason,
                cache_key=cache_key,
                candidate=Path(candidate),
            )
            receipt_path = Path(candidate).parent / "build_manifest.json"
            if receipt_path.is_file():
                artifacts["deepx_cache_receipt_json"] = (
                    _mirror_deepx_cache_receipt(
                        receipt_path,
                        local_dir=config_dir,
                    )
                )
        else:
            _record_cache_lookup(
                "MISS",
                reason=cache_reason,
                cache_key=cache_key,
                candidate=Path(candidate),
            )
            status_payload["cache_rejection"] = {
                "cache_key": cache_key,
                "reason": cache_reason,
                "candidate": str(candidate),
            }
            status_payload["warnings"].append(
                f"DeepX cache candidate rejected: {cache_reason}"
            )
    else:
        miss_reason = (
            "force_build_requested"
            if force_build_requested
            else ("identity_unresolved" if not cache_key else "not_found")
        )
        _record_cache_lookup(
            "MISS",
            reason=miss_reason,
            cache_key=cache_key,
        )
    if dxnn_ready and not parse_config_bool(cfg.get("force_build", False), field="deepx_build.force_build"):
        build_status = "ready_reused"
        status_payload["artifacts"].append({"variant":"full", "backend":"deepx_m1", "status":build_status, "cache_key":cache_key, "cache_contract_sha256":cache_contract.get("contract_sha256"), "dxnn_path":str(dxnn_ready)})
    elif not cache_key:
        if not onnx_path.is_file():
            build_status = "model_missing"
        elif not calib_dir:
            build_status = "calibration_dir_missing"
        elif (
            use_full_v2_contract
            and str(calibration_contract.get("status") or "") != "resolved"
        ):
            build_status = "calibration_manifest_contract_invalid"
        elif (
            use_full_v2_contract
            and str(compiler_identity.get("status") or "") != "resolved"
        ):
            build_status = "compiler_identity_unresolved"
        else:
            build_status = "cache_contract_preparation_failed"
        status_payload["queue"].append({
            "variant": "full",
            "backend": "deepx_m1",
            "status": build_status,
            "cache_key": "",
            "reason": preparation_error or str(
                calibration_contract.get("reason")
                or "v2 Full-DeepX cache contract unavailable"
            ),
        })
    elif mode in {"reuse_only", "reuse", "cache_verify_only"}:
        miss_status = (
            "cache_miss_blocked"
            if mode == "cache_verify_only"
            else "missing_reusable_dxnn"
        )
        status_payload["queue"].append({
            "variant": "full",
            "backend": "deepx_m1",
            "status": miss_status,
            "cache_key": cache_key,
            "reason": (
                "cache_miss_blocked[deepx_dx_com]: cache_verify_only forbids DX-COM dispatch"
                if mode == "cache_verify_only"
                else "deepx_build.mode is reuse_only"
            ),
        })
        build_status = miss_status
    elif not bool((env_status := inspect_deepx_environment(
        config=cfg, probe_import=True, process_registry=process_registry,
        cancel_event=cancel_event,
    )).get("compiler_ready")):
        cuda_status = dict(env_status.get("compiler_cuda_preflight") or {})
        compiler_reason = (
            compiler_cuda_preflight_message(cuda_status)
            if cuda_status.get("status") == "incompatible"
            else "dx_com import and dxcom entry point are required"
        )
        status_payload["queue"].append({"variant":"full", "backend":"deepx_m1", "status":"compiler_not_ready", "reason":compiler_reason})
        if callable(log):
            log(f"[deepx:{model_id}] compiler_not_ready: {compiler_reason}")
        build_status = "compiler_not_ready"
    elif not calib_dir:
        status_payload["queue"].append({"variant":"full", "backend":"deepx_m1", "status":"calibration_dir_missing", "reason":"DeepX full build needs calibration images unless a reusable DXNN exists"})
        build_status = "calibration_dir_missing"
    elif not onnx_path.is_file():
        build_status = "model_missing"
    else:
        if callable(log):
            log(
                f"[deepx:{model_id}] building full DXNN via dxcom "
                f"classification_preprocessing={classification_mode}"
            )
        res = compile_dxnn(
            onnx_path=build_onnx_path,
            config_path=cfg_path,
            output_dir=config_dir,
            compiler_root=cfg.get("compiler_root") or cfg.get("dx_all_suite_root") or env_status.get("dx_all_suite_root"),
            compiler_venv=cfg.get("compiler_venv") or env_status.get("compiler_venv"),
            compiler_overlay=env_status.get("compiler_overlay"), build_config=cfg,
            opt_level=opt_level,
            timeout_s=_as_int(cfg.get("timeout_s"), 7200),
            process_registry=process_registry,
            cancel_event=cancel_event,
        )
        artifacts["deepx_build_manifest_json"] = config_dir / "build_manifest.json"
        artifacts["deepx_build_log"] = Path(res.log_path) if res.log_path else config_dir / "deepx_build.log"
        # v52j: mirror a concise DX-COM log tail into the workflow log.
        try:
            _lp = Path(res.log_path) if res.log_path else config_dir / "deepx_build.log"
            if callable(log) and _lp.is_file():
                log(f"[deepx:{model_id}] dxcom compiler log: {_lp}")
                _lines = _lp.read_text(encoding="utf-8", errors="replace").splitlines()
                _interesting = [ln for ln in _lines if any(tok in ln.lower() for tok in ("error", "warning", "compile", "compiling", "final result", "added nodes", "skipped nodes", "npu", "cpu", "dxnn"))]
                for _ln in (_interesting[-30:] if _interesting else _lines[-30:]):
                    log(f"[deepx:{model_id}][dxcom] {_ln[:500]}")
        except Exception:
            pass
        if res.ok and res.dxnn_path:
            receipt_payload = (
                {
                    "schema": "onnx-splitpoint/deepx-full-cache-receipt",
                    "schema_version": 2,
                    "cache_key": cache_key,
                    "cache_contract": cache_contract,
                    "cache_contract_sha256": cache_contract.get("contract_sha256"),
                    "build_onnx_adapter": build_onnx_receipt,
                    "build": res.__dict__,
                }
                if use_full_v2_contract
                else {
                    "build_result": res.__dict__,
                    "cache_key": cache_key,
                }
            )
            try:
                cached = cache_dxnn_artifact(
                    dxnn_path=res.dxnn_path,
                    manifest=receipt_payload,
                    cache_root=cache_root,
                    cache_key=cache_key,
                    allow_overwrite=not use_full_v2_contract,
                )
                if use_full_v2_contract:
                    compatible, cache_reason, cache_receipt = (
                        deepx_cached_artifact_compatible(
                            cache_dir=cached.parent,
                            expected_contract=cache_contract,
                            require_artifact_identity=True,
                        )
                    )
                    if not compatible:
                        raise RuntimeError(
                            "published DeepX cache receipt failed verification: "
                            + cache_reason
                        )
                dxnn_ready = cached
                build_status = "ready_built"
                artifacts["deepx_cache_receipt_json"] = (
                    _mirror_deepx_cache_receipt(
                        cached.parent / "build_manifest.json",
                        local_dir=config_dir,
                    )
                )
                status_payload["artifacts"].append({"variant":"full", "backend":"deepx_m1", "status":build_status, "cache_key":cache_key, "cache_contract_sha256":cache_contract.get("contract_sha256"), "dxnn_path":str(cached), "build_output":res.dxnn_path})
            except Exception as exc:
                build_status = "cache_publish_failed"
                status_payload["queue"].append({
                    "variant": "full",
                    "backend": "deepx_m1",
                    "status": build_status,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
        else:
            build_status = str(res.status or "compile_failed")
            status_payload["queue"].append({"variant":"full", "backend":"deepx_m1", "status":build_status, "reason":res.message[-1000:]})

    contracts = []
    if dxnn_ready and Path(dxnn_ready).is_file():
        suite_dir = _legacy_suite_dir(model_dir, benchmark_set_contract)
        dst = suite_dir / "deepx" / "deepx_m1" / "full" / "model.dxnn"
        suite_copy_error = ""
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dxnn_ready, dst)
            artifacts["deepx_full_dxnn_in_suite"] = dst
        except Exception as exc:
            suite_copy_error = f"{type(exc).__name__}: {exc}"
            status_payload["warnings"].append(
                "could not copy DXNN into suite: " + suite_copy_error
            )
        source_onnx_sha256 = str(
            status_payload.get("source_onnx_sha256") or ""
        )
        artifact_sha256 = ""
        suite_artifact_sha256 = ""
        artifact_size_bytes = 0
        suite_artifact_size_bytes = 0
        artifact_identity_error = suite_copy_error
        if not artifact_identity_error:
            try:
                artifact_sha256 = strict_artifact_sha256(Path(dxnn_ready))
                suite_artifact_sha256 = strict_artifact_sha256(dst)
                artifact_size_bytes = int(Path(dxnn_ready).stat().st_size)
                suite_artifact_size_bytes = int(dst.stat().st_size)
            except Exception as exc:
                artifact_identity_error = f"{type(exc).__name__}: {exc}"
        identity_verified = bool(
            not artifact_identity_error
            and source_onnx_sha256
            and artifact_sha256
            and suite_artifact_sha256 == artifact_sha256
            and suite_artifact_size_bytes == artifact_size_bytes
        )
        status_payload["current_full_artifact_identity"] = {
            "schema": "onnx-splitpoint/deepx-current-artifact-identity",
            "schema_version": 1,
            "mode": "v2796_current_bytes_sha256",
            "status": "verified" if identity_verified else "failed",
            "source_onnx_sha256": source_onnx_sha256,
            "artifact_sha256": artifact_sha256,
            "artifact_size_bytes": artifact_size_bytes,
            "suite_artifact_sha256": suite_artifact_sha256,
            "suite_artifact_size_bytes": suite_artifact_size_bytes,
            "error": artifact_identity_error,
        }
        for artifact_row in status_payload["artifacts"]:
            if isinstance(artifact_row, dict) and artifact_row.get("variant") == "full":
                artifact_row.update({
                    "source_onnx_sha256": source_onnx_sha256,
                    "artifact_sha256": artifact_sha256,
                    "artifact_size_bytes": artifact_size_bytes,
                    "suite_artifact_sha256": suite_artifact_sha256,
                    "suite_artifact_size_bytes": suite_artifact_size_bytes,
                })
        if not identity_verified:
            build_status = "artifact_identity_failed"
            reason = (
                artifact_identity_error
                or "strict source/cache/suite artifact identity mismatch"
            )
            status_payload["queue"].append({
                "variant": "full",
                "backend": "deepx_m1",
                "status": build_status,
                "reason": reason,
            })
            for artifact_row in status_payload["artifacts"]:
                if (
                    isinstance(artifact_row, dict)
                    and artifact_row.get("variant") == "full"
                ):
                    artifact_row["status"] = build_status
            status_payload["status"] = "partial"
            status_payload["build_status"] = build_status
            p_status = write_json(
                out_dir / "deepx_artifact_status.json", status_payload
            )
            p_queue = write_json(
                out_dir / "deepx_build_queue.json",
                {
                    "schema": "onnx-splitpoint/deepx-build-queue",
                    "schema_version": 1,
                    "created_at": now_iso(),
                    "model_id": model_id,
                    "queue": status_payload["queue"],
                },
            )
            p_contracts = write_json(
                out_dir / "deepx_output_contracts.json",
                {
                    "schema": "onnx-splitpoint/deepx-output-contracts",
                    "schema_version": 1,
                    "created_at": now_iso(),
                    "model_id": model_id,
                    "contracts": [],
                },
            )
            artifacts.update({
                "deepx_artifact_status_json": p_status,
                "deepx_build_queue_json": p_queue,
                "deepx_output_contracts_json": p_contracts,
            })
            metrics.update({
                "deepx_build_status": build_status,
                "deepx_ready_artifact_count": 0,
                "deepx_queue_count": len(status_payload["queue"]),
            })
            return {
                "artifacts": artifacts,
                "metrics": metrics,
                "status": "partial",
                "message": (
                    "DeepX full DXNN strict artifact identity failed; "
                    "see deepx_build_queue.json."
                ),
            }
        # DX-COM's image loader is compiled into the DXNN.  dx_engine therefore
        # receives one raw RGB/HWC uint8 image, not the source ONNX NCHW float
        # tensor.  Archive that runtime contract explicitly so final evidence
        # never has to discover a working layout/preprocessing permutation.
        runtime_input_contract = task_bound_deepx_runtime_input_contract(
            task=task_name,
            input_name=input_name,
            source_shape=shape,
            classification_preprocessing=classification_mode,
        )
        endpoint_attestation = recorded_deepx_full_endpoint_attestation(
            model_dir=model_dir, suite_dir=suite_dir, model_id=model_id,
        )
        endpoint_mode = str(endpoint_attestation.get("endpoint_mode") or "").strip().lower()
        endpoint_attested = bool(endpoint_attestation.get("pass"))
        if task_name == "detection" and endpoint_attested and endpoint_mode == "decoded":
            authoritative_endpoint = (
                endpoint_attestation.get("authoritative_contract")
                if isinstance(
                    endpoint_attestation.get("authoritative_contract"), Mapping
                )
                else {}
            )
            endpoint_attestation = {
                **endpoint_attestation,
                "stage": "decoded_nms",
                "output_format": "bn6_detections",
                "declared_output_semantics": "xyxy_score_class",
                "source_coordinate_space": str(
                    authoritative_endpoint.get("source_coordinate_space")
                    or ""
                ),
                "semantic_attestation_policy": "recorded_endpoint_plus_runtime_value_validation",
            }
        if not endpoint_attested:
            status_payload["warnings"].append(
                "DeepX Full endpoint contract is not authoritatively attested: "
                + str(endpoint_attestation.get("reason") or "unknown")
            )
        if task_name == "detection" and endpoint_attested and endpoint_mode == "decoded":
            postprocessing_contract = {
                "type": "integrated_decoded_nms",
                "host_required": False,
                "nms_on_host": False,
            }
            contract_family = "decoded_nms"
        elif task_name == "detection" and endpoint_attested and endpoint_mode == "decoded_pre_nms":
            postprocessing_contract = {
                "type": "ultralytics_decoded_classaware_nms",
                "host_required": True,
                "nms_on_host": True,
            }
            contract_family = "decoded_pre_nms"
        elif task_name == "detection":
            postprocessing_contract = {
                "type": "yolo_host_decode_or_model_postprocess",
                "host_required": True,
                "nms_on_host": True,
            }
            contract_family = "raw_head_or_unattested"
        else:
            postprocessing_contract = {
                "type": "classification_topk",
                "host_required": False,
                "nms_on_host": None,
            }
            contract_family = "classification_logits"
        contract = {
            "backend": "deepx_m1",
            "model_id": str(model_id),
            "variant": "full",
            "artifact_kind": "dxnn",
            "artifact_path": str(dxnn_ready),
            "suite_artifact_path": str(dst),
            "artifact_sha256": artifact_sha256,
            "artifact_size_bytes": artifact_size_bytes,
            "suite_artifact_sha256": suite_artifact_sha256,
            "suite_artifact_size_bytes": suite_artifact_size_bytes,
            "input": runtime_input_contract,
            "preprocessing_contract": preprocessing_contract,
            "classification_preprocessing": (
                classification_mode
                if task_name == "classification" else "not_applicable"
            ),
            "full_cache_contract_mode": (
                "v2_exact_explicit"
                if use_full_v2_contract else "legacy_implicit"
            ),
            "cache_key": cache_key,
            "cache_contract_sha256": cache_contract.get("contract_sha256"),
            "calibration_manifest_identity_sha256": calibration_contract.get(
                "identity_sha256"
            ),
            "compiler_identity_sha256": compiler_identity.get("identity_sha256"),
            "source_onnx_sha256": (
                cache_contract.get("source_onnx_sha256")
                or source_onnx_sha256
            ),
            "build_onnx_sha256": (
                cache_contract.get("build_onnx_sha256")
                or build_onnx_receipt.get("build_onnx_sha256")
                or source_onnx_sha256
            ),
            "source_model_input": {
                "name": input_name, "shape": shape, "dtype": "float32", "layout": "NCHW",
            },
            "outputs": [
                {"name": "logits" if task_name == "classification" else "model_outputs", "shape": None, "dtype": "float32", "contract_source": "runtime_report_or_model_output"}
            ],
            "output": {"source": "runtime_report_or_model_output", "status": "declared_no_parse_model_pending_placeholder"},
            "endpoint_mode": endpoint_mode or "unattested",
            "contract_family": contract_family,
            "host_tail_required": endpoint_attestation.get("host_tail_required"),
            "postprocessing_required": endpoint_attestation.get("postprocessing_required"),
            "endpoint_semantic_attestation": endpoint_attestation,
            "postprocessing": postprocessing_contract,
            "contract_complete_enough_for_reports": True,
            "endpoint_contract_complete_enough_for_quality": endpoint_attested,
        }
        contracts.append(contract)
        status_payload["contracts"].append(contract)

        # v55w: materialize the DeepX full input contract inside the actual
        # legacy suite, next to the DXNN.  Remote benchmark bundles only see the
        # suite directory, not the workflow-side benchmark_set/deepx metadata.
        # Without this file the DeepX semantic validator falls back to the YOLO
        # default of 640x640 and classification full-DXNN validation reports
        # false Top1/Top5=0.0 for 224x224 models.
        try:
            write_json(dst.parent / "output_contract.json", contract)
        except Exception as exc:
            status_payload["warnings"].append(f"could not write suite DeepX output contract: {type(exc).__name__}: {exc}")
        try:
            write_json(suite_dir / "deepx" / "deepx_output_contracts.json", {
                "schema": "onnx-splitpoint/deepx-output-contracts",
                "schema_version": 1,
                "created_at": now_iso(),
                "model_id": model_id,
                "contracts": [contract],
            })
        except Exception as exc:
            status_payload["warnings"].append(f"could not write suite DeepX aggregate contracts: {type(exc).__name__}: {exc}")

    status_payload["status"] = "ok" if dxnn_ready and Path(dxnn_ready).is_file() else ("partial" if status_payload["queue"] else build_status)
    status_payload["build_status"] = build_status
    p_status = write_json(out_dir / "deepx_artifact_status.json", status_payload)
    write_json(p_env, env_status)
    status_payload["environment_status"] = env_status
    metrics["deepx_compiler_ready"] = bool(env_status.get("compiler_ready"))
    p_queue = write_json(out_dir / "deepx_build_queue.json", {"schema":"onnx-splitpoint/deepx-build-queue", "schema_version":1, "created_at":now_iso(), "model_id":model_id, "queue":status_payload["queue"]})
    p_contracts = write_json(out_dir / "deepx_output_contracts.json", {"schema":"onnx-splitpoint/deepx-output-contracts", "schema_version":1, "created_at":now_iso(), "model_id":model_id, "contracts":contracts})
    artifacts.update({"deepx_artifact_status_json":p_status, "deepx_build_queue_json":p_queue, "deepx_output_contracts_json":p_contracts})
    metrics.update({
        "deepx_build_status": build_status,
        "deepx_ready_artifact_count": 1 if dxnn_ready else 0,
        "deepx_queue_count": len(status_payload["queue"]),
        "deepx_classification_preprocessing": (
            classification_mode
            if task_name == "classification" else "not_applicable"
        ),
        "deepx_full_cache_contract_sha256": str(
            cache_contract.get("contract_sha256") or ""
        ),
        "deepx_full_cache_contract_mode": (
            "v2_exact_explicit"
            if use_full_v2_contract else "legacy_implicit"
        ),
        "deepx_cache_outcome": str(
            (status_payload.get("cache_lookup") or {}).get("outcome") or ""
        ),
        "deepx_cache_reason": str(
            (status_payload.get("cache_lookup") or {}).get("reason") or ""
        ),
    })
    status = "ok" if status_payload["status"] == "ok" else "partial"
    msg = "DeepX full DXNN artifact ready." if status == "ok" else "DeepX selected but full DXNN artifact is not ready; see deepx_build_queue.json."
    return {"artifacts": artifacts, "metrics": metrics, "status": status, "message": msg}


def _materialize_deepx_artifact_binding_legacy(**kwargs: Any) -> Dict[str, Any]:
    """Backward-compatible workflow entry point used by runner.py.

    v51h/v51i called this binding with a smaller argument set.  v52 keeps that
    call shape working and resolves the model path / benchmark contract from the
    model row and run directory.
    """
    row = kwargs.get("row") or {}
    run_dir = Path(kwargs.get("run_dir"))
    model_id = str(kwargs.get("model_id") or "")
    bdir = run_dir / "models" / model_id / "benchmark_set"
    model_path = str(kwargs.pop("model_path", "") or row.get("resolved_path") or row.get("onnx") or row.get("onnx_path") or row.get("path") or row.get("model_path") or "")
    benchmark_set_contract = kwargs.pop("benchmark_set_contract", None)
    if benchmark_set_contract is None:
        benchmark_set_contract = read_json(bdir / "benchmark_set.json", default={}) or {}
    return materialize_deepx_build_binding(
        run_dir=run_dir,
        model_id=model_id,
        model_path=model_path,
        row=row,
        profile_payload=kwargs.get("profile_payload") or {},
        targets=list(kwargs.get("targets") or []),
        benchmark_set_contract=benchmark_set_contract,
        log=kwargs.get("log"),
        process_registry=kwargs.get("process_registry"),
        cancel_event=kwargs.get("cancel_event"),
    )


# v60s unified artifact-library bridge ---------------------------------------
def _v60s_extract_dxnn_path(value):
    from pathlib import Path as _Path
    if isinstance(value, (_Path, str)):
        p = _Path(value).expanduser()
        if p.suffix.lower() == ".dxnn" and p.is_file():
            return p.resolve()
    if isinstance(value, dict):
        for key in ("dxnn", "dxnn_path", "artifact", "artifact_path", "path", "model"):
            if key in value:
                found = _v60s_extract_dxnn_path(value[key])
                if found:
                    return found
        for item in value.values():
            found = _v60s_extract_dxnn_path(item)
            if found:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _v60s_extract_dxnn_path(item)
            if found:
                return found
    return None


def _v60s_bound_arguments(fn, args, kwargs):
    import inspect as _inspect
    try:
        return dict(_inspect.signature(fn).bind_partial(*args, **kwargs).arguments)
    except Exception:
        return dict(kwargs)


def _v60s_deepx_cache_key(bound):
    for name in ("cache_key", "key", "artifact_key", "contract_key"):
        value = bound.get(name)
        if value:
            return str(value)
    for value in bound.values():
        if isinstance(value, dict):
            for name in ("cache_key", "key", "artifact_key"):
                if value.get(name):
                    return str(value[name])
    return ""


def _v60s_deepx_destination(bound, cache_key):
    from pathlib import Path as _Path
    for name in ("dxnn_path", "artifact_path", "expected_path", "output_path", "candidate"):
        value = bound.get(name)
        if value:
            p = _Path(value).expanduser()
            if p.suffix.lower() == ".dxnn":
                return p
    for name in ("cache_dir", "artifact_dir", "output_dir", "out_dir"):
        value = bound.get(name)
        if value:
            return _Path(value).expanduser() / "model.dxnn"
    for name in ("cache_root", "artifacts_root"):
        value = bound.get(name)
        if value and cache_key:
            return _Path(value).expanduser() / cache_key / "model.dxnn"
    return None


def _v60s_register_deepx_result(result, bound, *, source="deepx-binding"):
    import os as _os
    try:
        from ..artifact_store import ArtifactStore, artifact_store_enabled
        if not artifact_store_enabled():
            return
        path = _v60s_extract_dxnn_path(result)
        if path is None:
            return
        cache_key = _v60s_deepx_cache_key(bound)
        manifest = {}
        for name in ("build_manifest.json", "manifest.json", "build_status.json"):
            candidate = path.parent / name
            if candidate.is_file():
                try:
                    import json as _json
                    payload = _json.loads(candidate.read_text(encoding="utf-8"))
                    if isinstance(payload, dict): manifest = payload
                except Exception:
                    pass
                break
        contract = {
            "schema": "onnx-splitpoint/deepx-binding-contract/v1",
            "legacy_cache_key": cache_key,
            "task": str(bound.get("task") or bound.get("benchmark_task") or manifest.get("task") or "auto"),
            "target": str(bound.get("target") or bound.get("hardware") or manifest.get("target") or "deepx_m1"),
            "build_manifest": manifest,
        }
        store = ArtifactStore(_os.environ.get("ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT") or None)
        pin = str(_os.environ.get("ONNX_SPLITPOINT_ARTIFACT_PIN_FINAL", "0")).lower() in {"1","true","yes","on"}
        store.register(source_path=path, kind="deepx_dxnn", contract=contract,
                       metadata={"legacy_cache_key": cache_key, "binding": source},
                       source_run=_os.environ.get("ONNX_SPLITPOINT_RUN_ID", ""),
                       pin=pin, pin_label="final-campaign" if pin else "")
    except Exception:
        return


def _find_cached_dxnn(*args, **kwargs):
    result = _find_cached_dxnn_legacy(*args, **kwargs)
    bound = _v60s_bound_arguments(_find_cached_dxnn_legacy, args, kwargs)
    if result:
        _v60s_register_deepx_result(result, bound, source="legacy-cache-hit")
        return result
    try:
        import os as _os
        from ..artifact_store import ArtifactStore, artifact_store_enabled
        if not artifact_store_enabled():
            return result
        cache_key = _v60s_deepx_cache_key(bound)
        if not cache_key:
            return result
        store = ArtifactStore(_os.environ.get("ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT") or None)
        verify = _os.environ.get("ONNX_SPLITPOINT_ARTIFACT_VERIFY", "metadata")
        record = store.find_by_metadata(kind="deepx_dxnn", key="legacy_cache_key", value=cache_key, verify=verify)
        if record is None:
            return result
        destination = _v60s_deepx_destination(bound, cache_key)
        if destination is None:
            return result
        destination.parent.mkdir(parents=True, exist_ok=True)
        store.materialize(record, destination, reference=f"deepx-cache:{cache_key}")
        # The historical finder remains the authority for companion-manifest checks.
        result = _find_cached_dxnn_legacy(*args, **kwargs)
        if result:
            return result
    except Exception:
        pass
    return result



def materialize_deepx_artifact_binding(*args, **kwargs):
    result = _materialize_deepx_artifact_binding_legacy(*args, **kwargs)
    bound = _v60s_bound_arguments(_materialize_deepx_artifact_binding_legacy, args, kwargs)
    _v60s_register_deepx_result(result, bound, source="materialize-binding")
    return result
