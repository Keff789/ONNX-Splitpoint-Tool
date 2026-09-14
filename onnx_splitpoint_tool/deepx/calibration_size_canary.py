"""Fail-closed ResNet50 DeepX calibration-size canary verification.

The v2.75.41 canary changes one scientific variable: the number of
ImageNet calibration images used to build a DeepX Full artifact.  This module
loads the B500 and B1000 EvaluationRuns, verifies their exact-v2 build
receipts and full-only quality evidence, proves the calibration cohort
relationship from the actual manifest item lists, and emits one paired report.

An equal B500/B1000 DXNN digest is deliberately *not* an error.  A compiler is
allowed to produce byte-identical output for the two different, correctly
sealed build contracts; that is a scientifically meaningful result.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from onnx_splitpoint_tool.campaign import verify_dataset_manifest
from onnx_splitpoint_tool.dataset_provisioning import (
    IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG,
    _normalise_kaggle_username,
)
from onnx_splitpoint_tool.quality_cache import (
    image_ids_fingerprint,
    json_fingerprint,
)
from onnx_splitpoint_tool.quality_service import (
    _validate_candidate_execution_contract,
)

from .config import CLASSIFICATION_PREPROCESSING_IMAGENET
from .preprocessing_ab import ArmEvidence, load_arm_evidence


REPORT_SCHEMA = "onnx-splitpoint/deepx-classification-calibration-size-canary"
REPORT_SCHEMA_VERSION = 1
BASELINE_CALIBRATION_COUNT = 500
CANDIDATE_CALIBRATION_COUNT = 1000
VALIDATION_RECORD_COUNT = 500
BOOTSTRAP_REPETITIONS = 500
TOP1_BOOTSTRAP_SEED = 20260710
TOP5_BOOTSTRAP_SEED = 20260711
CONFIDENCE_LEVEL = 0.95
EXPECTED_QUALITY_POLICY_SHA256 = (
    "58bf34c5760ed51e4ff35faee4055891caf624b2722cb375df9d419863efa9c4"
)
QUALITY_MARGIN = 0.01
EXPECTED_QUALITY_ALGORITHM = (
    "management_paired_quality_v2:classification-accuracy-v2"
)
EXPECTED_SETUP_ID = "orin_nx_deepx_m1_01"
EXPECTED_SOURCE_ONNX_SHA256 = (
    "cebd9d5879ddd304a18f51e559ab74fba5cbd1f610b9359321bc5228acbe4f50"
)
EXPECTED_BUILD_ONNX_SHA256 = (
    "b5921d9f75da27c05b6a70fda31f4a0b119383eaf1ddc263fa5e20ec6bd61528"
)
EXPECTED_COMPILER_IDENTITY_SHA256 = (
    "298a0f8649edb95255129db076f3ffbee8a804698386b1f46077a3c0e4b970c2"
)
EXPECTED_COMPILER_CONTRACT_SHA256 = (
    "f5a67a84946d2f44f4578e1737a67778783f5c7c6ef47fb7e1de0beb6e31bb86"
)
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "64a7ef3bf55352bb39ef86f25fe73e6ff18a51ed1a02c8aca4940178f69e7b6c"
)
EXPECTED_VALIDATION_IMAGE_IDS_SHA256 = (
    "71032a98e158ca71711a5567de5d46fbf04f05777baa940f74f3c39fdf0c083f"
)
EXPECTED_VALIDATION_GROUND_TRUTH_SHA256 = (
    "87775e86ef3bcf3ec1e0d0a79696bf2f58fa215d5320ce80167bb1f417c7a177"
)
EXPECTED_PREPARED_INPUT_EVIDENCE_SHA256 = (
    "0833140afa44181b7163b2234197c8e64b308653e1e2656e46f1707e9e875724"
)
EXPECTED_PREPROCESSING_CONTRACT_SHA256 = (
    "ea28cf5ac35bd4c9a3321ac97fd559f32fd7a661dc4a93c324fe3ffc54188fa9"
)
EXPECTED_B500_CALIBRATION_ITEMS_SHA256 = (
    "943d1d1506250f3b6e29d0fb06b6118f01c3e70d769b00939d74e439cda9686c"
)
EXPECTED_B500_CALIBRATION_MANIFEST_SHA256 = (
    "6d6978cafd1b828d6fa9438b232120f2a769311c28492bea5a141af586c3d58c"
)
EXPECTED_B500_CALIBRATION_IDENTITY_SHA256 = (
    "913d753f34e6ab545ec38c5dcd27177b748eecf0c200625d579b7e6a6413efef"
)
EXPECTED_B500_CALIBRATION_CONTRACT_SHA256 = (
    "bea432f0cfbb7ec225a63e60c56c32f5c4c0fdf6e75c00e9852ec014e10677a7"
)
EXPECTED_B500_DXCOM_CONFIG_SHA256 = (
    "65b18a1721afea8432f2ac01631655bece6b8fb81326f78462491cb6aa771f9f"
)
EXPECTED_B500_BUILD_OPTIONS_SHA256 = (
    "55f9fade1ec02c5754a96f736c3233f5e3d4bad46b4980a01b1880e0340968b5"
)
EXPECTED_B500_CACHE_KEY = (
    "deepx_m1_full_imagenet_mean_std_v2_313fece50a4acf14bf4fc4076b10b063"
)
EXPECTED_B500_CACHE_CONTRACT_SHA256 = (
    "3804283b15e4b26040ec9775023d822a54df0111e564836054ba1e9f6cf34f49"
)
EXPECTED_B500_DXNN_SHA256 = (
    "a8bb14689d2d160f0b66c0f4b614c8e214808ca28adc25dce8cb6fa419af05bb"
)
EXPECTED_B500_DEEPX_TOP1_HITS = 401
EXPECTED_B500_DEEPX_TOP5_HITS = 474
EXPECTED_REFERENCE_TOP1_HITS = 406
EXPECTED_REFERENCE_TOP5_HITS = 476
BASELINE_PROFILE_NAME = "resnet50_v27540_deepx_preprocess_b_imagenet_mean_std"
CANDIDATE_PROFILE_NAME = "resnet50_v27541_deepx_calibration_1000_imagenet_mean_std"
B500_CACHE_NAMESPACE = (
    "~/Models/BackendArtifacts/deepx/v2.75.40/"
    "resnet50_preprocessing_ab/b_imagenet_mean_std"
)
B1000_CACHE_NAMESPACE = (
    "~/Models/BackendArtifacts/deepx/v2.75.41/"
    "resnet50_calibration_size/b1000_imagenet_mean_std"
)


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bare_sha(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def _require_sha256(value: Any, *, label: str) -> str:
    digest = _bare_sha(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} is not a SHA-256 identity")
    return digest


def _production_calibration_kernel_ref(value: Any, *, label: str) -> str:
    """Validate the public identity emitted by the production Kaggle writer.

    The owner is deliberately *not* pinned to one account: it is operational
    metadata and the production writer accepts any canonical Kaggle username.
    The writer slug is part of the scientific provisioning route, however, and
    must remain the exact production constant.  Cohort authority comes from the
    byte-addressed, self-hashed selection receipt verified separately below.
    """

    kernel_ref = str(value or "")
    if not kernel_ref or kernel_ref != kernel_ref.strip():
        raise ValueError(
            f"{label} provisioning kernel_ref is empty or not canonical: "
            f"observed={kernel_ref!r}"
        )
    if kernel_ref.count("/") != 1:
        raise ValueError(
            f"{label} provisioning kernel_ref must have canonical owner/slug "
            f"form: observed={kernel_ref!r}"
        )
    owner, slug = kernel_ref.split("/", 1)
    try:
        canonical_owner = _normalise_kaggle_username(owner)
    except ValueError as exc:
        raise ValueError(
            f"{label} provisioning kernel owner is invalid: "
            f"observed={owner!r}: {exc}"
        ) from exc
    if not canonical_owner or canonical_owner != owner:
        raise ValueError(
            f"{label} provisioning kernel owner is not canonical: "
            f"observed={owner!r}, canonical={canonical_owner!r}"
        )
    if slug != IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG:
        raise ValueError(
            f"{label} provisioning kernel slug mismatch: "
            f"expected={IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG!r}, "
            f"observed={slug!r}"
        )
    return kernel_ref


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not an integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not an integer") from exc
    if isinstance(value, float) and value != float(numeric):
        raise ValueError(f"{label} is not an integer")
    return numeric


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _single(paths: Sequence[Path], *, label: str) -> Path:
    existing = sorted({
        path.absolute() for path in paths
        if not path.is_symlink() and path.is_file()
    })
    if len(existing) != 1:
        raise ValueError(f"expected exactly one {label}, found {len(existing)}")
    return existing[0]


def _load_profile(root: Path) -> dict[str, Any]:
    profile_path = root / "profile.yaml"
    if profile_path.is_symlink() or not profile_path.is_file():
        raise FileNotFoundError(profile_path)
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to verify calibration canaries") from exc
    value = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("resolved profile.yaml is not an object")
    return dict(value)


def _load_profile_source(root: Path) -> dict[str, Any]:
    source_path = root / "profile_source.yaml"
    if source_path.is_symlink() or not source_path.is_file():
        raise FileNotFoundError(source_path)
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to verify calibration canaries") from exc
    value = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("profile_source.yaml is not an object")
    return dict(value)


def _validate_run_envelope(
    root: Path,
    run_manifest: Mapping[str, Any],
    *,
    label: str,
    expected_profile_name: str,
    profile: Mapping[str, Any],
) -> None:
    from onnx_splitpoint_tool.workflow.artifacts import sha256_json

    if (
        run_manifest.get("schema")
        != "onnx-splitpoint/evaluation-run-manifest"
        or _integer(
            run_manifest.get("schema_version"),
            label=f"{label} run manifest schema_version",
        ) != 1
        or run_manifest.get("status") not in {"ok", "completed"}
        or not isinstance(run_manifest.get("run_id"), str)
        or not str(run_manifest.get("run_id") or "").strip()
        or str(run_manifest.get("profile_id") or "") != expected_profile_name
        or _require_sha256(
            run_manifest.get("profile_hash"), label=f"{label} run profile",
        ) != _bare_sha(sha256_json(profile))
    ):
        raise ValueError(f"{label} EvaluationRun envelope is not exact/complete")
    run_status_path = root / "run_status.json"
    if run_status_path.is_symlink():
        raise ValueError(f"{label} run_status.json must not be a symlink")
    if run_status_path.exists():
        run_status = _read_object(run_status_path)
        if (
            run_status.get("schema") != "onnx-splitpoint/run-status-summary"
            or _integer(
                run_status.get("schema_version"),
                label=f"{label} run status schema_version",
            ) != 1
            or run_status.get("status") not in {"ok", "completed"}
            or run_status.get("technical_status") not in {"ok", "completed"}
            or str(run_status.get("profile_id") or "") != expected_profile_name
            or str(run_status.get("run_id") or "")
            != str(run_manifest.get("run_id") or "")
        ):
            raise ValueError(f"{label} run status is not exact/complete")


def _profile_calibration_manifest(profile: Mapping[str, Any]) -> str:
    value = _nested(
        profile, "campaign", "dataset_manifests", "classification", "calibration",
    )
    return str(value or "").strip()


def _frozen_profile_source(label: str) -> tuple[Path, dict[str, Any]]:
    name = BASELINE_PROFILE_NAME if label == "B500" else CANDIDATE_PROFILE_NAME
    path = Path(__file__).resolve().parents[2] / "profiles" / f"{name}.yaml"
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"frozen {label} source profile is unavailable: {path}")
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to verify calibration canaries") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("name") != name:
        raise ValueError(f"frozen {label} source profile identity is invalid")
    return path, dict(payload)


def _materialized_frozen_profile(label: str) -> tuple[Path, dict[str, Any]]:
    """Resolve the checked-in profile through the production run-mode projector."""

    source_path, source_profile = _frozen_profile_source(label)
    from onnx_splitpoint_tool.run_modes import (
        apply_run_mode,
        default_run_modes_config,
    )

    resolved, _ = apply_run_mode(
        source_profile,
        config=default_run_modes_config(),
    )
    return source_path, dict(resolved)


def _normalized_profile_for_source(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only run-location metadata not controlled by the source profile."""

    return _without_paths(profile, (
        ("campaign", "dataset_registry"),
        ("campaign", "dataset_registry_sha256"),
        ("campaign", "dataset_registry_binding_sha256"),
        ("execution_preset", "config_path"),
        ("execution_preset", "config_sha256"),
        ("execution_preset", "resolved_at"),
    ))


def _require_profile_subset(
    observed: Any,
    expected: Any,
    *,
    label: str,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping):
            raise ValueError(
                f"{label} resolved profile differs from frozen source at "
                + ".".join(path)
            )
        for key, value in expected.items():
            if key not in observed:
                raise ValueError(
                    f"{label} resolved profile omits frozen source field "
                    + ".".join((*path, str(key)))
                )
            _require_profile_subset(
                observed[key], value, label=label, path=(*path, str(key)),
            )
        return
    if observed != expected:
        raise ValueError(
            f"{label} resolved profile differs from frozen source at "
            + ".".join(path)
        )


def _validate_resolved_profile(
    profile: Mapping[str, Any],
    *,
    label: str,
    expected_count: int,
) -> dict[str, Any]:
    """Prove the exact resolved B500/B1000 Full-only profile axes."""

    expected_name = (
        BASELINE_PROFILE_NAME if label == "B500" else CANDIDATE_PROFILE_NAME
    )
    expected_cache = (
        B500_CACHE_NAMESPACE if label == "B500" else B1000_CACHE_NAMESPACE
    )
    if str(profile.get("name") or "") != expected_name:
        raise ValueError(f"{label} resolved profile name is not the frozen canary profile")
    campaign = profile.get("campaign")
    deepx = profile.get("deepx_build")
    execution = profile.get("execution_preset")
    validation = profile.get("validation_execution")
    quality = profile.get("quality_gate")
    native = profile.get("native_producers")
    energy = profile.get("energy")
    ranking = profile.get("ranking_validation")
    if not all(isinstance(value, Mapping) for value in (
        campaign, deepx, execution, validation, quality, native, energy, ranking,
    )):
        raise ValueError(f"{label} resolved profile is missing frozen canary axes")
    if str(campaign.get("id") or "") != expected_name:
        raise ValueError(f"{label} resolved campaign identity differs from its profile")

    expected_deepx = {
        "mode": "reuse_and_build_missing",
        "target": "deepx_m1",
        "calib_count": expected_count,
        "calibration_method": "ema",
        "opt_level": 0,
        "force_build": False,
        "classification_preprocessing": CLASSIFICATION_PREPROCESSING_IMAGENET,
    }
    for field, expected in expected_deepx.items():
        observed = deepx.get(field)
        if isinstance(expected, int) and not isinstance(expected, bool):
            observed = _integer(observed, label=f"{label} profile deepx_build.{field}")
        if observed != expected:
            raise ValueError(f"{label} resolved DeepX profile axis changed: {field}")
    cache_dir = str(deepx.get("cache_dir") or "").strip()
    if (
        not cache_dir
        or Path(cache_dir).expanduser().resolve()
        != Path(expected_cache).expanduser().resolve()
    ):
        raise ValueError(f"{label} resolved profile cache namespace changed")

    overrides = execution.get("overrides")
    max_items = validation.get("max_items")
    classification_gate = quality.get("classification")
    statistics = quality.get("statistics")
    if not all(isinstance(value, Mapping) for value in (
        overrides, max_items, classification_gate, statistics,
    )):
        raise ValueError(f"{label} resolved Standard quality axes are incomplete")
    if (
        execution.get("id") != "standard"
        or execution.get("follow_tool_config") is not False
        or dict(overrides) != {"native_enabled": False, "energy_enabled": False}
        or validation.get("mode") != "screening"
        or validation.get("cadence") != "once_per_artifact"
        or validation.get("cache_task_quality") is not True
        or _integer(
            max_items.get("classification"),
            label=f"{label} profile validation classification count",
        ) != VALIDATION_RECORD_COUNT
        or quality.get("schema") != "onnx-splitpoint/task-quality-policy"
        or _integer(
            quality.get("schema_version"), label=f"{label} quality schema_version",
        ) != 3
        or quality.get("profile_id") != "task_quality_development_500"
        or quality.get("diagnostic_only") is not False
        or dict(classification_gate) != {
            "primary_metric": "top1_accuracy",
            "non_inferiority_margin": QUALITY_MARGIN,
            "guardrails": {"top5_accuracy_margin": QUALITY_MARGIN},
        }
    ):
        raise ValueError(f"{label} resolved Standard quality axes changed")
    for field, expected in {
        "method": "paired_bootstrap",
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "seed": TOP1_BOOTSTRAP_SEED,
        "decision": "lower_one_sided_bound",
    }.items():
        observed = statistics.get(field)
        if isinstance(expected, int):
            observed = _integer(observed, label=f"{label} profile statistics.{field}")
        elif isinstance(expected, float):
            try:
                observed = float(observed)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{label} profile statistics.{field} is invalid"
                ) from exc
        if observed != expected:
            raise ValueError(f"{label} resolved quality statistics changed: {field}")
    if (
        native.get("enabled") is not False
        or energy.get("enabled") is not False
        or energy.get("generic_enabled") is not False
        or ranking.get("enabled") is not False
    ):
        raise ValueError(f"{label} resolved profile enables excluded claim axes")

    expected_run_profiles = [
        {
            "id": "ort_tensorrt", "type": "same_backend_reference",
            "full": "tensorrt", "stage1": "tensorrt", "stage2": "tensorrt",
            "required": True, "enabled": True,
        },
        {
            "id": "deepx_m1_full", "type": "same_backend_reference",
            "full": "deepx_m1", "stage1": "deepx_m1", "stage2": "deepx_m1",
            "required": True, "enabled": True,
        },
    ]
    if list(profile.get("run_profiles") or []) != expected_run_profiles:
        raise ValueError(f"{label} resolved profile run matrix is not exact")
    expected_quality_canary = {
        "schema": "onnx-splitpoint/full-only-quality-canary",
        "schema_version": 1,
        "enabled": True,
        "execution_scope": "full_only",
        "full_run_ids": [{
            "id": "deepx_m1_full", "run_id": "deepx_m1_full",
            "setup_id": "orin_nx_deepx_m1_01", "backend": "deepx_m1",
            "variant": "full", "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        }],
        "setup_local_tensorrt_companions": [{
            "id": "tensorrt_at_deepx_m1_full", "run_id": "ort_tensorrt",
            "source_run_id": "native_full_tensorrt",
            "setup_id": "orin_nx_deepx_m1_01", "backend": "tensorrt",
            "variant": "full", "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        }],
    }
    if dict(profile.get("quality_canary") or {}) != expected_quality_canary:
        raise ValueError(f"{label} resolved profile Full-only canary axes changed")
    source_path, _ = _frozen_profile_source(label)
    return {
        "profile_name": expected_name,
        "source_profile": str(source_path),
        "source_profile_sha256": _sha256_file(source_path),
        "force_build": False,
        "cache_namespace": expected_cache,
        "calibration_count": expected_count,
        "validation_count": VALIDATION_RECORD_COUNT,
        "execution_scope": "full_only",
    }


def _resolve_manifest_path(
    *,
    root: Path,
    profile: Mapping[str, Any],
    contract: Mapping[str, Any],
    explicit_path: str | Path | None,
    label: str,
) -> Path:
    expected_sha = _require_sha256(
        contract.get("manifest_file_sha256"), label=f"{label} manifest file",
    )
    candidates: list[Path] = []
    if explicit_path is not None:
        explicit = Path(explicit_path).expanduser()
        if explicit.is_symlink() or not explicit.is_file():
            raise FileNotFoundError(f"{label} calibration manifest not found: {explicit}")
        candidates.append(explicit.absolute())
    else:
        reference = _profile_calibration_manifest(profile)
        if reference:
            candidate = Path(reference).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.is_file():
                candidates.append(candidate)
        manifest_name = str(contract.get("manifest_file_name") or "").strip()
        if manifest_name:
            candidates.extend(root.rglob(manifest_name))
    matching = sorted({
        path.absolute() for path in candidates
        if not path.is_symlink() and path.is_file()
        and _sha256_file(path) == expected_sha
    })
    if len(matching) != 1:
        option = (
            "--baseline-calibration-manifest" if label == "B500"
            else "--candidate-calibration-manifest"
        )
        raise ValueError(
            f"{label} requires exactly one calibration manifest whose file SHA-256 "
            f"matches the run receipt; found {len(matching)}. Supply {option} with "
            "the frozen manifest used by that run."
        )
    return matching[0]


def _manifest_item_identity(item: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if set(item) != {
        "sample_id", "relative_path", "size_bytes", "sha256", "class_name",
    }:
        raise ValueError(f"{label} calibration manifest item shape is not frozen")
    sample_id = str(item.get("sample_id") or "").strip()
    relative_path = str(item.get("relative_path") or "").strip().replace("\\", "/")
    class_name = str(item.get("class_name") or "").strip()
    digest = _require_sha256(item.get("sha256"), label=f"{label} item content")
    try:
        size_bytes = int(item.get("size_bytes"))
    except Exception as exc:
        raise ValueError(f"{label} manifest item size is not an integer") from exc
    relative = Path(relative_path)
    if (
        not sample_id or not relative_path or relative.is_absolute()
        or ".." in relative.parts or len(class_name) != 9
        or not class_name.startswith("n") or not class_name[1:].isdigit()
        or size_bytes <= 0
    ):
        raise ValueError(f"{label} manifest item identity is incomplete")
    return {
        "sample_id": sample_id,
        "relative_path": relative_path,
        "sha256": digest,
        "size_bytes": size_bytes,
        "class_name": class_name,
    }


def _verify_provisioning_selection_manifest(
    *,
    dataset_manifest_path: Path,
    provisioning: Mapping[str, Any],
    manifest_items: Sequence[Mapping[str, Any]],
    expected_count: int,
    label: str,
) -> None:
    """Bind the private ImageNet selection receipt to the materialized root.

    The dataset manifest is written after the selected archive is extracted.
    Its ``provisioning_selection`` annotation is therefore only authoritative
    when the referenced, content-addressed selection receipt still exists and
    describes the same bytes/classes as the manifest items.
    """

    raw_path = Path(
        str(provisioning.get("selection_manifest") or "")
    ).expanduser()
    selection_path = (
        raw_path if raw_path.is_absolute()
        else dataset_manifest_path.parent / raw_path
    ).absolute()
    if selection_path.is_symlink() or not selection_path.is_file():
        raise ValueError(f"{label} provisioning selection manifest is unavailable")
    if _sha256_file(selection_path) != _require_sha256(
        provisioning.get("selection_manifest_sha256"),
        label=f"{label} provisioning selection manifest",
    ):
        raise ValueError(f"{label} provisioning selection manifest file hash mismatch")
    selection = _read_object(selection_path)
    expected_keys = {
        "schema", "schema_version", "generated_at_unix", "competition",
        "source_root", "strategy", "seed", "requested_count",
        "selected_count", "available_class_count", "selected_class_count",
        "class_counts", "selection_uses_model_predictions", "items",
        "selection_payload_sha256",
    }
    unhashed = dict(selection)
    declared_payload_sha = _require_sha256(
        unhashed.pop("selection_payload_sha256", None),
        label=f"{label} provisioning selection payload",
    )
    rows = selection.get("items")
    class_counts = selection.get("class_counts")
    if set(selection) != expected_keys:
        raise ValueError(
            f"{label} provisioning selection receipt shape differs: "
            f"missing={sorted(expected_keys - set(selection))}, "
            f"unexpected={sorted(set(selection) - expected_keys)}"
        )
    receipt_mismatches: list[str] = []
    exact_receipt_fields = {
        "schema": "onnx-splitpoint/imagenet-train-calibration-selection",
        "competition": "imagenet-object-localization-challenge",
        "strategy": "class_stratified_deterministic_hash",
        "selection_uses_model_predictions": False,
    }
    for field, expected_value in exact_receipt_fields.items():
        if selection.get(field) != expected_value:
            receipt_mismatches.append(
                f"{field}(expected={expected_value!r}, "
                f"observed={selection.get(field)!r})"
            )
    integer_receipt_fields = {
        "schema_version": 1,
        "seed": TOP1_BOOTSTRAP_SEED,
        "requested_count": expected_count,
        "selected_count": expected_count,
        "available_class_count": 1000,
        "selected_class_count": expected_count,
    }
    for field, expected_value in integer_receipt_fields.items():
        observed_value = _integer(
            selection.get(field), label=f"{label} selection {field}",
        )
        if observed_value != expected_value:
            receipt_mismatches.append(
                f"{field}(expected={expected_value!r}, "
                f"observed={observed_value!r})"
            )
    generated_at = selection.get("generated_at_unix")
    if (
        isinstance(generated_at, bool)
        or not isinstance(generated_at, int)
        or generated_at <= 0
    ):
        receipt_mismatches.append(
            f"generated_at_unix(expected=positive integer, observed={generated_at!r})"
        )
    if not str(selection.get("source_root") or "").strip():
        receipt_mismatches.append("source_root(expected=non-empty)")
    if not isinstance(rows, list) or len(rows) != expected_count:
        receipt_mismatches.append(
            f"items(expected=list[{expected_count}], "
            f"observed_type={type(rows).__name__}, "
            f"observed_count={len(rows) if isinstance(rows, list) else 'n/a'})"
        )
    if not isinstance(class_counts, Mapping):
        receipt_mismatches.append(
            "class_counts(expected=mapping, "
            f"observed_type={type(class_counts).__name__})"
        )
    actual_payload_sha = _canonical_sha256(unhashed)
    if declared_payload_sha != actual_payload_sha:
        receipt_mismatches.append(
            "selection_payload_sha256(self-hash mismatch: "
            f"declared={declared_payload_sha}, actual={actual_payload_sha})"
        )
    if receipt_mismatches:
        raise ValueError(
            f"{label} provisioning selection receipt is not frozen: "
            + "; ".join(receipt_mismatches)
        )

    selected_by_path: dict[str, dict[str, Any]] = {}
    observed_class_counts: dict[str, int] = {}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping) or set(raw_row) != {
            "class_name", "source_relative_path", "archive_path",
            "size_bytes", "sha256",
        }:
            raise ValueError(f"{label} provisioning selection item shape is not frozen")
        archive_path = str(raw_row.get("archive_path") or "").replace("\\", "/")
        source_path = str(
            raw_row.get("source_relative_path") or ""
        ).replace("\\", "/")
        relative = Path(archive_path)
        class_name = str(raw_row.get("class_name") or "")
        size_bytes = _integer(
            raw_row.get("size_bytes"), label=f"{label} selection item bytes",
        )
        digest = _require_sha256(
            raw_row.get("sha256"), label=f"{label} selection item content",
        )
        if (
            not archive_path or not source_path or source_path != archive_path
            or relative.is_absolute()
            or ".." in relative.parts or size_bytes <= 0
            or relative.parts[0] != class_name
            or archive_path in selected_by_path
        ):
            raise ValueError(f"{label} provisioning selection item is invalid")
        selected_by_path[archive_path] = {
            "class_name": class_name,
            "size_bytes": size_bytes,
            "sha256": digest,
        }
        observed_class_counts[class_name] = observed_class_counts.get(class_name, 0) + 1
    expected_by_path = {
        str(item["relative_path"]): {
            "class_name": str(item["class_name"]),
            "size_bytes": int(item["size_bytes"]),
            "sha256": str(item["sha256"]),
        }
        for item in manifest_items
    }
    normalized_class_counts = {
        str(key): _integer(value, label=f"{label} selection class count")
        for key, value in class_counts.items()
    }
    if (
        selected_by_path != expected_by_path
        or normalized_class_counts != observed_class_counts
    ):
        raise ValueError(
            f"{label} provisioning selection items differ from calibration manifest"
        )


def _validate_frozen_b500_manifest_authority(
    path: Path, payload: Mapping[str, Any],
) -> None:
    """Pin the historical B500 file bytes and selected-item cohort."""

    observed_manifest_sha = _sha256_file(path)
    if observed_manifest_sha != EXPECTED_B500_CALIBRATION_MANIFEST_SHA256:
        raise ValueError(
            "B500 calibration manifest differs from the frozen v2.75.40-B "
            "file authority: "
            f"expected={EXPECTED_B500_CALIBRATION_MANIFEST_SHA256}, "
            f"observed={observed_manifest_sha}, path={path}"
        )
    observed_items_sha = _require_sha256(
        payload.get("items_identity_sha256"),
        label="B500 calibration manifest items",
    )
    if observed_items_sha != EXPECTED_B500_CALIBRATION_ITEMS_SHA256:
        raise ValueError(
            "B500 calibration items differ from the frozen v2.75.40-B "
            "cohort authority: "
            f"expected={EXPECTED_B500_CALIBRATION_ITEMS_SHA256}, "
            f"observed={observed_items_sha}"
        )


def _load_manifest(
    path: Path,
    *,
    contract: Mapping[str, Any],
    expected_count: int,
    label: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    payload = _read_object(path)
    verification = verify_dataset_manifest(
        payload, verify_files=True, verification_mode="full",
    )
    if not bool(verification.get("ok")):
        raise ValueError(f"{label} calibration manifest self-verification failed")
    expected = {
        "schema": "onnx-splitpoint/dataset-manifest",
        "schema_version": 1,
        "task": "classification",
        "role": "calibration",
        "hash_mode": "content",
        "dataset_id": "ilsvrc2012-train-calibration",
        "split": "train",
        "source_kind": "directory_scan",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{label} calibration manifest {field} mismatch")
    expected_manifest_keys = {
        "schema", "schema_version", "created_at", "dataset_id", "task",
        "role", "split", "source_kind", "root", "root_name", "hash_mode",
        "item_count", "population_count", "selection", "items",
        "items_identity_sha256", "annotations", "labels", "final_use_note",
        "manifest_payload_sha256", "provisioning_selection",
    }
    annotations = payload.get("annotations")
    labels = payload.get("labels")
    if (
        set(payload) != expected_manifest_keys
        or not isinstance(payload.get("created_at"), str)
        or not str(payload.get("created_at") or "").strip()
        or not isinstance(annotations, Mapping)
        or set(annotations) != {"path", "sha256"}
        or not isinstance(labels, Mapping)
        or set(labels) != {"path", "sha256"}
        or payload.get("final_use_note")
        != (
            "A manifest is content-addressed evidence. Final status still "
            "depends on the campaign profile and a disjointness check."
        )
    ):
        raise ValueError(f"{label} calibration manifest shape is not frozen")
    raw_items = list(payload.get("items") or [])
    if (
        _integer(payload.get("item_count"), label=f"{label} item_count")
        != expected_count
        or len(raw_items) != expected_count
        or _integer(
            payload.get("population_count"), label=f"{label} population_count",
        ) != expected_count
    ):
        raise ValueError(
            f"{label} calibration manifest/root must contain exactly {expected_count} items"
        )
    items = tuple(
        _manifest_item_identity(item, label=label)
        for item in raw_items if isinstance(item, Mapping)
    )
    if len(items) != expected_count:
        raise ValueError(f"{label} calibration manifest contains non-object items")
    item_keys = {_canonical_sha256(item) for item in items}
    if len(item_keys) != expected_count:
        raise ValueError(f"{label} calibration manifest contains duplicate items")
    if len({item["relative_path"] for item in items}) != expected_count:
        raise ValueError(f"{label} calibration manifest contains duplicate paths")
    actual_class_count = len({str(item["class_name"]) for item in items})
    if actual_class_count != expected_count:
        raise ValueError(
            f"{label} calibration cohort must contain exactly one item in each of "
            f"{expected_count} distinct classes"
        )

    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError(f"{label} calibration selection contract is missing")
    provisioning = payload.get("provisioning_selection")
    if isinstance(provisioning, Mapping) and provisioning:
        expected_selection_keys = {
            "strategy", "seed", "requested_max_items",
            "selection_uses_model_predictions",
        }
        expected_provisioning_keys = {
            "source_split", "source_population", "strategy", "seed",
            "requested_items", "selected_items", "selected_class_count",
            "selection_manifest", "selection_manifest_sha256", "kernel_ref",
            "selection_uses_model_predictions",
        }
        if set(selection) != expected_selection_keys:
            raise ValueError(
                f"{label} calibration selection semantics shape differs: "
                f"missing={sorted(expected_selection_keys - set(selection))}, "
                f"unexpected={sorted(set(selection) - expected_selection_keys)}"
            )
        if set(provisioning) != expected_provisioning_keys:
            raise ValueError(
                f"{label} calibration provisioning semantics shape differs: "
                f"missing={sorted(expected_provisioning_keys - set(provisioning))}, "
                f"unexpected={sorted(set(provisioning) - expected_provisioning_keys)}"
            )
        _production_calibration_kernel_ref(
            provisioning.get("kernel_ref"), label=label,
        )
        selection_manifest = str(
            provisioning.get("selection_manifest") or ""
        ).strip()
        semantic_mismatches: list[str] = []
        exact_semantic_fields = (
            (selection, "selection.strategy", "strategy", "all"),
            (
                selection,
                "selection.selection_uses_model_predictions",
                "selection_uses_model_predictions",
                False,
            ),
            (provisioning, "source_split", "source_split", "train"),
            (
                provisioning,
                "source_population",
                "source_population",
                "ILSVRC2012 train",
            ),
            (
                provisioning,
                "strategy",
                "strategy",
                "class_stratified_deterministic_hash",
            ),
            (
                provisioning,
                "selection_uses_model_predictions",
                "selection_uses_model_predictions",
                False,
            ),
        )
        for owner, display_name, field, expected_value in exact_semantic_fields:
            if owner.get(field) != expected_value:
                semantic_mismatches.append(
                    f"{display_name}(expected={expected_value!r}, "
                    f"observed={owner.get(field)!r})"
                )
        integer_semantic_fields = (
            (
                selection,
                "selection.seed",
                "seed",
                TOP1_BOOTSTRAP_SEED,
            ),
            (
                selection,
                "selection.requested_max_items",
                "requested_max_items",
                0,
            ),
            (
                provisioning,
                "seed",
                "seed",
                TOP1_BOOTSTRAP_SEED,
            ),
            (
                provisioning,
                "requested_items",
                "requested_items",
                expected_count,
            ),
            (
                provisioning,
                "selected_items",
                "selected_items",
                expected_count,
            ),
            (
                provisioning,
                "selected_class_count",
                "selected_class_count",
                actual_class_count,
            ),
        )
        for owner, display_name, field, expected_value in integer_semantic_fields:
            observed_value = _integer(
                owner.get(field), label=f"{label} provisioning {display_name}",
            )
            if observed_value != expected_value:
                semantic_mismatches.append(
                    f"{display_name}(expected={expected_value!r}, "
                    f"observed={observed_value!r})"
                )
        if not selection_manifest or Path(selection_manifest).suffix.lower() != ".json":
            semantic_mismatches.append(
                "selection_manifest(expected=non-empty .json path, "
                f"observed={selection_manifest!r})"
            )
        _require_sha256(
            provisioning.get("selection_manifest_sha256"),
            label=f"{label} provisioning selection manifest",
        )
        if semantic_mismatches:
            raise ValueError(
                f"{label} calibration provisioning semantics are not frozen: "
                + "; ".join(semantic_mismatches)
            )
        _verify_provisioning_selection_manifest(
            dataset_manifest_path=path,
            provisioning=provisioning,
            manifest_items=items,
            expected_count=expected_count,
            label=label,
        )
    else:
        if (
            set(selection) != {
                "strategy", "seed", "requested_max_items",
                "selection_uses_model_predictions",
            }
            or str(selection.get("strategy") or "") != "class_stratified"
            or _integer(selection.get("seed"), label=f"{label} selection seed")
            != TOP1_BOOTSTRAP_SEED
            or _integer(
                selection.get("requested_max_items"),
                label=f"{label} requested_max_items",
            ) != expected_count
            or selection.get("selection_uses_model_predictions") is not False
        ):
            raise ValueError(f"{label} calibration selection semantics are not frozen")

    declared_root = Path(str(payload.get("root") or "")).expanduser()
    if declared_root.is_symlink() or not declared_root.is_dir():
        raise ValueError(f"{label} calibration manifest root is unavailable")
    root = declared_root.resolve()
    if payload.get("root_name") != root.name:
        raise ValueError(f"{label} calibration manifest root_name mismatch")
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    observed_paths: list[str] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"{label} calibration root contains a symlink")
        if not candidate.is_dir() and not candidate.is_file():
            raise ValueError(f"{label} calibration root contains a non-regular entry")
        if candidate.suffix.lower() not in allowed_extensions:
            continue
        if not candidate.is_file():
            raise ValueError(f"{label} calibration root contains a non-regular image")
        try:
            resolved = candidate.resolve(strict=True)
        except Exception as exc:
            raise ValueError(f"{label} calibration inventory is not resolvable") from exc
        if root not in resolved.parents:
            raise ValueError(f"{label} calibration inventory escapes its root")
        observed_paths.append(candidate.relative_to(root).as_posix())
    manifest_paths = sorted(str(item["relative_path"]) for item in items)
    if sorted(observed_paths) != manifest_paths:
        raise ValueError(f"{label} calibration root inventory differs from manifest items")

    contract_expectations = {
        "status": "resolved",
        "schema": "onnx-splitpoint/deepx-calibration-manifest-contract",
        "schema_version": 1,
        "task": "classification",
        "effective_count": expected_count,
        "item_count": expected_count,
        "root_inventory_count": expected_count,
        "manifest_file_sha256": _sha256_file(path),
        "manifest_payload_sha256": str(payload.get("manifest_payload_sha256") or ""),
        "items_identity_sha256": str(payload.get("items_identity_sha256") or ""),
    }
    for field, expected_value in contract_expectations.items():
        if contract.get(field) != expected_value:
            raise ValueError(f"{label} calibration contract {field} mismatch")
    expected_contract_keys = {
        "schema", "schema_version", "task", "effective_count", "status",
        "dataset_id", "split", "role", "hash_mode", "item_count",
        "items_identity_sha256", "manifest_payload_sha256",
        "manifest_file_sha256", "manifest_file_name", "dataset_root_name",
        "manifest_verification", "root_inventory_count",
        "root_inventory_sha256", "dataset_registry_binding_sha256",
        "identity_sha256",
    }
    if set(contract) != expected_contract_keys:
        raise ValueError(f"{label} calibration contract shape is not exact-v2")
    verification = contract.get("manifest_verification")
    expected_verification = {
        "ok": True,
        "schema_ok": True,
        "payload_hash_ok": True,
        "identity_hash_ok": True,
        "item_count_ok": True,
        "verification_mode": "full",
        "manifest_item_count": expected_count,
        "checked_item_count": expected_count,
        "missing_count": 0,
        "mismatch_count": 0,
    }
    unhashed_contract = dict(contract)
    unhashed_contract.pop("identity_sha256", None)
    if (
        contract.get("dataset_id") != "ilsvrc2012-train-calibration"
        or contract.get("split") != "train"
        or contract.get("role") != "calibration"
        or contract.get("hash_mode") != "content"
        or contract.get("manifest_file_name") != path.name
        or contract.get("dataset_root_name") != root.name
        or not isinstance(verification, Mapping)
        or dict(verification) != expected_verification
        or _require_sha256(
            contract.get("dataset_registry_binding_sha256"),
            label=f"{label} calibration registry binding",
        ) == ""
        or _require_sha256(
            contract.get("identity_sha256"),
            label=f"{label} calibration contract identity",
        ) != _canonical_sha256(unhashed_contract)
    ):
        raise ValueError(f"{label} calibration contract semantics are not frozen")
    inventory = sorted(
        (
            {
                "relative_path": item["relative_path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in items
        ),
        key=lambda row: row["relative_path"],
    )
    if _require_sha256(
        contract.get("root_inventory_sha256"), label=f"{label} calibration inventory",
    ) != _canonical_sha256(inventory):
        raise ValueError(f"{label} calibration inventory identity mismatch")
    return payload, items


def _find_dxcom_config(root: Path, expected_sha: str, *, label: str) -> Path:
    model_root = root / "models" / "resnet50"
    matches = [
        path for path in model_root.rglob("config_deepx.json")
        if path.is_file() and _sha256_file(path) == expected_sha
    ]
    return _single(matches, label=f"{label} sealed DX-COM config")


def _request_source_run_id(request: Mapping[str, Any]) -> str:
    producer = request.get("producer_identity")
    return str(
        request.get("source_run_id")
        or (producer.get("source_run_id") if isinstance(producer, Mapping) else "")
        or ""
    ).strip()


@dataclass(frozen=True)
class QualityEndpointEvidence:
    source_run_id: str
    backend: str
    setup_id: str
    request_path: Path
    candidate_path: Path
    request: dict[str, Any]
    producer: dict[str, Any]
    candidate: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    result: dict[str, Any]


def _classification_record_identity(
    records: Sequence[Mapping[str, Any]], *, label: str,
) -> tuple[list[str], str, str]:
    """Recompute the canonical classification dataset ID/GT hash domain."""

    image_ids: list[str] = []
    ground_truth: list[dict[str, Any]] = []
    for record in records:
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(f"{label} image_id must be a non-empty string")
        label_id = _integer(record.get("label_id"), label=f"{label} label_id")
        image_ids.append(image_id)
        ground_truth.append({"image_id": image_id, "label_id": label_id})
    if not image_ids or len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{label} image IDs are empty or duplicated")
    return (
        image_ids,
        image_ids_fingerprint(image_ids),
        json_fingerprint(sorted(ground_truth, key=lambda row: row["image_id"])),
    )


def _load_quality_endpoint(
    root: Path,
    *,
    source_run_id: str,
    expected_backend: str,
    expected_result_backend: str | None = None,
    expected_records: int,
    result: Mapping[str, Any],
) -> QualityEndpointEvidence:
    model_root = root / "models" / "resnet50"
    requests: list[Path] = []
    for path in model_root.rglob("full_request.json"):
        try:
            request = _read_object(path)
        except Exception:
            continue
        if _request_source_run_id(request) == source_run_id:
            requests.append(path)
    request_path = _single(requests, label=f"{source_run_id} Full quality request")
    request = _read_object(request_path)
    producer_raw = request.get("producer_identity")
    if not isinstance(producer_raw, Mapping):
        raise ValueError(f"{source_run_id} request producer identity is missing")
    producer = dict(producer_raw)
    if (
        request.get("schema") != "onnx-splitpoint/central-quality-evaluation-request"
        or _integer(
            request.get("schema_version"),
            label=f"{source_run_id} request schema_version",
        ) != 1
        or request.get("status") != "pending_central_evaluation"
        or request.get("execution_location") != "management_node"
        or request.get("producer_provenance_required") is not True
        or request.get("full_only_plan_identity_required") is not True
        or _integer(
            request.get("reference_record_count"),
            label=f"{source_run_id} reference record_count",
        ) != expected_records
    ):
        raise ValueError(f"{source_run_id} central request envelope is not frozen")
    request_producer_sha = _require_sha256(
        request.get("producer_identity_sha256"),
        label=f"{source_run_id} request producer identity",
    )
    declared_producer_sha = _require_sha256(
        producer.get("producer_identity_sha256"),
        label=f"{source_run_id} embedded producer identity",
    )
    unhashed_producer = dict(producer)
    unhashed_producer.pop("producer_identity_sha256", None)
    if (
        request_producer_sha != declared_producer_sha
        or _canonical_sha256(unhashed_producer) != declared_producer_sha
    ):
        raise ValueError(f"{source_run_id} producer identity self-hash mismatch")

    exact = {
        "schema": (
            "onnx-splitpoint/central-quality-producer-identity"
            if source_run_id == "deepx_m1_full"
            else "onnx-splitpoint/tensorrt-central-quality-producer-identity"
        ),
        "model_id": "resnet50",
        "source_run_id": source_run_id,
        "case_id": "full",
        "variant": "full",
        "task": "classification",
        "execution_role": "full_quality_only",
        "backend": expected_backend,
    }
    for field, expected in exact.items():
        observed = str(producer.get(field) or "").strip().lower()
        if observed != expected:
            raise ValueError(f"{source_run_id} is not exact Full-only quality evidence: {field}")
    if _integer(
        producer.get("schema_version"), label=f"{source_run_id} producer schema_version",
    ) != 1:
        raise ValueError(f"{source_run_id} producer schema version is not frozen")
    if (
        producer.get("performance_claims_emitted") is not False
        or request.get("performance_claims_emitted") is not False
    ):
        raise ValueError(f"{source_run_id} Full quality evidence emitted performance claims")
    setup_id = str(producer.get("setup_id") or "").strip()
    if setup_id != EXPECTED_SETUP_ID:
        raise ValueError(f"{source_run_id} setup identity is not frozen")
    request_exact = {
        "model_id": "resnet50",
        "source_run_id": source_run_id,
        "variant": "full",
        "task": "classification",
        "execution_role": "full_quality_only",
        "backend": expected_backend,
        "setup_id": setup_id,
    }
    for field, expected in request_exact.items():
        if str(request.get(field) or "").strip().lower() != expected:
            raise ValueError(f"{source_run_id} central request {field} mismatch")
    expected_request_case: str | None = (
        None if source_run_id == "deepx_m1_full" else "full"
    )
    if request.get("case_id") != expected_request_case:
        raise ValueError(f"{source_run_id} central request case_id mismatch")

    descriptor = request.get("candidate")
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{source_run_id} request candidate descriptor is missing")
    candidate_path = (
        request_path.parent / str(descriptor.get("path") or "")
    ).absolute()
    if (
        candidate_path.is_symlink()
        or candidate_path.parent.resolve() != request_path.parent.resolve()
        or not candidate_path.is_file()
    ):
        raise ValueError(f"{source_run_id} candidate escapes its quality evidence directory")
    if (
        _integer(
            descriptor.get("size_bytes"), label=f"{source_run_id} candidate bytes",
        ) != candidate_path.stat().st_size
        or _require_sha256(
            descriptor.get("sha256"), label=f"{source_run_id} candidate",
        ) != _sha256_file(candidate_path)
    ):
        raise ValueError(f"{source_run_id} candidate identity mismatch")
    candidate = _read_object(candidate_path)
    if (
        candidate.get("schema") != "onnx-splitpoint/task-quality-candidate-input"
        or _integer(
            candidate.get("schema_version"),
            label=f"{source_run_id} candidate schema_version",
        ) != 1
        or candidate.get("task") != "classification"
        or candidate.get("variant") != "full"
        or candidate.get("pairing_key") != "image_id"
        or not (
            candidate.get("producer_provenance_required") is True
            or candidate.get("provenance_required") is True
        )
    ):
        raise ValueError(f"{source_run_id} candidate envelope/provenance is invalid")
    candidate_producer = candidate.get("producer_identity")
    if not isinstance(candidate_producer, Mapping) or dict(candidate_producer) != producer:
        raise ValueError(f"{source_run_id} request/candidate producer identities differ")
    if _bare_sha(candidate.get("producer_identity_sha256")) != declared_producer_sha:
        raise ValueError(f"{source_run_id} candidate producer hash differs")
    if source_run_id == "deepx_m1_full":
        transforms = candidate.get("per_image_transforms")
        nested_prepared = producer.get("prepared_input_evidence")
        if (
            not isinstance(transforms, list)
            or len(transforms) != expected_records
            or not isinstance(nested_prepared, Mapping)
        ):
            raise ValueError("DeepX prepared-input transform evidence is incomplete")
        transforms_sha = json_fingerprint(transforms)
        prepared_digests = (
            request.get("prepared_input_evidence_sha256"),
            producer.get("prepared_input_evidence_sha256"),
            candidate.get("prepared_input_evidence_sha256"),
            nested_prepared.get("records_sha256"),
        )
        if any(
            _require_sha256(value, label="DeepX prepared-input transforms")
            != transforms_sha
            for value in prepared_digests
        ):
            raise ValueError("DeepX prepared-input transform identity differs")
    elif (
        candidate.get("per_image_transforms") is not None
        or candidate.get("prepared_input_evidence_sha256") not in {None, ""}
    ):
        raise ValueError("TensorRT candidate unexpectedly declares DeepX prepared input")
    records = tuple(
        dict(row) for row in list(candidate.get("records") or [])
        if isinstance(row, Mapping)
    )
    if (
        _integer(
            request.get("record_count"), label=f"{source_run_id} request record_count",
        ) != expected_records
        or _integer(
            candidate.get("record_count"), label=f"{source_run_id} candidate record_count",
        ) != expected_records
        or len(records) != expected_records
    ):
        raise ValueError(f"{source_run_id} requires exactly {expected_records} records")
    for record in records:
        prediction = record.get("candidate")
        if (
            not isinstance(prediction, Mapping)
            or not isinstance(prediction.get("top1_hit"), bool)
            or not isinstance(prediction.get("top5_hit"), bool)
        ):
            raise ValueError(f"{source_run_id} classification record is incomplete")
    ids, observed_ids_sha, observed_ground_truth_sha = (
        _classification_record_identity(records, label=source_run_id)
    )
    if list(request.get("expected_image_ids") or []) != ids:
        raise ValueError(f"{source_run_id} image cohort/order is invalid")
    producer_dataset = producer.get("dataset")
    if not isinstance(producer_dataset, Mapping):
        raise ValueError(f"{source_run_id} producer dataset identity is missing")
    if (
        _require_sha256(
            request.get("expected_image_ids_sha256"),
            label=f"{source_run_id} request image IDs",
        ) != observed_ids_sha
        or _require_sha256(
            producer_dataset.get("image_ids_sha256"),
            label=f"{source_run_id} producer image IDs",
        ) != observed_ids_sha
        or _require_sha256(
            producer_dataset.get("ground_truth_sha256"),
            label=f"{source_run_id} producer ground truth",
        ) != observed_ground_truth_sha
    ):
        raise ValueError(
            f"{source_run_id} records violate declared validation ID/ground-truth identity"
        )

    result_setup = str(result.get("source_setup_id") or result.get("setup_id") or "").strip()
    result_exact = {
        "source_run_id": source_run_id,
        "case_id": "full",
        "variant": "full",
        "task": "classification",
        "execution_role": "full_quality_only",
        "backend": expected_result_backend or expected_backend,
    }
    for field, expected in result_exact.items():
        if str(result.get(field) or "").strip().lower() != expected:
            raise ValueError(f"{source_run_id} central result {field} mismatch")
    if (
        result_setup != setup_id
        or result.get("performance_claims_emitted") is not False
        or _integer(result.get("n"), label=f"{source_run_id} result n")
        != expected_records
    ):
        raise ValueError(f"{source_run_id} central result is not setup-local Full evidence")
    if _bare_sha(result.get("source_request_sha256")) != _sha256_file(request_path):
        raise ValueError(f"{source_run_id} central result/request binding mismatch")
    return QualityEndpointEvidence(
        source_run_id=source_run_id,
        backend=expected_backend,
        setup_id=setup_id,
        request_path=request_path,
        candidate_path=candidate_path,
        request=request,
        producer=producer,
        candidate=candidate,
        records=records,
        result=dict(result),
    )


def _quality_summary(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary = _read_object(root / "quality_management" / "central_quality_summary.json")
    if (
        str(summary.get("schema") or "") != "onnx-splitpoint/central-quality-summary"
        or _integer(summary.get("schema_version"), label="quality summary schema_version") != 1
        or str(summary.get("status") or "") not in {"ok", "completed"}
        or _integer(summary.get("request_count"), label="quality request_count") != 2
        or _integer(summary.get("completed_count"), label="quality completed_count") != 2
        or _integer(
            summary.get("technical_completed_count"),
            label="quality technical_completed_count",
        ) != 2
        or _integer(summary.get("failed_count"), label="quality failed_count") != 0
        or _integer(
            _nested(summary, "merge", "unmatched_result_count"),
            label="quality unmatched_result_count",
        ) != 0
    ):
        raise ValueError("central quality summary is not an exact completed two-Full-endpoint run")
    raw_results = list(summary.get("results") or [])
    if len(raw_results) != 2 or not all(isinstance(row, Mapping) for row in raw_results):
        raise ValueError("central quality summary result cardinality/envelope is invalid")
    results = [dict(row) for row in raw_results]
    by_id: dict[str, dict[str, Any]] = {}
    for result in results:
        run_id = str(result.get("source_run_id") or "")
        if run_id in by_id:
            raise ValueError(f"duplicate central quality result: {run_id}")
        by_id[run_id] = result
    if set(by_id) != {"deepx_m1_full", "native_full_tensorrt"}:
        raise ValueError("only DeepX Full and setup-local TensorRT Full quality results are allowed")
    for run_id, result in by_id.items():
        if (
            str(result.get("schema") or "")
            != "onnx-splitpoint/management-paired-quality-result"
            or str(result.get("status") or "") != "completed"
            or str(result.get("technical_status") or "") != "completed"
            or str(result.get("decision") or "") not in {"pass", "inconclusive", "fail"}
            or str(result.get("algorithm_version") or "")
            != EXPECTED_QUALITY_ALGORITHM
        ):
            raise ValueError(f"central quality result is incomplete: {run_id}")
    return summary, by_id


@dataclass(frozen=True)
class CalibrationArmEvidence:
    label: str
    expected_calibration_count: int
    evaluation_run_id: str
    base: ArmEvidence
    profile: dict[str, Any]
    profile_source: dict[str, Any]
    profile_start_snapshot: dict[str, Any]
    cache_contract: dict[str, Any]
    calibration_contract: dict[str, Any]
    calibration_manifest_path: Path
    calibration_manifest: dict[str, Any]
    calibration_items: tuple[dict[str, Any], ...]
    dxcom_config_path: Path
    dxcom_config: dict[str, Any]
    build_options: dict[str, Any]
    artifact_status: str
    receipt_path: Path
    deepx_endpoint: QualityEndpointEvidence
    trt_endpoint: QualityEndpointEvidence


def load_calibration_arm(
    run_dir: str | Path,
    *,
    label: str,
    expected_calibration_count: int,
    calibration_manifest_path: str | Path | None = None,
) -> CalibrationArmEvidence:
    base = load_arm_evidence(
        run_dir,
        expected_mode=CLASSIFICATION_PREPROCESSING_IMAGENET,
        expected_records=VALIDATION_RECORD_COUNT,
    )
    root = base.run_dir
    profile = _load_profile(root)
    profile_source = _load_profile_source(root)
    profile_start_snapshot = _read_object(root / "profile_start_snapshot.json")
    run_manifest = _read_object(root / "run_manifest.json")
    expected_profile_name = (
        BASELINE_PROFILE_NAME if label == "B500" else CANDIDATE_PROFILE_NAME
    )
    _validate_run_envelope(
        root, run_manifest, label=label,
        expected_profile_name=expected_profile_name, profile=profile,
    )
    manifest_snapshot = run_manifest.get("profile_start_snapshot")
    if (
        not isinstance(manifest_snapshot, Mapping)
        or dict(manifest_snapshot) != profile_start_snapshot
    ):
        raise ValueError(f"{label} run manifest/profile start snapshot binding differs")
    _validate_resolved_profile(
        profile, label=label, expected_count=expected_calibration_count,
    )
    expected_profile_name = (
        BASELINE_PROFILE_NAME if label == "B500" else CANDIDATE_PROFILE_NAME
    )
    if base.profile_name != expected_profile_name:
        raise ValueError(f"{label} base receipt/profile identity differs")
    status_path = root / "models/resnet50/benchmark_set/deepx/deepx_artifact_status.json"
    status = _read_object(status_path)
    raw_contract = status.get("cache_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError(f"{label} exact-v2 cache contract is missing")
    cache_contract = dict(raw_contract)
    calibration_raw = cache_contract.get("calibration_manifest_contract")
    build_options_raw = cache_contract.get("build_options")
    if not isinstance(calibration_raw, Mapping) or not isinstance(build_options_raw, Mapping):
        raise ValueError(f"{label} calibration/build contracts are missing")
    calibration_contract = dict(calibration_raw)
    build_options = dict(build_options_raw)
    if set(build_options) != {"calibration_method", "calibration_count", "opt_level"}:
        raise ValueError(f"{label} DeepX build-options contract has unexpected fields")
    if (
        build_options.get("calibration_method") != "ema"
        or _integer(
            build_options.get("calibration_count"),
            label=f"{label} build calibration_count",
        ) != expected_calibration_count
        or _integer(
            build_options.get("opt_level"), label=f"{label} opt-level",
        ) != 0
    ):
        raise ValueError(f"{label} must use EMA, opt-level 0 and the exact calibration count")

    manifest_path = _resolve_manifest_path(
        root=root,
        profile=profile,
        contract=calibration_contract,
        explicit_path=calibration_manifest_path,
        label=label,
    )
    manifest, items = _load_manifest(
        manifest_path,
        contract=calibration_contract,
        expected_count=expected_calibration_count,
        label=label,
    )
    if label == "B500":
        _validate_frozen_b500_manifest_authority(manifest_path, manifest)
    if (
        base.calibration_identity_sha256
        != _require_sha256(
            calibration_contract.get("identity_sha256"),
            label=f"{label} calibration identity",
        )
        or base.calibration_contract_sha256 != _canonical_sha256(calibration_contract)
        or base.calibration_items_identity_sha256
        != _require_sha256(
            calibration_contract.get("items_identity_sha256"),
            label=f"{label} calibration items",
        )
    ):
        raise ValueError(f"{label} base/calibration manifest contract binding differs")
    dxcom_path = _find_dxcom_config(
        root, base.dxcom_config_sha256, label=label,
    )
    dxcom = _read_object(dxcom_path)
    if (
        _integer(
            dxcom.get("calibration_num"), label=f"{label} DX-COM calibration_num",
        ) != expected_calibration_count
        or dxcom.get("calibration_method") != "ema"
    ):
        raise ValueError(f"{label} DX-COM calibration count/method mismatch")
    dataset_path = str(_nested(dxcom, "default_loader", "dataset_path") or "").strip()
    manifest_root = str(manifest.get("root") or "").strip()
    if not dataset_path or not manifest_root or Path(dataset_path).expanduser().resolve() != Path(
        manifest_root
    ).expanduser().resolve():
        raise ValueError(f"{label} DX-COM dataset path differs from its sealed manifest root")

    artifact_rows = [
        dict(row) for row in list(status.get("artifacts") or [])
        if isinstance(row, Mapping)
    ]
    if len(artifact_rows) != 1 or artifact_rows[0].get("variant") != "full":
        raise ValueError(f"{label} requires exactly one Full DXNN artifact row")
    artifact_row = artifact_rows[0]
    artifact_status = str(artifact_row.get("status") or "")
    if artifact_status not in {"ready_built", "ready_reused"}:
        raise ValueError(f"{label} Full DXNN artifact status is not exact built/reused evidence")
    if (
        status.get("build_status") != artifact_status
        or list(status.get("queue") or [])
        or artifact_row.get("backend") != "deepx_m1"
        or str(artifact_row.get("cache_key") or "") != base.cache_key
        or _require_sha256(
            artifact_row.get("cache_contract_sha256"),
            label=f"{label} artifact cache contract",
        ) != base.cache_contract_sha256
    ):
        raise ValueError(f"{label} artifact status contains fallback/conflicting build evidence")
    artifact = _single(
        [Path(str(artifact_row.get("dxnn_path") or "")).expanduser()],
        label=f"{label} Full DXNN artifact",
    )
    receipt_path = artifact.parent / "build_manifest.json"
    receipt = _read_object(receipt_path)
    if (
        str(receipt.get("schema") or "")
        != "onnx-splitpoint/deepx-full-cache-receipt"
        or _integer(
            receipt.get("schema_version"), label=f"{label} receipt schema_version",
        ) != 2
        or receipt.get("cache_contract") != cache_contract
        or str(receipt.get("cache_key") or "") != base.cache_key
    ):
        raise ValueError(f"{label} exact-v2 cache receipt is invalid")

    _, quality_results = _quality_summary(root)
    deepx_endpoint = _load_quality_endpoint(
        root,
        source_run_id="deepx_m1_full",
        expected_backend="deepx_m1",
        expected_records=VALIDATION_RECORD_COUNT,
        result=quality_results["deepx_m1_full"],
    )
    trt_endpoint = _load_quality_endpoint(
        root,
        source_run_id="native_full_tensorrt",
        expected_backend="native_tensorrt",
        expected_result_backend="tensorrt",
        expected_records=VALIDATION_RECORD_COUNT,
        result=quality_results["native_full_tensorrt"],
    )
    if deepx_endpoint.setup_id != trt_endpoint.setup_id:
        raise ValueError(f"{label} TensorRT control is not setup-local to DeepX")
    if label == "B1000":
        actual_namespace = Path(base.cache_dir).expanduser().resolve()
        expected_namespace = Path(B1000_CACHE_NAMESPACE).expanduser().resolve()
        if actual_namespace != expected_namespace:
            raise ValueError(
                "B1000 cache root is not the exact v2.75.41 calibration-size namespace"
            )
    return CalibrationArmEvidence(
        label=label,
        expected_calibration_count=expected_calibration_count,
        evaluation_run_id=str(run_manifest["run_id"]),
        base=base,
        profile=profile,
        profile_source=profile_source,
        profile_start_snapshot=profile_start_snapshot,
        cache_contract=cache_contract,
        calibration_contract=calibration_contract,
        calibration_manifest_path=manifest_path,
        calibration_manifest=manifest,
        calibration_items=items,
        dxcom_config_path=dxcom_path,
        dxcom_config=dxcom,
        build_options=build_options,
        artifact_status=artifact_status,
        receipt_path=receipt_path,
        deepx_endpoint=deepx_endpoint,
        trt_endpoint=trt_endpoint,
    )


def _without_paths(value: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    for path in paths:
        cursor: Any = result
        for component in path[:-1]:
            if not isinstance(cursor, dict):
                break
            cursor = cursor.get(component)
        if isinstance(cursor, dict):
            cursor.pop(path[-1], None)
    return result


def _normalized_profile_for_pair(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the reviewed B500→B1000 profile deltas."""

    return _without_paths(profile, (
        ("name",),
        ("purpose",),
        ("campaign", "id"),
        ("campaign", "dataset_registry_sha256"),
        ("campaign", "dataset_registry_binding_sha256"),
        ("campaign", "dataset_manifests", "classification", "calibration"),
        ("execution_preset", "label"),
        ("execution_preset", "snapshot_sha256"),
        ("execution_preset", "resolved_at"),
        ("execution_preset", "snapshot", "label"),
        ("execution_preset", "snapshot", "description"),
        ("execution_preset", "snapshot", "recommended_for"),
        (
            "execution_preset", "snapshot", "data", "calibration_items",
            "classification",
        ),
        (
            "execution_preset", "snapshot", "build", "deepx",
            "calibration_items",
        ),
        ("execution_preset", "snapshot", "build", "deepx", "cache_dir"),
        ("deepx_build", "calib_count"),
        ("deepx_build", "cache_dir"),
        ("hailo_build", "calib_count"),
        (
            "execution_preset", "effective", "calibration_items",
            "classification",
        ),
    ))


def _record_identity(records: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[int]]:
    return (
        [str(row.get("image_id") or "") for row in records],
        [int(row.get("label_id")) for row in records],
    )


def _validate_profile_provenance(arm: CalibrationArmEvidence) -> None:
    """Bind runtime profile bytes to the exact source and immutable start snapshot."""

    label = arm.label
    expected_name = BASELINE_PROFILE_NAME if label == "B500" else CANDIDATE_PROFILE_NAME
    _, frozen_source = _frozen_profile_source(label)
    if arm.profile_source != frozen_source:
        raise ValueError(f"{label} profile_source.yaml differs from frozen source")
    snapshot = arm.profile_start_snapshot
    consistency = snapshot.get("consistency")
    if (
        snapshot.get("schema") != "onnx-splitpoint/evaluation-start-snapshot"
        or _integer(
            snapshot.get("schema_version"),
            label=f"{label} profile start snapshot schema_version",
        ) != 2
        or snapshot.get("profile_id") != expected_name
        or not isinstance(consistency, Mapping)
        or dict(consistency) != {"status": "ok", "mismatches": []}
    ):
        raise ValueError(f"{label} immutable profile start snapshot is not exact/complete")
    if (
        _require_sha256(
            snapshot.get("source_profile_sha256"),
            label=f"{label} source profile",
        ) != _canonical_sha256(arm.profile_source)
        or _require_sha256(
            snapshot.get("resolved_profile_sha256"),
            label=f"{label} resolved profile",
        ) != _canonical_sha256(arm.profile)
    ):
        raise ValueError(f"{label} source/resolved profile snapshot binding differs")
    expanded_snapshot = copy.deepcopy(snapshot)
    expanded_snapshot["source_profile"] = copy.deepcopy(arm.profile_source)
    expanded_snapshot["resolved_profile"] = copy.deepcopy(arm.profile)
    try:
        from onnx_splitpoint_tool.execution_plan import (
            build_effective_execution_plan,
        )
        from onnx_splitpoint_tool.workflow.start_snapshot import (
            validate_profile_start_snapshot,
        )

        expanded_snapshot["effective_execution_plan"] = (
            build_effective_execution_plan(arm.profile)
        )
        validate_profile_start_snapshot(expanded_snapshot)
    except Exception as exc:
        raise ValueError(f"{label} immutable profile start snapshot self-check failed") from exc

    bindings = snapshot.get("runtime_bindings")
    hardware = arm.profile.get("hardware")
    if not isinstance(bindings, Mapping) or bindings.get("runtime_materialized") is not True:
        raise ValueError(f"{label} runtime profile was not materialized at start")
    if not isinstance(hardware, Mapping):
        raise ValueError(f"{label} runtime hardware binding is missing")
    targets = hardware.get("resolved_targets")
    binding_targets = bindings.get("hardware_targets")
    if (
        hardware.get("resolution_frozen_at_start") is not True
        or not isinstance(targets, list)
        or not isinstance(binding_targets, list)
        or targets != binding_targets
        or _require_sha256(
            hardware.get("resolved_targets_sha256"),
            label=f"{label} hardware targets",
        ) != _canonical_sha256(targets)
        or _require_sha256(
            bindings.get("hardware_targets_sha256"),
            label=f"{label} snapshot hardware targets",
        ) != _canonical_sha256(targets)
    ):
        raise ValueError(f"{label} frozen runtime hardware target binding differs")
    enabled_targets = [
        row for row in targets
        if isinstance(row, Mapping) and row.get("enabled") is True
    ]
    if (
        len(enabled_targets) != 1
        or enabled_targets[0].get("id") != EXPECTED_SETUP_ID
        or enabled_targets[0].get("accelerator") != "deepx_m1"
    ):
        raise ValueError(f"{label} runtime hardware target is not the frozen DeepX setup")


def _hits(records: Sequence[Mapping[str, Any]], metric: str) -> np.ndarray:
    return np.asarray([
        int(bool(dict(row.get("candidate") or {}).get(metric))) for row in records
    ], dtype=np.int8)


def _paired_bootstrap(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    if reference.shape != candidate.shape or reference.ndim != 1 or not reference.size:
        raise ValueError("paired bootstrap requires non-empty equal one-dimensional vectors")
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, delta.size, size=(BOOTSTRAP_REPETITIONS, delta.size),
    )
    distribution = delta[indices].mean(axis=1)
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return ({
        "sample_count": int(delta.size),
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": seed,
        "confidence": CONFIDENCE_LEVEL,
        "point_delta_candidate_minus_reference": float(delta.mean()),
        "ci_low": float(np.quantile(distribution, tail)),
        "ci_high": float(np.quantile(distribution, 1.0 - tail)),
        "lower_one_sided_bound": float(
            np.quantile(distribution, 1.0 - CONFIDENCE_LEVEL)
        ),
        "reference_hits": int(reference.sum()),
        "candidate_hits": int(candidate.sum()),
        "corrected_by_candidate": int(np.sum((reference == 0) & (candidate == 1))),
        "regressed_by_candidate": int(np.sum((reference == 1) & (candidate == 0))),
    }, distribution)


def _quality_component(result: Mapping[str, Any], *, top5: bool) -> dict[str, Any]:
    raw = (
        _nested(result, "guardrails", "top5_accuracy") if top5
        else result.get("primary")
    )
    if not isinstance(raw, Mapping):
        raise ValueError("classification quality result component is missing")
    component = dict(raw)
    expected_metric = "top5_accuracy" if top5 else "top1_accuracy"
    if (
        str(component.get("metric") or "") != expected_metric
        or str(component.get("decision") or "") not in {"pass", "inconclusive", "fail"}
        or str(component.get("status") or component.get("decision") or "")
        != str(component.get("decision") or "")
        or _integer(
            component.get("sample_count")
            if component.get("sample_count") is not None else component.get("n"),
            label=f"classification {expected_metric} sample_count",
        ) != VALIDATION_RECORD_COUNT
    ):
        raise ValueError(f"classification {expected_metric} quality component is invalid")
    margin = float(component.get("margin"))
    if not math.isfinite(margin) or margin != QUALITY_MARGIN:
        raise ValueError(f"classification {expected_metric} margin is invalid")
    if _integer(
        component.get("bootstrap_repetitions_requested"),
        label=f"classification {expected_metric} requested bootstrap repetitions",
    ) != BOOTSTRAP_REPETITIONS:
        raise ValueError(
            f"classification {expected_metric} bootstrap contract is invalid"
        )
    return component


def _quality_policy_contract(endpoint: QualityEndpointEvidence) -> dict[str, Any]:
    request = endpoint.request
    policy_sha = _require_sha256(
        request.get("policy_sha256"),
        label=f"{endpoint.source_run_id} quality policy",
    )
    if policy_sha != EXPECTED_QUALITY_POLICY_SHA256:
        raise ValueError(
            f"{endpoint.source_run_id} does not use the frozen v2.75.40 quality policy"
        )
    statistics = request.get("statistics")
    gate = request.get("metric_gate_config")
    if not isinstance(statistics, Mapping) or not isinstance(gate, Mapping):
        raise ValueError(f"{endpoint.source_run_id} quality policy details are missing")
    expected_statistics = {
        "method": "paired_bootstrap",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "seed": TOP1_BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "decision": "lower_one_sided_bound",
    }
    for field, expected in expected_statistics.items():
        observed = statistics.get(field)
        if isinstance(expected, float):
            try:
                observed = float(observed)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{endpoint.source_run_id} quality statistics {field} is invalid"
                ) from exc
        elif isinstance(expected, int):
            observed = _integer(
                observed,
                label=f"{endpoint.source_run_id} quality statistics {field}",
            )
        if observed != expected:
            raise ValueError(
                f"{endpoint.source_run_id} quality statistics {field} changed"
            )
    expected_gate = {
        "primary_metric": "top1_accuracy",
        "non_inferiority_margin": QUALITY_MARGIN,
        "guardrails": {"top5_accuracy_margin": QUALITY_MARGIN},
    }
    if dict(gate) != expected_gate:
        raise ValueError(f"{endpoint.source_run_id} quality margins/metrics changed")
    return {
        "policy_sha256": policy_sha,
        "statistics": dict(statistics),
        "metric_gate_config": dict(gate),
        "central_guardrail_seed": TOP1_BOOTSTRAP_SEED,
        "external_b1000_b500_top5_seed": TOP5_BOOTSTRAP_SEED,
    }


def _validate_endpoint_scientific_binding(
    arm: CalibrationArmEvidence,
    endpoint: QualityEndpointEvidence,
) -> dict[str, Any]:
    producer = endpoint.producer
    if (
        endpoint.setup_id != EXPECTED_SETUP_ID
        or producer.get("setup_id") != EXPECTED_SETUP_ID
        or endpoint.request.get("setup_id") != EXPECTED_SETUP_ID
        or (
            endpoint.result.get("source_setup_id")
            or endpoint.result.get("setup_id")
        ) != EXPECTED_SETUP_ID
    ):
        raise ValueError(f"{endpoint.source_run_id} setup identity is not frozen")
    expected_producer_schema = (
        "onnx-splitpoint/central-quality-producer-identity"
        if endpoint.source_run_id == "deepx_m1_full"
        else "onnx-splitpoint/tensorrt-central-quality-producer-identity"
    )
    if (
        producer.get("schema") != expected_producer_schema
        or _integer(
            producer.get("schema_version"),
            label=f"{endpoint.source_run_id} producer schema_version",
        ) != 1
    ):
        raise ValueError(f"{endpoint.source_run_id} producer schema is not frozen")
    dataset = producer.get("dataset")
    model = producer.get("model")
    if not isinstance(dataset, Mapping) or not isinstance(model, Mapping):
        raise ValueError(f"{endpoint.source_run_id} producer model/dataset binding is missing")
    dataset_expected = {
        "manifest_sha256": arm.base.validation_manifest_sha256,
        "image_ids_sha256": arm.base.validation_image_ids_sha256,
        "ground_truth_sha256": arm.base.validation_ground_truth_sha256,
    }
    for field, expected in dataset_expected.items():
        if _require_sha256(
            dataset.get(field), label=f"{endpoint.source_run_id} dataset {field}",
        ) != expected:
            raise ValueError(f"{endpoint.source_run_id} validation {field} differs")
    try:
        _, records_ids_sha, records_ground_truth_sha = (
            _classification_record_identity(
                endpoint.records, label=endpoint.source_run_id,
            )
        )
    except Exception as exc:
        raise ValueError(
            f"{endpoint.source_run_id} validation record identity is invalid"
        ) from exc
    if (
        records_ids_sha != arm.base.validation_image_ids_sha256
        or records_ground_truth_sha != arm.base.validation_ground_truth_sha256
    ):
        raise ValueError(
            f"{endpoint.source_run_id} records violate frozen validation ID/GT authority"
        )
    if _integer(
        dataset.get("image_count"), label=f"{endpoint.source_run_id} dataset image_count",
    ) != VALIDATION_RECORD_COUNT:
        raise ValueError(f"{endpoint.source_run_id} validation image count differs")
    if _require_sha256(
        model.get("source_onnx_sha256"),
        label=f"{endpoint.source_run_id} source ONNX",
    ) != arm.base.source_onnx_sha256:
        raise ValueError(f"{endpoint.source_run_id} source ONNX differs")
    if _require_sha256(
        endpoint.request.get("expected_image_ids_sha256"),
        label=f"{endpoint.source_run_id} request image IDs",
    ) != arm.base.validation_image_ids_sha256:
        raise ValueError(f"{endpoint.source_run_id} request validation image IDs differ")
    request_preprocessing = _require_sha256(
        endpoint.request.get("preprocessing_contract_sha256"),
        label=f"{endpoint.source_run_id} preprocessing contract",
    )
    producer_preprocessing = _require_sha256(
        producer.get("preprocessing_contract_sha256"),
        label=f"{endpoint.source_run_id} producer preprocessing contract",
    )
    if (
        request_preprocessing != arm.base.preprocessing_contract_sha256
        or producer_preprocessing != arm.base.preprocessing_contract_sha256
    ):
        raise ValueError(f"{endpoint.source_run_id} preprocessing contract differs")
    if endpoint.source_run_id == "deepx_m1_full":
        transforms = endpoint.candidate.get("per_image_transforms")
        nested_prepared = producer.get("prepared_input_evidence")
        if (
            not isinstance(transforms, list)
            or len(transforms) != VALIDATION_RECORD_COUNT
            or not isinstance(nested_prepared, Mapping)
        ):
            raise ValueError("DeepX prepared-input transform evidence is incomplete")
        transforms_sha = json_fingerprint(transforms)
        prepared = _require_sha256(
            endpoint.request.get("prepared_input_evidence_sha256"),
            label="DeepX prepared-input evidence",
        )
        producer_prepared = _require_sha256(
            producer.get("prepared_input_evidence_sha256"),
            label="DeepX producer prepared-input evidence",
        )
        if (
            prepared != arm.base.prepared_input_evidence_sha256
            or producer_prepared != arm.base.prepared_input_evidence_sha256
            or _require_sha256(
                endpoint.candidate.get("prepared_input_evidence_sha256"),
                label="DeepX candidate prepared-input evidence",
            ) != arm.base.prepared_input_evidence_sha256
            or _require_sha256(
                nested_prepared.get("records_sha256"),
                label="DeepX nested prepared-input evidence",
            ) != arm.base.prepared_input_evidence_sha256
            or transforms_sha != arm.base.prepared_input_evidence_sha256
        ):
            raise ValueError("DeepX prepared-input transform identity differs")
        if _require_sha256(
            model.get("runtime_artifact_sha256"),
            label="DeepX runtime artifact",
        ) != arm.base.dxnn_sha256:
            raise ValueError("DeepX quality producer uses a different DXNN")
    else:
        # TensorRT intentionally consumes the source ONNX with an explicit
        # float input encoding; DeepX consumes the adapted build ONNX and its
        # sealed uint8 prepared-input set.  Freeze that documented mapping
        # instead of pretending the two runtime encodings are byte-identical.
        if (
            endpoint.request.get("prepared_input_evidence_sha256") not in {None, ""}
            or producer.get("prepared_input_evidence_sha256") not in {None, ""}
            or endpoint.candidate.get("prepared_input_evidence_sha256") not in {None, ""}
            or endpoint.candidate.get("per_image_transforms") is not None
        ):
            raise ValueError("TensorRT control unexpectedly declares DeepX prepared input")
        source_sha = arm.base.source_onnx_sha256
        trt_model_fields = (
            model.get("build_onnx_sha256"),
            endpoint.request.get("build_onnx_sha256"),
            endpoint.request.get("source_model_sha256"),
            _nested(producer, "build_onnx", "sha256"),
            _nested(producer, "build_onnx", "source_onnx_sha256"),
            _nested(producer, "engine", "build_onnx_sha256"),
            _nested(producer, "engine", "source_onnx_sha256"),
        )
        if any(
            _require_sha256(value, label="TensorRT source/build ONNX") != source_sha
            for value in trt_model_fields
        ):
            raise ValueError("TensorRT source/build ONNX mapping differs")

    policy = _quality_policy_contract(endpoint)
    if policy["policy_sha256"] != arm.base.policy_sha256:
        raise ValueError(f"{endpoint.source_run_id} policy differs from run authority")
    producer_policy = producer.get("policy_sha256")
    if endpoint.source_run_id == "native_full_tensorrt" and not producer_policy:
        raise ValueError("TensorRT producer quality policy identity is missing")
    if producer_policy and _require_sha256(
        producer_policy, label=f"{endpoint.source_run_id} producer policy",
    ) != policy["policy_sha256"]:
        raise ValueError(f"{endpoint.source_run_id} producer/request policy differs")
    _validate_quality_execution_contract(
        endpoint, expected_eval_run_id=arm.evaluation_run_id,
    )
    return policy


def _sealed_execution_component(
    producer: Mapping[str, Any], name: str, *, role: str,
    allow_runtime_input_encoding: bool = False,
) -> tuple[dict[str, Any], str]:
    component = producer.get(name)
    expected_keys = {"identity", "sha256"}
    if allow_runtime_input_encoding:
        expected_keys.update({
            "runtime_input_encoding", "runtime_input_encoding_sha256",
        })
    if (
        not isinstance(component, Mapping)
        or set(component) != expected_keys
        or not isinstance(component.get("identity"), Mapping)
    ):
        raise ValueError(f"{role} {name} execution component is incomplete")
    identity = dict(component["identity"])
    declared = _require_sha256(
        component.get("sha256"), label=f"{role} {name} execution component",
    )
    if _canonical_sha256(identity) != declared:
        raise ValueError(f"{role} {name} execution component hash mismatch")
    if allow_runtime_input_encoding:
        runtime_encoding = component.get("runtime_input_encoding")
        if (
            not isinstance(runtime_encoding, Mapping)
            or set(runtime_encoding) != {
                "schema", "schema_version", "image_scale", "input_dtype",
            }
            or runtime_encoding.get("schema")
            != "onnx-splitpoint/model-input-encoding-contract"
            or _integer(
                runtime_encoding.get("schema_version"),
                label=f"{role} runtime input encoding schema_version",
            ) != 1
            or not str(runtime_encoding.get("image_scale") or "").strip()
            or not str(runtime_encoding.get("input_dtype") or "").strip()
            or _require_sha256(
                component.get("runtime_input_encoding_sha256"),
                label=f"{role} runtime input encoding",
            ) != _canonical_sha256(runtime_encoding)
        ):
            raise ValueError(f"{role} runtime input encoding is invalid")
    return identity, declared


def _validate_full_only_plan_binding(
    endpoint: QualityEndpointEvidence, *, expected_eval_run_id: str,
) -> None:
    if not isinstance(expected_eval_run_id, str) or not expected_eval_run_id.strip():
        raise ValueError("EvaluationRun ID authority is missing")
    producer = endpoint.producer
    role = endpoint.source_run_id
    logical_backend = "deepx_m1" if role == "deepx_m1_full" else "tensorrt"
    expected_canary_id = (
        "deepx_m1_full"
        if role == "deepx_m1_full"
        else "tensorrt_at_deepx_m1_full"
    )
    expected_identity = {
        "schema": "onnx-splitpoint/full-only-quality-request-identity",
        "schema_version": 1,
        "model_id": "resnet50",
        "setup_id": EXPECTED_SETUP_ID,
        "source_run_id": role,
        "backend": logical_backend,
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    authorities: list[tuple[str, Mapping[str, Any]]] = [
        ("request", endpoint.request), ("candidate", endpoint.candidate),
    ]
    if role == "deepx_m1_full":
        authorities.append(("producer", producer))
    observed_identity: dict[str, Any] | None = None
    for owner_name, owner in authorities:
        identity = owner.get("full_only_plan_identity")
        if (
            owner.get("full_only_plan_identity_required") is not True
            or not isinstance(identity, Mapping)
            or set(identity) != {
                "schema", "schema_version", "quality_canary_id", "eval_run_id",
                "model_id", "setup_id", "source_run_id", "backend", "variant",
                "execution_role", "performance_claims_emitted",
            }
        ):
            raise ValueError(f"{role} {owner_name} Full-only plan identity is incomplete")
        identity = dict(identity)
        if (
            any(identity.get(field) != value for field, value in expected_identity.items())
            or identity.get("quality_canary_id") != expected_canary_id
            or identity.get("eval_run_id") != expected_eval_run_id
            or _require_sha256(
                owner.get("full_only_plan_identity_sha256"),
                label=f"{role} {owner_name} Full-only plan identity",
            ) != _canonical_sha256(identity)
            or owner.get("quality_canary_id") != identity["quality_canary_id"]
        ):
            raise ValueError(f"{role} {owner_name} Full-only plan identity differs")
        if observed_identity is None:
            observed_identity = identity
        elif identity != observed_identity:
            raise ValueError(f"{role} request/candidate Full-only plan identities differ")
        for field in (
            "eval_run_id", "model_id", "setup_id", "source_run_id",
            "variant", "execution_role", "performance_claims_emitted",
        ):
            if owner.get(field) != identity[field]:
                raise ValueError(f"{role} {owner_name} Full-only {field} mirror differs")
        expected_outer_backend = (
            producer.get("backend") if owner_name != "producer"
            else "deepx_m1"
        )
        if owner.get("backend") != expected_outer_backend:
            raise ValueError(f"{role} {owner_name} Full-only backend mirror differs")


def _validate_quality_execution_contract(
    endpoint: QualityEndpointEvidence,
    *,
    expected_eval_run_id: str,
) -> None:
    """Revalidate producer execution semantics, not only its outer self-hash."""

    producer = endpoint.producer
    role = endpoint.source_run_id
    components = {
        name: _sealed_execution_component(
            producer, name, role=role,
            allow_runtime_input_encoding=(
                role == "native_full_tensorrt" and name == "preprocessing"
            ),
        )
        for name in (
            "preprocessing", "endpoint", "quality_record_endpoint", "precision",
        )
    }
    quality_contract = producer.get("quality_contract")
    if not isinstance(quality_contract, Mapping):
        raise ValueError(f"{role} quality execution contract is missing")
    try:
        _validate_candidate_execution_contract(
            producer, role=role, task="classification",
        )
        quality_sha = _require_sha256(
            quality_contract.get("quality_contract_sha256"),
            label=f"{role} quality contract",
        )
    except Exception as exc:
        raise ValueError(f"{role} producer execution contract is invalid: {exc}") from exc

    preprocessing_identity, preprocessing_sha = components["preprocessing"]
    endpoint_identity, endpoint_sha = components["endpoint"]
    quality_endpoint_identity, quality_endpoint_sha = components[
        "quality_record_endpoint"
    ]
    if (
        _require_sha256(
            producer.get("quality_contract_sha256"),
            label=f"{role} producer quality contract",
        ) != quality_sha
        or _require_sha256(
            producer.get("preprocessing_contract_sha256"),
            label=f"{role} producer preprocessing contract",
        ) != preprocessing_sha
        or _require_sha256(
            producer.get("endpoint_contract_hash"),
            label=f"{role} producer runtime endpoint",
        ) != endpoint_sha
        or _require_sha256(
            producer.get("quality_record_endpoint_contract_sha256"),
            label=f"{role} producer quality-record endpoint",
        ) != quality_endpoint_sha
        or endpoint_identity.get("schema")
        != "onnx-splitpoint/output-endpoint-contract"
        or _integer(
            endpoint_identity.get("schema_version"),
            label=f"{role} endpoint schema_version",
        ) != 3
        or endpoint_identity.get("task") != "classification"
        or endpoint_identity.get("stage") not in {
            "classification_logits", "classification_probabilities",
        }
        or not isinstance(endpoint_identity.get("tensor_signature"), Mapping)
        or quality_endpoint_identity.get("canonical_record_endpoint")
        != "classification_topk_hits"
    ):
        raise ValueError(f"{role} producer execution component bindings differ")
    runner_sha = _require_sha256(
        producer.get("implementation_runner_sha256"),
        label=f"{role} implementation runner",
    )
    if _require_sha256(
        quality_endpoint_identity.get("implementation_runner_sha256"),
        label=f"{role} quality-record implementation runner",
    ) != runner_sha:
        raise ValueError(f"{role} quality-record endpoint runner differs")
    quality_preprocessing = quality_contract.get("preprocessing")
    quality_endpoint = quality_contract.get("quality_record_endpoint")
    quality_dataset = quality_contract.get("dataset")
    quality_model = quality_contract.get("model")
    precision_identity = components["precision"][0]
    quality_model_sha = (
        quality_model.get("source_onnx_sha256")
        if endpoint.source_run_id == "deepx_m1_full"
        else quality_model.get("sha256")
    ) if isinstance(quality_model, Mapping) else None
    if (
        not isinstance(quality_preprocessing, Mapping)
        or dict(quality_preprocessing) != dict(producer["preprocessing"])
        or not isinstance(quality_endpoint, Mapping)
        or dict(quality_endpoint) != dict(producer["quality_record_endpoint"])
        or not isinstance(quality_dataset, Mapping)
        or any(
            quality_dataset.get(field) != producer["dataset"].get(field)
            for field in (
                "manifest_sha256", "image_ids_sha256",
                "ground_truth_sha256", "image_count",
            )
        )
        or _require_sha256(
            quality_model_sha, label=f"{role} nested quality model",
        ) != _require_sha256(
            producer["model"].get("source_onnx_sha256"),
            label=f"{role} producer source model",
        )
        or preprocessing_identity.get("task") != "classification"
    ):
        raise ValueError(f"{role} nested quality execution contract differs")
    expected_precision_schema = (
        "onnx-splitpoint/deepx-runtime-precision-contract"
        if role == "deepx_m1_full"
        else "onnx-splitpoint/tensorrt-runtime-precision-contract"
    )
    if (
        precision_identity.get("schema") != expected_precision_schema
        or _integer(
            precision_identity.get("schema_version"),
            label=f"{role} precision schema_version",
        ) != 1
    ):
        raise ValueError(f"{role} precision execution contract differs")

    for owner_name, owner in (
        ("request", endpoint.request), ("candidate", endpoint.candidate),
    ):
        expected_mirrors = {
            "quality_contract_sha256": quality_sha,
            "preprocessing_contract_sha256": preprocessing_sha,
            "quality_record_endpoint_contract_sha256": quality_endpoint_sha,
        }
        if owner_name == "request" or role == "native_full_tensorrt":
            expected_mirrors["endpoint_contract_hash"] = endpoint_sha
        for field, expected in expected_mirrors.items():
            if _require_sha256(
                owner.get(field), label=f"{role} {owner_name} {field}",
            ) != expected:
                raise ValueError(f"{role} {owner_name} execution contract differs")
        if (
            owner.get("runtime_precision_identity")
            != producer.get("runtime_precision_identity")
            or not isinstance(owner.get("quality_contract"), Mapping)
            or dict(owner["quality_contract"]) != dict(quality_contract)
        ):
            raise ValueError(f"{role} {owner_name} execution contract differs")
        for field in ("decoder_contract_sha256", "nms_contract_sha256"):
            if field not in owner or str(owner.get(field) or "") != str(
                producer.get(field) or ""
            ):
                raise ValueError(f"{role} {owner_name} {field} differs")
        if role == "deepx_m1_full":
            for field in (
                "prepared_input_evidence_sha256",
                "prepared_input_join_binding",
                "prepared_input_join_binding_sha256",
            ):
                if field not in owner or owner.get(field) != producer.get(field):
                    raise ValueError(
                        f"{role} {owner_name} prepared-input duplicate differs"
                    )
        else:
            source_onnx = producer["source_onnx"]
            build_onnx = producer["build_onnx"]
            engine = producer["engine"]
            trtexec = producer["trtexec"]
            receipt_binding = producer["engine_build_receipt"]
            receipt = receipt_binding["receipt"]
            duplicates = {
                "eval_run_id": producer["eval_run_id"],
                "model_id": producer["model_id"],
                "setup_id": producer["setup_id"],
                "source_run_id": producer["source_run_id"],
                "execution_role": producer["execution_role"],
                "backend": producer["backend"],
                "variant": producer["variant"],
                "case_id": producer["case_id"],
                "task": producer["task"],
                "source_model_sha256": source_onnx["sha256"],
                "build_onnx_sha256": build_onnx["sha256"],
                "runtime_artifact_sha256": engine["sha256"],
                "trtexec_sha256": trtexec["sha256"],
                "engine_build_receipt_sha256": receipt_binding["sha256"],
                "engine_build_receipt_file_sha256": producer[
                    "engine_build_receipt_file_sha256"
                ],
                "trt_engine_build_receipt_sha256": receipt["receipt_sha256"],
            }
            for field, expected in duplicates.items():
                if field not in owner or str(owner.get(field) or "") != str(expected or ""):
                    raise ValueError(
                        f"{role} {owner_name} TensorRT {field} duplicate differs"
                    )
    _validate_full_only_plan_binding(
        endpoint, expected_eval_run_id=expected_eval_run_id,
    )


def _quality_execution_invariant(
    endpoint: QualityEndpointEvidence,
) -> dict[str, Any]:
    """Project the producer fields that must not change with calibration size."""

    producer = endpoint.producer
    quality_record_endpoint = copy.deepcopy(
        producer["quality_record_endpoint"]
    )
    quality_contract = copy.deepcopy(producer["quality_contract"])
    if endpoint.source_run_id == "deepx_m1_full":
        # ``output_contract_sha256`` binds the quality evidence to the exact
        # output-contract file emitted beside one compiled DXNN.  That file is
        # necessarily arm-local when calibration size changes, just like the
        # DXNN bytes normalized below.  Each arm has already passed the strict
        # producer/component/request validation above, so only neutralize this
        # artifact digest and recompute its deterministic containing hashes for
        # the cross-arm semantic projection.  Runtime endpoint stage/tensors,
        # observed outputs, runner and postprocessor remain byte-for-byte
        # comparable.
        quality_endpoint_identity = dict(
            quality_record_endpoint["identity"]
        )
        if "output_contract_sha256" in quality_endpoint_identity:
            _require_sha256(
                quality_endpoint_identity["output_contract_sha256"],
                label="DeepX quality-record output-contract artifact",
            )
            quality_endpoint_identity["output_contract_sha256"] = (
                "<calibration-arm-output-contract-sha256>"
            )
            quality_record_endpoint = {
                "identity": quality_endpoint_identity,
                "sha256": _canonical_sha256(quality_endpoint_identity),
            }
            quality_contract["quality_record_endpoint"] = copy.deepcopy(
                quality_record_endpoint
            )
            quality_contract[
                "quality_record_endpoint_contract_sha256"
            ] = quality_record_endpoint["sha256"]
            quality_contract.pop("quality_contract_sha256", None)
            quality_contract["quality_contract_sha256"] = (
                _canonical_sha256(quality_contract)
            )
    precision = dict(producer["precision"])
    precision_identity = dict(precision["identity"])
    if endpoint.source_run_id == "deepx_m1_full":
        for field in (
            "artifact_name", "artifact_sha256", "artifact_size_bytes",
        ):
            precision_identity.pop(field, None)
        precision = {"identity": precision_identity}
    invariant = {
        "preprocessing": copy.deepcopy(producer["preprocessing"]),
        "endpoint": copy.deepcopy(producer["endpoint"]),
        "quality_record_endpoint": quality_record_endpoint,
        "quality_contract": quality_contract,
        "precision": precision,
        "implementation_runner_sha256": producer["implementation_runner_sha256"],
        "preprocessing_contract_sha256": producer[
            "preprocessing_contract_sha256"
        ],
        "endpoint_contract_hash": producer["endpoint_contract_hash"],
        "quality_contract_sha256": quality_contract[
            "quality_contract_sha256"
        ],
        "quality_record_endpoint_contract_sha256": (
            quality_record_endpoint["sha256"]
        ),
    }
    full_only_identity = copy.deepcopy(
        endpoint.request["full_only_plan_identity"]
    )
    full_only_identity.pop("quality_canary_id", None)
    full_only_identity.pop("eval_run_id", None)
    invariant["full_only_plan_identity"] = full_only_identity
    if endpoint.source_run_id == "native_full_tensorrt":
        def artifact_identity(name: str) -> dict[str, Any]:
            artifact = dict(producer[name])
            artifact.pop("path", None)
            return artifact

        receipt = copy.deepcopy(
            producer["engine_build_receipt"]["receipt"]
        )
        receipt.pop("receipt_sha256", None)
        receipt["source_onnx"] = "<build-onnx>"
        receipt["engine"] = "<engine>"
        receipt["trtexec"] = "<trtexec>"
        normalized_command: list[str] = []
        for index, argument in enumerate(list(receipt.get("command") or [])):
            if index == 0:
                normalized_command.append("<trtexec>")
            elif str(argument).startswith("--onnx="):
                normalized_command.append("--onnx=<build-onnx>")
            elif str(argument).startswith("--saveEngine="):
                normalized_command.append("--saveEngine=<engine>")
            elif str(argument).startswith("--timingCacheFile="):
                normalized_command.append("--timingCacheFile=<cache>")
            else:
                normalized_command.append(str(argument))
        receipt["command"] = normalized_command
        endpoint_attestor = copy.deepcopy(producer["endpoint_attestor"])
        endpoint_attestor.pop("path", None)
        invariant.update({
            "policy_sha256": producer["policy_sha256"],
            "runtime_precision_identity": producer[
                "runtime_precision_identity"
            ],
            "model": copy.deepcopy(producer["model"]),
            "dataset": copy.deepcopy(producer["dataset"]),
            "source_onnx": artifact_identity("source_onnx"),
            "build_onnx": artifact_identity("build_onnx"),
            "engine": artifact_identity("engine"),
            "trtexec": artifact_identity("trtexec"),
            "engine_build_receipt": receipt,
            "endpoint_contract_complete": producer[
                "endpoint_contract_complete"
            ],
            "endpoint_authority": copy.deepcopy(
                producer["endpoint_authority"]
            ),
            "endpoint_attestor": endpoint_attestor,
            "vendored_endpoint_attestor_sha256": producer[
                "vendored_endpoint_attestor_sha256"
            ],
            "decoder_contract_sha256": producer["decoder_contract_sha256"],
            "nms_contract_sha256": producer["nms_contract_sha256"],
        })
    return invariant


def _validate_result_hit_count(
    endpoint: QualityEndpointEvidence,
    *,
    top5: bool,
) -> dict[str, Any]:
    component = _quality_component(endpoint.result, top5=top5)
    metric = "top5_hit" if top5 else "top1_hit"
    observed_hits = int(_hits(endpoint.records, metric).sum())
    if _integer(
        component.get("candidate_hits"),
        label=f"{endpoint.source_run_id} {metric} candidate_hits",
    ) != observed_hits:
        raise ValueError(
            f"{endpoint.source_run_id} {metric} records/result hit counts differ"
        )
    expected_reference_hits = (
        EXPECTED_REFERENCE_TOP5_HITS if top5 else EXPECTED_REFERENCE_TOP1_HITS
    )
    if _integer(
        component.get("reference_hits"),
        label=f"{endpoint.source_run_id} {metric} reference_hits",
    ) != expected_reference_hits:
        raise ValueError(
            f"{endpoint.source_run_id} {metric} CPU/ORT reference authority differs"
        )
    return component


def _validated_result_decision(endpoint: QualityEndpointEvidence) -> str:
    if (
        endpoint.result.get("status") != "completed"
        or endpoint.result.get("technical_status") != "completed"
        or endpoint.result.get("algorithm_version") != EXPECTED_QUALITY_ALGORITHM
    ):
        raise ValueError(f"{endpoint.source_run_id} central result is not exact/complete")
    primary = _quality_component(endpoint.result, top5=False)
    top5 = _quality_component(endpoint.result, top5=True)
    decisions = {str(primary.get("decision") or ""), str(top5.get("decision") or "")}
    expected = (
        "fail" if "fail" in decisions
        else "pass" if decisions == {"pass"}
        else "inconclusive"
    )
    if str(endpoint.result.get("decision") or "") != expected:
        raise ValueError(
            f"{endpoint.source_run_id} aggregate/component quality decisions disagree"
        )
    return expected


def _non_inferiority_decision(
    report: Mapping[str, Any], *, margin: float,
) -> str:
    point = float(report["point_delta_candidate_minus_reference"])
    lower = float(report["lower_one_sided_bound"])
    tolerance = 1e-12
    if point < -float(margin) - tolerance:
        return "fail"
    if lower >= -float(margin) - tolerance:
        return "pass"
    return "inconclusive"


def compare_calibration_arms(
    baseline: CalibrationArmEvidence,
    candidate: CalibrationArmEvidence,
) -> dict[str, Any]:
    if (
        baseline.label != "B500"
        or baseline.expected_calibration_count != BASELINE_CALIBRATION_COUNT
        or candidate.label != "B1000"
        or candidate.expected_calibration_count != CANDIDATE_CALIBRATION_COUNT
    ):
        raise ValueError("calibration canary requires exact B500 and B1000 arms")
    if baseline.artifact_status not in {"ready_built", "ready_reused"} or (
        candidate.artifact_status not in {"ready_built", "ready_reused"}
    ):
        raise ValueError("B500/B1000 artifact status is not exact built/reused evidence")
    _validate_profile_provenance(baseline)
    _validate_profile_provenance(candidate)
    baseline_profile_contract = _validate_resolved_profile(
        baseline.profile, label="B500", expected_count=BASELINE_CALIBRATION_COUNT,
    )
    candidate_profile_contract = _validate_resolved_profile(
        candidate.profile, label="B1000", expected_count=CANDIDATE_CALIBRATION_COUNT,
    )
    if _normalized_profile_for_pair(baseline.profile) != _normalized_profile_for_pair(
        candidate.profile
    ):
        raise ValueError(
            "B500/B1000 resolved profiles differ outside the reviewed "
            "calibration-size allowlist"
        )
    for arm, contract in (
        (baseline, baseline_profile_contract),
        (candidate, candidate_profile_contract),
    ):
        if arm.base.profile_name != contract["profile_name"] or (
            Path(arm.base.cache_dir).expanduser().resolve()
            != Path(str(contract["cache_namespace"])).expanduser().resolve()
        ):
            raise ValueError(f"{arm.label} resolved profile/base cache binding differs")

    invariant_base_fields = (
        "mode", "task", "target", "source_onnx_sha256", "build_onnx_sha256",
        "compiler_identity_sha256", "compiler_contract_sha256",
        "validation_manifest_sha256", "validation_image_ids_sha256",
        "validation_ground_truth_sha256", "prepared_input_evidence_sha256",
        "preprocessing_contract_sha256", "policy_sha256",
    )
    mismatches = [
        field for field in invariant_base_fields
        if not getattr(baseline.base, field)
        or getattr(baseline.base, field) != getattr(candidate.base, field)
    ]
    if mismatches:
        raise ValueError(
            "B500/B1000 invariant run evidence differs: " + ",".join(mismatches)
        )
    frozen_authority = {
        "source_onnx_sha256": EXPECTED_SOURCE_ONNX_SHA256,
        "build_onnx_sha256": EXPECTED_BUILD_ONNX_SHA256,
        "compiler_identity_sha256": EXPECTED_COMPILER_IDENTITY_SHA256,
        "compiler_contract_sha256": EXPECTED_COMPILER_CONTRACT_SHA256,
        "validation_manifest_sha256": EXPECTED_VALIDATION_MANIFEST_SHA256,
        "validation_image_ids_sha256": EXPECTED_VALIDATION_IMAGE_IDS_SHA256,
        "validation_ground_truth_sha256": EXPECTED_VALIDATION_GROUND_TRUTH_SHA256,
        "prepared_input_evidence_sha256": EXPECTED_PREPARED_INPUT_EVIDENCE_SHA256,
        "preprocessing_contract_sha256": EXPECTED_PREPROCESSING_CONTRACT_SHA256,
        "policy_sha256": EXPECTED_QUALITY_POLICY_SHA256,
    }
    authority_mismatches = [
        field for field, expected in frozen_authority.items()
        if getattr(baseline.base, field) != expected
        or getattr(candidate.base, field) != expected
    ]
    if authority_mismatches:
        raise ValueError(
            "B500/B1000 differ from frozen v2.75.40-B scientific authority: "
            + ",".join(authority_mismatches)
        )
    frozen_b500 = {
        "calibration_identity_sha256": EXPECTED_B500_CALIBRATION_IDENTITY_SHA256,
        "calibration_contract_sha256": EXPECTED_B500_CALIBRATION_CONTRACT_SHA256,
        "calibration_items_identity_sha256": EXPECTED_B500_CALIBRATION_ITEMS_SHA256,
        "dxcom_config_sha256": EXPECTED_B500_DXCOM_CONFIG_SHA256,
        "build_options_sha256": EXPECTED_B500_BUILD_OPTIONS_SHA256,
        "cache_key": EXPECTED_B500_CACHE_KEY,
        "cache_contract_sha256": EXPECTED_B500_CACHE_CONTRACT_SHA256,
        "dxnn_sha256": EXPECTED_B500_DXNN_SHA256,
    }
    b500_mismatches = [
        field for field, expected in frozen_b500.items()
        if getattr(baseline.base, field) != expected
    ]
    if b500_mismatches:
        raise ValueError(
            "B500 differs from frozen v2.75.40-B calibration authority: "
            + ",".join(b500_mismatches)
        )
    if baseline.deepx_endpoint.setup_id != candidate.deepx_endpoint.setup_id:
        raise ValueError("B500/B1000 DeepX setup identity differs")

    manifest_invariants = (
        "dataset_id", "task", "role", "split", "source_kind", "hash_mode",
    )
    manifest_mismatches = [
        field for field in manifest_invariants
        if baseline.calibration_manifest.get(field)
        != candidate.calibration_manifest.get(field)
    ]
    for field in ("annotations", "labels"):
        a = baseline.calibration_manifest.get(field)
        b = candidate.calibration_manifest.get(field)
        if not isinstance(a, Mapping) or not isinstance(b, Mapping) or _bare_sha(
            a.get("sha256")
        ) != _bare_sha(b.get("sha256")):
            manifest_mismatches.append(field)
    a_selection = dict(baseline.calibration_manifest.get("selection") or {})
    b_selection = dict(candidate.calibration_manifest.get("selection") or {})
    for field in ("strategy", "seed", "selection_uses_model_predictions"):
        if a_selection.get(field) != b_selection.get(field):
            manifest_mismatches.append(f"selection.{field}")
    a_provisioning = baseline.calibration_manifest.get("provisioning_selection")
    b_provisioning = candidate.calibration_manifest.get("provisioning_selection")
    if isinstance(a_provisioning, Mapping) != isinstance(b_provisioning, Mapping):
        manifest_mismatches.append("provisioning_selection.shape")
    elif isinstance(a_provisioning, Mapping) and isinstance(b_provisioning, Mapping):
        for field in (
            "source_split", "source_population", "strategy", "seed",
            "selection_uses_model_predictions",
        ):
            if a_provisioning.get(field) != b_provisioning.get(field):
                manifest_mismatches.append(f"provisioning_selection.{field}")
    if _integer(
        baseline.calibration_manifest.get("population_count"),
        label="B500 population_count",
    ) != BASELINE_CALIBRATION_COUNT:
        manifest_mismatches.append("B500.population_count")
    if _integer(
        candidate.calibration_manifest.get("population_count"),
        label="B1000 population_count",
    ) != CANDIDATE_CALIBRATION_COUNT:
        manifest_mismatches.append("B1000.population_count")
    if manifest_mismatches:
        raise ValueError(
            "B500/B1000 calibration population/selection semantics differ: "
            + ",".join(manifest_mismatches)
        )
    baseline_items = {_canonical_sha256(item): item for item in baseline.calibration_items}
    candidate_items = {_canonical_sha256(item): item for item in candidate.calibration_items}
    missing = sorted(set(baseline_items) - set(candidate_items))
    if missing:
        raise ValueError(
            f"B500 calibration cohort is not a subset of B1000 ({len(missing)} item(s) missing)"
        )

    normalized_dxcom_a = _without_paths(
        baseline.dxcom_config,
        (("calibration_num",), ("default_loader", "dataset_path")),
    )
    normalized_dxcom_b = _without_paths(
        candidate.dxcom_config,
        (("calibration_num",), ("default_loader", "dataset_path")),
    )
    if normalized_dxcom_a != normalized_dxcom_b:
        raise ValueError("DX-COM configs differ outside calibration count/dataset path")
    if baseline.base.dxcom_config_sha256 == candidate.base.dxcom_config_sha256:
        raise ValueError("B500/B1000 DX-COM config identities must differ")

    normalized_options_a = dict(baseline.build_options)
    normalized_options_b = dict(candidate.build_options)
    normalized_options_a.pop("calibration_count", None)
    normalized_options_b.pop("calibration_count", None)
    if normalized_options_a != normalized_options_b:
        raise ValueError("DeepX build options differ outside calibration count")
    normalized_contract_a = _without_paths(
        baseline.cache_contract,
        (
            ("contract_sha256",), ("dxcom_config_sha256",),
            ("calibration_manifest_contract",),
            ("build_options", "calibration_count"),
        ),
    )
    normalized_contract_b = _without_paths(
        candidate.cache_contract,
        (
            ("contract_sha256",), ("dxcom_config_sha256",),
            ("calibration_manifest_contract",),
            ("build_options", "calibration_count"),
        ),
    )
    if normalized_contract_a != normalized_contract_b:
        raise ValueError("v2 cache contracts differ outside allowed calibration-derived fields")
    if (
        baseline.base.cache_contract_sha256 == candidate.base.cache_contract_sha256
        or baseline.base.cache_key == candidate.base.cache_key
        or baseline.base.cache_dir == candidate.base.cache_dir
    ):
        raise ValueError("B500/B1000 cache contract, key and root must differ")

    cohort_identity = _record_identity(baseline.deepx_endpoint.records)
    endpoints = (
        candidate.deepx_endpoint, baseline.trt_endpoint, candidate.trt_endpoint,
    )
    if any(_record_identity(endpoint.records) != cohort_identity for endpoint in endpoints):
        raise ValueError("B500/B1000/DeepX/TRT validation sample order or ground truth differs")
    baseline_deepx_hits = (
        int(_hits(baseline.deepx_endpoint.records, "top1_hit").sum()),
        int(_hits(baseline.deepx_endpoint.records, "top5_hit").sum()),
    )
    if baseline_deepx_hits != (
        EXPECTED_B500_DEEPX_TOP1_HITS,
        EXPECTED_B500_DEEPX_TOP5_HITS,
    ):
        raise ValueError("B500 DeepX prediction records differ from historical authority")
    baseline_trt_top1 = _hits(baseline.trt_endpoint.records, "top1_hit")
    baseline_trt_top5 = _hits(baseline.trt_endpoint.records, "top5_hit")
    candidate_trt_top1 = _hits(candidate.trt_endpoint.records, "top1_hit")
    candidate_trt_top5 = _hits(candidate.trt_endpoint.records, "top5_hit")
    if (
        int(baseline_trt_top1.sum()) != EXPECTED_REFERENCE_TOP1_HITS
        or int(baseline_trt_top5.sum()) != EXPECTED_REFERENCE_TOP5_HITS
        or not np.array_equal(baseline_trt_top1, candidate_trt_top1)
        or not np.array_equal(baseline_trt_top5, candidate_trt_top5)
    ):
        raise ValueError(
            "B500/B1000 setup-local TensorRT hit vectors differ from historical authority"
        )

    policy_contracts = [
        _validate_endpoint_scientific_binding(arm, endpoint)
        for arm in (baseline, candidate)
        for endpoint in (arm.deepx_endpoint, arm.trt_endpoint)
    ]
    if any(contract != policy_contracts[0] for contract in policy_contracts[1:]):
        raise ValueError("B500/B1000/DeepX/TRT quality policy contracts differ")
    frozen_policy = policy_contracts[0]
    for source_run_id, baseline_endpoint, candidate_endpoint in (
        (
            "deepx_m1_full", baseline.deepx_endpoint,
            candidate.deepx_endpoint,
        ),
        (
            "native_full_tensorrt", baseline.trt_endpoint,
            candidate.trt_endpoint,
        ),
    ):
        if _quality_execution_invariant(
            baseline_endpoint
        ) != _quality_execution_invariant(candidate_endpoint):
            raise ValueError(
                f"B500/B1000 {source_run_id} quality execution contracts differ"
            )

    baseline_top1 = _hits(baseline.deepx_endpoint.records, "top1_hit")
    candidate_top1 = _hits(candidate.deepx_endpoint.records, "top1_hit")
    baseline_top5 = _hits(baseline.deepx_endpoint.records, "top5_hit")
    candidate_top5 = _hits(candidate.deepx_endpoint.records, "top5_hit")
    paired_top1, _ = _paired_bootstrap(
        baseline_top1, candidate_top1, seed=TOP1_BOOTSTRAP_SEED,
    )
    paired_top5, _ = _paired_bootstrap(
        baseline_top5, candidate_top5, seed=TOP5_BOOTSTRAP_SEED,
    )

    # Bind central result hit counts to the exact candidate records before
    # using their margins or decisions.
    for endpoint in (
        baseline.deepx_endpoint, baseline.trt_endpoint,
        candidate.deepx_endpoint, candidate.trt_endpoint,
    ):
        _validate_result_hit_count(endpoint, top5=False)
        _validate_result_hit_count(endpoint, top5=True)
    endpoint_decisions = {
        (arm.label, endpoint.source_run_id): _validated_result_decision(endpoint)
        for arm in (baseline, candidate)
        for endpoint in (arm.deepx_endpoint, arm.trt_endpoint)
    }
    if endpoint_decisions[(baseline.label, "deepx_m1_full")] != "inconclusive":
        raise ValueError("B500 central DeepX quality decision is not historical inconclusive")
    trt_top1 = _hits(candidate.trt_endpoint.records, "top1_hit")
    trt_top5 = _hits(candidate.trt_endpoint.records, "top5_hit")
    trt_guard_top1, _ = _paired_bootstrap(
        trt_top1, candidate_top1, seed=TOP1_BOOTSTRAP_SEED,
    )
    trt_guard_top5, _ = _paired_bootstrap(
        trt_top5, candidate_top5, seed=TOP1_BOOTSTRAP_SEED,
    )
    trt_guard_top1["margin"] = float(
        frozen_policy["metric_gate_config"]["non_inferiority_margin"]
    )
    trt_guard_top5["margin"] = float(
        frozen_policy["metric_gate_config"]["guardrails"][
            "top5_accuracy_margin"
        ]
    )
    trt_guard_top1["decision"] = _non_inferiority_decision(
        trt_guard_top1, margin=float(trt_guard_top1["margin"]),
    )
    trt_guard_top5["decision"] = _non_inferiority_decision(
        trt_guard_top5, margin=float(trt_guard_top5["margin"]),
    )
    b1000_vs_trt_decision = (
        "fail" if "fail" in {trt_guard_top1["decision"], trt_guard_top5["decision"]}
        else "pass" if {trt_guard_top1["decision"], trt_guard_top5["decision"]} == {"pass"}
        else "inconclusive"
    )

    technical_complete = all(
        arm.base.workflow_status in {"ok", "completed"}
        and all(
            str(endpoint.result.get("status") or "") == "completed"
            and str(endpoint.result.get("technical_status") or "") == "completed"
            for endpoint in (arm.deepx_endpoint, arm.trt_endpoint)
        )
        for arm in (baseline, candidate)
    )
    trt_controls_pass = all(
        endpoint_decisions[(arm.label, "native_full_tensorrt")] == "pass"
        for arm in (baseline, candidate)
    )
    central_b1000_pass = (
        endpoint_decisions[(candidate.label, "deepx_m1_full")] == "pass"
    )
    b1000_quality_pass = bool(
        central_b1000_pass and b1000_vs_trt_decision == "pass"
    )
    standard_plus_ready = bool(
        technical_complete and trt_controls_pass and b1000_quality_pass
    )
    same_dxnn = baseline.base.dxnn_sha256 == candidate.base.dxnn_sha256
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified",
        "experiment": {
            "changed_variable": "deepx_full_calibration_item_count",
            "baseline_count": BASELINE_CALIBRATION_COUNT,
            "candidate_count": CANDIDATE_CALIBRATION_COUNT,
            "classification_preprocessing": CLASSIFICATION_PREPROCESSING_IMAGENET,
            "calibration_method": "ema",
            "opt_level": 0,
            "validation_records": VALIDATION_RECORD_COUNT,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "top1_seed": TOP1_BOOTSTRAP_SEED,
            "top5_seed": TOP5_BOOTSTRAP_SEED,
        },
        "invariant_evidence": {
            field: getattr(baseline.base, field) for field in invariant_base_fields
        },
        "resolved_profile_contracts": {
            "baseline_b500": baseline_profile_contract,
            "candidate_b1000": candidate_profile_contract,
        },
        "frozen_quality_policy": frozen_policy,
        "calibration_cohort": {
            "selection_strategy": a_selection["strategy"],
            "selection_seed": int(a_selection["seed"]),
            "selection_semantics": (
                "provisioning_selection"
                if isinstance(a_provisioning, Mapping)
                else "direct_manifest_selection"
            ),
            "source_population": (
                str(a_provisioning.get("source_population") or "")
                if isinstance(a_provisioning, Mapping) else ""
            ),
            "b500_population_count": _integer(
                baseline.calibration_manifest.get("population_count"),
                label="B500 population_count",
            ),
            "b1000_population_count": _integer(
                candidate.calibration_manifest.get("population_count"),
                label="B1000 population_count",
            ),
            "b500_item_count": len(baseline.calibration_items),
            "b1000_item_count": len(candidate.calibration_items),
            "b500_is_subset_of_b1000": True,
            "additional_b1000_items": len(candidate_items) - len(baseline_items),
        },
        "baseline_b500": {
            "run_dir": str(baseline.base.run_dir),
            "calibration_manifest": str(baseline.calibration_manifest_path),
            "calibration_manifest_sha256": _sha256_file(
                baseline.calibration_manifest_path
            ),
            "cache_dir": baseline.base.cache_dir,
            "cache_key": baseline.base.cache_key,
            "cache_contract_sha256": baseline.base.cache_contract_sha256,
            "dxcom_config_sha256": baseline.base.dxcom_config_sha256,
            "dxnn_sha256": baseline.base.dxnn_sha256,
            "artifact_status": baseline.artifact_status,
            "deepx_quality_decision": str(
                baseline.deepx_endpoint.result.get("decision") or ""
            ),
            "trt_quality_decision": str(
                baseline.trt_endpoint.result.get("decision") or ""
            ),
        },
        "candidate_b1000": {
            "run_dir": str(candidate.base.run_dir),
            "calibration_manifest": str(candidate.calibration_manifest_path),
            "calibration_manifest_sha256": _sha256_file(
                candidate.calibration_manifest_path
            ),
            "cache_dir": candidate.base.cache_dir,
            "cache_key": candidate.base.cache_key,
            "cache_contract_sha256": candidate.base.cache_contract_sha256,
            "dxcom_config_sha256": candidate.base.dxcom_config_sha256,
            "dxnn_sha256": candidate.base.dxnn_sha256,
            "artifact_status": candidate.artifact_status,
            "deepx_quality_decision": str(
                candidate.deepx_endpoint.result.get("decision") or ""
            ),
            "trt_quality_decision": str(
                candidate.trt_endpoint.result.get("decision") or ""
            ),
            "exact_namespace": B1000_CACHE_NAMESPACE,
        },
        "dxnn_identity": {
            "same_sha256": same_dxnn,
            "scientifically_valid": True,
            "interpretation": (
                "calibration_count_change_produced_identical_dxnn_bytes"
                if same_dxnn else "calibration_count_change_produced_different_dxnn_bytes"
            ),
        },
        "paired_b1000_minus_b500": {
            "top1": paired_top1,
            "top5": paired_top5,
        },
        "authoritative_b1000_vs_setup_local_trt_guardrail": {
            "top1": trt_guard_top1,
            "top5": trt_guard_top5,
            "decision": b1000_vs_trt_decision,
        },
        "technical_evidence_complete": technical_complete,
        "setup_local_tensorrt_controls_pass": trt_controls_pass,
        "central_b1000_deepx_quality_pass": central_b1000_pass,
        "b1000_quality_pass": b1000_quality_pass,
        "standard_plus_ready": standard_plus_ready,
        "next_step": (
            "standard_plus_confirmation"
            if standard_plus_ready
            else "keep_standard_plus_blocked"
        ),
    }


def compare_run_directories(
    *,
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    baseline_calibration_manifest: str | Path | None = None,
    candidate_calibration_manifest: str | Path | None = None,
) -> dict[str, Any]:
    baseline = load_calibration_arm(
        baseline_dir,
        label="B500",
        expected_calibration_count=BASELINE_CALIBRATION_COUNT,
        calibration_manifest_path=baseline_calibration_manifest,
    )
    candidate = load_calibration_arm(
        candidate_dir,
        label="B1000",
        expected_calibration_count=CANDIDATE_CALIBRATION_COUNT,
        calibration_manifest_path=candidate_calibration_manifest,
    )
    return compare_calibration_arms(baseline, candidate)


__all__ = [
    "BASELINE_CALIBRATION_COUNT",
    "BOOTSTRAP_REPETITIONS",
    "B1000_CACHE_NAMESPACE",
    "CANDIDATE_CALIBRATION_COUNT",
    "CalibrationArmEvidence",
    "QualityEndpointEvidence",
    "compare_calibration_arms",
    "compare_run_directories",
    "load_calibration_arm",
]
