from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
import pytest

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.benchmark.classification_validation_presets import (
    project_classification_validation_subset,
    provision_classification_validation_source_to_suite,
)
from onnx_splitpoint_tool.campaign import (
    create_dataset_manifest,
    verify_dataset_manifest,
)
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.quality_cache import (
    image_ids_fingerprint,
    json_fingerprint,
)
from onnx_splitpoint_tool.workflow.artifacts import sha256_json


ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT
    / "profiles"
    / "resnet50_v27541_deepx_calibration_1000_imagenet_mean_std.yaml"
)
BASELINE_PROFILE = (
    ROOT
    / "profiles"
    / "resnet50_v27540_deepx_preprocess_b_imagenet_mean_std.yaml"
)
SCRIPT = ROOT / "scripts" / "preflight_v27541_deepx_calibration_1000.py"
PROFILE_ID = "resnet50_v27541_deepx_calibration_1000_imagenet_mean_std"
CALIBRATION_MANIFEST = (
    "/home/kmika/.onnx_splitpoint_tool/final_datasets/"
    "v27541_imagenet_n1000_s20260710/manifests/"
    "imagenet_train_calibration_manifest.json"
)
VALIDATION_MANIFEST = (
    "/home/kmika/.onnx_splitpoint_tool/final_datasets/manifests/"
    "imagenet_val_manifest.json"
)
CACHE_DIR = (
    "~/Models/BackendArtifacts/deepx/v2.75.41/"
    "resnet50_calibration_size/b1000_imagenet_mean_std"
)


def _load_preflight_module():
    spec = importlib.util.spec_from_file_location("v27541_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _leaf_differences(
    left: Any,
    right: Any,
    *,
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[tuple[str, ...]] = set()
        for key in sorted(set(left) | set(right)):
            path = (*prefix, str(key))
            if key not in left or key not in right:
                differences.add(path)
            else:
                differences.update(
                    _leaf_differences(left[key], right[key], prefix=path)
                )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = set()
        for index in range(max(len(left), len(right))):
            path = (*prefix, str(index))
            if index >= len(left) or index >= len(right):
                differences.add(path)
            else:
                differences.update(
                    _leaf_differences(
                        left[index], right[index], prefix=path
                    )
                )
        return differences
    return set() if left == right else {prefix}


def _write_images(root: Path, *, start: int, stop: int, prefix: str) -> None:
    for index in range(start, stop):
        image = root / f"n{index % 1000:08d}" / f"image_{index:08d}.JPEG"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"{prefix}-{index}".encode("utf-8"))


def _manifest(
    *,
    data_root: Path,
    output: Path,
    role: str,
    seed: int = 20260710,
) -> Path:
    path = create_dataset_manifest(
        task="classification",
        role=role,
        dataset_id=(
            "ilsvrc2012-train-calibration"
            if role == "calibration" else "ilsvrc2012-validation"
        ),
        split="train" if role == "calibration" else "val",
        root=data_root,
        output=output,
        max_items=0,
        selection_strategy="class_stratified",
        selection_seed=seed,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if role == "calibration":
        payload["selection"] = {
            "strategy": "class_stratified",
            "seed": seed,
            "requested_max_items": len(payload["items"]),
            "selection_uses_model_predictions": False,
        }
        payload["provisioning_selection"] = {}
    else:
        # The production ImageNet manifest resolves these values through its
        # frozen synset mapping.  Synthetic fixtures carry the equivalent
        # explicit identities so the runtime cohort can be projected exactly.
        for index, item in enumerate(payload["items"]):
            item["label_id"] = index % 1000
            item["label_name"] = str(item["class_name"])
    payload["manifest_payload_sha256"] = sha256_json({
        key: value for key, value in payload.items()
        if key != "manifest_payload_sha256"
    })
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _write_baseline_run(
    run_dir: Path,
    *,
    baseline_root: Path,
) -> Path:
    manifest = _manifest(
        data_root=baseline_root,
        output=(
            run_dir
            / "dataset_evidence"
            / "imagenet_train_calibration_manifest.json"
        ),
        role="calibration",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    (run_dir / "profile.yaml").write_text(
        yaml.safe_dump({
            "name": "resnet50_v27540_deepx_preprocess_b_imagenet_mean_std",
            "deepx_build": {
                "calib_count": 500,
                "calibration_method": "ema",
                "opt_level": 0,
                "classification_preprocessing": "imagenet_mean_std",
            },
        }),
        encoding="utf-8",
    )
    calibration_contract = {
        "schema": "onnx-splitpoint/deepx-calibration-manifest-contract",
        "schema_version": 1,
        "task": "classification",
        "status": "resolved",
        "effective_count": 500,
        "dataset_id": "ilsvrc2012-train-calibration",
        "split": "train",
        "role": "calibration",
        "hash_mode": "content",
        "item_count": 500,
        "items_identity_sha256": payload["items_identity_sha256"],
        "manifest_payload_sha256": payload["manifest_payload_sha256"],
        "manifest_file_sha256": _sha256_file(manifest),
        "manifest_file_name": manifest.name,
        "dataset_root_name": baseline_root.name,
        "manifest_verification": {
            key: verify_dataset_manifest(
                payload, verify_files=True, verification_mode="full",
            ).get(key)
            for key in (
                "ok", "schema_ok", "payload_hash_ok", "identity_hash_ok",
                "item_count_ok", "verification_mode", "manifest_item_count",
                "checked_item_count", "missing_count", "mismatch_count",
            )
        },
        "root_inventory_count": 500,
        "root_inventory_sha256": _canonical_sha256(sorted(({
            "relative_path": str(item["relative_path"]),
            "size_bytes": int(item["size_bytes"]),
            "sha256": str(item["sha256"]).split(":", 1)[-1],
        } for item in payload["items"]), key=lambda row: row["relative_path"])),
        "dataset_registry_binding_sha256": "a" * 64,
    }
    calibration_contract["identity_sha256"] = _canonical_sha256(
        calibration_contract
    )
    compiler = {"status": "resolved", "evidence": {"version": "test"}}
    compiler["identity_sha256"] = _canonical_sha256(compiler)
    contract = {
        "schema": "onnx-splitpoint/deepx-full-cache-contract",
        "schema_version": 2,
        "task": "classification",
        "target": "deepx_m1",
        "classification_preprocessing": "imagenet_mean_std",
        "calibration_manifest_contract": calibration_contract,
        "compiler_identity": compiler,
        "build_options": {
            "calibration_count": 500,
            "calibration_method": "ema",
            "opt_level": 0,
        },
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    status = (
        run_dir
        / "models"
        / "resnet50"
        / "benchmark_set"
        / "deepx"
        / "deepx_artifact_status.json"
    )
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(
        json.dumps({"status": "ok", "cache_contract": contract}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _tree_identity(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            _sha256_file(path),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _runtime_projection(
    validation_manifest: Path,
    *,
    seed: int = 20260710,
) -> dict[str, Any]:
    source = json.loads(validation_manifest.read_text(encoding="utf-8"))
    source_root = Path(str(source["root"])).resolve()
    projection = project_classification_validation_subset(
        resolved=source_root,
        requested=str(source_root),
        max_images=500,
        selection_seed=seed,
        explicit_manifest=validation_manifest,
        base_dir=validation_manifest.parent,
    )
    assert projection is not None
    return projection


def _projection_authorities(
    projection: dict[str, Any],
) -> dict[str, str]:
    payload = projection["manifest"]
    manifest_bytes = (
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    records = sorted(({
        "image_id": Path(str(sample["image"])).name,
        "label_id": int(sample["label_id"]),
    } for sample in payload["samples"]), key=lambda row: row["image_id"])
    image_ids = [str(row["image_id"]) for row in records]
    return {
        "manifest": hashlib.sha256(manifest_bytes).hexdigest(),
        "image_ids": image_ids_fingerprint(image_ids),
        "ground_truth": json_fingerprint(records),
    }


def _runtime_projection_authorities(
    validation_manifest: Path,
) -> dict[str, str]:
    return _projection_authorities(_runtime_projection(validation_manifest))


def _bind_frozen_candidate_profile_fixture(
    preflight,
    *,
    tmp_path: Path,
    calibration_manifest: Path,
    validation_manifest: Path,
    monkeypatch,
) -> None:
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    profile["campaign"]["dataset_manifests"]["classification"] = {
        "calibration": str(calibration_manifest.resolve()),
        "validation": str(validation_manifest.resolve()),
    }
    profile_path = tmp_path / "frozen_candidate_profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    monkeypatch.setattr(preflight, "CANDIDATE_PROFILE_SOURCE", profile_path)
    authorities = _runtime_projection_authorities(validation_manifest)
    monkeypatch.setattr(
        preflight,
        "EXPECTED_VALIDATION_MANIFEST_SHA256",
        authorities["manifest"],
    )
    monkeypatch.setattr(
        preflight,
        "EXPECTED_VALIDATION_IMAGE_IDS_SHA256",
        authorities["image_ids"],
    )
    monkeypatch.setattr(
        preflight,
        "EXPECTED_VALIDATION_GROUND_TRUTH_SHA256",
        authorities["ground_truth"],
    )


def test_v27541_packaged_profile_is_exact_full_only_calibration_canary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    assert validate_evaluation_profile_payload(raw, source_path=PROFILE) == raw
    assert raw["name"] == PROFILE_ID
    assert raw["campaign"]["dataset_manifests"]["classification"] == {
        "calibration": CALIBRATION_MANIFEST,
        "validation": VALIDATION_MANIFEST,
    }
    snapshot = raw["execution_preset"]["snapshot"]
    assert raw["execution_preset"]["id"] == "standard"
    assert raw["execution_preset"]["follow_tool_config"] is False
    assert raw["execution_preset"]["overrides"] == {
        "native_enabled": False,
        "energy_enabled": False,
    }
    assert snapshot["data"]["calibration_items"] == {
        "classification": 1000,
        "detection": 500,
    }
    assert snapshot["data"]["validation_items"] == {
        "classification": 500,
        "detection": 500,
    }
    # Hailo is inactive in this Full-only canary and remains bit-for-bit on
    # the v2.75.40 B-arm setting. Only the active DeepX build mirror changes.
    assert snapshot["build"]["hailo"]["calibration_items"] == 500
    assert snapshot["build"]["deepx"] == {
        "mode": "reuse_and_build_missing",
        "optimization_level": 0,
        "calibration_items": 1000,
        "calibration_method": "ema",
        "classification_preprocessing": "imagenet_mean_std",
        "cache_dir": CACHE_DIR,
        "force_build": False,
    }
    assert snapshot["quality"]["bootstrap_repetitions"] == 500
    assert snapshot["ranking"]["enabled"] is False
    assert [row["id"] for row in raw["run_profiles"]] == [
        "ort_tensorrt",
        "deepx_m1_full",
    ]
    assert raw["quality_canary"]["execution_scope"] == "full_only"
    assert raw["quality_canary"]["full_run_ids"] == [{
        "id": "deepx_m1_full",
        "run_id": "deepx_m1_full",
        "setup_id": "orin_nx_deepx_m1_01",
        "backend": "deepx_m1",
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }]

    monkeypatch.setenv(
        "ONNX_SPLITPOINT_RUN_MODES_FILE",
        str(tmp_path / "missing_user_run_modes.yaml"),
    )
    loaded = load_evaluation_profile(PROFILE, validate=True)
    assert loaded is not None
    assert loaded.profile_id == PROFILE_ID
    assert loaded.start_snapshot is not None
    assert loaded.start_snapshot["consistency"] == {
        "status": "ok",
        "mismatches": [],
    }
    profile = loaded.raw_profile
    assert profile["deepx_build"] == {
        "mode": "reuse_and_build_missing",
        "target": "deepx_m1",
        "calib_dir": "",
        "calib_count": 1000,
        "calibration_method": "ema",
        "opt_level": 0,
        "force_build": False,
        "classification_preprocessing": "imagenet_mean_std",
        "cache_dir": CACHE_DIR,
    }
    plan = build_effective_execution_plan(profile)
    assert plan["models"] == ["resnet50"]
    assert plan["generic_rows_total"] == 0
    assert plan["expected_generic_result_rows_total"] == 0
    assert plan["expected_full_quality_results_per_model"] == 2
    assert plan["setup_groups"] == {
        "orin_nx_deepx_m1_01": ["deepx_m1_full", "ort_tensorrt"],
    }
    assert plan["native_enabled"] is False
    assert plan["native_full_baselines"] is False
    assert plan["native_energy_enabled"] is False
    assert plan["generic_energy_enabled"] is False
    assert plan["energy_measurement_path"] == "disabled"
    assert plan["ranking_enabled"] is False
    assert plan["performance_claims_emitted"] is False
    assert plan["calibration_items"] == {
        "classification": 1000,
        "detection": 500,
    }
    assert plan["validation_items"] == {
        "classification": 500,
        "detection": 500,
    }
    assert plan["bootstrap_repetitions"] == 500
    assert plan["deepx_classification_preprocessing"] == "imagenet_mean_std"
    assert plan["deepx_full_cache_contract"] == "v2_exact_explicit"
    assert plan["deepx_cache_dir"] == CACHE_DIR


def test_v27541_profile_diff_from_v27540_b_is_strictly_allowlisted() -> None:
    baseline = yaml.safe_load(BASELINE_PROFILE.read_text(encoding="utf-8"))
    current = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    assert _leaf_differences(baseline, current) == {
        ("name",),
        ("purpose",),
        ("campaign", "dataset_manifests", "classification", "calibration"),
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
        (
            "execution_preset", "snapshot", "build", "deepx", "cache_dir",
        ),
    }


def test_v27541_preflight_is_read_only_and_proves_actual_b500_subset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    preflight = _load_preflight_module()
    current_root = tmp_path / "datasets" / "train_calibration_n1000_s20260710"
    baseline_root = tmp_path / "datasets" / "train_calibration_n500_s20260710"
    validation_root = tmp_path / "datasets" / "val"
    _write_images(current_root, start=0, stop=1000, prefix="train")
    _write_images(baseline_root, start=0, stop=500, prefix="train")
    _write_images(validation_root, start=2000, stop=3000, prefix="validation")
    current_manifest = _manifest(
        data_root=current_root,
        output=tmp_path / "dedicated" / "imagenet_train_calibration_manifest.json",
        role="calibration",
    )
    validation_manifest = _manifest(
        data_root=validation_root,
        output=tmp_path / "validation" / "imagenet_val_manifest.json",
        role="validation",
    )
    _bind_frozen_candidate_profile_fixture(
        preflight,
        tmp_path=tmp_path,
        calibration_manifest=current_manifest,
        validation_manifest=validation_manifest,
        monkeypatch=monkeypatch,
    )
    baseline_run = tmp_path / "baseline_run"
    run_local_baseline = _write_baseline_run(
        baseline_run, baseline_root=baseline_root
    )
    status = json.loads((
        baseline_run
        / "models/resnet50/benchmark_set/deepx/deepx_artifact_status.json"
    ).read_text(encoding="utf-8"))
    contract = status["cache_contract"]
    calibration_contract = contract["calibration_manifest_contract"]
    monkeypatch.setattr(
        preflight.canary_authority,
        "load_calibration_arm",
        lambda *_args, **_kwargs: SimpleNamespace(
            base=SimpleNamespace(
                cache_contract_sha256=contract["contract_sha256"],
                calibration_identity_sha256=calibration_contract["identity_sha256"],
                calibration_items_identity_sha256=(
                    calibration_contract["items_identity_sha256"]
                ),
                dxnn_sha256="d" * 64,
            ),
            calibration_items=tuple(range(500)),
        ),
    )
    monkeypatch.setattr(
        preflight,
        "validate_frozen_b500_authority",
        lambda _evidence: {
            "ok": True,
            "authority": "delivered_v2.75.40_arm_b",
            "observed": {},
        },
    )
    pinned_baseline = (
        tmp_path
        / "v27541_imagenet_n500_s20260710"
        / "manifests"
        / "imagenet_train_calibration_manifest.json"
    )
    pinned_baseline.parent.mkdir(parents=True, exist_ok=True)
    pinned_baseline.write_bytes(run_local_baseline.read_bytes())
    before = _tree_identity(tmp_path)

    report = preflight.build_preflight_report(
        calibration_manifest=current_manifest,
        validation_manifest=validation_manifest,
        baseline_run=baseline_run,
        baseline_calibration_manifest=pinned_baseline,
    )

    assert report["status"] == "ready"
    assert report["ready"] is True
    assert report["dataset_only_ready"] is True
    assert report["read_only"] is True
    assert report["network_used"] is False
    assert report["errors"] == []
    assert report["baseline"]["status"] == "pass"
    assert report["baseline"]["baseline_manifest_source"] == "pinned_or_explicit"
    assert report["baseline"]["resolved_baseline_manifest"] == str(
        pinned_baseline.resolve()
    )
    assert report["baseline"]["deepx_cache_binding"]["ok"] is True
    assert report["baseline"]["subset"] == {
        "ok": True,
        "relation": "B500_proper_subset_of_B1000",
        "baseline_unique_item_count": 500,
        "current_unique_item_count": 1000,
        "intersection_count": 500,
        "missing_from_b1000_count": 0,
        "missing_from_b1000_preview": [],
    }
    validation_check = next(
        row for row in report["checks"]
        if row["id"] == "validation_manifest_integrity"
    )
    validation_evidence = validation_check["evidence"]
    assert validation_evidence["manifest_verification"][
        "verification_mode"
    ] == "full"
    assert validation_evidence["manifest_verification"][
        "checked_item_count"
    ] == 1000
    assert validation_evidence["runtime_cohort"]["source_item_count"] == 1000
    assert validation_evidence["runtime_cohort"]["runtime_item_count"] == 500
    assert validation_evidence["runtime_cohort"][
        "source_and_runtime_cardinality_are_distinct"
    ] is True
    assert validation_evidence["runtime_cohort"]["read_only"] is True
    assert _tree_identity(tmp_path) == before


def test_v27541_valid_dataset_without_baseline_is_not_launch_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    preflight = _load_preflight_module()
    current_root = tmp_path / "train_calibration_n1000_s20260710"
    validation_root = tmp_path / "val"
    _write_images(current_root, start=0, stop=1000, prefix="train")
    _write_images(validation_root, start=2000, stop=2500, prefix="validation")
    current_manifest = _manifest(
        data_root=current_root,
        output=tmp_path / "dedicated" / "imagenet_train_calibration_manifest.json",
        role="calibration",
    )
    validation_manifest = _manifest(
        data_root=validation_root,
        output=tmp_path / "validation" / "imagenet_val_manifest.json",
        role="validation",
    )
    _bind_frozen_candidate_profile_fixture(
        preflight,
        tmp_path=tmp_path,
        calibration_manifest=current_manifest,
        validation_manifest=validation_manifest,
        monkeypatch=monkeypatch,
    )

    report = preflight.build_preflight_report(
        calibration_manifest=current_manifest,
        validation_manifest=validation_manifest,
    )

    assert report["dataset_only_ready"] is True
    assert report["ready"] is False
    assert report["status"] == "blocked"
    baseline = next(
        row for row in report["checks"]
        if row["id"] == "baseline_b500_proper_subset_of_b1000"
    )
    assert baseline["status"] == "fail"
    assert baseline["error"] == "required --baseline-run was not supplied"
    assert "--baseline-run" in report["remediation"]["launch_gate_note"]


def test_v27541_preflight_blocks_500_and_only_explains_safe_isolated_routes(
    tmp_path: Path,
) -> None:
    preflight = _load_preflight_module()
    calibration_root = tmp_path / "train_calibration_n500_s20260710"
    validation_root = tmp_path / "val"
    _write_images(calibration_root, start=0, stop=500, prefix="train")
    _write_images(validation_root, start=2000, stop=2001, prefix="validation")
    calibration_manifest = _manifest(
        data_root=calibration_root,
        output=tmp_path / "dedicated" / "imagenet_train_calibration_manifest.json",
        role="calibration",
    )
    validation_manifest = _manifest(
        data_root=validation_root,
        output=tmp_path / "validation" / "imagenet_val_manifest.json",
        role="validation",
    )
    before = _tree_identity(tmp_path)

    report = preflight.build_preflight_report(
        calibration_manifest=calibration_manifest,
        validation_manifest=validation_manifest,
    )

    assert report["status"] == "blocked"
    assert report["ready"] is False
    profile_binding = next(
        row for row in report["checks"]
        if row["id"] == "candidate_profile_manifest_paths"
    )
    assert profile_binding["status"] == "fail"
    calibration_check = next(
        row for row in report["checks"]
        if row["id"] == "calibration_manifest_exact_1000"
    )
    assert calibration_check["status"] == "fail"
    assert "expected_exactly_1000_items" in calibration_check["evidence"]["errors"]
    remediation = report["remediation"]
    assert remediation["automatic_action_taken"] is False
    assert "register-imagenet" in remediation["register_existing_exact_1000_item_root"]
    assert "provision-imagenet-kaggle" in remediation["provision_isolated_kaggle_copy"]
    assert "v27541_imagenet_n1000_s20260710/dataset_registry.json" in (
        remediation["register_existing_exact_1000_item_root"]
    )
    assert str(preflight.LEGACY_CALIBRATION_POINTER) not in (
        remediation["register_existing_exact_1000_item_root"]
    )
    assert _tree_identity(tmp_path) == before


def test_v27541_preflight_rejects_mutable_legacy_manifest_pointer() -> None:
    preflight = _load_preflight_module()
    report = preflight.build_preflight_report(
        calibration_manifest=preflight.LEGACY_CALIBRATION_POINTER,
        validation_manifest=Path("/definitely/missing/validation.json"),
    )
    dedicated = next(
        row for row in report["checks"]
        if row["id"] == "dedicated_calibration_manifest_namespace"
    )
    assert report["ready"] is False
    assert dedicated["status"] == "fail"
    assert any("must not use" in error for error in report["errors"])


def test_v27541_preflight_rejects_symlinked_manifest_leaf(
    tmp_path: Path,
) -> None:
    preflight = _load_preflight_module()
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    calibration_link = tmp_path / "calibration-link.json"
    calibration_link.symlink_to(target)
    report = preflight.build_preflight_report(
        calibration_manifest=calibration_link,
        validation_manifest=target,
        baseline_run=None,
    )
    check = next(
        row for row in report["checks"]
        if row["id"] == "calibration_manifest_exact_1000"
    )
    assert report["ready"] is False
    assert check["status"] == "fail"
    assert "not a regular file" in check["error"]


def test_v27541_baseline_binding_requires_full_receipt_and_dxnn_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    preflight = _load_preflight_module()
    run = tmp_path / "baseline"
    run.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")

    def reject(*_args, **_kwargs):
        raise ValueError("tampered build receipt")

    monkeypatch.setattr(
        preflight.canary_authority, "load_calibration_arm", reject,
    )
    with pytest.raises(
        preflight.PreflightError,
        match="exact cache/receipt/DXNN/quality validation",
    ):
        preflight._baseline_cache_binding_audit(run, manifest, {})


def test_v27541_baseline_binding_rejects_non_authoritative_b500(
    tmp_path: Path,
    monkeypatch,
) -> None:
    preflight = _load_preflight_module()
    run = tmp_path / "baseline"
    run.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        preflight.canary_authority,
        "load_calibration_arm",
        lambda *_args, **_kwargs: SimpleNamespace(
            base=SimpleNamespace(), calibration_items=(),
        ),
    )

    def reject(_evidence):
        raise preflight.PinError("source_onnx_sha256")

    monkeypatch.setattr(preflight, "validate_frozen_b500_authority", reject)
    with pytest.raises(
        preflight.PreflightError,
        match="not the frozen v2.75.40 Arm-B authority",
    ):
        preflight._baseline_cache_binding_audit(run, manifest, {})


def test_v27541_frozen_profile_paths_and_validation_authority_are_literal() -> None:
    preflight = _load_preflight_module()
    assert preflight._frozen_candidate_manifest_paths() == {
        "calibration": Path(CALIBRATION_MANIFEST),
        "validation": Path(VALIDATION_MANIFEST),
    }
    assert preflight.EXPECTED_VALIDATION_MANIFEST_SHA256 == (
        "64a7ef3bf55352bb39ef86f25fe73e6ff18a51ed1a02c8aca4940178f69e7b6c"
    )
    assert preflight.EXPECTED_VALIDATION_IMAGE_IDS_SHA256 == (
        "71032a98e158ca71711a5567de5d46fbf04f05777baa940f74f3c39fdf0c083f"
    )
    assert preflight.EXPECTED_VALIDATION_GROUND_TRUTH_SHA256 == (
        "87775e86ef3bcf3ec1e0d0a79696bf2f58fa215d5320ce80167bb1f417c7a177"
    )


def test_validation_projection_is_byte_identical_to_production_materializer(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "imagenet_val"
    _write_images(source_root, start=0, stop=600, prefix="validation")
    source_manifest = _manifest(
        data_root=source_root,
        output=tmp_path / "manifests" / "imagenet_val_manifest.json",
        role="validation",
    )
    source_before = _tree_identity(tmp_path)

    projection = _runtime_projection(source_manifest)
    expected_bytes = (
        json.dumps(
            projection["manifest"], indent=2, ensure_ascii=False,
        ) + "\n"
    ).encode("utf-8")
    expected_authorities = _projection_authorities(projection)

    # Projection is the preflight operation and must not materialise anything.
    assert _tree_identity(tmp_path) == source_before
    suite_dir = tmp_path / "suite"
    relative = provision_classification_validation_source_to_suite(
        suite_dir,
        str(source_root.resolve()),
        base_dir=suite_dir,
        max_images=500,
        manifest_path=source_manifest,
        selection_seed=20260710,
    )
    assert relative == projection["destination_relative"].as_posix()
    materialized_manifest = suite_dir / str(relative) / "manifest.json"
    assert materialized_manifest.read_bytes() == expected_bytes
    assert _sha256_file(materialized_manifest) == expected_authorities["manifest"]

    materialized = json.loads(materialized_manifest.read_text(encoding="utf-8"))
    records = sorted(({
        "image_id": Path(str(sample["image"])).name,
        "label_id": int(sample["label_id"]),
    } for sample in materialized["samples"]), key=lambda row: row["image_id"])
    assert image_ids_fingerprint([
        row["image_id"] for row in records
    ]) == expected_authorities["image_ids"]
    assert json_fingerprint(records) == expected_authorities["ground_truth"]
    # Materialising the transport copy must not mutate the source authority.
    assert {
        key: value for key, value in _tree_identity(tmp_path).items()
        if not key.startswith("suite/")
    } == source_before


def test_validation_projection_resolves_labels_through_synset_authority(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "imagenet_val"
    _write_images(source_root, start=0, stop=6, prefix="validation")
    source_manifest = _manifest(
        data_root=source_root,
        output=tmp_path / "manifests" / "imagenet_val_manifest.json",
        role="validation",
    )
    synsets = source_manifest.parent / "LOC_synset_mapping.txt"
    synsets.write_text("".join(
        f"n{index:08d} synthetic label {index}\n"
        for index in range(6)
    ), encoding="utf-8")
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    for item in payload["items"]:
        item.pop("label_id", None)
        item.pop("label_name", None)
    payload["labels"] = {
        "path": synsets.name,
        "sha256": _sha256_file(synsets),
    }
    payload["manifest_payload_sha256"] = sha256_json({
        key: value for key, value in payload.items()
        if key != "manifest_payload_sha256"
    })
    source_manifest.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )
    before = _tree_identity(tmp_path)

    projection = project_classification_validation_subset(
        resolved=source_root,
        requested=str(source_root),
        max_images=4,
        selection_seed=20260710,
        explicit_manifest=source_manifest,
        base_dir=source_manifest.parent,
    )
    assert projection is not None
    assert all(
        "label_id" not in item and "label_name" not in item
        for item in payload["items"]
    )
    for sample in projection["manifest"]["samples"]:
        expected_id = int(str(sample["class_name"])[1:])
        assert sample["label_id"] == expected_id
        assert sample["label_name"] == f"synthetic label {expected_id}"
    expected_bytes = (
        json.dumps(
            projection["manifest"], indent=2, ensure_ascii=False,
        ) + "\n"
    ).encode("utf-8")
    assert _tree_identity(tmp_path) == before

    suite_dir = tmp_path / "suite"
    relative = provision_classification_validation_source_to_suite(
        suite_dir,
        str(source_root.resolve()),
        base_dir=suite_dir,
        max_images=4,
        manifest_path=source_manifest,
        selection_seed=20260710,
    )
    assert relative == projection["destination_relative"].as_posix()
    assert (suite_dir / str(relative) / "manifest.json").read_bytes() == (
        expected_bytes
    )
    assert {
        key: value for key, value in _tree_identity(tmp_path).items()
        if not key.startswith("suite/")
    } == before


def test_validation_projection_distinguishes_500_runtime_from_50000_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    preflight = _load_preflight_module()
    source_root = tmp_path / "imagenet_val_50000"
    source_root.mkdir()
    items: list[dict[str, Any]] = []
    identity_rows: list[dict[str, str]] = []
    for index in range(50000):
        label_id = index % 1000
        class_name = f"n{label_id:08d}"
        relative_path = (
            f"{class_name}/ILSVRC2012_val_{index + 1:08d}.JPEG"
        )
        digest = hashlib.sha256(
            f"validation-image-{index}".encode("utf-8")
        ).hexdigest()
        items.append({
            "sample_id": relative_path,
            "relative_path": relative_path,
            "size_bytes": 1000 + index,
            "sha256": digest,
            "class_name": class_name,
            "label_id": label_id,
            "label_name": class_name,
        })
        identity_rows.append({
            "sample_id": relative_path,
            "relative_path": relative_path,
            "sha256": digest,
            "class_name": class_name,
        })
    payload: dict[str, Any] = {
        "schema": "onnx-splitpoint/dataset-manifest",
        "schema_version": 1,
        "dataset_id": "ilsvrc2012-val",
        "task": "classification",
        "role": "validation",
        "split": "val",
        "root": str(source_root.resolve()),
        "hash_mode": "content",
        "item_count": 50000,
        "items": items,
        "items_identity_sha256": _canonical_sha256(identity_rows),
        "annotations": {"path": "", "sha256": ""},
        "labels": {"path": "", "sha256": ""},
    }
    payload["manifest_payload_sha256"] = sha256_json(payload)
    source_manifest = tmp_path / "imagenet_val_manifest.json"
    source_manifest.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )
    before = _tree_identity(tmp_path)
    projection = _runtime_projection(source_manifest)
    authorities = _projection_authorities(projection)
    monkeypatch.setattr(
        preflight, "EXPECTED_VALIDATION_MANIFEST_SHA256",
        authorities["manifest"],
    )
    monkeypatch.setattr(
        preflight, "EXPECTED_VALIDATION_IMAGE_IDS_SHA256",
        authorities["image_ids"],
    )
    monkeypatch.setattr(
        preflight, "EXPECTED_VALIDATION_GROUND_TRUTH_SHA256",
        authorities["ground_truth"],
    )

    audit = preflight._runtime_validation_cohort_audit(
        source_manifest, payload,
    )
    assert audit["ok"] is True
    assert audit["source_item_count"] == 50000
    assert audit["runtime_item_count"] == 500
    assert audit["selection"] == {
        "type": "deterministic_class_stratified",
        "seed": 20260710,
        "requested_images": 500,
        "selected_images": 500,
        "source_population": 50000,
    }
    assert audit["source_and_runtime_cardinality_are_distinct"] is True
    assert preflight._portable_dataset_manifest_sha256(payload) != (
        audit["projected_manifest_sha256"]
    )
    assert _tree_identity(tmp_path) == before


def test_runtime_validation_authority_blocks_semantic_and_byte_tampering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    preflight = _load_preflight_module()
    source_root = tmp_path / "imagenet_val"
    _write_images(source_root, start=0, stop=600, prefix="validation")
    source_manifest = _manifest(
        data_root=source_root,
        output=tmp_path / "manifests" / "imagenet_val_manifest.json",
        role="validation",
    )
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    clean_projection = _runtime_projection(source_manifest)
    authorities = _projection_authorities(clean_projection)
    for attribute, key in (
        ("EXPECTED_VALIDATION_MANIFEST_SHA256", "manifest"),
        ("EXPECTED_VALIDATION_IMAGE_IDS_SHA256", "image_ids"),
        ("EXPECTED_VALIDATION_GROUND_TRUTH_SHA256", "ground_truth"),
    ):
        monkeypatch.setattr(preflight, attribute, authorities[key])
    original_projector = preflight.project_classification_validation_subset
    before = _tree_identity(tmp_path)

    clean = preflight._runtime_validation_cohort_audit(
        source_manifest, source_payload,
    )
    assert clean["ok"] is True

    def run_tamper(mutator) -> dict[str, Any]:
        def tampered_projector(**kwargs):
            projected = original_projector(**kwargs)
            assert projected is not None
            result = dict(projected)
            result["manifest"] = json.loads(json.dumps(projected["manifest"]))
            mutator(result["manifest"])
            return result

        monkeypatch.setattr(
            preflight,
            "project_classification_validation_subset",
            tampered_projector,
        )
        audit = preflight._runtime_validation_cohort_audit(
            source_manifest, source_payload,
        )
        monkeypatch.setattr(
            preflight,
            "project_classification_validation_subset",
            original_projector,
        )
        return audit

    def reverse_order(manifest: dict[str, Any]) -> None:
        manifest["samples"].reverse()

    def change_label(manifest: dict[str, Any]) -> None:
        manifest["samples"][0]["label_id"] += 1

    def change_path(manifest: dict[str, Any]) -> None:
        sample = manifest["samples"][0]
        sample["image"] = "images/changed/" + Path(sample["image"]).name

    def change_sha_prefix(manifest: dict[str, Any]) -> None:
        sample = manifest["samples"][0]
        value = str(sample["source_sha256"])
        sample["source_sha256"] = (
            value.removeprefix("sha256:")
            if value.startswith("sha256:") else f"sha256:{value}"
        )

    def change_label_name(manifest: dict[str, Any]) -> None:
        manifest["samples"][0]["label_name"] += "-tampered"

    def duplicate_basename(manifest: dict[str, Any]) -> None:
        first = Path(manifest["samples"][0]["image"]).name
        manifest["samples"][1]["image"] = f"images/duplicate/{first}"

    for mutator in (
        reverse_order,
        change_path,
        change_sha_prefix,
        change_label_name,
    ):
        audit = run_tamper(mutator)
        assert audit["ok"] is False
        assert (
            "validation_runtime_cohort_manifest_differs_from_frozen_v27540_b_authority"
            in audit["errors"]
        )

    label_audit = run_tamper(change_label)
    assert label_audit["ok"] is False
    assert (
        "validation_runtime_cohort_ground_truth_differs_from_frozen_v27540_b_authority"
        in label_audit["errors"]
    )

    duplicate_audit = run_tamper(duplicate_basename)
    assert duplicate_audit["ok"] is False
    assert "validation_runtime_cohort_image_ids_not_unique" in (
        duplicate_audit["errors"]
    )

    seed_audit = preflight._runtime_validation_cohort_audit(
        source_manifest,
        source_payload,
        selection_seed=20260711,
    )
    assert seed_audit["ok"] is False
    assert (
        "validation_runtime_cohort_cardinality_seed_or_population_invalid"
        in seed_audit["errors"]
    )
    assert (
        "validation_runtime_cohort_manifest_differs_from_frozen_v27540_b_authority"
        in seed_audit["errors"]
    )
    assert _tree_identity(tmp_path) == before


def test_v27541_portable_validation_identity_accepts_production_legacy_hash(
) -> None:
    preflight = _load_preflight_module()
    items = []
    for index in range(500):
        digest = hashlib.sha256(f"validation-image-{index}".encode()).hexdigest()
        relative_path = (
            f"n{index % 1000:08d}/ILSVRC2012_val_{index + 1:08d}.JPEG"
        )
        items.append({
            "class_name": f"n{index % 1000:08d}",
            "relative_path": relative_path,
            "sample_id": relative_path,
            "sha256": f"sha256:{digest}",
            "size_bytes": 1000 + index,
        })

    legacy_rows = [{
        "sample_id": item["sample_id"],
        "relative_path": item["relative_path"],
        "sha256": item["sha256"],
        "class_name": item["class_name"],
    } for item in items]
    normalized_rows = [{
        "sample_id": item["sample_id"],
        "relative_path": item["relative_path"],
        "sha256": str(item["sha256"]).removeprefix("sha256:"),
        "class_name": item["class_name"],
    } for item in items]
    legacy_items_sha = _canonical_sha256(legacy_rows)
    normalized_items_sha = _canonical_sha256(normalized_rows)
    assert legacy_items_sha != normalized_items_sha

    payload = {
        "schema": "onnx-splitpoint/dataset-manifest",
        "schema_version": 1,
        "dataset_id": "ilsvrc2012-val",
        "task": "classification",
        "role": "validation",
        "split": "val",
        "hash_mode": "content",
        "item_count": len(items),
        "items": items,
        "items_identity_sha256": f"sha256:{legacy_items_sha}",
        "annotations": {"path": "", "sha256": ""},
        "labels": {
            "path": "/dataset/LOC_synset_mapping.txt",
            "sha256": f"sha256:{'a' * 64}",
        },
    }
    expected_portable = _canonical_sha256({
        "schema": "onnx-splitpoint/portable-dataset-identity",
        "schema_version": 1,
        "dataset_id": "ilsvrc2012-val",
        "task": "classification",
        "role": "validation",
        "split": "val",
        "hash_mode": "content",
        "item_count": len(items),
        "items_identity_sha256": normalized_items_sha,
        "annotations_sha256": "",
        "labels_sha256": "a" * 64,
    })

    assert preflight._portable_dataset_manifest_sha256(payload) == (
        expected_portable
    )
    payload["items_identity_sha256"] = f"sha256:{normalized_items_sha}"
    assert preflight._portable_dataset_manifest_sha256(payload) == (
        expected_portable
    )


def test_v27541_portable_validation_identity_rejects_unknown_declared_hash(
) -> None:
    preflight = _load_preflight_module()
    digest = hashlib.sha256(b"validation-image").hexdigest()
    payload = {
        "schema": "onnx-splitpoint/dataset-manifest",
        "schema_version": 1,
        "dataset_id": "ilsvrc2012-val",
        "task": "classification",
        "role": "validation",
        "split": "val",
        "hash_mode": "content",
        "item_count": 1,
        "items": [{
            "class_name": "n01440764",
            "relative_path": "n01440764/ILSVRC2012_val_00000001.JPEG",
            "sample_id": "n01440764/ILSVRC2012_val_00000001.JPEG",
            "sha256": f"sha256:{digest}",
            "size_bytes": 1234,
        }],
        "items_identity_sha256": f"sha256:{'f' * 64}",
        "annotations": {"path": "", "sha256": ""},
        "labels": {"path": "", "sha256": ""},
    }

    with pytest.raises(
        preflight.PreflightError,
        match="portable validation item identity SHA-256 mismatch",
    ):
        preflight._portable_dataset_manifest_sha256(payload)


def test_v27541_strict_calibration_manifest_rejects_lookalike_dataset(
    tmp_path: Path,
) -> None:
    preflight = _load_preflight_module()
    root = tmp_path / "train"
    _write_images(root, start=0, stop=1, prefix="train")
    manifest = _manifest(
        data_root=root,
        output=tmp_path / "manifest.json",
        role="calibration",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert preflight._strict_calibration_manifest_audit(
        manifest, payload, expected_count=1, label="fixture",
    )["ok"] is True
    payload["dataset_id"] = "lookalike-train-calibration"
    payload["manifest_payload_sha256"] = sha256_json({
        key: value for key, value in payload.items()
        if key != "manifest_payload_sha256"
    })
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(preflight.PreflightError, match="dataset_id mismatch"):
        preflight._strict_calibration_manifest_audit(
            manifest, payload, expected_count=1, label="fixture",
        )


def test_v27541_subset_identity_includes_path_and_size() -> None:
    preflight = _load_preflight_module()

    def item(index: int) -> dict[str, object]:
        return {
            "sample_id": f"train-{index:04d}",
            "relative_path": f"n{index:08d}/train-{index:04d}.JPEG",
            "size_bytes": 100 + index,
            "sha256": hashlib.sha256(f"image-{index}".encode()).hexdigest(),
            "class_name": f"n{index:08d}",
        }

    baseline = {"items": [item(index) for index in range(500)]}
    candidate = {"items": [item(index) for index in range(1000)]}
    assert preflight._subset_audit(baseline, candidate)["ok"] is True
    for field, value in (
        ("relative_path", "n00000000/renamed.JPEG"),
        ("size_bytes", 999999),
    ):
        changed = json.loads(json.dumps(candidate))
        changed["items"][0][field] = value
        assert preflight._subset_audit(baseline, changed)["ok"] is False
