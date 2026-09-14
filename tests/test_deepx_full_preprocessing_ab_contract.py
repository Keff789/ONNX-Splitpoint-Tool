from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.campaign import create_dataset_manifest
from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
import onnx_splitpoint_tool.workflow.deepx_build_binding as deepx_binding
from onnx_splitpoint_tool.deepx.artifacts import (
    cache_dxnn_artifact,
    deepx_cache_key,
    deepx_cached_artifact_compatible,
)
from onnx_splitpoint_tool.deepx.config import (
    CLASSIFICATION_PREPROCESSING_CURRENT,
    CLASSIFICATION_PREPROCESSING_IMAGENET,
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    deepx_classification_preprocessing_contract,
    image_model_dxcom_config,
    materialize_imagenet_normalized_build_onnx,
)
from onnx_splitpoint_tool.workflow.deepx_build_binding import (
    _deepx_calibration_manifest_contract,
    _deepx_compiler_identity,
    _explicit_classification_v2_contract_requested,
    _lookup_full_deepx_cache_candidate,
    _deepx_full_preprocessing_contract,
    build_full_deepx_cache_contract,
    task_bound_deepx_runtime_input_contract,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_calibration_contract(token: str = "a") -> dict[str, object]:
    return {
        "schema": "onnx-splitpoint/deepx-calibration-manifest-contract",
        "schema_version": 1,
        "status": "resolved",
        "task": "classification",
        "effective_count": 500,
        "manifest_file_sha256": token * 64,
        "items_identity_sha256": "sha256:" + token * 64,
        "identity_sha256": ("b" if token == "a" else "c") * 64,
    }


def _resolved_compiler_identity(token: str = "d") -> dict[str, object]:
    return {
        "schema": "onnx-splitpoint/deepx-compiler-identity",
        "schema_version": 1,
        "status": "resolved",
        "evidence": {"dx_com_version": f"2.3.{token}"},
        "identity_sha256": token * 64,
    }


@pytest.mark.parametrize(
    ("mode", "cache_dir"),
    (
        (CLASSIFICATION_PREPROCESSING_CURRENT, "/cache/deepx-ab/current"),
        (CLASSIFICATION_PREPROCESSING_IMAGENET, "/cache/deepx-ab/imagenet"),
    ),
)
def test_frozen_run_mode_projects_each_explicit_ab_arm_into_plan(
    mode: str, cache_dir: str,
) -> None:
    config = default_run_modes_config()
    snapshot = copy.deepcopy(config["modes"]["standard"])
    snapshot["build"]["deepx"]["classification_preprocessing"] = mode
    snapshot["build"]["deepx"]["cache_dir"] = cache_dir
    source = {
        "name": f"deepx-ab-{mode}",
        "purpose": "test",
        "model_suite": {
            "primary": [{
                "id": "resnet50", "task": "classification", "enabled": True,
            }],
        },
        "selection_policy": {
            "max_accepted_cases_per_model": 1,
            "preferred_shortlist": 1,
        },
        "run_profiles": [{
            "id": "deepx_m1_full", "type": "same_backend_reference",
            "full": "deepx_m1", "enabled": True,
        }],
        "execution_preset": {
            "id": "standard",
            "follow_tool_config": False,
            "snapshot": snapshot,
            "overrides": {"native_enabled": False, "energy_enabled": False},
        },
    }
    resolved, _ = apply_run_mode(source, config=config)
    assert validate_evaluation_profile_payload(resolved, source="A/B test") == resolved
    assert resolved["deepx_build"]["classification_preprocessing"] == mode
    assert resolved["deepx_build"]["cache_dir"] == cache_dir
    plan = build_effective_execution_plan(resolved)
    assert plan["deepx_classification_preprocessing"] == mode
    assert plan["deepx_classification_preprocessing_explicit"] is True
    assert plan["deepx_full_cache_contract"] == "v2_exact_explicit"
    assert plan["deepx_cache_dir"] == cache_dir


def test_legacy_and_detection_profiles_do_not_enter_classification_v2_cache() -> None:
    assert not _explicit_classification_v2_contract_requested(
        task="classification", cfg={},
    )
    assert _explicit_classification_v2_contract_requested(
        task="classification",
        cfg={"classification_preprocessing": "current_scale_only"},
    )
    assert _explicit_classification_v2_contract_requested(
        task="classification",
        cfg={"classification_preprocessing": "imagenet_mean_std"},
    )
    assert not _explicit_classification_v2_contract_requested(
        task="detection",
        cfg={"classification_preprocessing": "imagenet_mean_std"},
    )

    config = default_run_modes_config()
    source = {
        "name": "legacy-standard",
        "purpose": "test",
        "model_suite": {
            "primary": [{"id": "resnet50", "task": "classification"}],
        },
        "run_profiles": [],
        "execution_preset": {
            "id": "standard",
            "follow_tool_config": False,
            "snapshot": config["modes"]["standard"],
            "overrides": {"native_enabled": False, "energy_enabled": False},
        },
    }
    resolved, _ = apply_run_mode(source, config=config)
    # v31 regular profiles now explicitly select the verified numeric arm.
    assert resolved["deepx_build"]["classification_preprocessing"] == "imagenet_mean_std"
    assert build_effective_execution_plan(resolved)[
        "deepx_full_cache_contract"
    ] == "v2_exact_explicit"


def test_v2_lookup_never_materializes_artifact_store_and_force_skips_all_lookup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[str] = []
    exact = tmp_path / "exact.dxnn"
    legacy = tmp_path / "legacy.dxnn"

    def exact_lookup(_root: Path, _key: str) -> Path:
        calls.append("exact_local")
        return exact

    def wrapped_lookup(_root: Path, _key: str) -> Path:
        calls.append("artifact_store_capable")
        return legacy

    monkeypatch.setattr(deepx_binding, "_find_cached_dxnn_legacy", exact_lookup)
    monkeypatch.setattr(deepx_binding, "_find_cached_dxnn", wrapped_lookup)
    assert _lookup_full_deepx_cache_candidate(
        cache_root=tmp_path, cache_key="v2", use_v2=True, force_build=False,
    ) == exact
    assert calls == ["exact_local"]

    calls.clear()
    assert _lookup_full_deepx_cache_candidate(
        cache_root=tmp_path, cache_key="legacy", use_v2=False,
        force_build=False,
    ) == legacy
    assert calls == ["artifact_store_capable"]

    calls.clear()
    assert _lookup_full_deepx_cache_candidate(
        cache_root=tmp_path, cache_key="v2", use_v2=True, force_build=True,
    ) is None
    assert calls == []


def test_classification_build_binding_emits_model_id_in_each_input_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "mobilenet_v3_large.onnx"
    source.write_bytes(b"classification-onnx")
    cached = tmp_path / "cache" / "entry" / "model.dxnn"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"classification-dxnn")
    suite = tmp_path / "suite"
    suite.mkdir()

    monkeypatch.setattr(
        deepx_binding,
        "inspect_deepx_environment",
        lambda **_kwargs: {
            "compiler_ready": False,
            "runtime_ready": True,
            "cache_dir": str(tmp_path / "cache"),
        },
    )
    monkeypatch.setattr(
        deepx_binding,
        "_onnx_first_input_info",
        lambda _path: ("images", [1, 3, 224, 224]),
    )
    monkeypatch.setattr(
        deepx_binding,
        "_lookup_full_deepx_cache_candidate",
        lambda **_kwargs: cached,
    )
    monkeypatch.setattr(
        deepx_binding,
        "deepx_cached_artifact_identity_compatible",
        lambda **_kwargs: (True, "", {}),
    )

    # This existing fixture exercises only the archived legacy output identity.
    # v31 regular mean/std routing has its own real-ONNX integration tests.
    monkeypatch.setattr(deepx_binding, "_explicit_classification_v2_contract_requested", lambda **_kwargs: False)
    result = deepx_binding.materialize_deepx_build_binding(
        run_dir=tmp_path / "run",
        model_id="mobilenet_v3_large",
        model_path=str(source),
        row={"task": "classification", "input_shape": [1, 3, 224, 224]},
        profile_payload={
            "deepx_build": {
                "mode": "reuse_only",
                "classification_preprocessing": "current_scale_only",
                "diagnostic_only": True,
                "cache_dir": str(tmp_path / "cache"),
            },
        },
        targets=["deepx_m1"],
        benchmark_set_contract={"suite_dir": str(suite)},
    )

    assert result["status"] == "ok"
    aggregate = json.loads(
        Path(result["artifacts"]["deepx_output_contracts_json"])
        .read_text(encoding="utf-8")
    )
    assert aggregate["model_id"] == "mobilenet_v3_large"
    assert len(aggregate["contracts"]) == 1
    contract = aggregate["contracts"][0]
    assert contract["model_id"] == "mobilenet_v3_large"
    assert contract["contract_family"] == "classification_logits"
    suite_contract = json.loads(
        (suite / "deepx" / "deepx_m1" / "full" / "output_contract.json")
        .read_text(encoding="utf-8")
    )
    assert suite_contract["model_id"] == "mobilenet_v3_large"


def test_ab_modes_keep_proven_dxcom_loader_identical_and_explicit(tmp_path: Path) -> None:
    common = {
        "task": "classification",
        "input_name": "images",
        "input_shape": [1, 3, 224, 224],
        "calibration_dir": tmp_path,
        "calibration_num": 500,
        "calibration_method": "ema",
    }
    current = image_model_dxcom_config(
        **common,
        classification_preprocessing=CLASSIFICATION_PREPROCESSING_CURRENT,
    )
    corrected = image_model_dxcom_config(
        **common,
        classification_preprocessing=CLASSIFICATION_PREPROCESSING_IMAGENET,
    )

    assert current == corrected
    operations = current["default_loader"]["preprocessings"]
    assert operations == [
        {"resize": {"width": 224, "height": 224}},
        {"div": {"x": 255.0}},
        {"convertColor": {"form": "BGR2RGB"}},
        {"transpose": {"axis": [2, 0, 1]}},
        {"expandDim": {"axis": 0}},
    ]
    # No guessed vendor-side mean/std opcode may enter the corrected arm.
    serialized = json.dumps(corrected, sort_keys=True).lower()
    assert "mean" not in serialized
    assert "std" not in serialized
    assert "subtract" not in serialized

    current_contract = deepx_classification_preprocessing_contract(
        CLASSIFICATION_PREPROCESSING_CURRENT
    )
    corrected_contract = deepx_classification_preprocessing_contract(
        CLASSIFICATION_PREPROCESSING_IMAGENET
    )
    assert current_contract["build_onnx_adapter"]["kind"] == "identity"
    assert corrected_contract["build_onnx_adapter"] == {
        "kind": "imagenet_rgb_mean_std",
        "implementation": "onnx_sub_then_div_before_original_graph",
        "input_domain": "rgb_float32_0_1",
        "mean": list(IMAGENET_RGB_MEAN),
        "std": list(IMAGENET_RGB_STD),
        "broadcast_shape": [1, 3, 1, 1],
    }
    with pytest.raises(ValueError, match="classification_preprocessing"):
        image_model_dxcom_config(
            **common, classification_preprocessing="imagenet_typo",
        )


def test_corrected_runtime_contract_names_the_exact_embedded_numeric_path() -> None:
    current = task_bound_deepx_runtime_input_contract(
        task="classification",
        input_name="images",
        source_shape=[1, 3, 224, 224],
        classification_preprocessing=CLASSIFICATION_PREPROCESSING_CURRENT,
    )
    corrected = task_bound_deepx_runtime_input_contract(
        task="classification",
        input_name="images",
        source_shape=[1, 3, 224, 224],
        classification_preprocessing=CLASSIFICATION_PREPROCESSING_IMAGENET,
    )
    assert current["shape"] == corrected["shape"] == [224, 224, 3]
    assert current["dtype"] == corrected["dtype"] == "uint8"
    assert current["color_space"] == corrected["color_space"] == "RGB"
    assert current["embedded_numeric_path"] == "dxcom_div255_only"
    assert (
        corrected["embedded_numeric_path"]
        == "dxcom_div255_then_build_onnx_imagenet_mean_std"
    )


def test_generated_imagenet_build_onnx_is_content_addressed_and_ort_equivalent(
    tmp_path: Path,
) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    source = tmp_path / "source.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["logits"])],
        "source",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3, 2, 2])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="deepx-ab-test",
    )
    model.ir_version = min(int(model.ir_version), 9)
    onnx.save_model(model, str(source))
    source_sha256 = _sha256(source)

    wrapped, receipt = materialize_imagenet_normalized_build_onnx(
        source_onnx=source,
        output_dir=tmp_path / "build",
        input_name="images",
    )
    repeated, repeated_receipt = materialize_imagenet_normalized_build_onnx(
        source_onnx=source,
        output_dir=tmp_path / "build",
        input_name="images",
    )
    assert wrapped == repeated
    assert receipt["build_onnx_sha256"] == repeated_receipt["build_onnx_sha256"]
    assert wrapped.name == f"imagenet_mean_std_{_sha256(wrapped)}.onnx"
    assert _sha256(source) == source_sha256

    wrapped_model = onnx.load(str(wrapped))
    assert [node.op_type for node in wrapped_model.graph.node[:2]] == ["Sub", "Div"]
    rng = np.random.default_rng(42)
    scaled = rng.random((1, 3, 2, 2), dtype=np.float32)
    mean = np.asarray(IMAGENET_RGB_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(IMAGENET_RGB_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    normalized = (scaled - mean) / std
    source_session = ort.InferenceSession(
        str(source), providers=["CPUExecutionProvider"],
    )
    wrapped_session = ort.InferenceSession(
        str(wrapped), providers=["CPUExecutionProvider"],
    )
    expected = source_session.run(None, {"images": normalized})[0]
    observed = wrapped_session.run(None, {"images": scaled})[0]
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-6)


def test_full_v2_cache_contract_isolates_modes_manifest_and_compiler(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.onnx"
    corrected = tmp_path / "corrected.onnx"
    config = tmp_path / "config.json"
    source.write_bytes(b"source-model")
    corrected.write_bytes(b"corrected-build-model")
    config.write_text('{"loader":"proven"}\n', encoding="utf-8")

    def make_contract(
        *, mode: str, calibration: str = "a", compiler: str = "d",
    ) -> dict[str, object]:
        runtime = task_bound_deepx_runtime_input_contract(
            task="classification",
            input_name="images",
            source_shape=[1, 3, 224, 224],
            classification_preprocessing=mode,
        )
        preprocessing = _deepx_full_preprocessing_contract(
            task="classification",
            classification_mode=mode,
            runtime_input=runtime,
        )
        return build_full_deepx_cache_contract(
            task="classification",
            classification_mode=mode,
            source_onnx_path=source,
            build_onnx_path=(
                corrected
                if mode == CLASSIFICATION_PREPROCESSING_IMAGENET else source
            ),
            config_path=config,
            preprocessing_contract=preprocessing,
            calibration_contract=_resolved_calibration_contract(calibration),
            compiler_identity=_resolved_compiler_identity(compiler),
            calibration_method="ema",
            calibration_count=500,
            opt_level=0,
        )

    current_contract = make_contract(mode=CLASSIFICATION_PREPROCESSING_CURRENT)
    corrected_contract = make_contract(mode=CLASSIFICATION_PREPROCESSING_IMAGENET)
    changed_calibration = make_contract(
        mode=CLASSIFICATION_PREPROCESSING_CURRENT, calibration="e",
    )
    changed_compiler = make_contract(
        mode=CLASSIFICATION_PREPROCESSING_CURRENT, compiler="f",
    )
    contracts = [
        current_contract, corrected_contract, changed_calibration, changed_compiler,
    ]
    keys = {
        deepx_cache_key(
            onnx_path=(
                corrected
                if contract["classification_preprocessing"]
                == CLASSIFICATION_PREPROCESSING_IMAGENET else source
            ),
            config_path=config,
            target="deepx_m1",
            variant=(
                "full_imagenet_mean_std"
                if contract["classification_preprocessing"]
                == CLASSIFICATION_PREPROCESSING_IMAGENET
                else "full_current_scale_only"
            ),
            cache_contract=contract,
        )
        for contract in contracts
    }
    assert len(keys) == 4
    assert all("_v2_" in key for key in keys)
    assert all(not key.startswith("deepx_m1_full_v2_") for key in keys)

    dxnn = tmp_path / "model.dxnn"
    dxnn.write_bytes(b"a/b artifact")
    current_key = deepx_cache_key(
        onnx_path=source,
        config_path=config,
        target="deepx_m1",
        variant="full_current_scale_only",
        cache_contract=current_contract,
    )
    cache_root = tmp_path / "cache"
    cached = cache_dxnn_artifact(
        dxnn_path=dxnn,
        cache_root=cache_root,
        cache_key=current_key,
        manifest={
            "schema": "onnx-splitpoint/deepx-full-cache-receipt",
            "schema_version": 2,
            "cache_contract": current_contract,
        },
        allow_overwrite=False,
    )
    ok, reason, _ = deepx_cached_artifact_compatible(
        cache_dir=cached.parent,
        expected_contract=current_contract,
        require_artifact_identity=True,
    )
    assert ok, reason
    cached.write_bytes(b"tampered")
    ok, reason, _ = deepx_cached_artifact_compatible(
        cache_dir=cached.parent,
        expected_contract=current_contract,
        require_artifact_identity=True,
    )
    assert not ok
    assert reason == "artifact_identity_mismatch"


def test_full_cache_rejects_legacy_receipt_without_artifact_identity(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "legacy"
    cache_dir.mkdir()
    (cache_dir / "model.dxnn").write_bytes(b"legacy")
    contract = {"schema": "example", "status": "resolved"}
    (cache_dir / "build_manifest.json").write_text(
        json.dumps({"cache_contract": contract}), encoding="utf-8",
    )
    ok, reason, _ = deepx_cached_artifact_compatible(
        cache_dir=cache_dir,
        expected_contract=contract,
        require_artifact_identity=True,
    )
    assert not ok
    assert reason == "artifact_identity_missing"


def test_calibration_contract_binds_manifest_bytes_and_rejects_root_mismatch(
    tmp_path: Path,
) -> None:
    calibration_root = tmp_path / "calibration"
    (calibration_root / "class_a").mkdir(parents=True)
    (calibration_root / "class_b").mkdir(parents=True)
    (calibration_root / "class_a" / "a.jpg").write_bytes(b"image-a")
    (calibration_root / "class_b" / "b.png").write_bytes(b"image-b")
    manifest = tmp_path / "calibration.json"
    create_dataset_manifest(
        task="classification",
        role="calibration",
        dataset_id="imagenet-calibration",
        split="train",
        root=calibration_root,
        output=manifest,
        hash_mode="content",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    profile = {
        "campaign": {
            "dataset_manifests": {
                "classification": {"calibration": str(manifest)},
            }
        }
    }
    contract = _deepx_calibration_manifest_contract(
        profile_payload=profile,
        task="classification",
        calibration_dir=str(calibration_root),
        effective_count=2,
    )
    assert contract["status"] == "resolved"
    assert contract["manifest_file_sha256"] == _sha256(manifest)
    assert contract["items_identity_sha256"] == payload["items_identity_sha256"]
    assert contract["manifest_verification"]["ok"] is True
    assert contract["manifest_verification"]["checked_item_count"] == 2
    assert contract["root_inventory_count"] == 2
    assert len(contract["root_inventory_sha256"]) == 64

    wrong_root = tmp_path / "wrong"
    wrong_root.mkdir()
    rejected = _deepx_calibration_manifest_contract(
        profile_payload=profile,
        task="classification",
        calibration_dir=str(wrong_root),
        effective_count=2,
    )
    assert rejected["status"] == "invalid"
    assert "calibration_dir_manifest_root_mismatch" in rejected["reason"]

    # An image that DX-COM would scan but the manifest does not bind must make
    # the v2 cache contract unavailable.
    (calibration_root / "class_b" / "extra.jpeg").write_bytes(b"extra")
    extra_rejected = _deepx_calibration_manifest_contract(
        profile_payload=profile,
        task="classification",
        calibration_dir=str(calibration_root),
        effective_count=2,
    )
    assert extra_rejected["status"] == "invalid"
    assert "calibration_inventory_manifest_mismatch" in extra_rejected["reason"]


def test_calibration_contract_rejects_tampered_manifest_payload(
    tmp_path: Path,
) -> None:
    root = tmp_path / "calibration"
    root.mkdir()
    (root / "one.jpg").write_bytes(b"one")
    manifest = tmp_path / "calibration.json"
    create_dataset_manifest(
        task="classification",
        role="calibration",
        dataset_id="original",
        split="train",
        root=root,
        output=manifest,
        hash_mode="content",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["dataset_id"] = "tampered-without-rehash"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    profile = {
        "campaign": {
            "dataset_manifests": {
                "classification": {"calibration": str(manifest)},
            }
        }
    }
    rejected = _deepx_calibration_manifest_contract(
        profile_payload=profile,
        task="classification",
        calibration_dir=str(root),
        effective_count=1,
    )
    assert rejected["status"] == "invalid"
    assert "manifest_full_verification_failed" in rejected["reason"]
    assert rejected["manifest_verification"]["payload_hash_ok"] is False


def test_compiler_identity_changes_with_active_dxcom_version(tmp_path: Path) -> None:
    cli = tmp_path / "dxcom"
    cli.write_text("#!/bin/sh\n", encoding="utf-8")
    base_environment = {
        "compiler_python_tag": "cp311",
        "compiler_cli": str(cli),
        "compiler_imports": [{
            "module": "dx_com",
            "ok": True,
            "distribution": "dx-com",
            "package_version": "2.3.0",
            "module_file": "dx_com/__init__.py",
            "module_file_sha256": "a" * 64,
        }],
    }
    first = _deepx_compiler_identity(
        cfg={}, environment_status=base_environment,
    )
    changed_environment = json.loads(json.dumps(base_environment))
    changed_environment["compiler_imports"][0]["package_version"] = "2.3.1"
    second = _deepx_compiler_identity(
        cfg={}, environment_status=changed_environment,
    )
    assert first["status"] == second["status"] == "resolved"
    assert first["identity_sha256"] != second["identity_sha256"]
