from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pytest

from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    prepare_rgb_uint8_image,
    preprocessing_contract_sha256,
    runtime_numeric_input_identity,
)
from scripts import native_deepx_full_energy_hotloop as deepx_energy
from scripts import native_full_baseline_eval_runner as full_runner


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"


def _template_functions(*names: str) -> Dict[str, Any]:
    tree = ast.parse(SUITE.read_text(encoding="utf-8"), filename=str(SUITE))
    wanted = set(names)
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    namespace: Dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "Mapping": Mapping,
        "Optional": Optional,
        "Tuple": Tuple,
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "re": re,
        "np": np,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SUITE), "exec"),
        namespace,
    )
    return namespace


def _sealed_input_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], np.ndarray]:
    image = tmp_path / "source.jpg"
    image.write_bytes(b"sealed-source-image")
    feed = np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)
    tensor = tmp_path / "runtime_input.bin"
    tensor.write_bytes(feed.tobytes())
    preprocessing = canonical_image_preprocessing_contract(
        "classification", [2, 2],
    )
    preprocessing_sha = preprocessing_contract_sha256(preprocessing)
    numeric, numeric_sha = runtime_numeric_input_identity(
        backend="native_full_deepx",
        task="classification",
        preprocessing_contract_sha256_value=preprocessing_sha,
        runtime_input_name="images",
        runtime_input_shape=list(feed.shape),
        runtime_input_dtype=str(feed.dtype),
        runtime_input_layout="NHWC",
        runtime_color_space="RGB",
        runtime_normalization="raw",
    )
    manifest = tmp_path / "native_full_input_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "onnx-splitpoint/native-full-input-dump",
        "schema_version": 2,
        "backend": "native_full_deepx",
        "task": "classification",
        "input_image": str(image),
        "input_image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "runtime_input_name": "images",
        "runtime_input_shape": list(feed.shape),
        "runtime_input_dtype": str(feed.dtype),
        "runtime_input_layout": "NHWC",
        "runtime_input_file": str(tensor),
        "runtime_input_sha256": hashlib.sha256(tensor.read_bytes()).hexdigest(),
        "runtime_input_bytes": int(feed.nbytes),
        "runtime_preprocess_mode": "resize_rgb_uint8",
        "runtime_color_space": "RGB",
        "runtime_normalization": "raw",
        "preprocess": {
            "mode": "resize_rgb_uint8",
            "layout": "NHWC",
            "color_space": "RGB",
            "normalization": "raw",
        },
        "runtime_preprocessing_identity": preprocessing,
        "runtime_preprocessing_sha256": preprocessing_sha,
        "runtime_numeric_input_identity": numeric,
        "runtime_numeric_input_sha256": numeric_sha,
    }), encoding="utf-8")
    contract = {"input": {
        "name": "images",
        "shape": list(feed.shape),
        "dtype": "uint8",
        "layout": "NHWC",
        "normalization": "raw",
        "color_space": "RGB",
        "preprocess_mode": "resize",
        "letterbox_pad_value": 0,
    }}
    return manifest, image, contract, feed


def test_performance_loader_replays_exact_writable_semantic_tensor(
    tmp_path: Path,
) -> None:
    functions = _template_functions(
        "_read_json", "_deepx_load_sealed_prepared_feed",
    )
    manifest, image, contract, expected = _sealed_input_fixture(tmp_path)

    feed, binding = functions["_deepx_load_sealed_prepared_feed"](
        manifest,
        input_contract=contract,
        image_path=image,
        task="classification",
        np=np,
    )

    assert np.array_equal(feed, expected)
    assert feed.flags.c_contiguous
    assert feed.flags.writeable
    assert binding["prepared_input_binding_verified"] is True
    assert binding["prepared_input_source"] == (
        "sealed_semantic_dump_runtime_tensor"
    )
    assert binding["prepared_input_sha256"] == hashlib.sha256(
        expected.tobytes()
    ).hexdigest()
    assert binding["prepared_input_bytes"] == expected.nbytes


def test_performance_and_energy_reject_one_byte_tensor_mutation(
    tmp_path: Path,
) -> None:
    functions = _template_functions(
        "_read_json", "_deepx_load_sealed_prepared_feed",
    )
    manifest, image, contract, expected = _sealed_input_fixture(tmp_path)
    tensor = tmp_path / "runtime_input.bin"
    mutated = bytearray(tensor.read_bytes())
    mutated[-1] ^= 0x01
    tensor.write_bytes(bytes(mutated))

    with pytest.raises(ValueError, match="prepared_input_tensor_sha256_mismatch"):
        functions["_deepx_load_sealed_prepared_feed"](
            manifest,
            input_contract=contract,
            image_path=image,
            task="classification",
            np=np,
        )
    with pytest.raises(ValueError, match="prepared_input_sha256_mismatch"):
        deepx_energy._load_prepared_input(
            tensor,
            expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
            expected_bytes=expected.nbytes,
            expected_name="images",
            expected_shape=list(expected.shape),
            expected_dtype="uint8",
            expected_layout="NHWC",
        )


def test_sealed_manifest_and_tensor_roles_reject_symlink_components(
    tmp_path: Path,
) -> None:
    functions = _template_functions(
        "_read_json", "_deepx_load_sealed_prepared_feed",
    )

    manifest_root = tmp_path / "manifest-leaf"
    manifest_root.mkdir()
    manifest, image, contract, _expected = _sealed_input_fixture(
        manifest_root
    )
    real_manifest = manifest.with_name("manifest-real.json")
    manifest.rename(real_manifest)
    manifest.symlink_to(real_manifest)
    with pytest.raises(ValueError, match="prepared_input_manifest_missing"):
        functions["_deepx_load_sealed_prepared_feed"](
            manifest,
            input_contract=contract,
            image_path=image,
            task="classification",
            np=np,
        )
    assert full_runner._validated_runtime_input_manifest(
        manifest, allowed_root=manifest_root,
    )[1] == "runtime_input_manifest_missing"

    tensor_root = tmp_path / "tensor-leaf"
    tensor_root.mkdir()
    manifest, image, contract, _expected = _sealed_input_fixture(
        tensor_root
    )
    tensor = tensor_root / "runtime_input.bin"
    real_tensor = tensor.with_name("tensor-real.bin")
    tensor.rename(real_tensor)
    tensor.symlink_to(real_tensor)
    with pytest.raises(
        ValueError, match="prepared_input_tensor_metadata_invalid",
    ):
        functions["_deepx_load_sealed_prepared_feed"](
            manifest,
            input_contract=contract,
            image_path=image,
            task="classification",
            np=np,
        )
    assert full_runner._validated_runtime_input_manifest(
        manifest, allowed_root=tensor_root,
    )[1] == "runtime_input_file_missing"

    linked_root = tmp_path / "semantic-output"
    linked_root.mkdir()
    manifest, image, contract, _expected = _sealed_input_fixture(
        linked_root
    )
    real_root = tmp_path / "semantic-output-real"
    linked_root.rename(real_root)
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ValueError, match="prepared_input_manifest_missing"):
        functions["_deepx_load_sealed_prepared_feed"](
            manifest,
            input_contract=contract,
            image_path=image,
            task="classification",
            np=np,
        )
    assert full_runner._validated_runtime_input_manifest(
        manifest, allowed_root=linked_root,
    )[1] == "runtime_input_manifest_missing"


def test_nested_deepx_v3_contract_cannot_be_promoted_from_top_level() -> None:
    top_level_only = {
        "prepared_feed_contract_version": "deepx-sealed-runtime-input-v3",
        "deepx_prepared_feed_benchmark": {
            "performance_benchmark_source": "dx_engine_prepared_feed",
        },
    }
    projection, conflicts = full_runner._deepx_prepared_feed_projection(
        top_level_only
    )
    assert "prepared_feed_contract_version" not in projection
    assert any(
        conflict.get("field") == "prepared_feed_contract_version"
        and conflict.get("reason") == "nested_contract_version_missing"
        for conflict in conflicts
    )

    legacy_nested = {
        **top_level_only,
        "deepx_prepared_feed_benchmark": {
            "performance_benchmark_source": "dx_engine_prepared_feed",
            "prepared_feed_contract_version": (
                "deepx-explicit-image-input-v2"
            ),
        },
    }
    projection, conflicts = full_runner._deepx_prepared_feed_projection(
        legacy_nested
    )
    assert projection["prepared_feed_contract_version"] == (
        "deepx-explicit-image-input-v2"
    )
    assert any(
        conflict.get("field") == "prepared_feed_contract_version"
        for conflict in conflicts
    )
@pytest.mark.parametrize("task", ["classification", "detection"])
def test_quality_feed_records_canonical_exact_tensor_identity(
    task: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = SimpleNamespace(
        COLOR_BGR2RGB=1,
        cvtColor=lambda value, _mode: np.ascontiguousarray(value[..., ::-1]),
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    functions = _template_functions(
        "_deepx_input_geometry", "_deepx_prepare_image_from_contract",
    )
    height = width = 8
    source_bgr = np.arange(5 * 9 * 3, dtype=np.uint8).reshape(5, 9, 3)
    preprocessing = canonical_image_preprocessing_contract(
        task, [height, width],
    )
    contract = {"input": {
        "name": "images",
        "shape": [1, height, width, 3],
        "dtype": "uint8",
        "layout": "NHWC",
        "normalization": "raw",
        "color_space": "RGB",
        "preprocess_mode": preprocessing["preprocess_mode"],
        "letterbox_pad_value": preprocessing["letterbox_pad_value"],
    }}

    feed, audit = functions["_deepx_prepare_image_from_contract"](
        source_bgr, height, contract, task,
    )
    expected_rgb, _geometry = prepare_rgb_uint8_image(
        np.ascontiguousarray(source_bgr[..., ::-1]), preprocessing,
    )
    expected = expected_rgb[None, ...]
    binding = audit["prepared_tensor_binding"]

    assert np.array_equal(feed, expected)
    assert audit["runtime_preprocessing_identity"] == preprocessing
    assert audit["runtime_preprocessing_sha256"] == (
        preprocessing_contract_sha256(preprocessing)
    )
    assert binding["binding_verified"] is True
    assert binding["prepared_input_sha256"] == hashlib.sha256(
        feed.tobytes()
    ).hexdigest()
    assert binding["prepared_input_bytes"] == feed.nbytes


@pytest.mark.parametrize("completion_kind", ["raw_head", "direct_bn6"])
def test_completed_v2_persistence_omission_and_tamper_fail_closed(
    tmp_path: Path,
    completion_kind: str,
) -> None:
    artifact = {
        "schema": "onnx-splitpoint/frozen-completed-detection-result-artifact",
        "schema_version": 1,
        "record_schema": "xyxy_score_class_id_v1",
        "coordinate_space": "original_image_xyxy_pixels",
        "sort_policy": "score_desc_class_id_asc_xyxy_lexicographic_v1",
        "detections": [],
    }
    sha = full_runner._canonical_json_sha256(artifact)
    path = tmp_path / f"{completion_kind}.completed.json"
    path.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    sealed_result = {
        "completed_result_artifact": artifact,
        "completed_result_artifact_sha256": sha,
        (
            "postprocess_contract_sha256"
            if completion_kind == "raw_head"
            else "normalization_contract_sha256"
        ): "a" * 64,
    }
    payload = {
        "completed_task_result_artifact_saved": True,
        "completed_task_result_artifact": artifact,
        "completed_task_result_artifact_sha256": sha,
        "completed_task_result_artifact_path": str(path),
        "completed_task_result_artifact_file_sha256": sha,
    }

    assert full_runner._completed_result_artifact_persistence_status(
        payload, sealed_result=sealed_result,
        allowed_root=tmp_path, expected_path=path,
    ) == (True, "verified_exact")
    for missing in (
        "completed_task_result_artifact_saved",
        "completed_task_result_artifact",
        "completed_task_result_artifact_sha256",
        "completed_task_result_artifact_path",
        "completed_task_result_artifact_file_sha256",
    ):
        broken = dict(payload)
        broken.pop(missing)
        ok, _status = full_runner._completed_result_artifact_persistence_status(
            broken, sealed_result=sealed_result,
            allowed_root=tmp_path, expected_path=path,
        )
        assert ok is False, missing

    path.write_text("{}", encoding="utf-8")
    assert full_runner._completed_result_artifact_persistence_status(
        payload, sealed_result=sealed_result,
        allowed_root=tmp_path, expected_path=path,
    ) == (False, "completed_task_result_artifact_persistence_mismatch")


def test_old_deepx_v2_contract_is_rejected_before_preflight(
    tmp_path: Path,
) -> None:
    sha = "a" * 64
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/native_deepx_full_energy_hotloop.py"),
            "--dxnn", str(tmp_path / "missing.dxnn"),
            "--prepared-input-file", str(tmp_path / "missing.bin"),
            "--expected-prepared-input-sha256", sha,
            "--expected-prepared-input-bytes", "1",
            "--expected-prepared-input-name", "images",
            "--expected-prepared-input-shape-json", "[1,1,1,1]",
            "--expected-prepared-input-dtype", "uint8",
            "--expected-prepared-input-layout", "NHWC",
            "--runtime-preprocessing-identity-json", "{}",
            "--expected-runtime-preprocessing-sha256", sha,
            "--runtime-numeric-input-identity-json", "{}",
            "--expected-runtime-numeric-input-sha256", sha,
            "--original-image-wh-json", "[1,1]",
            "--prepared-feed-contract-version", "deepx-explicit-image-input-v2",
            "--frames", "1",
            "--task", "classification",
            "--json-out", str(tmp_path / "result.json"),
            "--expected-runner-sha256", sha,
            "--expected-dxnn-sha256", sha,
            "--source-contract-sha256", sha,
            "--preflight-attestation", str(tmp_path / "missing-preflight.json"),
            "--preflight-nonce", "nonce",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 5
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "prepared_feed_contract_version_mismatch"
    assert deepx_energy.CONTRACT_VERSION == "deepx-sealed-runtime-input-v3"
    assert full_runner.DEEPX_PREPARED_FEED_CONTRACT_VERSION == (
        "deepx-sealed-runtime-input-v3"
    )
