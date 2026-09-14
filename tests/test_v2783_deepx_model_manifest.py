from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any, Dict, Mapping

import pytest

from onnx_splitpoint_tool.benchmark.schema import (
    build_benchmark_artifact_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE_TEMPLATE = (
    ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
)


def _deepx_identity_verifier() -> Any:
    tree = ast.parse(
        SUITE_TEMPLATE.read_text(encoding="utf-8"),
        filename=str(SUITE_TEMPLATE),
    )
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_deepx_verified_suite_run_identity"
    ]
    assert len(functions) == 1
    namespace: dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "Mapping": Mapping,
        "re": re,
    }
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            str(SUITE_TEMPLATE),
            "exec",
        ),
        namespace,
    )
    return namespace["_deepx_verified_suite_run_identity"]


def _deepx_run() -> dict[str, Any]:
    return {
        "backend": "deepx_m1",
        "provider": "deepx_m1",
        "stage1": {"backend": "deepx_m1"},
        "stage2": {"backend": "deepx_m1"},
        "variants": ["full"],
    }


def _benchmark_set(
    suite: Path, model_id: str,
) -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/benchmark-set",
        "schema_version": 2,
        "model_name": model_id,
        "model": f"models/{model_id}.onnx",
        "artifact_manifest": build_benchmark_artifact_manifest(suite),
    }


@pytest.mark.parametrize(
    "model_id", ("mobilenet_v3_large", "regnet_x_1_6gf"),
)
def test_classification_model_sidecars_do_not_change_deepx_model_identity(
    tmp_path: Path, model_id: str,
) -> None:
    suite = tmp_path / model_id
    models = suite / "models"
    models.mkdir(parents=True)
    (models / f"{model_id}.onnx").write_bytes(b"onnx")
    (models / f"{model_id}.categories.json").write_text(
        "[]\n", encoding="utf-8",
    )
    (models / f"{model_id}.export.json").write_text(
        "{}\n", encoding="utf-8",
    )

    benchmark_set = _benchmark_set(suite, model_id)
    manifest = benchmark_set["artifact_manifest"]

    assert manifest["files"]["models"] == [
        f"models/{model_id}.onnx",
    ]
    assert manifest["counts"]["models"] == 1
    assert manifest["files"]["model_sidecars"] == [
        f"models/{model_id}.categories.json",
        f"models/{model_id}.export.json",
    ]
    assert manifest["counts"]["model_sidecars"] == 2
    assert _deepx_identity_verifier()(
        benchmark_set, _deepx_run(),
    ) == {
        "model_id": model_id,
        "backend": "deepx_m1",
        "variant": "full",
    }


def test_model_manifest_without_sidecars_preserves_existing_valid_contract(
    tmp_path: Path,
) -> None:
    model_id = "resnet50"
    suite = tmp_path / model_id
    models = suite / "models"
    models.mkdir(parents=True)
    (models / f"{model_id}.onnx").write_bytes(b"onnx")

    benchmark_set = _benchmark_set(suite, model_id)
    manifest = benchmark_set["artifact_manifest"]

    assert manifest["files"]["models"] == [
        f"models/{model_id}.onnx",
    ]
    assert manifest["counts"]["models"] == 1
    assert manifest["files"]["model_sidecars"] == []
    assert manifest["counts"]["model_sidecars"] == 0
    assert _deepx_identity_verifier()(
        benchmark_set, _deepx_run(),
    )["model_id"] == model_id


def test_second_onnx_model_remains_a_strict_identity_conflict(
    tmp_path: Path,
) -> None:
    model_id = "mobilenet_v3_large"
    suite = tmp_path / model_id
    models = suite / "models"
    models.mkdir(parents=True)
    (models / f"{model_id}.onnx").write_bytes(b"onnx")
    (models / "other.onnx").write_bytes(b"other")

    benchmark_set = _benchmark_set(suite, model_id)

    assert benchmark_set["artifact_manifest"]["counts"]["models"] == 2
    with pytest.raises(
        RuntimeError, match="deepx_suite_model_identity_invalid",
    ):
        _deepx_identity_verifier()(benchmark_set, _deepx_run())
