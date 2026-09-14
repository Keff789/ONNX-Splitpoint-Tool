from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import native_full_baseline_eval_runner as full_runner
from scripts import native_producer_energy_plan as energy_plan


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _benchmark_set(tmp_path: Path, *, model: str = "yolo26s") -> Path:
    root = tmp_path / "benchmark_set"
    source = root / "models" / f"{model}.onnx"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"sealed-benchmark-set-source")
    (root / "benchmark_set.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/benchmark-set",
            "schema_version": 2,
            "model_name": model,
            "model": f"models/{model}.onnx",
            "artifact_manifest": {
                "schema": "onnx-splitpoint/benchmark-set",
                "schema_version": 2,
                "files": {"models": [f"models/{model}.onnx"]},
                "counts": {"models": 1},
            },
        }, sort_keys=True),
        encoding="utf-8",
    )
    return root


def _source_contract(root: Path, *, model: str = "yolo26s") -> dict[str, Any]:
    artifacts, status = (
        full_runner._verified_benchmark_set_source_onnx_artifacts(
            root, model,
        )
    )
    assert status == "benchmark_set_source_onnx_verified_exact"
    dxnn = root / "deepx/deepx_m1/full/model.dxnn"
    dxnn.parent.mkdir(parents=True)
    dxnn.write_bytes(b"compiled-dxnn")
    artifacts["dxnn"] = {"path": str(dxnn), "sha256": _sha(dxnn)}
    source_sha = str(artifacts["source_onnx"]["sha256"])
    return {
        "model": model,
        "benchmark_set": str(root),
        "source_model_sha256": source_sha,
        "artifacts": artifacts,
        "model_binding": {
            "source_artifact": "source_onnx",
            "source_onnx_sha256": source_sha,
            "compiled_artifact": "dxnn",
            "compiled_artifact_sha256": _sha(dxnn),
            "status": "source_and_compiled_artifact_hash_bound",
        },
        "energy_workload": {
            "source_model_artifact": "source_onnx",
            "benchmark_set_manifest_artifact": "benchmark_set_manifest",
            "dxnn_artifact": "dxnn",
            "source_model_binding_status": status,
        },
    }


def test_first_deepx_full_contract_source_is_exact_benchmark_set_model(
    tmp_path: Path,
) -> None:
    root = _benchmark_set(tmp_path)
    artifacts, status = (
        full_runner._verified_benchmark_set_source_onnx_artifacts(
            root, "yolo26s",
        )
    )

    assert status == "benchmark_set_source_onnx_verified_exact"
    assert artifacts["source_onnx"]["path"] == str(
        (root / "models/yolo26s.onnx").resolve()
    )
    assert artifacts["source_onnx"]["sha256"] == _sha(
        root / "models/yolo26s.onnx"
    )
    assert artifacts["source_onnx"][
        "benchmark_set_manifest_sha256"
    ] == artifacts["benchmark_set_manifest"]["sha256"]


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    (
        (
            lambda payload: payload.update({"model_name": "yolov7_paper"}),
            "benchmark_set_source_onnx_manifest_identity_invalid",
        ),
        (
            lambda payload: payload["artifact_manifest"]["files"].update({
                "models": ["models/yolo26s.onnx", "models/other.onnx"],
            }),
            "benchmark_set_source_onnx_manifest_identity_invalid",
        ),
        (
            lambda payload: payload.update({
                "model": "models/../models/yolo26s.onnx",
            }),
            "benchmark_set_source_onnx_manifest_identity_invalid",
        ),
    ),
)
def test_first_deepx_full_contract_rejects_benchmark_set_identity_drift(
    tmp_path: Path, mutation: Any, expected_status: str,
) -> None:
    root = _benchmark_set(tmp_path)
    manifest = root / "benchmark_set.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutation(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    artifacts, status = (
        full_runner._verified_benchmark_set_source_onnx_artifacts(
            root, "yolo26s",
        )
    )

    assert artifacts == {}
    assert status == expected_status


def test_deepx_planner_source_binding_rejects_hash_path_and_manifest_drift(
    tmp_path: Path,
) -> None:
    root = _benchmark_set(tmp_path)
    pristine = _source_contract(root)
    workload = pristine["energy_workload"]
    artifacts = pristine["artifacts"]
    expected = {
        "model": "yolo26s",
        "model_sha256": pristine["source_model_sha256"],
    }
    assert energy_plan._sealed_deepx_source_model_binding(
        pristine, workload, artifacts, expected_identity=expected,
    )

    mutations = []
    bad_source_hash = copy.deepcopy(pristine)
    bad_source_hash["artifacts"]["source_onnx"]["sha256"] = "f" * 64
    mutations.append(bad_source_hash)
    bad_source_path = copy.deepcopy(pristine)
    bad_source_path["artifacts"]["source_onnx"]["path"] = str(
        root / "models/other.onnx"
    )
    mutations.append(bad_source_path)
    bad_manifest_link = copy.deepcopy(pristine)
    bad_manifest_link["artifacts"]["source_onnx"][
        "benchmark_set_manifest_sha256"
    ] = "e" * 64
    mutations.append(bad_manifest_link)
    bad_binding = copy.deepcopy(pristine)
    bad_binding["model_binding"]["source_onnx_sha256"] = "d" * 64
    mutations.append(bad_binding)

    for tampered in mutations:
        assert not energy_plan._sealed_deepx_source_model_binding(
            tampered,
            tampered["energy_workload"],
            tampered["artifacts"],
            expected_identity=expected,
        )

