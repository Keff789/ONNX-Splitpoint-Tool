from __future__ import annotations

import hashlib
import json
import math
import tarfile
from pathlib import Path, PurePosixPath

import onnx
import pytest
from onnx import TensorProto, helper

from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
)
from onnx_splitpoint_tool.remote.bundle import (
    build_suite_bundle,
    remote_minimal_bundle_patterns,
)
from onnx_splitpoint_tool.workflow.benchmark_binding import (
    materialize_backend_artifact_decisions,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


def _constant_model(path: Path, output_shapes: list[list[int]]) -> None:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 8, 8],
    )
    outputs = []
    nodes = []
    for index, shape in enumerate(output_shapes):
        output_name = f"output_{index}"
        outputs.append(
            helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, shape,
            )
        )
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                [output_name],
                value=helper.make_tensor(
                    f"value_{index}",
                    TensorProto.FLOAT,
                    shape,
                    [0.0] * math.prod(shape),
                ),
            )
        )
    graph = helper.make_graph(nodes, "contract_graph", [input_info], outputs)
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)],
        ),
        path,
    )


def _contract_runner(run_dir: Path) -> EvaluationWorkflowRunner:
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.manifest = {}
    runner.profile_payload = {
        "run_profiles": [{
            "id": "ort_tensorrt",
            "full": "tensorrt",
            "stage1": "tensorrt",
            "stage2": "tensorrt",
        }],
    }
    runner._profile_with_cli_hardware_overrides = (
        lambda: runner.profile_payload
    )
    runner._prepared_full_hailo_info = lambda _path: {}
    return runner


def _extract_regular_files(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            logical = PurePosixPath(member.name)
            assert not logical.is_absolute()
            assert logical.parts
            assert all(part not in {"", ".", ".."} for part in logical.parts)
            if not member.isfile():
                continue
            source = handle.extractfile(member)
            assert source is not None
            target = destination.joinpath(*logical.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())


def test_materializer_registers_rewritten_formal_output_contract(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    model_id = "yolo11l"
    model_dir = run_dir / "models" / model_id
    suite_dir = model_dir / "benchmark_set" / "legacy_suite"
    suite_dir.mkdir(parents=True)
    formal_contract = model_dir / "full_baselines" / "output_contracts.json"
    formal_contract.parent.mkdir(parents=True)
    formal_contract.write_text(
        json.dumps({"contracts": [{"endpoint_mode": "stale"}]}) + "\n",
        encoding="utf-8",
    )
    stale_sha256 = hashlib.sha256(formal_contract.read_bytes()).hexdigest()

    output_contracts = {
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": model_id,
        "task": "detection",
        "contracts": [{
            "model_id": model_id,
            "backend": "tensorrt",
            "variant": "full",
            "endpoint_mode": "decoded_pre_nms",
        }],
    }
    materialized = materialize_backend_artifact_decisions(
        run_dir=run_dir,
        model_id=model_id,
        targets=["tensorrt"],
        full_baseline_plan={"task": "detection", "baselines": []},
        output_contracts=output_contracts,
        benchmark_set_contract={
            "materialized": True,
            "legacy_suite_dir": str(suite_dir),
        },
    )

    artifacts = materialized["artifacts"]
    suite_contract = suite_dir / "output_contracts.json"
    assert artifacts["formal_output_contracts_json"] == formal_contract
    assert artifacts["suite_output_contracts_json"] == suite_contract
    assert formal_contract.read_bytes() == suite_contract.read_bytes()
    assert (
        hashlib.sha256(formal_contract.read_bytes()).hexdigest()
        != stale_sha256
    )


@pytest.mark.parametrize(
    ("model_id", "task", "family", "shapes", "stage", "output_format"),
    (
        (
            "resnet50", "classification", "resnet",
            [[1, 1000]], "classification_logits", None,
        ),
        (
            "yolo26s", "detection", "yolo26",
            [[1, 300, 6]], "decoded_nms", "bn6_detections",
        ),
        (
            "yolo11l", "detection", "yolo11",
            [[1, 84, 8400]], "decoded_pre_nms",
            "ultralytics_decoded",
        ),
        (
            "yolov7_paper", "detection", "yolov7",
            [[1, 3, 4, 5, 6], [1, 3, 2, 3, 6], [1, 3, 1, 2, 6]],
            "raw_head", None,
        ),
    ),
)
def test_real_tensorrt_contract_survives_materializer_and_remote_bundle(
    tmp_path: Path,
    model_id: str,
    task: str,
    family: str,
    shapes: list[list[int]],
    stage: str,
    output_format: str | None,
) -> None:
    """Exercise the production writer, suite overlay, transport, and loader."""

    model_path = tmp_path / f"{model_id}.onnx"
    _constant_model(model_path, shapes)
    run_dir = tmp_path / "run"
    runner = _contract_runner(run_dir)

    writer_artifacts, _details, _message, writer_status = (
        runner._stage_prepare_full_baselines(
            model_id,
            {
                "id": model_id,
                "family": family,
                "task": task,
                "resolved_path": str(model_path),
            },
        )
    )
    assert writer_status == "ok"
    full_baseline_plan = json.loads(
        writer_artifacts["full_baseline_plan_json"].read_text(
            encoding="utf-8",
        )
    )
    writer_contracts = json.loads(
        writer_artifacts["output_contracts_json"].read_text(
            encoding="utf-8",
        )
    )
    assert runner._targets() == ["tensorrt"]
    assert runner._baseline_backends(runner._targets()) == ["cuda_ort"]

    suite_root = (
        run_dir / "models" / model_id / "benchmark_set" / "legacy_suite"
    )
    suite_root.mkdir(parents=True)
    benchmark_set = {
        "schema": "onnx-splitpoint/benchmark-set",
        "schema_version": 1,
        "model_id": model_id,
        "benchmark_task": task,
        "cases": [],
    }
    (suite_root / "benchmark_set.json").write_text(
        json.dumps(benchmark_set, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    materialized = materialize_backend_artifact_decisions(
        run_dir=run_dir,
        model_id=model_id,
        targets=runner._targets(),
        full_baseline_plan=full_baseline_plan,
        output_contracts=writer_contracts,
        benchmark_set_contract={
            **benchmark_set,
            "materialized": True,
            "legacy_suite_dir": str(suite_root),
        },
    )
    suite_contract_path = Path(
        materialized["artifacts"]["suite_output_contracts_json"]
    )
    assert suite_contract_path == suite_root / "output_contracts.json"
    materialized_contracts = json.loads(
        suite_contract_path.read_text(encoding="utf-8")
    )
    assert materialized_contracts == writer_contracts

    includes, excludes = remote_minimal_bundle_patterns()
    archive = tmp_path / "transport" / "suite_bundle.tar.gz"
    stats = build_suite_bundle(
        suite_root,
        archive,
        includes=includes,
        excludes=excludes,
        reuse_if_unchanged=False,
    )
    bundle_manifest = json.loads(
        Path(stats.manifest_path).read_text(encoding="utf-8")
    )
    assert "output_contracts.json" in {
        row["rel"] for row in bundle_manifest["files"]
    }

    extracted_root = tmp_path / "extracted_suite"
    _extract_regular_files(archive, extracted_root)
    extracted_contract_path = extracted_root / "output_contracts.json"
    assert extracted_contract_path.read_bytes() == suite_contract_path.read_bytes()

    declaration = load_authoritative_output_contract(
        extracted_root,
        backend="tensorrt",
        model_id=model_id,
        variant="full",
        task=task,
    )
    assert declaration["contract_resolution_status"] == "attested"
    assert declaration["authoritative_output_contract"] is True
    assert declaration["backend"] == "cuda_ort"
    assert declaration["stage"] == stage
    assert declaration.get("output_format") == output_format
    assert declaration["source_contracts_sha256"] == hashlib.sha256(
        extracted_contract_path.read_bytes()
    ).hexdigest()
    needs_postprocess = stage in {"raw_head", "decoded_pre_nms"}
    assert declaration["host_tail_required"] is needs_postprocess
    assert declaration["postprocessing_required"] is needs_postprocess
    assert declaration["requires_external_postprocess"] is needs_postprocess
