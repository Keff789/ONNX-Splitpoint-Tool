from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper

from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
    runtime_output_contract,
)
from onnx_splitpoint_tool.hailo_backend import (
    _hailo_cache_key,
    _write_hailo_receipt,
)
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
)
from onnx_splitpoint_tool.workflow.benchmark_binding import (
    _validate_hailo_build_receipt,
    materialize_backend_artifact_decisions,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_TEMPLATE = (
    ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"
)


def _seal_test_hailo_hef(
    hef: Path, *, task: str, hw_arch: str
) -> tuple[Path, Path]:
    source = hef.parent / "receipt_source.onnx"
    source.write_bytes(f"{task}-source-onnx".encode("utf-8"))
    compiler = hef.parent / "receipt_compiler.onnx"
    compiler.write_bytes(f"{task}-compiler-fixed-onnx".encode("utf-8"))
    contract = canonical_image_preprocessing_contract(
        task, (224, 224) if task == "classification" else (640, 640)
    )
    cache_key, cache_payload = _hailo_cache_key(
        model_path=compiler,
        activation_part1=None,
        hw_arch=hw_arch,
        opt_level=1,
        calib_dir=None,
        calib_count=64,
        calib_batch_size=8,
        extra_model_script="",
        start_nodes=None,
        end_nodes=None,
        preprocessing_contract=contract,
    )
    _write_hailo_receipt(
        hef_path=hef,
        source_onnx=source,
        compiler_onnx=compiler,
        hw_arch=hw_arch,
        net_name="receipt_test",
        preprocessing_contract=contract,
        preprocessing_sha256=preprocessing_contract_sha256(contract),
        cache_key=cache_key,
        cache_payload=cache_payload,
        calibration_identity=str(cache_payload["calibration_identity"]),
        calibration_count=int(cache_payload["calibration_count"]),
    )
    return source, compiler


def _row(
    *, model: str = "resnet50", backend: str = "cuda_ort",
    task: str = "classification", mode: str = "decoded",
    status: str = "recorded",
) -> dict[str, Any]:
    raw = mode in {"raw_head", "raw_detection_head"}
    return {
        "model_id": model, "backend": backend, "variant": "full",
        "task": task, "endpoint_mode": mode, "contract_status": status,
        "host_tail_required": raw, "postprocessing_required": raw,
    }


def _write_contracts(
    root: Path, rows: Sequence[Mapping[str, Any]], *, model: str, task: str,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "output_contracts.json"
    path.write_text(json.dumps({
        "schema": "onnx-splitpoint/output-contracts", "schema_version": 1,
        "model_id": model, "task": task,
        "contracts": [dict(row) for row in rows],
    }, sort_keys=True), encoding="utf-8")
    return path


def _load_script(name: str):
    path = ROOT / "scripts" / name
    module_name = "_osp_endpoint_audit_" + path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _template_functions(*names: str) -> dict[str, Any]:
    tree = ast.parse(RUNNER_TEMPLATE.read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in set(names)
    ]
    assert {node.name for node in selected} == set(names)
    namespace: dict[str, Any] = {
        "Any": Any, "Dict": Dict, "List": List, "Mapping": Mapping,
        "Optional": Optional, "Sequence": Sequence, "Path": Path,
        "np": np, "onnx": onnx,
        "load_authoritative_output_contract": load_authoritative_output_contract,
        "runtime_output_contract": runtime_output_contract,
        "_read_json": lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
        "_quality_file_sha256": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "_quality_contract_sha256": lambda value: hashlib.sha256(
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(RUNNER_TEMPLATE), "exec"),
        namespace,
    )
    return namespace


def _save_classifier(path: Path, *, output_name: str = "logits") -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], [output_name])],
        "classifier",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1000])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 1000])],
    )
    onnx.save(helper.make_model(graph), path)


def test_classification_rejects_values_only_or_local_stage_declarations() -> None:
    outputs = {"logits": np.zeros((1, 1000), dtype=np.float32)}
    for declaration in (None, "classification_logits", {"stage": "classification_logits"}):
        contract = runtime_output_contract(
            "classification", outputs, raw_fallback=False,
            declared_contract=declaration,
        )
        assert contract["endpoint_contract_complete"] is False
        assert contract["contract_family"] == "unknown"
        assert contract["stage"] == "unknown"


def test_runtime_revalidates_root_contract_and_rejects_forged_copies(
    tmp_path: Path,
) -> None:
    _write_contracts(
        tmp_path, [_row()], model="resnet50", task="classification",
    )
    declaration = load_authoritative_output_contract(
        tmp_path, backend="tensorrt", model_id="resnet50",
        variant="full", task="classification",
    )
    outputs = {"logits": np.zeros((1, 1000), dtype=np.float32)}
    original = runtime_output_contract(
        "classification", outputs, raw_fallback=False,
        declared_contract=declaration,
    )
    assert original["endpoint_contract_complete"] is True

    mutations = {
        "model_id": "another_model",
        "backend": "deepx_m1",
        "variant": "composed",
        "task": "detection",
        "stage": "classification_probabilities",
        "contract_family": "classification_probabilities",
        "endpoint_mode": "classification_probabilities",
        "host_tail_required": True,
        "postprocessing_required": True,
        "source_contract_sha256": "1" * 64,
        "source_contracts_sha256": "2" * 64,
    }
    for key, value in mutations.items():
        forged = dict(declaration)
        forged[key] = value
        rejected = runtime_output_contract(
            "classification", outputs, raw_fallback=False,
            declared_contract=forged,
        )
        assert rejected["endpoint_contract_complete"] is False, key
        assert rejected["contract_family"] == "unknown", key

    copied = dict(declaration)
    copied["declaration_source"] = str(tmp_path / "other/output_contracts.json")
    rejected = runtime_output_contract(
        "classification", outputs, raw_fallback=False,
        declared_contract=copied,
    )
    assert rejected["endpoint_contract_complete"] is False
    assert "source_unavailable" in rejected["output_endpoint_attestation"]["reason"]


def test_runtime_rejects_stale_container_and_raw_row_hashes(tmp_path: Path) -> None:
    path = _write_contracts(
        tmp_path, [_row()], model="resnet50", task="classification",
    )
    declaration = load_authoritative_output_contract(
        tmp_path, backend="tensorrt", model_id="resnet50",
        variant="full", task="classification",
    )
    outputs = {"logits": np.zeros((1, 1000), dtype=np.float32)}

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_note"] = "container changed after declaration load"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    stale = runtime_output_contract(
        "classification", outputs, raw_fallback=False,
        declared_contract=declaration,
    )
    assert stale["endpoint_contract_complete"] is False
    assert "container_sha256_mismatch" in stale["output_endpoint_attestation"]["reason"]

    refreshed = load_authoritative_output_contract(
        tmp_path, backend="tensorrt", model_id="resnet50",
        variant="full", task="classification",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Even a row field which is not itself endpoint-semantic changes the exact
    # recorded-row identity and must invalidate the previously issued digest.
    payload["contracts"][0]["review_note"] = "mutated-row"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    copied = dict(refreshed)
    copied["source_contracts_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    stale_row = runtime_output_contract(
        "classification", outputs, raw_fallback=False,
        declared_contract=copied,
    )
    assert stale_row["endpoint_contract_complete"] is False
    assert "field_mismatch:source_contract_sha256" in (
        stale_row["output_endpoint_attestation"]["reason"]
    )


def test_auxiliary_classification_hints_cannot_override_attested_stage(
    tmp_path: Path,
) -> None:
    row = {
        **_row(),
        # These auxiliary fields used to be scanned after authoritative
        # revalidation and could relabel an attested logits endpoint as
        # probabilities.  Only the loader-normalized stage is authoritative.
        "output_format": "classification_probabilities",
        "outputs": [{"name": "softmax", "stage": "classification_probabilities"}],
    }
    _write_contracts(
        tmp_path, [row], model="resnet50", task="classification",
    )
    declaration = load_authoritative_output_contract(
        tmp_path, backend="tensorrt", model_id="resnet50",
        variant="full", task="classification",
    )
    assert declaration["stage"] == "classification_logits"

    result = runtime_output_contract(
        "classification",
        {"logits": np.full((1, 3), 1.0 / 3.0, dtype=np.float32)},
        raw_fallback=False, declared_contract=declaration,
    )

    assert result["endpoint_contract_complete"] is True
    assert result["stage"] == "classification_logits"
    assert result["contract_family"] == "classification_logits"


def test_valid_plus_invalid_exact_rows_conflict_without_stage(tmp_path: Path) -> None:
    valid = _row()
    invalid = {**valid, "contract_status": "inferred"}
    _write_contracts(tmp_path, [valid, invalid], model="resnet50", task="classification")

    declaration = load_authoritative_output_contract(
        tmp_path, backend="tensorrt", model_id="resnet50",
        variant="full", task="classification",
    )

    assert declaration["contract_resolution_status"] == "conflict"
    assert declaration["contract_resolution_reason"] == (
        "suite_output_contract_exact_matches_conflict"
    )
    assert "endpoint_contract_not_recorded" in declaration["contract_resolution_errors"]
    assert "stage" not in declaration


def test_stage_task_and_flag_contradictions_conflict_without_stage(tmp_path: Path) -> None:
    cases = (
        {**_row(), "stage": "classification_probabilities"},
        {**_row(), "task": "detection"},
        {**_row(), "host_tail_required": True},
    )
    for index, row in enumerate(cases):
        root = tmp_path / str(index)
        _write_contracts(root, [row], model="resnet50", task="classification")
        declaration = load_authoritative_output_contract(
            root, backend="tensorrt", model_id="resnet50",
            variant="full", task="classification",
        )
        assert declaration["contract_resolution_status"] == "conflict"
        assert "stage" not in declaration


def test_raw_head_declaration_requires_root_mapping_and_raw_runtime_structure(
    tmp_path: Path,
) -> None:
    local_declaration = {
        "stage": "raw_head", "contract_family": "raw_head",
        "endpoint_mode": "raw_head",
        "host_tail_required": True, "postprocessing_required": True,
    }
    plausible_local = runtime_output_contract(
        "detection", {"output0": np.zeros((1, 84, 8400), dtype=np.float32)},
        raw_fallback=False, declared_contract=local_declaration,
    )
    assert plausible_local["endpoint_contract_complete"] is False
    assert plausible_local["contract_family"] == "unknown"

    _write_contracts(
        tmp_path,
        [_row(model="yolov7_paper", task="detection", mode="raw_head")],
        model="yolov7_paper", task="detection",
    )
    declaration = load_authoritative_output_contract(
        tmp_path, backend="tensorrt", model_id="yolov7_paper",
        variant="full", task="detection",
    )
    assert declaration["contract_resolution_status"] == "attested"
    invalid = (
        {"scalar": np.asarray(1.0, dtype=np.float32)},
        {"classes": np.zeros((1, 1000), dtype=np.float32)},
        {"output0": np.zeros((1, 300, 6), dtype=np.float32)},
    )
    for outputs in invalid:
        contract = runtime_output_contract(
            "detection", outputs, raw_fallback=False,
            declared_contract=declaration,
        )
        assert contract["endpoint_contract_complete"] is False
        assert contract["contract_family"] == "unknown"
        assert contract["stage"] == "unknown"

    valid = runtime_output_contract(
        "detection",
        {
            "head_80": np.zeros((1, 3, 80, 80, 85), dtype=np.float32),
            "head_40": np.zeros((1, 3, 40, 40, 85), dtype=np.float32),
            "head_20": np.zeros((1, 3, 20, 20, 85), dtype=np.float32),
        },
        raw_fallback=False, declared_contract=declaration,
    )
    assert valid["endpoint_contract_complete"] is True
    assert valid["contract_family"] == "raw_head"


def test_hailo_runner_rejects_adjacent_standalone_contract(tmp_path: Path) -> None:
    adjacent = tmp_path / "output_contract.json"
    adjacent.write_text(json.dumps({
        "model_id": "resnet50", "backend": "hailo8", "task": "classification",
        "variant": "full", "contract_status": "recorded",
        "stage": "classification_logits",
        "host_tail_required": False, "postprocessing_required": False,
    }), encoding="utf-8")
    runner = _load_script("smoke_hailo10_hef_runner.py")

    declaration = runner._resolve_declared_output_contract(
        adjacent, hw_arch="hailo8", model="resnet50", task="classification",
    )

    assert declaration["contract_resolution_status"] == "conflict"
    assert declaration["contract_resolution_reason"] == (
        "standalone_output_contract_not_authoritative"
    )
    assert "stage" not in declaration
    endpoint = runner._output_contract(
        "classification", {"logits": np.zeros((1, 1000), dtype=np.float32)},
        declaration,
    )
    assert endpoint["endpoint_contract_complete"] is False
    assert endpoint["claim_eligible_e2e"] is False


def _hailo_contract_inputs(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    full_plan = {
        "task": "classification",
        "baselines": [{
            "model_id": "resnet50", "backend": "hailo8", "variant": "full",
            "endpoint_mode": "decoded", "reuse_policy": "reuse_first",
        }],
    }
    contracts = {
        "model_id": "resnet50", "task": "classification",
        "contracts": [{
            **_row(backend="hailo8"),
            "contract_status": "pending_build_or_prepare",
        }],
    }
    return full_plan, contracts


def test_hailo8_verified_reuse_becomes_recorded_and_tamper_fails(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prepared.hef"
    source.write_bytes(b"verified-prepared-hef")
    source_onnx, compiler_onnx = _seal_test_hailo_hef(
        source, task="classification", hw_arch="hailo8"
    )
    full_plan, contracts = _hailo_contract_inputs(tmp_path)
    full_plan["baselines"][0]["artifact_path"] = str(source)
    full_plan["baselines"][0]["source_onnx_path"] = str(source_onnx)
    full_plan["baselines"][0]["compiler_onnx_path"] = str(compiler_onnx)
    run_root = tmp_path / "run"

    result = materialize_backend_artifact_decisions(
        run_dir=run_root, model_id="resnet50", targets=["hailo8"],
        full_baseline_plan=full_plan, output_contracts=contracts,
        benchmark_set_contract={
            "materialized": False, "materialization_scope": "contract_only",
            "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
        },
    )
    suite = run_root / "models/resnet50/benchmark_set"
    payload = json.loads((suite / "output_contracts.json").read_text(encoding="utf-8"))
    recorded = payload["contracts"][0]
    assert result["metrics"]["recorded_hailo_full_contracts"] == 1
    assert recorded["contract_status"] == "recorded"
    assert recorded["artifact_binding_status"] == "verified"
    assert "artifact_binding_error" not in recorded
    assert len(recorded["recorded_artifact_sha256"]) == 64
    assert len(recorded["hailo_build_receipt_file_sha256"]) == 64
    assert len(recorded["hailo_build_receipt_identity_sha256"]) == 64
    assert (suite / recorded["hailo_build_receipt_path"]).is_file()
    decisions = json.loads(
        (suite / "backend_artifact_decisions.json").read_text(encoding="utf-8")
    )
    baseline_decision = decisions["baseline_decisions"][0]
    copied = decisions["copied_artifacts"][0]
    for row in (baseline_decision, copied):
        assert len(row["hailo_build_receipt_file_sha256"]) == 64
        assert len(row["hailo_build_receipt_identity_sha256"]) == 64
        assert row["preprocessing_contract_sha256"] == recorded[
            "preprocessing_contract_sha256"
        ]
    benchmark_set = json.loads(
        (suite / "benchmark_set.json").read_text(encoding="utf-8")
    )
    full_meta = benchmark_set["hailo"]["hefs"]["hailo8"]
    assert full_meta["full_build"]["receipt_validated"] is True
    assert full_meta["full_receipt_attested"] is True
    assert full_meta["full_output_contract"][
        "hailo_build_receipt_identity_sha256"
    ] == recorded["hailo_build_receipt_identity_sha256"]

    declaration = load_authoritative_output_contract(
        suite, backend="hailo8", model_id="resnet50",
        variant="full", task="classification",
    )
    assert declaration["stage"] == "classification_logits"

    artifact = suite / recorded["recorded_artifact_path"]
    artifact.write_bytes(b"tampered-hef")
    tampered = load_authoritative_output_contract(
        suite, backend="hailo8", model_id="resnet50",
        variant="full", task="classification",
    )
    assert tampered["contract_resolution_status"] == "conflict"
    assert "stage" not in tampered
    assert "hailo8_recorded_artifact_sha256_mismatch" in tampered[
        "contract_resolution_errors"
    ]


def test_hailo8_successful_build_result_becomes_recorded(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    suite = run_root / "models/resnet50/benchmark_set"
    artifact = suite / "hailo/hailo8/full/compiled.hef"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"successful-build-hef")
    source_onnx, compiler_onnx = _seal_test_hailo_hef(
        artifact, task="classification", hw_arch="hailo8"
    )
    (artifact.parent / "hailo_hef_build_result.json").write_text(json.dumps({
        "ok": True, "skipped": False, "returncode": 0,
        "hef_path": str(artifact.resolve()),
        "source_onnx_path": str(source_onnx),
        "fixed_onnx_path": str(compiler_onnx),
    }), encoding="utf-8")
    full_plan, contracts = _hailo_contract_inputs(tmp_path)

    result = materialize_backend_artifact_decisions(
        run_dir=run_root, model_id="resnet50", targets=["hailo8"],
        full_baseline_plan=full_plan, output_contracts=contracts,
        benchmark_set_contract={
            "materialized": False, "materialization_scope": "contract_only",
            "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
        },
    )

    assert result["metrics"]["recorded_hailo_full_contracts"] == 1
    declaration = load_authoritative_output_contract(
        suite, backend="hailo8", model_id="resnet50",
        variant="full", task="classification",
    )
    assert declaration["contract_resolution_status"] == "attested"


def test_hailo8l_receipt_keeps_exact_physical_backend_binding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prepared-8l.hef"
    source.write_bytes(b"verified-hailo8l-hef")
    source_onnx, compiler_onnx = _seal_test_hailo_hef(
        source, task="classification", hw_arch="hailo8l"
    )
    full_plan, contracts = _hailo_contract_inputs(tmp_path)
    full_plan["baselines"][0].update({
        "backend": "hailo8l",
        "artifact_path": str(source),
        "source_onnx_path": str(source_onnx),
        "compiler_onnx_path": str(compiler_onnx),
    })
    contracts["contracts"][0]["backend"] = "hailo8l"
    run_root = tmp_path / "run"

    result = materialize_backend_artifact_decisions(
        run_dir=run_root,
        model_id="resnet50",
        targets=["hailo8l"],
        full_baseline_plan=full_plan,
        output_contracts=contracts,
        benchmark_set_contract={
            "materialized": False,
            "materialization_scope": "contract_only",
            "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
        },
    )

    assert result["metrics"]["copied_artifacts"] == 1
    assert result["metrics"]["recorded_hailo_full_contracts"] == 1
    suite = run_root / "models/resnet50/benchmark_set"
    contract = json.loads(
        (suite / "output_contracts.json").read_text(encoding="utf-8")
    )["contracts"][0]
    assert contract["backend"] == "hailo8l"
    assert contract["hailo_build_receipt_hw_arch"] == "hailo8l"
    assert contract["artifact_binding_status"] == "verified"


def test_hailo8_reuse_without_build_receipt_is_not_compile_ok_or_recorded(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prepared.hef"
    source.write_bytes(b"unsealed-prepared-hef")
    full_plan, contracts = _hailo_contract_inputs(tmp_path)
    full_plan["baselines"][0].update({
        "artifact_path": str(source),
        "compile_ok": True,
    })
    run_root = tmp_path / "run"
    result = materialize_backend_artifact_decisions(
        run_dir=run_root,
        model_id="resnet50",
        targets=["hailo8"],
        full_baseline_plan=full_plan,
        output_contracts=contracts,
        benchmark_set_contract={
            "materialized": False,
            "materialization_scope": "contract_only",
            "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
        },
    )
    suite = run_root / "models/resnet50/benchmark_set"
    decisions = json.loads(
        (suite / "backend_artifact_decisions.json").read_text(encoding="utf-8")
    )
    decision = decisions["baseline_decisions"][0]
    assert decision["decision"] == "reuse_build_receipt_missing"
    assert decision["compile_ok"] is False
    assert decision["artifact_copy_verified"] is False
    assert result["metrics"]["copied_artifacts"] == 0
    assert result["metrics"]["recorded_hailo_full_contracts"] == 0
    assert not (suite / "hailo/hailo8/full/compiled.hef").exists()


def test_hailo8_reuse_without_declared_onnx_paths_is_not_promoted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prepared.hef"
    source.write_bytes(b"receipt-bound-but-unresolved-onnx")
    _seal_test_hailo_hef(
        source, task="classification", hw_arch="hailo8"
    )
    full_plan, contracts = _hailo_contract_inputs(tmp_path)
    full_plan["baselines"][0].update({
        "artifact_path": str(source),
        "compile_ok": True,
    })
    run_root = tmp_path / "run"

    result = materialize_backend_artifact_decisions(
        run_dir=run_root,
        model_id="resnet50",
        targets=["hailo8"],
        full_baseline_plan=full_plan,
        output_contracts=contracts,
        benchmark_set_contract={
            "materialized": False,
            "materialization_scope": "contract_only",
            "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
        },
    )

    suite = run_root / "models/resnet50/benchmark_set"
    decisions = json.loads(
        (suite / "backend_artifact_decisions.json").read_text(encoding="utf-8")
    )
    decision = decisions["baseline_decisions"][0]
    assert decision["decision"] == "reuse_build_receipt_invalid"
    assert decision["compile_ok"] is False
    assert "source_onnx_file_missing" in decision["error_detail"]
    # The compiler ONNX is now recoverable from the receipt-signed sibling
    # filename and SHA-256.  The undeclared source ONNX remains a hard blocker.
    assert "compiler_onnx_file_missing" not in decision["error_detail"]
    assert result["metrics"]["copied_artifacts"] == 0
    assert result["metrics"]["recorded_hailo_full_contracts"] == 0
    assert not (suite / "hailo/hailo8/full/compiled.hef").exists()


def test_hailo_receipt_requires_exact_hw_and_strict_positive_fields(
    tmp_path: Path,
) -> None:
    hef = tmp_path / "compiled.hef"
    hef.write_bytes(b"hef")
    source_onnx, compiler_onnx = _seal_test_hailo_hef(
        hef, task="classification", hw_arch="hailo8r"
    )
    receipt_path = hef.parent / "hailo_hef_build_receipt.json"
    original = json.loads(receipt_path.read_text(encoding="utf-8"))

    wrong_hw = _validate_hailo_build_receipt(
        hef,
        source_onnx_candidates=[source_onnx],
        compiler_onnx_candidates=[compiler_onnx],
        expected_backend="hailo8l",
        expected_task="classification",
    )
    assert wrong_hw["valid"] is False
    assert "hailo_build_receipt_hw_arch_mismatch" in wrong_hw["errors"]

    cases = []
    fractional_size = json.loads(json.dumps(original))
    fractional_size["hef_size_bytes"] = f"{hef.stat().st_size}.9"
    cases.append((fractional_size, "hailo_build_receipt_hef_size_mismatch"))

    missing_count = json.loads(json.dumps(original))
    missing_count.pop("calibration_count", None)
    missing_count["cache_payload"].pop("calibration_count", None)
    cases.append((missing_count, "hailo_build_receipt_calibration_count_invalid"))

    zero_count = json.loads(json.dumps(original))
    zero_count["calibration_count"] = 0
    zero_count["cache_payload"]["calibration_count"] = 0
    cases.append((zero_count, "hailo_build_receipt_calibration_count_invalid"))

    empty_identity = json.loads(json.dumps(original))
    empty_identity["calibration_identity"] = ""
    empty_identity["cache_payload"]["calibration_identity"] = ""
    prepared_sha = hashlib.sha256(json.dumps({
        "calibration_identity": "",
        "preprocessing_contract_sha256": empty_identity[
            "preprocessing_contract_sha256"
        ],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    empty_identity["prepared_calibration_identity_sha256"] = prepared_sha
    empty_identity["cache_payload"][
        "prepared_calibration_identity_sha256"
    ] = prepared_sha
    cases.append((
        empty_identity,
        "hailo_build_receipt_calibration_identity_missing",
    ))

    for receipt, expected_error in cases:
        receipt["cache_key"] = hashlib.sha256(json.dumps(
            receipt["cache_payload"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        evidence = _validate_hailo_build_receipt(
            hef,
            source_onnx_candidates=[source_onnx],
            compiler_onnx_candidates=[compiler_onnx],
            expected_backend="hailo8r",
            expected_task="classification",
        )
        assert evidence["valid"] is False
        assert expected_error in evidence["errors"]


def test_invalid_receipt_demotes_stale_contract_and_benchmark_claims(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unsealed.hef"
    source.write_bytes(b"unsealed-stale-hef")
    full_plan, contracts = _hailo_contract_inputs(tmp_path)
    full_plan["baselines"][0]["artifact_path"] = str(source)
    stale = contracts["contracts"][0]
    stale.update({
        "contract_status": "recorded",
        "artifact_binding_status": "verified",
        "artifact_path": "hailo/hailo8/full/compiled.hef",
        "artifact_sha256": "a" * 64,
        "recorded_artifact_path": "hailo/hailo8/full/compiled.hef",
        "recorded_artifact_sha256": "a" * 64,
        "artifact_binding_sha256": "b" * 64,
        "hailo_build_receipt_identity_sha256": "c" * 64,
    })
    run_root = tmp_path / "run"
    suite = run_root / "models/resnet50/benchmark_set"
    suite.mkdir(parents=True)
    (suite / "benchmark_set.json").write_text(json.dumps({
        "model_id": "resnet50",
        "benchmark_task": "classification",
        "hailo": {"hefs": {"hailo8": {
            "full": str(source),
            "full_prepared_baseline": True,
            "full_receipt_attested": True,
            "full_build": {
                "ok": True,
                "receipt_validated": True,
                "artifact_hash": "a" * 64,
            },
            "full_build_receipt": {"receipt_validated": True},
            "full_output_contract": dict(stale),
        }}},
    }), encoding="utf-8")

    result = materialize_backend_artifact_decisions(
        run_dir=run_root,
        model_id="resnet50",
        targets=["hailo8"],
        full_baseline_plan=full_plan,
        output_contracts=contracts,
        benchmark_set_contract={
            "materialized": False,
            "materialization_scope": "contract_only",
            "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
        },
    )
    assert result["metrics"]["recorded_hailo_full_contracts"] == 0

    for path in (
        suite / "output_contracts.json",
        run_root / "models/resnet50/full_baselines/output_contracts.json",
    ):
        contract = json.loads(path.read_text(encoding="utf-8"))["contracts"][0]
        assert contract["contract_status"] == "pending_build_or_prepare"
        assert contract["artifact_binding_status"] == "pending_receipt_validation"
        assert contract["artifact_binding_error"] == (
            "hailo_full_receipt_not_verified"
        )
        assert "recorded_artifact_sha256" not in contract
        assert "hailo_build_receipt_identity_sha256" not in contract

    benchmark = json.loads(
        (suite / "benchmark_set.json").read_text(encoding="utf-8")
    )
    meta = benchmark["hailo"]["hefs"]["hailo8"]
    assert meta["full_build"]["ok"] is False
    assert meta["full_build"]["receipt_validated"] is False
    assert meta["full_receipt_attested"] is False
    assert "full_build_receipt" not in meta
    assert meta["full_output_contract"]["contract_status"] == (
        "pending_build_or_prepare"
    )
    declaration = load_authoritative_output_contract(
        suite,
        backend="hailo8",
        model_id="resnet50",
        variant="full",
        task="classification",
    )
    assert declaration["contract_resolution_status"] != "attested"


def test_hailo8_reuse_with_tampered_build_receipt_is_not_promoted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prepared.hef"
    source.write_bytes(b"receipt-bound-prepared-hef")
    source_onnx, compiler_onnx = _seal_test_hailo_hef(
        source, task="classification", hw_arch="hailo8"
    )
    receipt_path = source.parent / "hailo_hef_build_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["cache_key"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    full_plan, contracts = _hailo_contract_inputs(tmp_path)
    full_plan["baselines"][0].update({
        "artifact_path": str(source),
        "source_onnx_path": str(source_onnx),
        "compiler_onnx_path": str(compiler_onnx),
        "compile_ok": True,
    })
    run_root = tmp_path / "run"
    result = materialize_backend_artifact_decisions(
        run_dir=run_root,
        model_id="resnet50",
        targets=["hailo8"],
        full_baseline_plan=full_plan,
        output_contracts=contracts,
        benchmark_set_contract={
            "materialized": False,
            "materialization_scope": "contract_only",
            "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
        },
    )
    suite = run_root / "models/resnet50/benchmark_set"
    decisions = json.loads(
        (suite / "backend_artifact_decisions.json").read_text(encoding="utf-8")
    )
    decision = decisions["baseline_decisions"][0]
    assert decision["decision"] == "reuse_build_receipt_invalid"
    assert decision["compile_ok"] is False
    assert "cache_key_payload_mismatch" in decision["error_detail"]
    assert result["metrics"]["copied_artifacts"] == 0
    assert result["metrics"]["recorded_hailo_full_contracts"] == 0
    contract = json.loads(
        (suite / "output_contracts.json").read_text(encoding="utf-8")
    )["contracts"][0]
    assert contract.get("artifact_binding_status") != "verified"


def test_hailo8_reuse_with_receipt_sdk_mismatch_is_not_promoted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prepared.hef"
    source.write_bytes(b"receipt-bound-prepared-hef")
    source_onnx, compiler_onnx = _seal_test_hailo_hef(
        source, task="classification", hw_arch="hailo8"
    )
    receipt_path = source.parent / "hailo_hef_build_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["hailo_sdk_version"] = "tampered-sdk-version"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    full_plan, contracts = _hailo_contract_inputs(tmp_path)
    full_plan["baselines"][0].update({
        "artifact_path": str(source),
        "source_onnx_path": str(source_onnx),
        "compiler_onnx_path": str(compiler_onnx),
        "compile_ok": True,
    })
    run_root = tmp_path / "run"

    result = materialize_backend_artifact_decisions(
        run_dir=run_root,
        model_id="resnet50",
        targets=["hailo8"],
        full_baseline_plan=full_plan,
        output_contracts=contracts,
        benchmark_set_contract={
            "materialized": False,
            "materialization_scope": "contract_only",
            "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
        },
    )

    suite = run_root / "models/resnet50/benchmark_set"
    decisions = json.loads(
        (suite / "backend_artifact_decisions.json").read_text(encoding="utf-8")
    )
    decision = decisions["baseline_decisions"][0]
    assert decision["decision"] == "reuse_build_receipt_invalid"
    assert decision["compile_ok"] is False
    assert "cache_sdk_version_mismatch" in decision["error_detail"]
    assert result["metrics"]["copied_artifacts"] == 0
    assert result["metrics"]["recorded_hailo_full_contracts"] == 0


def test_hailo8_reuse_with_changed_declared_onnx_is_not_promoted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prepared.hef"
    source.write_bytes(b"receipt-bound-prepared-hef")
    source_onnx, compiler_onnx = _seal_test_hailo_hef(
        source, task="classification", hw_arch="hailo8"
    )
    source_onnx.write_bytes(b"changed-after-receipt-was-sealed")
    compiler_onnx.write_bytes(b"compiler-changed-after-receipt-was-sealed")
    full_plan, contracts = _hailo_contract_inputs(tmp_path)
    full_plan["baselines"][0].update({
        "artifact_path": str(source),
        "source_onnx_path": str(source_onnx),
        "compiler_onnx_path": str(compiler_onnx),
        "compile_ok": True,
    })
    run_root = tmp_path / "run"
    result = materialize_backend_artifact_decisions(
        run_dir=run_root,
        model_id="resnet50",
        targets=["hailo8"],
        full_baseline_plan=full_plan,
        output_contracts=contracts,
        benchmark_set_contract={
            "materialized": False,
            "materialization_scope": "contract_only",
            "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
        },
    )
    suite = run_root / "models/resnet50/benchmark_set"
    decisions = json.loads(
        (suite / "backend_artifact_decisions.json").read_text(encoding="utf-8")
    )
    decision = decisions["baseline_decisions"][0]
    assert decision["decision"] == "reuse_build_receipt_invalid"
    assert decision["compile_ok"] is False
    assert "source_onnx_file_mismatch" in decision["error_detail"]
    assert "compiler_onnx_file_mismatch" in decision["error_detail"]
    assert result["metrics"]["copied_artifacts"] == 0
    assert result["metrics"]["recorded_hailo_full_contracts"] == 0
    assert not (suite / "hailo/hailo8/full/compiled.hef").exists()


def test_generic_resnet_full_composed_and_native_full_share_exact_endpoint(
    tmp_path: Path,
) -> None:
    funcs = _template_functions(
        "_recorded_suite_endpoint_declaration",
        "_central_quality_endpoint_contract",
    )
    suite = tmp_path / "suite"
    case = suite / "b010"
    case.mkdir(parents=True)
    full = suite / "full.onnx"
    part2 = case / "part2.onnx"
    _save_classifier(full)
    _save_classifier(part2)
    (suite / "benchmark_set.json").write_text(
        json.dumps({"model_id": "resnet50", "benchmark_task": "classification"}),
        encoding="utf-8",
    )
    (case / "split_manifest.json").write_text(json.dumps({
        "full_model": "../full.onnx", "part2_model": "part2.onnx",
    }), encoding="utf-8")
    _write_contracts(
        suite, [_row()], model="resnet50", task="classification",
    )
    output = np.zeros((1, 1000), dtype=np.float32)
    generic_contracts = []
    for variant, terminal in (("full", full), ("composed", part2)):
        declaration = funcs["_recorded_suite_endpoint_declaration"](
            base_dir=case, full_model=full, terminal_model=terminal,
            variant=variant, provider="tensorrt", task="classification",
            output_names=["logits"], outputs=[output],
        )
        assert declaration["contract_resolution_status"] == "attested"
        generic = funcs["_central_quality_endpoint_contract"](
            task="classification", output_names=["logits"], outputs=[output],
            detected_output_format="classification_logits",
            declared_endpoint_contract=declaration,
        )
        assert generic["endpoint_contract_complete"] is True
        generic_contracts.append(generic)

    native = _load_script("native_full_semantic_dump.py")
    native_declaration = load_authoritative_output_contract(
        suite, backend="tensorrt", model_id="resnet50",
        variant="full", task="classification",
    )
    native_contract = native._contract(
        "classification", {"logits": output}, native_declaration,
    )
    assert native_contract["claim_eligible_e2e"] is True
    assert {
        generic_contracts[0]["endpoint_contract_hash"],
        generic_contracts[1]["endpoint_contract_hash"],
        native_contract["endpoint_contract_hash"],
    } == {native_contract["endpoint_contract_hash"]}
