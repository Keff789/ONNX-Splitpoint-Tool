from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from PIL import Image

from onnx_splitpoint_tool.hailo_backend import (
    _effective_hailo_calibration_count,
    _hailo_authoritative_calibration_sample_count,
    _hailo_cache_key,
    _hailo_calibration_shape_candidates,
    _load_valid_hailo_receipt,
    _verify_hailo_calibration_count_after_translation,
    _verify_hailo_materialized_calibration_count,
    _write_hailo_receipt,
)
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
)
from scripts import native_full_baseline_eval_runner as full_runner
from onnx_splitpoint_tool.workflow.artifacts import sha256_payload


ROOT = Path(__file__).resolve().parents[1]
MEMORY_CAP = 256 * 1024 * 1024


def _runtime_receipt_fixture(
    tmp_path: Path,
    *,
    receipt_hw: str = "hailo10h",
    end_nodes: list[str] | None = None,
    raw_contract: bool = False,
    source_raw_head: bool = False,
    source_bn6: bool = False,
) -> tuple[Path, Path, Path, Path]:
    benchmark_set = tmp_path / "benchmark_set"
    model = "yolov7_paper"
    source = benchmark_set / "models" / f"{model}.onnx"
    compiler = benchmark_set / "compiler" / f"{model}_hailo_fixed.onnx"
    hef = benchmark_set / "hailo" / receipt_hw / "full" / "compiled.hef"
    source.parent.mkdir(parents=True)
    compiler.parent.mkdir(parents=True)
    hef.parent.mkdir(parents=True)
    if source_raw_head or source_bn6:
        if source_bn6:
            output_specs = [("detections", [1, 300, 6])]
        else:
            output_specs = [
                ("head_80", [1, 3, 80, 80, 85]),
                ("head_40", [1, 3, 40, 40, 85]),
                ("head_20", [1, 3, 20, 20, 85]),
            ]
        graph_outputs = [
            onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, shape,
            )
            for name, shape in output_specs
        ]
        source.write_bytes(onnx.helper.make_model(
            onnx.helper.make_graph(
                [], "raw_source", graph_outputs, graph_outputs,
            )
        ).SerializeToString())
    else:
        source.write_bytes(b"selected-source-onnx")
    compiler.write_bytes(b"distinct-fixed-compiler-onnx")
    hef.write_bytes(b"compiled-hef")
    contract = canonical_image_preprocessing_contract(
        "detection", (640, 640),
    )
    cache_key, cache_payload = _hailo_cache_key(
        model_path=compiler,
        activation_part1=None,
        hw_arch=receipt_hw,
        opt_level=1,
        calib_dir=None,
        calib_count=64,
        effective_calib_count=54,
        calibration_storage="memory",
        calibration_memory_cap_bytes=MEMORY_CAP,
        calib_batch_size=8,
        extra_model_script="",
        start_nodes=None,
        end_nodes=end_nodes,
        preprocessing_contract=contract,
    )
    receipt = _write_hailo_receipt(
        hef_path=hef,
        source_onnx=source,
        compiler_onnx=compiler,
        hw_arch=receipt_hw,
        net_name=model,
        preprocessing_contract=contract,
        preprocessing_sha256=preprocessing_contract_sha256(contract),
        cache_key=cache_key,
        cache_payload=cache_payload,
        calibration_identity=str(cache_payload["calibration_identity"]),
        calibration_count=int(cache_payload["calibration_count"]),
    )
    receipt_path = hef.parent / "hailo_hef_build_receipt.json"
    projected = {
        "hailo_build_receipt_file_sha256": full_runner._sha256_file(
            receipt_path
        ),
        "hailo_build_receipt_identity_sha256": (
            sha256_payload(receipt)
        ),
        "hailo_build_receipt_schema": receipt["schema"],
        "hailo_build_receipt_cache_key": receipt["cache_key"],
        "hailo_build_receipt_cache_payload_sha256": (
            full_runner._canonical_json_sha256(receipt["cache_payload"])
        ),
        "hailo_build_receipt_hw_arch": receipt["hw_arch"],
        "hailo_build_receipt_sdk_version": receipt["hailo_sdk_version"],
        "hailo_build_receipt_calibration_identity": (
            receipt["calibration_identity"]
        ),
        "hailo_build_receipt_prepared_calibration_identity_sha256": (
            receipt["prepared_calibration_identity_sha256"]
        ),
        "hailo_build_receipt_calibration_count": (
            receipt["calibration_count"]
        ),
        "hailo_build_receipt_requested_calibration_count": (
            receipt["requested_calibration_count"]
        ),
        "hailo_build_receipt_calibration_storage": (
            receipt["calibration_storage"]
        ),
        "hailo_build_receipt_calibration_memory_cap_bytes": (
            receipt["calibration_memory_cap_bytes"]
        ),
        "source_onnx_sha256": receipt["source_onnx_sha256"],
        "compiler_onnx_sha256": receipt["compiler_onnx_sha256"],
        "compiler_onnx_path": str(compiler.relative_to(benchmark_set)),
        "hailo_build_receipt_end_nodes": list(end_nodes or []),
    }
    full_meta: dict[str, Any] = {
        "full": str(hef.relative_to(benchmark_set)),
        "full_build_receipt": dict(projected),
    }
    contracts: list[dict[str, Any]] = []
    if end_nodes or raw_contract:
        full_meta.update({
            "full_endpoint_mode": "raw_detection_head",
            "full_end_node_names": list(end_nodes),
            "source_onnx_multiscale_raw_head": source_raw_head,
            "full_output_contract": {
                **projected,
                "endpoint_mode": "raw_detection_head",
                "requires_external_postprocess": True,
                "hailo_build_receipt_end_nodes": list(end_nodes),
                "end_node_names": list(end_nodes),
                "raw_endpoint_origin": (
                    "source_onnx_graph_outputs"
                    if source_raw_head else "compiler_end_nodes"
                ),
                "source_onnx_multiscale_raw_head": source_raw_head,
            },
        })
        contracts.append({
            **projected,
            "model_id": model,
            "task": "detection",
            "backend": receipt_hw,
            "variant": "full",
            "endpoint_mode": "raw_detection_head",
            "requires_external_postprocess": True,
            "hailo_build_receipt_end_nodes": list(end_nodes),
            "end_node_names": list(end_nodes),
            "raw_endpoint_origin": (
                "source_onnx_graph_outputs"
                if source_raw_head else "compiler_end_nodes"
            ),
            "source_onnx_multiscale_raw_head": source_raw_head,
        })
    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps({"hailo": {"hefs": {receipt_hw: full_meta}}}),
        encoding="utf-8",
    )
    (benchmark_set / "output_contracts.json").write_text(
        json.dumps({"model_id": model, "contracts": contracts}),
        encoding="utf-8",
    )
    return benchmark_set, source, compiler, hef


def _receipt_payload(hef: Path) -> dict[str, Any]:
    return json.loads(
        (hef.parent / "hailo_hef_build_receipt.json").read_text(
            encoding="utf-8",
        )
    )


def _save_receipt(hef: Path, receipt: dict[str, Any]) -> None:
    (hef.parent / "hailo_hef_build_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8",
    )


def test_runtime_receipt_accepts_distinct_source_and_compiler_only_when_bound(
    tmp_path: Path,
) -> None:
    benchmark_set, source, compiler, hef = _runtime_receipt_fixture(tmp_path)
    assert source.read_bytes() != compiler.read_bytes()

    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch="hailo10",
    )

    assert status == "hailo_hef_build_receipt_verified_exact"
    assert verified is not None
    assert verified["source_onnx_sha256"] != verified["compiler_onnx_sha256"]
    assert (
        verified["receipt"]["cache_payload"]["model_sha256"]
        == verified["compiler_onnx_sha256"]
    )
    assert verified["requested_calibration_count"] == 64
    assert verified["calibration_count"] == 54


def test_runtime_receipt_accepts_receipt_bound_source_raw_heads_without_cut(
    tmp_path: Path,
) -> None:
    benchmark_set, _source, _compiler, hef = _runtime_receipt_fixture(
        tmp_path,
        end_nodes=[],
        raw_contract=True,
        source_raw_head=True,
    )

    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch="hailo10h",
    )

    assert status == "hailo_hef_build_receipt_verified_exact"
    assert verified is not None
    assert verified["compiler_end_nodes"] == []
    assert verified["raw_endpoint_origin"] == "source_onnx_graph_outputs"
    attestation = verified["source_onnx_raw_head_attestation"]
    assert [row["shape"] for row in attestation["outputs"]] == [
        [1, 3, 80, 80, 85],
        [1, 3, 40, 40, 85],
        [1, 3, 20, 20, 85],
    ]
    assert len(attestation["attestation_sha256"]) == 64


def test_runtime_receipt_rejects_empty_cut_without_source_raw_declaration(
    tmp_path: Path,
) -> None:
    benchmark_set, _source, _compiler, hef = _runtime_receipt_fixture(
        tmp_path,
        end_nodes=[],
        raw_contract=True,
        source_raw_head=False,
    )

    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch="hailo10h",
    )

    assert verified is None
    assert status == "hailo_hef_build_receipt_raw_end_nodes_missing"


def test_runtime_receipt_rejects_source_raw_flag_for_decoded_bn6_graph(
    tmp_path: Path,
) -> None:
    benchmark_set, _source, _compiler, hef = _runtime_receipt_fixture(
        tmp_path,
        end_nodes=[],
        raw_contract=True,
        source_raw_head=True,
        source_bn6=True,
    )

    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch="hailo10h",
    )

    assert verified is None
    assert status == "hailo_source_raw_head_signature_not_multiscale"


def test_runtime_receipt_rejects_joint_compiler_cache_key_reseal(
    tmp_path: Path,
) -> None:
    benchmark_set, _source, compiler, hef = _runtime_receipt_fixture(tmp_path)
    compiler.write_bytes(b"attacker-compiler-onnx")
    attacker_sha = full_runner._sha256_file(compiler)
    receipt = _receipt_payload(hef)
    receipt["compiler_onnx_sha256"] = attacker_sha
    receipt["cache_payload"]["model_sha256"] = attacker_sha
    receipt["cache_key"] = full_runner._canonical_json_sha256(
        receipt["cache_payload"]
    )
    _save_receipt(hef, receipt)

    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch="hailo10h",
    )

    assert verified is None
    assert "projected_claim_mismatch" in status


def test_runtime_receipt_rejects_missing_physical_compiler_onnx(
    tmp_path: Path,
) -> None:
    benchmark_set, _source, compiler, hef = _runtime_receipt_fixture(tmp_path)
    compiler.unlink()

    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch="hailo10h",
    )

    assert verified is None
    assert status == "hailo_hef_build_receipt_compiler_onnx_mismatch"


@pytest.mark.parametrize("axis", ["sdk", "calibration_identity", "counts"])
def test_runtime_receipt_rejects_jointly_resealed_projected_claims(
    tmp_path: Path, axis: str,
) -> None:
    benchmark_set, _source, _compiler, hef = _runtime_receipt_fixture(tmp_path)
    receipt = _receipt_payload(hef)
    cache = receipt["cache_payload"]
    if axis == "sdk":
        receipt["hailo_sdk_version"] = "attacker-sdk"
        cache["hailo_sdk_version"] = "attacker-sdk"
    elif axis == "calibration_identity":
        receipt["calibration_identity"] = "attacker-calibration"
        cache["calibration_identity"] = "attacker-calibration"
        prepared = full_runner._canonical_json_sha256({
            "calibration_identity": "attacker-calibration",
            "preprocessing_contract_sha256": (
                receipt["preprocessing_contract_sha256"]
            ),
        })
        receipt["prepared_calibration_identity_sha256"] = prepared
        cache["prepared_calibration_identity_sha256"] = prepared
    elif axis == "counts":
        receipt["calibration_count"] = 53
        cache["calibration_count"] = 53
    else:  # pragma: no cover
        raise AssertionError(axis)
    receipt["cache_key"] = full_runner._canonical_json_sha256(cache)
    _save_receipt(hef, receipt)

    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch="hailo10h",
    )
    assert verified is None
    assert "projected_claim_mismatch" in status


@pytest.mark.parametrize(
    "axis",
    [
        "receipt_schema",
        "source_sha",
        "compiler_sha",
        "cache_schema",
        "cache_key",
        "cache_model",
        "cache_hw",
        "sdk",
        "calibration_count",
        "requested_count",
        "storage",
        "memory_cap",
        "calibration_identity",
        "prepared_identity",
        "raw_end_nodes",
    ],
)
def test_runtime_receipt_rejects_adversarial_tampering(
    tmp_path: Path, axis: str,
) -> None:
    nodes = ["/head/box", "/head/class"]
    benchmark_set, _source, _compiler, hef = _runtime_receipt_fixture(
        tmp_path, end_nodes=nodes,
    )
    receipt = _receipt_payload(hef)
    cache = receipt["cache_payload"]
    reseal_cache = False
    if axis == "receipt_schema":
        receipt["schema"] = "onnx-splitpoint/hailo-hef-build-receipt/v1"
    elif axis == "source_sha":
        receipt["source_onnx_sha256"] = "0" * 64
    elif axis == "compiler_sha":
        receipt["compiler_onnx_sha256"] = "0" * 64
    elif axis == "cache_schema":
        cache["schema"] = "attacker/cache"
        reseal_cache = True
    elif axis == "cache_key":
        receipt["cache_key"] = "0" * 64
    elif axis == "cache_model":
        cache["model_sha256"] = "0" * 64
        reseal_cache = True
    elif axis == "cache_hw":
        cache["hw_arch"] = "hailo8"
        reseal_cache = True
    elif axis == "sdk":
        receipt["hailo_sdk_version"] = "attacker-sdk"
    elif axis == "calibration_count":
        receipt["calibration_count"] = 53
    elif axis == "requested_count":
        receipt["requested_calibration_count"] = 63
    elif axis == "storage":
        receipt["calibration_storage"] = "memmap"
    elif axis == "memory_cap":
        receipt["calibration_memory_cap_bytes"] = MEMORY_CAP - 1
    elif axis == "calibration_identity":
        receipt["calibration_identity"] = "attacker-calibration"
    elif axis == "prepared_identity":
        receipt["prepared_calibration_identity_sha256"] = "0" * 64
    elif axis == "raw_end_nodes":
        cache["end_nodes"] = ["/attacker/box", "/attacker/class"]
        reseal_cache = True
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(axis)
    if reseal_cache:
        receipt["cache_key"] = full_runner._canonical_json_sha256(cache)
    _save_receipt(hef, receipt)

    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch="hailo10h",
    )

    assert verified is None
    assert status != "hailo_hef_build_receipt_verified_exact"


@pytest.mark.parametrize(
    ("expected_hw", "receipt_hw", "accepted"),
    [
        ("hailo10h", "hailo10", True),
        ("hailo10", "hailo10h", True),
        ("hailo8", "hailo8", True),
        ("hailo8l", "hailo8l", True),
        ("hailo8r", "hailo8r", True),
        ("hailo8l", "hailo8", False),
        ("hailo8r", "hailo8", False),
        ("hailo8", "hailo8l", False),
        ("hailo8l", "hailo8r", False),
    ],
)
def test_runtime_receipt_only_allows_the_hailo10_family_alias(
    tmp_path: Path, expected_hw: str, receipt_hw: str, accepted: bool,
) -> None:
    benchmark_set, _source, _compiler, hef = _runtime_receipt_fixture(
        tmp_path, receipt_hw=receipt_hw,
    )
    verified, _status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch=expected_hw,
    )
    assert (verified is not None) is accepted
    assert full_runner._hailo_aliases(expected_hw) == (
        [expected_hw, "hailo10"] if expected_hw == "hailo10h"
        else [expected_hw, "hailo10h"] if expected_hw == "hailo10"
        else [expected_hw]
    )


def test_memory_calibration_64_clamps_to_54_and_seals_small_dataset_reuse(
    tmp_path: Path,
) -> None:
    compiler = tmp_path / "compiler.onnx"
    source = tmp_path / "source.onnx"
    hef = tmp_path / "compiled.hef"
    compiler.write_bytes(b"compiler")
    source.write_bytes(b"source")
    hef.write_bytes(b"hef")
    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()
    for index in range(7):
        Image.fromarray(
            np.full((8, 9, 3), index, dtype=np.uint8), mode="RGB",
        ).save(calib_dir / f"{index:03d}.png")
    contract = canonical_image_preprocessing_contract(
        "detection", (640, 640),
    )
    shapes = _hailo_calibration_shape_candidates(
        {"input": [640, 640, 3]}, contract,
    )
    memory_effective = _effective_hailo_calibration_count(
        requested=64,
        shapes=shapes,
        storage="memory",
        cap_bytes=MEMORY_CAP,
    )
    assert memory_effective == 54
    available, source_kind = _hailo_authoritative_calibration_sample_count(
        calib_dir,
    )
    assert (available, source_kind) == (7, "image_files")
    sealed_effective = min(memory_effective, int(available or 0))

    def cache_identity() -> tuple[str, dict[str, Any]]:
        return _hailo_cache_key(
            model_path=compiler,
            activation_part1=None,
            hw_arch="hailo10h",
            opt_level=1,
            calib_dir=calib_dir,
            calib_count=64,
            effective_calib_count=sealed_effective,
            calibration_storage="memory",
            calibration_memory_cap_bytes=MEMORY_CAP,
            calib_batch_size=8,
            extra_model_script="",
            start_nodes=None,
            end_nodes=None,
            preprocessing_contract=contract,
        )

    key1, payload1 = cache_identity()
    _write_hailo_receipt(
        hef_path=hef,
        source_onnx=source,
        compiler_onnx=compiler,
        hw_arch="hailo10h",
        net_name="detector",
        preprocessing_contract=contract,
        preprocessing_sha256=preprocessing_contract_sha256(contract),
        cache_key=key1,
        cache_payload=payload1,
        calibration_identity=str(payload1["calibration_identity"]),
        calibration_count=sealed_effective,
    )
    key2, payload2 = cache_identity()

    assert key2 == key1
    assert payload2 == payload1
    assert payload1["requested_calibration_count"] == 64
    assert payload1["calibration_count"] == 7
    assert payload1["calibration_storage"] == "memory"
    assert payload1["calibration_memory_cap_bytes"] == MEMORY_CAP
    loaded = _load_valid_hailo_receipt(
        hef,
        preprocessing_sha256=preprocessing_contract_sha256(contract),
        source_onnx_sha256=payload1.get("unused", None)
        or full_runner._sha256_file(source),
        cache_key=key2,
        cache_payload=payload2,
    )
    assert loaded is not None
    assert loaded["calibration_count"] == 7
    assert loaded["requested_calibration_count"] == 64


def test_post_translation_shape_and_materialized_count_drift_fail_closed() -> None:
    with pytest.raises(
        RuntimeError,
        match="hailo_calibration_effective_count_changed_after_translation",
    ):
        _verify_hailo_calibration_count_after_translation(
            sealed_effective_count=54,
            requested=64,
            expected_shapes={"input": [320, 320, 3]},
            storage="memory",
            cap_bytes=MEMORY_CAP,
        )
    with pytest.raises(
        RuntimeError,
        match="hailo_calibration_materialized_count_mismatch",
    ):
        _verify_hailo_materialized_calibration_count(
            sealed_effective_count=7,
            calib_inputs={"input": np.zeros((6, 2, 2, 3), np.float32)},
        )


def test_hailo_parse_path_has_no_build_only_calibration_state() -> None:
    source_path = ROOT / "onnx_splitpoint_tool" / "hailo_backend.py"
    source = source_path.read_text(encoding="utf-8")
    compile(source, str(source_path), "exec")
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # There is no separate `_hailo_parse_legacy` in this module: the local
    # legacy parse implementation is `hailo_parse_check`, while the legacy
    # HEF build implementation is `_hailo_build_hef_legacy`.
    assert "_hailo_parse_legacy" not in functions
    assert "hailo_parse_check" in functions
    assert "_hailo_build_hef_legacy" in functions
    owners = {
        name for name, node in functions.items()
        # Match the state variable, not a diagnostic string such as
        # "calibration_identity_shapes_unresolved" in workspace admission.
        if any(
            isinstance(child, ast.Name)
            and child.id == "calibration_identity_shapes"
            for child in ast.walk(node)
        )
    }
    assert owners == {"_hailo_build_hef_legacy"}
    for name in (
        "hailo_parse_check", "hailo_parse_check_auto",
        "hailo_parse_check_via_venv", "hailo_parse_check_via_wsl",
    ):
        parse_source = ast.get_source_segment(source, functions[name]) or ""
        assert "effective_calib_count" not in parse_source
        assert "calibration_available_count" not in parse_source
