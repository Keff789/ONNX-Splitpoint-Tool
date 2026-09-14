from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from onnx_splitpoint_tool.gui.controller import _copy_runner_lib
from onnx_splitpoint_tool.runners.native_full_input import (
    load_sealed_deepx_native_full_input,
    prepare_and_seal_deepx_native_full_input,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE_TEMPLATE = (
    ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _load_full_runner():
    path = ROOT / "scripts" / "native_full_baseline_eval_runner.py"
    spec = importlib.util.spec_from_file_location(
        "test_v2751_native_full_runner", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FULL_RUNNER = _load_full_runner()


def _deepx_input_contract() -> dict[str, object]:
    return {
        "input": {
            "name": "images",
            "shape": [1, 4, 4, 3],
            "dtype": "uint8",
            "layout": "NHWC",
            "color_space": "RGB",
            "normalization": "embedded_dxcom_preprocessing",
            "preprocess_mode": "letterbox",
            "letterbox_pad_value": 114,
        }
    }


def test_deepx_generic_sealer_and_native_loader_replay_exact_bytes(
    tmp_path: Path,
) -> None:
    image = tmp_path / "validation.png"
    Image.new("RGB", (7, 3), color=(11, 22, 33)).save(image)
    sealed = prepare_and_seal_deepx_native_full_input(
        image_path=image,
        input_contract=_deepx_input_contract(),
        task="detection",
        out_dir=tmp_path / "results" / "run" / "prepared_input",
        model="yolo26s",
        setup_id="orin_nx_deepx_m1_01",
        comparison_backend="deepx",
    )

    loaded = load_sealed_deepx_native_full_input(
        sealed["manifest_path"], image_path=image,
        input_contract=_deepx_input_contract(), task="detection",
        expected_model="yolo26s",
        expected_setup_id="orin_nx_deepx_m1_01",
        expected_comparison_backend="deepx",
    )

    assert loaded["runtime_input"].tobytes() == (
        sealed["runtime_input"].tobytes()
    )
    assert loaded["payload"]["runtime_input_sha256"] == hashlib.sha256(
        loaded["runtime_input"].tobytes()
    ).hexdigest()
    assert loaded["payload"]["contract_source"] == (
        "shared_pre_timing_deepx_input_sealer"
    )


def test_generated_suite_vendors_shared_deepx_input_runtime_and_seals_pre_timing(
    tmp_path: Path,
) -> None:
    _copy_runner_lib(tmp_path)
    vendored = tmp_path / "splitpoint_runners" / "native_full_input.py"
    canonical = (
        ROOT / "onnx_splitpoint_tool/runners/native_full_input.py"
    )
    assert vendored.read_bytes() == canonical.read_bytes()

    source = SUITE_TEMPLATE.read_text(encoding="utf-8")
    seal_call = source.index("sealed = prepare_and_seal_deepx_native_full_input(")
    strict_load = source.index("strict_sealed = load_sealed_deepx_native_full_input(")
    sealed_load = source.index("feed, prepared_input_binding = _deepx_load_sealed_prepared_feed(")
    engine_start = source.index("engine = InferenceEngine(str(dxnn))", seal_call)
    timed_loop = source.index("for _ in range(runs):", engine_start)
    assert seal_call < strict_load < sealed_load < engine_start < timed_loop


@pytest.mark.parametrize(
    "mutation", ["tensor", "identity", "path", "absolute_path", "symlink"],
)
def test_deepx_shared_input_tampering_fails_closed(
    tmp_path: Path, mutation: str,
) -> None:
    image = tmp_path / "validation.png"
    Image.new("RGB", (7, 3), color=(11, 22, 33)).save(image)
    sealed = prepare_and_seal_deepx_native_full_input(
        image_path=image,
        input_contract=_deepx_input_contract(),
        task="detection",
        out_dir=tmp_path / "prepared_input",
        model="yolo26s",
    )
    manifest = Path(sealed["manifest_path"])
    target_manifest = manifest
    if mutation == "tensor":
        tensor = manifest.parent / "runtime_input.bin"
        data = bytearray(tensor.read_bytes())
        data[0] ^= 0xFF
        tensor.write_bytes(data)
    elif mutation == "identity":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["runtime_preprocessing_identity"]["pad_value"] = 0
        semantic_sha = _canonical_sha(
            payload["runtime_preprocessing_identity"]
        )
        payload["runtime_preprocessing_sha256"] = semantic_sha
        payload["runtime_numeric_input_identity"][
            "preprocessing_contract_sha256"
        ] = semantic_sha
        payload["runtime_numeric_input_sha256"] = _canonical_sha(
            payload["runtime_numeric_input_identity"]
        )
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "path":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["runtime_input_file"] = "other.bin"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "absolute_path":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["runtime_input_file"] = str(
            manifest.parent / "runtime_input.bin"
        )
        payload["input_dump"] = str(
            manifest.parent / "input_rgb_uint8.bin"
        )
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    else:
        link = tmp_path / "linked_manifest.json"
        link.symlink_to(manifest)
        target_manifest = link

    with pytest.raises(ValueError, match="prepared_input_"):
        load_sealed_deepx_native_full_input(
            target_manifest, image_path=image,
            input_contract=_deepx_input_contract(), task="detection",
        )


@pytest.mark.parametrize("dtype", ["float16", "fp16", "int8"])
def test_deepx_shared_input_rejects_dtype_not_supported_by_both_loaders(
    tmp_path: Path, dtype: str,
) -> None:
    image = tmp_path / "validation.png"
    Image.new("RGB", (7, 3), color=(11, 22, 33)).save(image)
    contract = _deepx_input_contract()
    contract["input"]["dtype"] = dtype  # type: ignore[index]
    with pytest.raises(ValueError, match="prepared_input_contract_dtype_invalid"):
        prepare_and_seal_deepx_native_full_input(
            image_path=image, input_contract=contract, task="detection",
            out_dir=tmp_path / "prepared_input", model="yolo26s",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_model", "yolov7_paper"),
        ("expected_setup_id", "other_setup"),
        ("expected_comparison_backend", "other_backend"),
    ],
)
def test_deepx_shared_input_rejects_cross_identity_replay(
    tmp_path: Path, field: str, value: str,
) -> None:
    image = tmp_path / "validation.png"
    Image.new("RGB", (7, 3), color=(11, 22, 33)).save(image)
    sealed = prepare_and_seal_deepx_native_full_input(
        image_path=image, input_contract=_deepx_input_contract(),
        task="detection", out_dir=tmp_path / "prepared_input",
        model="yolo26s", setup_id="setup_a",
        comparison_backend="deepx",
    )
    expectations = {
        "expected_model": "yolo26s",
        "expected_setup_id": "setup_a",
        "expected_comparison_backend": "deepx",
    }
    expectations[field] = value
    with pytest.raises(ValueError, match="prepared_input_manifest_identity_mismatch"):
        load_sealed_deepx_native_full_input(
            sealed["manifest_path"], image_path=image,
            input_contract=_deepx_input_contract(), task="detection",
            **expectations,
        )


def _semantic_dump_namespace(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dump_outputs=True,
        image_map_data={},
        setup_id="setup_a",
        comparison_backend="deepx",
        out_dir=str(tmp_path / "evidence"),
        trt_precision="fp16",
        timeout=30,
        diagnostic_deepx_input_probes=False,
    )


def test_deepx_semantic_dump_rejects_symlinked_prepared_input_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_set = tmp_path / "benchmark_set"
    benchmark_set.mkdir()
    image = benchmark_set / "validation.png"
    Image.new("RGB", (7, 3), color=(11, 22, 33)).save(image)
    outside = tmp_path / "outside" / "run_a" / "prepared_input"
    sealed = prepare_and_seal_deepx_native_full_input(
        image_path=image, input_contract=_deepx_input_contract(),
        task="detection", out_dir=outside, model="yolo26s",
        setup_id="setup_a", comparison_backend="deepx",
    )
    (benchmark_set / "results").mkdir()
    (benchmark_set / "results" / "run_a").symlink_to(outside.parent)
    monkeypatch.setattr(FULL_RUNNER, "_first_case", lambda _root: ("full", _root, _root))
    monkeypatch.setattr(
        FULL_RUNNER, "_resolve_image",
        lambda *_args, **_kwargs: (image, "fixture"),
    )
    monkeypatch.setattr(
        FULL_RUNNER, "_run_plan_meta",
        lambda *_args, **_kwargs: ("detection", "norm"),
    )
    monkeypatch.setattr(
        FULL_RUNNER, "_run",
        lambda *_args, **_kwargs: pytest.fail("child must not start"),
    )

    result = FULL_RUNNER._semantic_full_dump(
        benchmark_set, "yolo26s", "native_full_deepx", "run_a",
        _semantic_dump_namespace(tmp_path), force=True,
    )

    assert Path(sealed["manifest_path"]).is_file()
    assert result["failure_reason"] == (
        "deepx_shared_prepared_input_manifest_missing"
    )


def test_deepx_semantic_dump_selects_exact_run_and_exact_output_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_set = tmp_path / "benchmark_set"
    benchmark_set.mkdir()
    image = benchmark_set / "validation.png"
    Image.new("RGB", (7, 3), color=(11, 22, 33)).save(image)
    expected = prepare_and_seal_deepx_native_full_input(
        image_path=image, input_contract=_deepx_input_contract(),
        task="detection",
        out_dir=benchmark_set / "results" / "run_a" / "prepared_input",
        model="yolo26s", setup_id="setup_a", comparison_backend="deepx",
    )
    prepare_and_seal_deepx_native_full_input(
        image_path=image, input_contract=_deepx_input_contract(),
        task="detection",
        out_dir=benchmark_set / "results" / "foreign" / "prepared_input",
        model="yolo26s", setup_id="setup_a", comparison_backend="deepx",
    )
    monkeypatch.setattr(FULL_RUNNER, "_first_case", lambda _root: ("full", _root, _root))
    monkeypatch.setattr(
        FULL_RUNNER, "_resolve_image",
        lambda *_args, **_kwargs: (image, "fixture"),
    )
    monkeypatch.setattr(
        FULL_RUNNER, "_run_plan_meta",
        lambda *_args, **_kwargs: ("detection", "norm"),
    )
    monkeypatch.setattr(
        FULL_RUNNER, "_suite_python_env",
        lambda *_args, **_kwargs: (Path(sys.executable), {}, []),
    )
    seen: dict[str, object] = {}

    def _fake_run(command, **_kwargs):
        seen["command"] = list(command)
        out_dir = Path(command[command.index("--out-dir") + 1])
        report = Path(command[command.index("--json-out") + 1])
        out_dir.mkdir(parents=True)
        output_manifest = out_dir / "native_full_outputs_manifest.json"
        input_manifest = out_dir / "native_full_input_manifest.json"
        output_manifest.write_text(json.dumps({
            "model": "yolo26s", "backend": "native_full_deepx",
            "setup_id": "setup_a", "comparison_backend": "deepx",
            "case": "full", "execution_mode": "native_full_baseline",
            "task": "detection",
        }), encoding="utf-8")
        input_manifest.write_text(json.dumps({
            "schema": "onnx-splitpoint/native-full-input-dump",
            "schema_version": 2,
            "model": "yolo26s", "backend": "native_full_deepx",
            "setup_id": "setup_a", "comparison_backend": "deepx",
            "case": "full", "task": "detection",
        }), encoding="utf-8")
        report.write_text(json.dumps({
            "ok": True, "output_manifest": str(output_manifest),
            "input_manifest": str(input_manifest),
        }), encoding="utf-8")
        return {"rc": 0, "timed_out": False}

    monkeypatch.setattr(FULL_RUNNER, "_run", _fake_run)
    result = FULL_RUNNER._semantic_full_dump(
        benchmark_set, "yolo26s", "native_full_deepx", "run_a",
        _semantic_dump_namespace(tmp_path), force=True,
    )

    assert result["ok"] is True
    command = seen["command"]
    manifest_arg = Path(
        command[command.index("--prepared-input-manifest") + 1]  # type: ignore[union-attr]
    )
    assert manifest_arg == Path(expected["manifest_path"])


def test_deepx_semantic_dump_rejects_symlinked_output_parent_before_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_set = tmp_path / "benchmark_set"
    benchmark_set.mkdir()
    image = benchmark_set / "validation.png"
    Image.new("RGB", (7, 3), color=(11, 22, 33)).save(image)
    prepare_and_seal_deepx_native_full_input(
        image_path=image, input_contract=_deepx_input_contract(),
        task="detection",
        out_dir=benchmark_set / "results" / "run_a" / "prepared_input",
        model="yolo26s", setup_id="setup_a", comparison_backend="deepx",
    )
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    outside = tmp_path / "outside-output"
    outside.mkdir()
    (evidence / "native_full_outputs").symlink_to(outside)
    monkeypatch.setattr(FULL_RUNNER, "_first_case", lambda _root: ("full", _root, _root))
    monkeypatch.setattr(
        FULL_RUNNER, "_resolve_image",
        lambda *_args, **_kwargs: (image, "fixture"),
    )
    monkeypatch.setattr(
        FULL_RUNNER, "_run_plan_meta",
        lambda *_args, **_kwargs: ("detection", "norm"),
    )
    monkeypatch.setattr(
        FULL_RUNNER, "_run",
        lambda *_args, **_kwargs: pytest.fail("child must not start"),
    )

    result = FULL_RUNNER._semantic_full_dump(
        benchmark_set, "yolo26s", "native_full_deepx", "run_a",
        _semantic_dump_namespace(tmp_path), force=True,
    )

    assert result["failure_reason"] == "native_full_output_path_contains_symlink"
    assert list(outside.iterdir()) == []


def _partial_binding_set() -> dict[str, object]:
    key = "native_full_deepx|yolo26s"
    payload: dict[str, object] = {
        "schema": "onnx-splitpoint/native-full-quality-request-binding-set",
        "schema_version": 2,
        "eval_run_id": "run_2751",
        "setup_id": "orin_nx_deepx_m1_01",
        "comparison_backend": "deepx",
        "central_quality_summary": "",
        "central_quality_summary_sha256": "",
        "required_binding_keys": [key],
        "binding_status_by_backend_model": {key: "unavailable"},
        "binding_errors_by_backend_model": {
            key: ["missing_vendor_full_quality:native_full_deepx|yolo26s"],
        },
        "complete": False,
        "bindings_by_backend_model": {},
    }
    payload["binding_set_sha256"] = _canonical_sha(payload)
    return payload


def test_partial_quality_set_vetoes_claims_without_rewriting_runtime_success(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run_2751"
    path = (
        run_root / "quality_first"
        / "vendor_full_quality_request_binding_set.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_partial_binding_set()), encoding="utf-8")
    ns = argparse.Namespace(
        quality_request_binding_set=str(path),
        root=str(run_root),
        setup_id="orin_nx_deepx_m1_01",
        comparison_backend="deepx",
    )
    payload, status = FULL_RUNNER._load_full_quality_binding_set(ns)
    assert status == "quality_request_binding_set_verified_partial"
    ns.quality_request_binding_set_data = payload
    ns.quality_request_binding_set_load_status = status

    row, attach_status = FULL_RUNNER._attach_full_quality_request_binding(
        {
            "backend": "native_full_deepx",
            "model": "yolo26s",
            "ok": True,
            "status": "ok",
            "latency_mean_ms": 1.25,
        },
        {},
        model="yolo26s",
        ns=ns,
    )

    assert attach_status == "quality_request_binding_failed"
    assert row["ok"] is True
    assert row["status"] == "ok"
    assert row["latency_mean_ms"] == 1.25
    assert row["quality_request_binding_status"] == "unavailable"
    assert row["performance_claim_eligible"] is False
    assert row["energy_claim_eligible"] is False
    assert row["scientific_claim_eligible"] is False
    assert row["eligible_for_ranking"] is False
    assert row["quality_request_binding_errors"] == [
        "missing_vendor_full_quality:native_full_deepx|yolo26s"
    ]


def test_resealed_inconsistent_partial_quality_set_is_rejected(
    tmp_path: Path,
) -> None:
    payload = _partial_binding_set()
    key = "native_full_deepx|yolo26s"
    payload["binding_status_by_backend_model"] = {key: "verified_exact"}
    payload.pop("binding_set_sha256")
    payload["binding_set_sha256"] = _canonical_sha(payload)
    run_root = tmp_path / "run_2751"
    path = (
        run_root / "quality_first"
        / "vendor_full_quality_request_binding_set.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    ns = argparse.Namespace(
        quality_request_binding_set=str(path), root=str(run_root),
        setup_id="orin_nx_deepx_m1_01", comparison_backend="deepx",
    )

    loaded, status = FULL_RUNNER._load_full_quality_binding_set(ns)

    assert loaded == {}
    assert status == "quality_request_binding_set_status_map_invalid"
