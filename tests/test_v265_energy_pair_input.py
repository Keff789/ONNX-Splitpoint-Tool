from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _sealed_input_fixture(
    monkeypatch,
    module,
    tmp_path: Path,
    *,
    source_image: Path | None = None,
    setup_id: str = "",
):
    from onnx_splitpoint_tool.preprocessing_contract import (
        canonical_image_preprocessing_contract,
        preprocessing_contract_sha256,
        runtime_numeric_input_identity,
    )

    if source_image is None:
        source_image = tmp_path / "prepared-source.jpg"
        source_image.write_bytes(b"sealed-prepared-feed-source")
    assert source_image.is_file()
    semantic_root = module._native_full_dump_dir(
        tmp_path,
        "resnet50",
        "native_full_deepx",
        SimpleNamespace(setup_id=setup_id),
    )
    semantic_root.mkdir(parents=True)
    tensor = semantic_root / "runtime_input.bin"
    tensor.write_bytes(bytes(range(12)))
    digest = hashlib.sha256(tensor.read_bytes()).hexdigest()
    source_image_sha256 = hashlib.sha256(source_image.read_bytes()).hexdigest()
    semantic_identity = canonical_image_preprocessing_contract(
        "classification", [2, 2],
    )
    semantic_sha256 = preprocessing_contract_sha256(semantic_identity)
    numeric_identity, numeric_sha256 = runtime_numeric_input_identity(
        backend="native_full_deepx",
        task="classification",
        preprocessing_contract_sha256_value=semantic_sha256,
        runtime_input_name="input",
        runtime_input_shape=[2, 2, 3],
        runtime_input_dtype="uint8",
        runtime_input_layout="HWC",
        runtime_color_space="RGB",
        runtime_normalization="embedded_dxcom_preprocessing",
    )
    preprocess = {
        "mode": "resize_rgb_uint8",
        "layout": "HWC",
        "normalization": "embedded_dxcom_preprocessing",
        "color_space": "RGB",
        "pad_value": 0,
    }
    manifest = semantic_root / "native_full_input_manifest.json"
    manifest_payload = {
        "schema": "onnx-splitpoint/native-full-input-dump",
        "schema_version": 2,
        "backend": "native_full_deepx",
        "task": "classification",
        "runtime_input_file": str(tensor),
        "runtime_input_name": "input",
        "runtime_input_shape": [2, 2, 3],
        "runtime_input_dtype": "uint8",
        "runtime_input_bytes": 12,
        "runtime_input_sha256": digest,
        "runtime_input_layout": "HWC",
        "runtime_color_space": "RGB",
        "runtime_normalization": "embedded_dxcom_preprocessing",
        "runtime_preprocessing_identity": semantic_identity,
        "runtime_preprocessing_sha256": semantic_sha256,
        "runtime_numeric_input_identity": numeric_identity,
        "runtime_numeric_input_sha256": numeric_sha256,
        "input_image": str(source_image),
        "input_image_sha256": source_image_sha256,
        "preprocess": preprocess,
    }
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True), encoding="utf-8",
    )
    validated = {
        "runtime_input_file": str(tensor),
        "runtime_input_name": "input",
        "runtime_input_shape": [2, 2, 3],
        "runtime_input_dtype": "uint8",
        "runtime_input_bytes": 12,
        "runtime_input_sha256": digest,
        "runtime_input_layout": "HWC",
        "runtime_color_space": "RGB",
        "runtime_normalization": "embedded_dxcom_preprocessing",
        "input_image": str(source_image),
        "input_image_sha256": source_image_sha256,
        "preprocess": preprocess,
        "runtime_preprocessing_identity": semantic_identity,
        "runtime_preprocessing_sha256": semantic_sha256,
        "runtime_numeric_input_identity": numeric_identity,
        "runtime_numeric_input_sha256": numeric_sha256,
    }

    def _validated_manifest(path: Path, *, allowed_root: Path):
        assert path == manifest
        assert allowed_root == semantic_root
        return dict(validated), "runtime_input_manifest_and_tensor_verified"

    monkeypatch.setattr(
        module,
        "_validated_runtime_input_manifest",
        _validated_manifest,
    )
    return manifest, {
        "prepared_input_manifest": str(manifest),
        "prepared_input_manifest_sha256": hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest(),
        "prepared_input_file": str(tensor),
        "prepared_input_sha256": digest,
        "prepared_input_file_sha256": digest,
        "prepared_input_bytes": 12,
        "prepared_input_name": "input",
        "prepared_input_shape": [2, 2, 3],
        "prepared_input_dtype": "uint8",
        "prepared_input_layout": "HWC",
        "prepared_input_source_image_id": source_image.name,
        "prepared_input_source_image_sha256": source_image_sha256,
        "runtime_color_space": "RGB",
        "runtime_normalization": "embedded_dxcom_preprocessing",
        "runtime_preprocess_mode": "resize",
        "runtime_preprocessing_identity": semantic_identity,
        "runtime_preprocessing_sha256": semantic_sha256,
        "runtime_numeric_input_identity": numeric_identity,
        "runtime_numeric_input_sha256": numeric_sha256,
        "prepared_input_binding_verified": True,
        "prepared_input_source": "sealed_semantic_dump_runtime_tensor",
        "preprocess": preprocess,
    }


def _load_runner():
    path = ROOT / "scripts" / "native_full_baseline_eval_runner.py"
    spec = importlib.util.spec_from_file_location("v265_native_full_pair_input", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generated_deepx_full_supports_exact_prepared_feed_image() -> None:
    template = (
        ROOT / "onnx_splitpoint_tool" / "resources" / "templates"
        / "benchmark_suite.py.txt"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--prepared-feed-image"' in template
    assert 'return candidate.resolve(), "explicit_prepared_feed_image"' in template
    assert template.index('getattr(args, "prepared_feed_image"') < template.index(
        'td / "test_image_imagenet.png"'
    )


def test_deepx_full_sealed_manifest_image_wins_over_case_map(
    monkeypatch, tmp_path: Path,
) -> None:
    module = _load_runner()
    benchmark_set = tmp_path / "suite"
    benchmark_set.mkdir()
    expected = benchmark_set / "expected.jpg"
    expected.write_bytes(b"expected-pair-image")
    # The workflow may stage the byte-sealed validation image beside, rather
    # than below, the BenchmarkSet.  Its manifest+SHA identity remains the
    # authority; the performance child must not substitute a suite case.
    observed = tmp_path / "external-observed.jpg"
    observed.write_bytes(b"different-prepared-feed")
    prepared_manifest, prepared_binding = _sealed_input_fixture(
        monkeypatch, module, benchmark_set, source_image=observed,
        setup_id="orin_nx_deepx_m1_01",
    )

    captured: dict[str, object] = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        return {"rc": 0, "timed_out": False}

    monkeypatch.setattr(
        module, "_suite_python_env",
        lambda _ns, _backend: (sys.executable, {}, []),
    )
    monkeypatch.setattr(
        module,
        "_first_case",
        lambda _root: (_ for _ in ()).throw(
            AssertionError("DeepX Full must not select an image from a case")
        ),
    )
    monkeypatch.setattr(
        module,
        "_resolve_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("DeepX Full must not select an image from the case map")
        ),
    )
    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "_metrics_from_rows",
        lambda _paths: {
            "fps_makespan": 25.0,
            "latency_mean_ms": 40.0,
            "result_source": str(tmp_path / "benchmark_results_deepx.json"),
            "result_row": {
                "run_id": "deepx_m1_full",
                "runtime_ok": True,
                "performance_benchmark_source": "dx_engine_prepared_feed",
                "prepared_feed_contract_version": "deepx-sealed-runtime-input-v3",
                "completed_frames": 10,
                "deepx_prepared_feed_benchmark": {
                    "image": str(observed),
                    **prepared_binding,
                    "makespan_s": 0.4,
                    "fps_makespan": 25.0,
                    "mean_ms": 40.0,
                    "completed_frames": 10,
                    "completed_work_units": 10,
                    "completed_work_units_source": (
                        "dx_engine_prepared_feed_timed_loop"
                    ),
                    "completed_work_units_status": "exact_runtime_counter",
                    "performance_benchmark_source": "dx_engine_prepared_feed",
                    "prepared_feed_contract_version": (
                        "deepx-sealed-runtime-input-v3"
                    ),
                    "input_contract": {"input": {"dtype": "uint8", "shape": [2, 2, 3]}},
                },
            },
        },
    )
    ns = SimpleNamespace(
        frames=10,
        warmup=0,
        timeout=30,
        duration_s=60.0,
        setup_id="orin_nx_deepx_m1_01",
        engine_build_python="auto",
        diagnostic_deepx_input_probes=False,
        image_map_data={"resnet50": {"b001": str(expected)}},
    )

    row = module._generic_full_via_suite(
        benchmark_set, "resnet50", "native_full_deepx", "deepx_m1_full", ns,
        prepared_input_manifest=prepared_manifest,
    )

    assert row["ok"] is True
    assert row["status"] == "ok"
    assert row["failure_reason"] == ""
    assert row["prepared_feed_image_binding_ok"] is True
    assert row["prepared_input_binding_verified"] is True
    assert row["prepared_input_binding_status"] == "verified_exact"
    assert row["input_image"] == str(observed.resolve())
    command = captured["cmd"]
    assert isinstance(command, list)
    index = command.index("--prepared-feed-image")
    assert command[index + 1] == str(observed.resolve())
    manifest_index = command.index("--prepared-input-manifest")
    assert command[manifest_index + 1] == str(prepared_manifest.resolve())
    model_index = command.index("--quality-evidence-model-id")
    assert command[model_index + 1] == "resnet50"
    setup_index = command.index("--quality-evidence-setup-id")
    assert command[setup_index + 1] == "orin_nx_deepx_m1_01"
    assert str(expected) not in command


@pytest.mark.parametrize("mutation", ("missing", "tampered"))
def test_deepx_full_invalid_manifest_source_stops_before_child(
    monkeypatch, tmp_path: Path, mutation: str,
) -> None:
    module = _load_runner()
    real_validator = module._validated_runtime_input_manifest
    source_image = tmp_path / "sealed-source.jpg"
    source_image.write_bytes(b"sealed-source-bytes")
    prepared_manifest, _prepared_binding = _sealed_input_fixture(
        monkeypatch, module, tmp_path, source_image=source_image,
    )
    monkeypatch.setattr(
        module, "_validated_runtime_input_manifest", real_validator,
    )
    if mutation == "missing":
        source_image.unlink()
    else:
        source_image.write_bytes(b"tampered-after-manifest-seal")

    child_calls: list[list[str]] = []

    def child_must_not_run(cmd, **_kwargs):
        child_calls.append(list(cmd))
        raise AssertionError("invalid sealed source must stop before child")

    monkeypatch.setattr(
        module, "_suite_python_env",
        lambda _ns, _backend: (sys.executable, {}, []),
    )
    monkeypatch.setattr(module, "_run", child_must_not_run)
    ns = SimpleNamespace(
        frames=10,
        warmup=0,
        timeout=30,
        duration_s=60.0,
        engine_build_python="auto",
        diagnostic_deepx_input_probes=False,
        image_map_data={},
    )

    row = module._generic_full_via_suite(
        tmp_path, "resnet50", "native_full_deepx", "deepx_m1_full", ns,
        prepared_input_manifest=prepared_manifest,
    )

    assert row["ok"] is False
    assert row["status"] == "prepared_input_source_image_unavailable"
    assert row["failure_reason"] == (
        "deepx_prepared_input_source_image_missing_or_mismatched"
    )
    assert row["prepared_input_binding_status"] == (
        "runtime_input_source_image_binding_invalid"
    )
    assert child_calls == []


def test_native_full_remote_runner_and_energy_plan_mirrors_match() -> None:
    for name in ("native_full_baseline_eval_runner.py", "native_producer_energy_plan.py"):
        assert (ROOT / "scripts" / name).read_bytes() == (
            ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts" / name
        ).read_bytes()
