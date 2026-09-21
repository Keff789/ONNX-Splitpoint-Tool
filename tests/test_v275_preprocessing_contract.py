from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from onnx_splitpoint_tool import hailo_backend
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    prepare_rgb_uint8_image,
    preprocessing_contract_sha256,
    resolve_image_preprocessing_contract,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"


def _runner_module():
    name = "_osp_v275_runner_contract_test"
    loader = importlib.machinery.SourceFileLoader(name, str(RUNNER))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def test_detection_and_classification_contracts_are_explicit_and_stable():
    detection = canonical_image_preprocessing_contract("detection", (640, 640))
    classification = canonical_image_preprocessing_contract("classification", (224, 224))

    assert detection["spatial_transform"] == "centered_letterbox"
    assert detection["color_space"] == "RGB"
    assert detection["pad_value"] == 114
    assert detection["schema_version"] == 2
    assert classification["spatial_transform"] == "direct_resize"
    assert classification["pad_value"] == 0
    assert preprocessing_contract_sha256(detection) == preprocessing_contract_sha256(
        dict(reversed(list(detection.items())))
    )


def test_hailo_320x320_requires_explicit_task_and_never_uses_size_heuristic(
    tmp_path: Path, monkeypatch
):
    for name in (
        "ONNX_SPLITPOINT_HAILO_CALIB_TASK",
        "SPLITPOINT_HAILO_CALIB_TASK",
        "ONNX_SPLITPOINT_HAILO_PREPROCESSING_CONTRACT_JSON",
        "SPLITPOINT_HAILO_PREPROCESSING_CONTRACT_JSON",
    ):
        monkeypatch.delenv(name, raising=False)

    model = tmp_path / "opaque_320.onnx"
    with pytest.raises(ValueError, match="input dimensions must not infer semantics"):
        hailo_backend._resolve_hailo_image_contract(
            model_path=model,
            activation_part1=None,
            net_input_shapes=[1, 3, 320, 320],
        )

    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED", "0")
    monkeypatch.setattr(
        hailo_backend,
        "hailo_build_hef_via_venv",
        lambda *_args, **_kwargs: pytest.fail("compiler bridge must not run"),
    )
    refused = hailo_backend.hailo_build_hef_auto(
        model,
        outdir=tmp_path,
        backend="venv",
        net_input_shapes=[1, 3, 320, 320],
    )
    assert refused.ok is False
    assert refused.failure_kind == "invalid_preprocessing_contract"
    assert refused.last_stage == "preprocessing_contract_preflight"
    assert "input dimensions must not infer semantics" in str(refused.error)

    detection, _ = hailo_backend._resolve_hailo_image_contract(
        model_path=model,
        activation_part1=None,
        net_input_shapes=[1, 3, 320, 320],
        task="detection",
    )
    classification, _ = hailo_backend._resolve_hailo_image_contract(
        model_path=model,
        activation_part1=None,
        net_input_shapes=[1, 3, 320, 320],
        task="classification",
    )
    assert detection["preprocess_mode"] == "letterbox"
    assert detection["pad_value"] == 114
    assert classification["preprocess_mode"] == "resize"

    declared = canonical_image_preprocessing_contract("detection", (320, 320))
    from_contract, _ = hailo_backend._resolve_hailo_image_contract(
        model_path=model,
        activation_part1=None,
        net_input_shapes=[1, 3, 320, 320],
        declared=declared,
    )
    assert from_contract == declared


def test_full_hailo_probe_forwards_explicit_task(tmp_path: Path, monkeypatch):
    from onnx_splitpoint_tool.benchmark.model_preparation import (
        PreparationRuntimeOptions,
        _probe_full_hailo,
    )

    captured = {}

    def _fake_build(model_path, **kwargs):
        captured["model_path"] = Path(model_path)
        captured.update(kwargs)
        return SimpleNamespace(
            ok=True,
            error=None,
            result_json_path=str(tmp_path / "result.json"),
            hef_path=str(tmp_path / "compiled.hef"),
        )

    monkeypatch.setattr(hailo_backend, "hailo_build_hef_auto", _fake_build)
    model = tmp_path / "detector.onnx"
    ok, error, *_ = _probe_full_hailo(
        model,
        "current",
        runtime=PreparationRuntimeOptions(),
        screening_dir=tmp_path / "screening",
        task="detection",
    )
    assert ok is True
    assert error is None
    assert captured["model_path"] == model
    assert captured["task"] == "detection"


def test_every_production_hailo_build_call_site_supplies_task_keyword():
    expected = {
        ROOT / "onnx_splitpoint_tool/benchmark/model_preparation.py": {
            "hailo_build_hef_auto": 1
        },
        ROOT / "onnx_splitpoint_tool/benchmark/services.py": {
            "hailo_build_hef_fn": 5
        },
        ROOT / "onnx_splitpoint_tool/gui_app.py": {"hailo_build_hef_auto": 3},
    }
    for path, names in expected.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = {name: [] for name in names}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            if name in found:
                found[name].append(node)
        for name, expected_count in names.items():
            calls = found[name]
            assert len(calls) == expected_count, (path, name, len(calls))
            assert all(any(kw.arg == "task" for kw in call.keywords) for call in calls)

    for relative in (
        "onnx_splitpoint_tool/gui/benchmark_workflow.py",
        "onnx_splitpoint_tool/workflow/legacy_benchmarkset_binding.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        execution_configs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "BenchmarkGenerationExecutionConfig"
        ]
        assert execution_configs
        assert all(
            any(kw.arg == "benchmark_task" for kw in call.keywords)
            for call in execution_configs
        )


def test_declared_detection_resize_or_wrong_padding_fails_closed():
    expected = canonical_image_preprocessing_contract("detection", (640, 640))
    for field, value in (("preprocess_mode", "resize"), ("pad_value", 0)):
        bad = dict(expected)
        bad[field] = value
        with pytest.raises(ValueError, match="contract mismatch"):
            resolve_image_preprocessing_contract(
                task="detection", target_hw=(640, 640), declared=bad
            )


def test_centered_letterbox_has_exact_odd_padding_geometry_and_bytes():
    # 640x483 -> 640x483 resized content and 157 vertical padding pixels.
    source = np.zeros((483, 640, 3), dtype=np.uint8)
    source[..., 0] = 17
    contract = canonical_image_preprocessing_contract("detection", (640, 640))
    prepared, geometry = prepare_rgb_uint8_image(source, contract)

    assert prepared.shape == (640, 640, 3)
    assert geometry["resized_hw"] == [483, 640]
    assert geometry["pad_ltrb"] == [0, 78, 0, 79]
    assert np.all(prepared[:78] == 114)
    assert np.all(prepared[-79:] == 114)
    assert np.all(prepared[78 : 78 + 483, :, 0] == 17)


def test_hailo_full_image_calibration_uses_same_contract(tmp_path: Path):
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[..., 1] = 200
    Image.fromarray(image, mode="RGB").save(tmp_path / "sample.png")
    contract = canonical_image_preprocessing_contract("detection", (64, 64))

    dataset = hailo_backend._try_build_calib_from_dir(
        calib_dir=tmp_path,
        expected_shape=[64, 64, 3],
        limit=1,
        preprocess="norm",
        preprocessing_contract=contract,
    )
    assert dataset is not None
    assert dataset.shape == (1, 64, 64, 3)
    # 20x40 becomes 32x64, centered with 16-pixel top/bottom padding.
    assert np.allclose(dataset[0, :16], 114.0 / 255.0)
    assert np.allclose(dataset[0, -16:], 114.0 / 255.0)
    assert np.allclose(dataset[0, 16:48, :, 1], 200.0 / 255.0)


def test_activation_part1_preparation_matches_shared_prepared_pixels():
    class Input:
        shape = [1, 3, 64, 64]
        type = "tensor(float)"

    source = np.zeros((20, 40, 3), dtype=np.uint8)
    source[..., 2] = 99
    contract = canonical_image_preprocessing_contract("detection", (64, 64))
    prepared, _ = prepare_rgb_uint8_image(source, contract)
    actual = hailo_backend._prepare_part1_input_for_activation_calib(
        source, Input(), preprocess="norm", preprocessing_contract=contract
    )
    expected = np.transpose(prepared.astype(np.float32) / 255.0, (2, 0, 1))[None]
    assert np.array_equal(actual, expected)


def test_hailo_cache_key_v2_includes_preprocessing_contract(tmp_path: Path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    detection = canonical_image_preprocessing_contract("detection", (640, 640))
    classification = canonical_image_preprocessing_contract("classification", (640, 640))

    def key(contract):
        return hailo_backend._hailo_cache_key(
            model_path=model,
            activation_part1=None,
            hw_arch="hailo10h",
            opt_level=1,
            calib_dir=None,
            calib_count=64,
            calib_batch_size=8,
            extra_model_script="",
            start_nodes=None,
            end_nodes=None,
            preprocessing_contract=contract,
        )

    detection_key, payload = key(detection)
    classification_key, _ = key(classification)
    assert payload["schema"].endswith("cache-key-v2")
    assert payload["preprocessing_contract_sha256"] == preprocessing_contract_sha256(
        detection
    )
    assert detection_key != classification_key


def test_bare_or_stale_hef_is_not_reusable_without_matching_receipt(tmp_path: Path):
    source = tmp_path / "source.onnx"
    compiler = tmp_path / "compiler.onnx"
    hef = tmp_path / "compiled.hef"
    source.write_bytes(b"source")
    compiler.write_bytes(b"compiler")
    hef.write_bytes(b"hef")
    contract = canonical_image_preprocessing_contract("detection", (640, 640))
    contract_sha = preprocessing_contract_sha256(contract)
    cache_key, cache_payload = hailo_backend._hailo_cache_key(
        model_path=compiler,
        activation_part1=None,
        hw_arch="hailo10h",
        opt_level=1,
        calib_dir=None,
        calib_count=1,
        calib_batch_size=1,
        extra_model_script="",
        start_nodes=None,
        end_nodes=None,
        preprocessing_contract=contract,
    )

    assert hailo_backend._load_valid_hailo_receipt(hef) is None
    hailo_backend._write_hailo_receipt(
        hef_path=hef,
        source_onnx=source,
        compiler_onnx=compiler,
        hw_arch="hailo10h",
        net_name="yolo",
        preprocessing_contract=contract,
        preprocessing_sha256=contract_sha,
        cache_key=cache_key,
        cache_payload=cache_payload,
        calibration_identity=str(cache_payload["calibration_identity"]),
        calibration_count=int(cache_payload["calibration_count"]),
    )
    assert hailo_backend._load_valid_hailo_receipt(
        hef,
        preprocessing_sha256=contract_sha,
        cache_key=cache_key,
        cache_payload=cache_payload,
    ) is not None
    assert hailo_backend._load_valid_hailo_receipt(
        hef, preprocessing_sha256="0" * 64
    ) is None


def test_runner_bn6_geometry_is_task_driven_and_yolov7_heads_are_attested():
    module = _runner_module()
    if type(module.ort).__name__ == "_MissingOnnxRuntime":
        with pytest.raises(
            ModuleNotFoundError,
            match="onnxruntime is required to execute the generated runner",
        ):
            module.ort.InferenceSession
    contract = module._canonical_image_preprocessing_contract(
        "detection", (640, 640)
    )
    assert contract["preprocess_mode"] == "letterbox"
    assert contract["pad_value"] == 114

    heads = [
        np.zeros((1, 3, 80, 80, 85), dtype=np.float32),
        np.zeros((1, 3, 40, 40, 85), dtype=np.float32),
        np.zeros((1, 3, 20, 20, 85), dtype=np.float32),
    ]
    attestation = module._attest_detection_full_endpoint(
        ["head_s", "head_m", "head_l"], heads, declared_mode=""
    )
    assert attestation["detected_output_format"] == "multiscale_head"
    assert attestation["effective_raw_detection_head"] is True
    source = RUNNER.read_text(encoding="utf-8")
    assert "val_letterbox = output_format in" not in source
    assert "validation_preprocessing_contract[\"preprocess_mode\"]" in source
    assert source.index(
        "_hailo_full_untimed_probe_outputs = hailo_full.infer(full_inputs_hailo)"
    ) < source.index("# ----------------------------\n    # Timings")
    assert "hailo_full_raw_detection_head = bool(" in source
    assert '"full_rawhead_plus_frozen_decode_nms"' in source


def test_central_quality_and_hailo_preparation_have_identical_rgb_bytes(
    tmp_path: Path,
):
    module = _runner_module()
    yy, xx = np.indices((37, 53))
    source = np.stack(
        [
            (xx * 7 + yy * 3) % 256,
            (xx * 11 + yy * 5 + 17) % 256,
            (xx * 13 + yy * 19 + 29) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    image_path = tmp_path / "asymmetric.png"
    Image.fromarray(source, mode="RGB").save(image_path)
    contract = canonical_image_preprocessing_contract("detection", (63, 79))

    hailo_rgb, _ = prepare_rgb_uint8_image(source, contract)
    central_nchw = module._load_image_as_nchw(
        image_path,
        target_hw=(63, 79),
        dtype=np.dtype(np.float32),
        scale="norm",
        letterbox=True,
        preprocessing_contract=contract,
    )
    assert central_nchw is not None
    expected_nchw = np.transpose(
        hailo_rgb.astype(np.float32) / 255.0, (2, 0, 1)
    )[None]
    assert np.array_equal(central_nchw, expected_nchw)
    central_rgb = np.rint(
        np.transpose(central_nchw[0], (1, 2, 0)) * 255.0
    ).astype(np.uint8)
    assert hashlib.sha256(central_rgb.tobytes()).hexdigest() == hashlib.sha256(
        hailo_rgb.tobytes()
    ).hexdigest()
    assert np.array_equal(central_rgb, hailo_rgb)
