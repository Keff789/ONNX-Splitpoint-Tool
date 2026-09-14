from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import yaml
from onnx import TensorProto, helper

from onnx_splitpoint_tool.deepx import compiler as deepx_compiler
from onnx_splitpoint_tool.deepx import env_status as deepx_env_status
from onnx_splitpoint_tool.energy.config import (
    EnergyDefaults,
    load_hardware_registry,
)
from onnx_splitpoint_tool.gui import benchmark_workflow
from onnx_splitpoint_tool.run_modes import (
    RUN_MODE_SCHEMA_VERSION,
    apply_run_mode,
    default_run_modes_config,
)
from onnx_splitpoint_tool.workflow import legacy_benchmarkset_binding


def _tiny_part1(path: Path) -> Path:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["features"])],
        "deepx_part1",
        [helper.make_tensor_value_info(
            "images", TensorProto.FLOAT, [1, 3, 4, 4],
        )],
        [helper.make_tensor_value_info(
            "features", TensorProto.FLOAT, [1, 3, 4, 4],
        )],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, str(path))
    return path


def _suite(tmp_path: Path, name: str) -> tuple[Path, Path]:
    suite = tmp_path / name
    case = suite / "b001"
    case.mkdir(parents=True)
    part1 = _tiny_part1(case / "part1.onnx")
    (case / "split_manifest.json").write_text(
        json.dumps({"part1_model": part1.name}) + "\n",
        encoding="utf-8",
    )
    return suite, part1


def _build_part1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    suite_name: str,
    classification_preprocessing: str,
    compiled: list[Path],
) -> tuple[dict, Path]:
    monkeypatch.delenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", raising=False)
    suite, source = _suite(tmp_path, suite_name)
    calibration = tmp_path / "calibration"
    calibration.mkdir(exist_ok=True)
    (calibration / "sample.jpg").write_bytes(b"image")
    cache = tmp_path / "cache"

    monkeypatch.setattr(
        deepx_env_status,
        "inspect_deepx_environment",
        lambda **_kwargs: {
            "compiler_ready": True,
            "cache_dir": str(cache),
            "compiler_version": "test-dxcom",
        },
    )

    def fake_compile(*, onnx_path, output_dir, **_kwargs):
        compiled.append(Path(onnx_path))
        build = Path(output_dir)
        build.mkdir(parents=True, exist_ok=True)
        dxnn = build / "model.dxnn"
        dxnn.write_bytes(b"dxnn:" + Path(onnx_path).read_bytes()[:32])
        return SimpleNamespace(
            ok=True,
            dxnn_path=str(dxnn),
            status="ok",
            message="ok",
            log_path="",
        )

    monkeypatch.setattr(deepx_compiler, "compile_dxnn", fake_compile)
    result = benchmark_workflow._materialize_manual_deepx_part1_artifacts(
        out_dir=suite,
        bench_plan_runs=[{
            "id": "deepx_m1_to_tensorrt",
            "type": "matrix",
            "stage1": "deepx_m1",
            "stage2": "tensorrt",
        }],
        validation_images="",
        fallback_calib_dir=str(calibration),
        calibration_num=1,
        task_hint="classification",
        classification_preprocessing=classification_preprocessing,
    )
    return result, source


def test_deepx_part1_imagenet_compiles_adapter_and_records_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compiled: list[Path] = []
    result, source = _build_part1(
        monkeypatch,
        tmp_path,
        suite_name="imagenet_suite",
        classification_preprocessing="imagenet_mean_std",
        compiled=compiled,
    )

    assert result["status"] == "ok"
    assert result["classification_preprocessing"] == "imagenet_mean_std"
    assert len(compiled) == 1
    assert compiled[0] != source
    adapted = onnx.load(str(compiled[0]), load_external_data=False)
    assert [node.op_type for node in adapted.graph.node[:2]] == ["Sub", "Div"]
    assert [node.op_type for node in onnx.load(str(source)).graph.node] == [
        "Identity"
    ]

    case = result["cases"][0]
    assert case["classification_preprocessing"] == "imagenet_mean_std"
    assert case["source_onnx_sha256"] != case["build_onnx_sha256"]
    receipt = (
        tmp_path
        / "imagenet_suite"
        / "b001"
        / case["build_onnx_adapter_receipt"]
    )
    assert receipt.is_file()
    contract = json.loads(
        (
            tmp_path
            / "imagenet_suite"
            / "b001"
            / case["output_contract"]
        ).read_text(encoding="utf-8")
    )
    assert contract["classification_preprocessing"] == "imagenet_mean_std"
    assert contract["input"]["embedded_numeric_path"] == (
        "dxcom_div255_then_build_onnx_imagenet_mean_std"
    )
    assert contract["preprocessing_contract"]["build_onnx_adapter"][
        "kind"
    ] == "imagenet_rgb_mean_std"


def test_deepx_part1_scale_only_keeps_source_and_legacy_cache_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compiled: list[Path] = []
    result, source = _build_part1(
        monkeypatch,
        tmp_path,
        suite_name="scale_suite",
        classification_preprocessing="current_scale_only",
        compiled=compiled,
    )

    assert result["status"] == "ok"
    assert compiled == [source]
    case = result["cases"][0]
    assert case["classification_preprocessing"] == "current_scale_only"
    assert case["source_onnx_sha256"] == case["build_onnx_sha256"]
    assert "classification_preprocessing" not in case["cache_contract"]
    assert case["build_onnx_adapter_receipt"] == ""


def test_formal_profile_passes_deepx_part1_preprocessing() -> None:
    source = inspect.getsource(
        legacy_benchmarkset_binding.materialize_legacy_benchmark_set
    )
    assert "classification_preprocessing=str(" in source
    assert 'deepx_build_cfg.get("classification_preprocessing")' in source


def test_fs_is_the_current_default_and_legacy_registry_is_migrated(
    tmp_path: Path,
) -> None:
    assert EnergyDefaults().physical_scope == "FS"
    registry_path = tmp_path / "hardware_setups.yaml"
    registry_path.write_text(
        yaml.safe_dump({
            "schema": "onnx-splitpoint/hardware-setups",
            "schema_version": 2,
            "energy_defaults": {
                "physical_scope": "MB",
                "window_label": "command",
            },
            "hardware_setups": [],
        }, sort_keys=False),
        encoding="utf-8",
    )

    loaded = load_hardware_registry(registry_path)
    persisted = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert loaded["energy_defaults"]["physical_scope"] == "FS"
    assert persisted["energy_defaults"]["physical_scope"] == "FS"


def test_fs_migration_does_not_relabel_invalid_or_custom_scope(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "hardware_setups.yaml"
    registry_path.write_text(
        yaml.safe_dump({
            "schema": "onnx-splitpoint/hardware-setups",
            "schema_version": 2,
            "energy_defaults": {"physical_scope": " MB"},
            "hardware_setups": [],
        }, sort_keys=False),
        encoding="utf-8",
    )

    loaded = load_hardware_registry(registry_path)
    persisted = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert loaded["energy_defaults"]["physical_scope"] == " MB"
    assert persisted["energy_defaults"]["physical_scope"] == " MB"


def test_run_mode_materializes_explicit_fs_command_scope() -> None:
    config = default_run_modes_config()
    assert RUN_MODE_SCHEMA_VERSION == 13
    profile = {
        "name": "fs_scope_test",
        "model_suite": {
            "primary": [{
                "id": "resnet50", "task": "classification",
            }],
        },
        "run_profiles": [],
        "execution_preset": {
            "id": "standard",
            "follow_tool_config": False,
            "snapshot": config["modes"]["standard"],
            "overrides": {
                "native_enabled": True,
                "energy_enabled": True,
            },
        },
    }
    resolved, _audit = apply_run_mode(profile, config=config)
    assert resolved["energy"]["scope"] == "row_variant"
    assert resolved["energy"]["physical_scope"] == "FS"
    assert resolved["energy"]["window_label"] == "command"
    assert resolved["native_producers"]["energy"]["physical_scope"] == "FS"
    assert resolved["native_producers"]["energy"]["window_label"] == "command"
