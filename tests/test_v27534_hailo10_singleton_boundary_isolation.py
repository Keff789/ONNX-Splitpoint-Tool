from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.native_split_quality import (
    resolve_native_boundary_layout,
)


def _native_split_selection(
    *, applicable: bool, split_backends: list[str], source: str,
) -> dict:
    return {
        "schema": "onnx-splitpoint/native-split-quality-selection",
        "schema_version": 1,
        "applicable": applicable,
        "split_backends": split_backends,
        "split_selection_source": source,
    }


def _load_generated_suite(tmp_path: Path, name: str):
    suite_dir = tmp_path / name
    script_path = Path(write_benchmark_suite_script(suite_dir))
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    return suite, suite_dir


def _quality_args(tmp_path: Path, *, model_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        energy_measurement_only=False,
        quality_evidence_eval_id="eval-1",
        quality_evidence_setup_id="hailo8_setup",
        quality_evidence_model_id=model_id,
        trt_cache_root=str((tmp_path / "trt-cache").resolve()),
        native_trt_workspace_mb=1024,
        native_trt_build_timeout_s=60,
    )


@pytest.mark.parametrize(
    "runtime_shape",
    ([2048], [1, 2048], [2048, 1], [2048, 1, 1], [1, 2048, 1, 1]),
)
def test_hailo_channel_vector_singleton_squeeze_is_identity(
    runtime_shape: list[int],
) -> None:
    canonical_shape = [1, 2048, 1, 1]
    assert resolve_native_boundary_layout(
        runtime_shape, canonical_shape,
    ) == "as_input"
    source = np.arange(2048, dtype=np.float32).reshape(runtime_shape)
    canonical = source.reshape(canonical_shape)
    np.testing.assert_array_equal(canonical.reshape(-1), source.reshape(-1))


def test_singleton_squeeze_rule_does_not_flatten_spatial_tensor() -> None:
    with pytest.raises(ValueError, match="boundary_layout_shape_invalid"):
        resolve_native_boundary_layout([401408], [1, 2048, 7, 7])
    with pytest.raises(ValueError, match="boundary_layout_shape_invalid"):
        resolve_native_boundary_layout([1, 2048, 1, 1, 1], [1, 2048, 1, 1])
    with pytest.raises(ValueError, match="boundary_layout_unresolved"):
        resolve_native_boundary_layout([28, 512, 28], [1, 512, 28, 28])
    with pytest.raises(ValueError, match="boundary_layout_ambiguous"):
        resolve_native_boundary_layout([32, 32, 32], [1, 32, 32, 32])


def test_generated_suite_isolates_layout_failure_without_generic_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_dir = tmp_path / "generated-suite"
    script_path = Path(write_benchmark_suite_script(suite_dir))
    spec = importlib.util.spec_from_file_location(
        "benchmark_suite_v27534_isolation_test", script_path,
    )
    assert spec is not None and spec.loader is not None
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)

    package = ModuleType("splitpoint_runners")
    package.__path__ = []  # type: ignore[attr-defined]
    runtime_module = ModuleType(
        "splitpoint_runners.native_split_quality_runtime"
    )

    def _layout_failure(**_kwargs):
        raise ValueError(
            "native_split_quality_boundary_layout_shape_invalid:"
            "[2048]->[1, 2048, 1, 1]"
        )

    runtime_module.prepare_native_split_quality_binding = _layout_failure  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "splitpoint_runners", package)
    monkeypatch.setitem(
        sys.modules,
        "splitpoint_runners.native_split_quality_runtime",
        runtime_module,
    )

    case_dir = suite_dir / "b119"
    case_dir.mkdir()
    args = SimpleNamespace(
        energy_measurement_only=False,
        quality_evidence_eval_id="eval-1",
        quality_evidence_setup_id="hailo10h_setup",
        quality_evidence_model_id="resnet50",
        trt_cache_root=str((tmp_path / "trt-cache").resolve()),
        native_trt_workspace_mb=1024,
        native_trt_build_timeout_s=60,
    )

    preparation = suite._prepare_native_split_quality_for_case(
        root=suite_dir,
        case_dir=case_dir,
        bench={"model_id": "resnet50"},
        plan={},
        run={},
        args=args,
        run_id="hailo10_to_tensorrt",
        stage1="hailo10",
        stage2="tensorrt",
        variants=["composed"],
    )
    assert preparation["status"] == "technical_failure"
    failure = preparation["failure"]
    assert failure["suite_execution_continues"] is True
    assert failure["generic_execution_continues"] is False
    assert failure["claim_eligible"] is False

    failure_path = Path(preparation["failure_path"])
    assert json.loads(failure_path.read_text(encoding="utf-8")) == failure

    # There is deliberately no generated per-case runner.  The typed failure
    # must become an explicit no-claim row before any runner/fallback is used.
    row = suite._run_case(
        case_dir,
        run_id="hailo10_to_tensorrt",
        provider="hailo10_to_trt",
        image="",
        preset="balanced",
        image_scale="0_1",
        warmup=1,
        runs=1,
        timeout_s=1,
        stage1="hailo10",
        stage2="tensorrt",
        variants=["composed"],
        native_split_quality_preparation_failure=failure,
    )
    assert row["error_class"] == "native_split_quality_binding_materialization_failed"
    assert row["runtime_ok"] is False
    assert row["claim_eligible"] is False
    assert row["performance_eligible"] is False
    assert row["native_split_quality_preparation_failure"] == failure

    template = script_path.read_text(encoding="utf-8")
    assert template.count("native_split_quality_preparation_failure=(") == 3
    assert template.count(
        '"reason": "native_split_quality_binding_materialization_failed"'
    ) == 3


@pytest.mark.parametrize(
    "selection",
    (
        _native_split_selection(
            applicable=False,
            split_backends=[],
            source="native_disabled",
        ),
        _native_split_selection(
            applicable=True,
            split_backends=["deepx"],
            source="evaluation_profile.run_profiles",
        ),
    ),
)
def test_generic_hailo_split_does_not_enter_unselected_native_quality_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: dict,
) -> None:
    suite, suite_dir = _load_generated_suite(
        tmp_path, "benchmark_suite_native_unselected_test",
    )
    package = ModuleType("splitpoint_runners")
    package.__path__ = []  # type: ignore[attr-defined]
    runtime_module = ModuleType(
        "splitpoint_runners.native_split_quality_runtime"
    )
    calls: list[dict] = []

    def forbidden_binding(**kwargs):
        calls.append(dict(kwargs))
        raise AssertionError("unselected Native Quality producer was called")

    runtime_module.prepare_native_split_quality_binding = forbidden_binding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "splitpoint_runners", package)
    monkeypatch.setitem(
        sys.modules,
        "splitpoint_runners.native_split_quality_runtime",
        runtime_module,
    )
    case_dir = suite_dir / "b135"
    case_dir.mkdir()

    result = suite._prepare_native_split_quality_for_case(
        root=suite_dir,
        case_dir=case_dir,
        bench={"model_id": "mobilenet_v3_large"},
        plan={"native_split_quality_selection": selection},
        run={},
        args=_quality_args(tmp_path, model_id="mobilenet_v3_large"),
        run_id="hailo8_to_trt",
        stage1="hailo8",
        stage2="tensorrt",
        variants=["composed"],
    )

    assert result is None
    assert calls == []
    assert not (case_dir / "results_hailo8_to_trt").exists()


@pytest.mark.parametrize(
    "plan",
    (
        {
            "native_split_quality_selection": _native_split_selection(
                applicable=True,
                split_backends=["hailo8"],
                source="evaluation_profile.run_profiles",
            ),
        },
        {},
    ),
)
def test_selected_and_legacy_hailo_split_keep_native_quality_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan: dict,
) -> None:
    suite, suite_dir = _load_generated_suite(
        tmp_path, "benchmark_suite_native_selected_test",
    )
    package = ModuleType("splitpoint_runners")
    package.__path__ = []  # type: ignore[attr-defined]
    runtime_module = ModuleType(
        "splitpoint_runners.native_split_quality_runtime"
    )
    calls: list[dict] = []
    prepared = {
        "precision": "float32_layout_fp16",
        "binding": {"binding_sha256": "a" * 64},
        "binding_path": str(tmp_path / "binding.json"),
    }

    def selected_binding(**kwargs):
        calls.append(dict(kwargs))
        return prepared

    runtime_module.prepare_native_split_quality_binding = selected_binding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "splitpoint_runners", package)
    monkeypatch.setitem(
        sys.modules,
        "splitpoint_runners.native_split_quality_runtime",
        runtime_module,
    )
    case_dir = suite_dir / "b038"
    case_dir.mkdir()

    result = suite._prepare_native_split_quality_for_case(
        root=suite_dir,
        case_dir=case_dir,
        bench={"model_id": "resnet50"},
        plan=plan,
        run={},
        args=_quality_args(tmp_path, model_id="resnet50"),
        run_id="hailo8_to_trt",
        stage1="hailo8",
        stage2="tensorrt",
        variants=["composed"],
    )

    assert result is prepared
    assert len(calls) == 1
    assert calls[0]["backend"] == "hailo8"
    assert calls[0]["model_id"] == "resnet50"


def test_explicit_malformed_native_split_quality_selection_fails_closed(
    tmp_path: Path,
) -> None:
    suite, suite_dir = _load_generated_suite(
        tmp_path, "benchmark_suite_native_selection_invalid_test",
    )
    case_dir = suite_dir / "b038"
    case_dir.mkdir()
    malformed = _native_split_selection(
        applicable=False,
        split_backends=["hailo8"],
        source="native_disabled",
    )

    with pytest.raises(
        RuntimeError, match="^native_split_quality_selection_invalid$",
    ):
        suite._prepare_native_split_quality_for_case(
            root=suite_dir,
            case_dir=case_dir,
            bench={"model_id": "resnet50"},
            plan={"native_split_quality_selection": malformed},
            run={},
            args=_quality_args(tmp_path, model_id="resnet50"),
            run_id="hailo8_to_trt",
            stage1="hailo8",
            stage2="tensorrt",
            variants=["composed"],
        )
