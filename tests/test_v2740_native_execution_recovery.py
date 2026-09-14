from __future__ import annotations

from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_performance_identity import (
    bind_native_split_expected_setups,
    native_performance_identity,
)
from onnx_splitpoint_tool.remote_runtime_closure import (
    native_remote_package_closure,
)
from onnx_splitpoint_tool.workflow.runner import (
    _native_expected_full_rows_v61b,
)


ROOT = Path(__file__).resolve().parents[1]


def test_withdrawn_endpoint_join_files_and_product_hooks_are_absent() -> None:
    forbidden_paths = (
        "onnx_splitpoint_tool/dataset_sample_identity.py",
        "onnx_splitpoint_tool/evidence_binding.py",
    )
    for relative in forbidden_paths:
        assert not (ROOT / relative).exists(), relative

    assert not list((ROOT / "tests").glob("test_p04_*.py"))

    production_roots = (ROOT / "onnx_splitpoint_tool", ROOT / "scripts")
    forbidden_tokens = (
        "dataset_sample_identity",
        "onnx_splitpoint_tool.evidence_binding",
        "central_join_gate",
        "central_join_binding",
        "native_expected_matrix_attestation.json",
    )
    for production_root in production_roots:
        for path in production_root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".txt", ".sh"}:
                continue
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                assert token not in source, f"{token!r} remains in {path}"


def test_shared_remote_closure_is_exactly_the_recovery_runtime_set() -> None:
    closure = native_remote_package_closure()
    paths = [row[0] for row in closure]
    modules = [row[1] for row in closure]

    assert len(closure) == 20
    assert len(paths) == len(set(paths))
    assert len(modules) == len(set(modules))
    assert modules == [
        "onnx_splitpoint_tool.validation.host_postprocess",
        "onnx_splitpoint_tool.validation.accuracy_gates",
        "onnx_splitpoint_tool.preprocessing_contract",
        "onnx_splitpoint_tool.native_split_quality",
        "onnx_splitpoint_tool.native_command_contract",
        "onnx_splitpoint_tool.native_output_endpoint",
        "onnx_splitpoint_tool.runners._types",
        "onnx_splitpoint_tool.resources_utils",
        "onnx_splitpoint_tool.runners.backends.base",
        "onnx_splitpoint_tool.runners.backends.hailo_utils",
        "onnx_splitpoint_tool.cache_verify_policy",
        "onnx_splitpoint_tool.hailo_attempt_receipts",
        "onnx_splitpoint_tool.hailo_timeout_policy",
        "onnx_splitpoint_tool.runners.backends.hailo_backend",
        "onnx_splitpoint_tool.runners.harness.base",
        "onnx_splitpoint_tool.runners.harness.yolo",
        "onnx_splitpoint_tool.runners.native_full_input",
        "onnx_splitpoint_tool.runners.native_split_quality_runtime",
        "onnx_splitpoint_tool.native_detection_postprocess",
        "onnx_splitpoint_tool.hailo_full_contract_promotion",
    ]
    assert "onnx_splitpoint_tool/evidence_binding.py" not in paths
    assert "onnx_splitpoint_tool/dataset_sample_identity.py" not in paths
    assert "onnx_splitpoint_tool.evidence_binding" not in modules
    assert "onnx_splitpoint_tool.dataset_sample_identity" not in modules
    assert "onnx_splitpoint_tool.workflow.benchmark_binding" not in modules
    assert "onnx_splitpoint_tool.management_reference" not in modules
    assert "onnx_splitpoint_tool.native_full_quality" not in modules
    assert "onnx_splitpoint_tool/preprocessing_contract.py" in paths
    assert "onnx_splitpoint_tool.preprocessing_contract" in modules
    assert all((ROOT / relative).is_file() for relative in paths)

    for relative in (
        "onnx_splitpoint_tool/workflow/runner.py",
        "scripts/update_evalset_native_producers.py",
        (
            "onnx_splitpoint_tool/resources/remote_scripts/"
            "update_evalset_native_producers.py"
        ),
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "native_remote_package_closure()" in source
        assert "package_runtime_closure =" not in source


@pytest.mark.parametrize(
    ("models", "cases", "expected_total"),
    (
        (("resnet50", "yolo26s"), ("b001",), 18),
        (("resnet50", "yolo26s", "yolov7_paper"), ("b001",), 27),
        (
            ("resnet50", "yolo26s", "yolov7_paper"),
            ("b001", "b002", "b003"),
            45,
        ),
        (
            ("resnet50", "yolo26s", "yolov7_paper"),
            ("b001", "b002", "b003", "b004", "b005"),
            63,
        ),
    ),
)
def test_split_setup_binding_completes_dynamic_matrix_identity(
    models: tuple[str, ...],
    cases: tuple[str, ...],
    expected_total: int,
) -> None:
    rows = [
        {
            "backend_key": producer,
            "backend": backend,
            "model": model,
            "case": case,
            "execution_mode": "native_split",
        }
        for producer, backend in (
            ("hailo8", "hailo8_to_trt"),
            ("hailo10h", "hailo10h_to_trt"),
            ("deepx", "deepx_to_trt"),
        )
        for model in models
        for case in cases
    ]
    setup_ids = {
        "hailo8": "orin_nx_hailo8_01",
        "hailo10h": "orin_nx_hailo10_01",
        "deepx": "orin_nx_deepx_m1_01",
    }

    bound = bind_native_split_expected_setups(rows, setup_ids)
    full = _native_expected_full_rows_v61b(
        models,
        {
            "hailo8": ("hailo8", "tensorrt"),
            "hailo10h": ("hailo10h", "tensorrt"),
            "deepx": ("deepx", "tensorrt"),
        },
        setup_ids,
    )

    assert len(bound) + len(full) == expected_total
    assert all(native_performance_identity(row) is not None for row in bound)
    assert all(native_performance_identity(row) is not None for row in full)
    assert {row["comparison_backend"] for row in bound} == {
        "hailo8", "hailo10h", "deepx",
    }
    assert all("setup_id" in row for row in bound)


def test_split_setup_binding_rejects_conflicting_identity() -> None:
    row = {
        "backend_key": "hailo8",
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b001",
        "execution_mode": "native_split",
        "setup_id": "wrong-setup",
    }
    with pytest.raises(ValueError, match="setup identity conflicts"):
        bind_native_split_expected_setups(
            [row], {"hailo8": "orin_nx_hailo8_01"},
        )


def test_trt_full_runner_does_not_reintroduce_p04_input_replay() -> None:
    source = (ROOT / "scripts/native_full_baseline_eval_runner.py").read_text(
        encoding="utf-8"
    )
    # The withdrawn join feature added an exact-input replay through
    # --extra-run-arg and caused the ResNet Full rows to fail in argparse
    # before TensorRT started.  That whole path is removed rather than left as
    # a dormant fixed variant.
    assert "--extra-run-arg" not in source
    # The older direct trtexec Energy hotloop legitimately uses loadInputs;
    # it must not be wrapped in the withdrawn outer argparse contract.
    assert "f\"--loadInputs={runtime_input_name}:{runtime_input}\"" in source
