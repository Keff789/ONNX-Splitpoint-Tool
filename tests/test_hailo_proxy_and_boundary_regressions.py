from __future__ import annotations

from pathlib import Path


def test_template_uses_proxy_for_hailo_detection_outputs() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert 'provider_l.startswith("hailo")' in src
    assert 'output_format == "multiscale_head" and accel_proxy' in src
    assert 'mode = "proxy_detections"' in src


def test_template_no_longer_soft_passes_proxy_failures_as_off() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert 'proxy_validation_decode_failed' in src
    assert 'proxy_validation_skipped_fallback=off' not in src


def test_template_uses_manifest_cut_name_mapping_for_stage2_inputs() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert 'part1_cut_names_manifest' in src
    assert 'part2_cut_names_manifest' in src
    assert 'manifest_cut_name_map_p2_to_p1' in src
    assert 'used_stage1_sources' in src


def test_runner_interface_transfer_exports_legacy_default_transfer() -> None:
    from onnx_splitpoint_tool.runners.interface_transfer import DefaultTransfer, TransferMeta

    assert DefaultTransfer is not None
    assert TransferMeta is not None


def test_template_cpu_stage2_completion_uses_unique_stage1_sources() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert '_build_stage2_inputs_cpu' in src
    assert 'used_stage1_sources: set[str] = set()' in src
    assert '_overlay_boundary_slot_mapping(' in src


def test_template_auto_validation_uses_proxy_for_cuda_detection_outputs() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert 'provider_l in ("cuda", "gpu", "tensorrt", "trt")' in src
