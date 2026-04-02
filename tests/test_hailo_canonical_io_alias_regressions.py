from __future__ import annotations

from pathlib import Path


def test_runner_hailo_io_name_map_uses_slot_aliases() -> None:
    from onnx_splitpoint_tool.runners.backends.hailo_backend import _build_hailo_io_name_map

    mapping = _build_hailo_io_name_map(
        ["output_layer1", "output_layer2", "output_layer3"],
        ["cut_small", "cut_medium", "cut_large"],
    )

    assert mapping == {
        "output_layer1": "cut_small",
        "output_layer2": "cut_medium",
        "output_layer3": "cut_large",
    }


def test_runner_hailo_io_name_map_can_match_unique_numel_signatures() -> None:
    from onnx_splitpoint_tool.runners.backends.hailo_backend import _build_hailo_io_name_map

    mapping = _build_hailo_io_name_map(
        ["hailo_a", "hailo_b", "hailo_c"],
        ["head_20", "head_80", "head_40"],
        hailo_shapes={
            "hailo_a": (80, 80, 255),
            "hailo_b": (20, 20, 255),
            "hailo_c": (40, 40, 255),
        },
        onnx_shapes={
            "head_20": (1, 255, 20, 20),
            "head_80": (1, 255, 80, 80),
            "head_40": (1, 255, 40, 40),
        },
    )

    assert mapping == {
        "hailo_a": "head_80",
        "hailo_b": "head_20",
        "hailo_c": "head_40",
    }


def test_template_hailo_sessions_bind_to_source_onnx_io_names() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert "def _build_hailo_io_name_map" in src
    assert "onnx_model_path=p1_path" in src
    assert "onnx_model_path=p2_path" in src
    assert "onnx_model_path=full_path" in src
    assert "self._hef_output_names" in src
    assert "self._output_name_hef_to_canonical" in src


def test_runner_hailo_io_name_map_prefers_manifest_slot_order_for_generic_inputs() -> None:
    from onnx_splitpoint_tool.runners.backends.hailo_backend import _build_hailo_io_name_map

    mapping = _build_hailo_io_name_map(
        ["input_layer1", "input_layer2", "input_layer3", "input_layer4"],
        ["mul_24", "mul_25", "mul_26", "mul_58"],
        hailo_shapes={
            "input_layer1": (40, 40, 128),
            "input_layer2": (40, 40, 128),
            "input_layer3": (20, 20, 256),
            "input_layer4": (20, 20, 256),
        },
        onnx_shapes={
            "mul_24": (1, 128, 40, 40),
            "mul_25": (1, 128, 40, 40),
            "mul_26": (1, 256, 20, 20),
            "mul_58": (1, 256, 20, 20),
        },
        slot_names=["mul_58", "mul_25", "mul_24", "mul_26"],
    )

    assert mapping == {
        "input_layer1": "mul_58",
        "input_layer2": "mul_25",
        "input_layer3": "mul_24",
        "input_layer4": "mul_26",
    }


def test_template_hailo_part2_session_uses_part2_onnx_input_order_for_slot_aliasing() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert "canonical_input_slot_names=part2_cut_names_manifest" not in src
    assert "splitbench shows that Hailo part2 input_layerN" in src
    assert 'print(f"[hailo][io] part2' in src


def test_runner_hailo_part2_slot_aliasing_should_follow_onnx_input_order_for_current_build_pipeline() -> None:
    from onnx_splitpoint_tool.runners.backends.hailo_backend import _build_hailo_io_name_map

    onnx_input_order = ["mul_24", "mul_25", "mul_27", "mul_58"]
    manifest_cut_order = ["mul_27", "mul_25", "mul_24", "mul_58"]

    mapping = _build_hailo_io_name_map(
        ["input_layer1", "input_layer2", "input_layer3", "input_layer4"],
        onnx_input_order,
        slot_names=None,
    )

    assert mapping == {
        "input_layer1": "mul_24",
        "input_layer2": "mul_25",
        "input_layer3": "mul_27",
        "input_layer4": "mul_58",
    }

    wrong_if_manifest_order_is_used = _build_hailo_io_name_map(
        ["input_layer1", "input_layer2", "input_layer3", "input_layer4"],
        onnx_input_order,
        slot_names=manifest_cut_order,
    )

    assert wrong_if_manifest_order_is_used["input_layer1"] == "mul_27"
    assert wrong_if_manifest_order_is_used["input_layer3"] == "mul_24"
