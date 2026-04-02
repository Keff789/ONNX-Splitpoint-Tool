from __future__ import annotations

import ast
import py_compile
from pathlib import Path


ROOT = Path("onnx_splitpoint_tool")


def _class_assign_names(src: str, class_name: str) -> set[str]:
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            names: set[str] = set()
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    names.add(item.target.id)
            return names
    raise AssertionError(f"class {class_name!r} not found")



def test_hailo_hef_result_dataclass_has_timeout_and_failure_metadata() -> None:
    src = (ROOT / "hailo_backend.py").read_text(encoding="utf-8")
    names = _class_assign_names(src, "HailoHefBuildResult")
    for field_name in {
        "returncode",
        "debug_log",
        "timed_out",
        "timeout_kind",
        "last_stage",
        "failure_kind",
        "unsupported_reason",
        "details",
    }:
        assert field_name in names
    assert "def _make_hef_result" in src
    assert "extra_fields" in src



def test_hailo_backend_source_contains_activation_preflight_and_shared_mapping() -> None:
    src = (ROOT / "hailo_backend.py").read_text(encoding="utf-8")
    assert "def _map_part2_inputs_to_part1_outputs" in src
    assert "def _activation_calib_preflight" in src
    assert "def _format_activation_calib_preflight_error" in src
    assert "Unsupported Hailo Part2 activation-calibration splitpoint" in src
    assert "failure_kind='unsupported_splitpoint'" in src
    assert "unsupported_reason='activation_preflight_missing_inputs'" in src



def test_hailo_backend_source_uses_shared_timeout_policy_and_streamed_subprocess_runner() -> None:
    src = (ROOT / "hailo_backend.py").read_text(encoding="utf-8")
    assert "def _resolve_hef_timeout_policy" in src
    assert "requested == 3600" in src
    assert "def _run_streamed_subprocess" in src
    assert "def _kill_process_tree" in src
    assert "def _hailo_stage_from_line" in src
    assert "run = _run_streamed_subprocess(" in src
    assert "failure_kind='timeout'" in src
    assert "Last active stage:" in src



def test_hailo_backend_python_compiles_after_runtime_patch() -> None:
    py_compile.compile(str(ROOT / "hailo_backend.py"), doraise=True)
    py_compile.compile(str(ROOT / "wsl_inline_build_hef"), doraise=True)



def test_gui_source_uses_shared_hef_timeout_helper() -> None:
    src = (ROOT / "gui_app.py").read_text(encoding="utf-8")
    assert "def _hailo_hef_timeout_seconds" in src
    assert "return 10800" in src
    assert "wsl_timeout_s=3600" not in src
    assert '"timeout_s": int(hef_timeout_s)' in src
