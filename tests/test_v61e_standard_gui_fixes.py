from __future__ import annotations

import ast
from pathlib import Path

from onnx_splitpoint_tool import (
    __development_lineage__,
    __next_major_version__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.cli import main as cli_main
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan, execution_plan_text
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


def _profile(cases: int, *, ranking: bool = True) -> dict:
    return {
        "model_suite": {"primary": [{"id": "resnet50", "enabled": True, "task": "classification"}]},
        "selection_policy": {"max_accepted_cases_per_model": cases, "preferred_shortlist": max(3, cases)},
        "run_profiles": [
            {"id": "hailo8", "enabled": True},
            {"id": "hailo8_to_tensorrt", "enabled": True},
        ],
        "execution_preset": {
            "id": "standard",
            "label": "Standard",
            "snapshot": {
                "defaults": {"native_enabled": True, "energy_enabled": False},
                "quality": {"bootstrap_repetitions": 500},
                "ranking": {"enabled": ranking, "minimum_candidates_for_correlation": 3},
                "runtime": {"native": {"full_baselines": True}, "benchmark": {"warmup": 3, "runs": 5}},
                "data": {"validation_items": {"classification": 500, "detection": 500}},
                "build": {"hailo": {"preset": "balanced", "optimization_level": 1}},
            },
            "overrides": {},
        },
    }


def test_current_version_and_workflow() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __next_major_version__ == "3.0.0"
    assert WORKFLOW_VERSION == (
        "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    )


def test_cli_reports_public_and_package_version(capsys) -> None:
    assert cli_main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "2.75.47" in out
    assert "v2.75.47" in out
    assert "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent" in out


def test_native_console_help_tracks_current_release() -> None:
    source = Path("scripts/native_console_smoke.py").read_text(encoding="utf-8")
    assert 'help="Run the narrow 2.75.47 source regression block."' in source


def test_tool_config_builder_has_no_free_self_reference() -> None:
    path = Path("onnx_splitpoint_tool/gui/panels/panel_hardware.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    build = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_panel")
    assert not [node for node in ast.walk(build) if isinstance(node, ast.Name) and node.id == "self"]
    text = path.read_text(encoding="utf-8")
    assert "artifact_library_tab = ArtifactLibraryPanel(categories)" in text
    assert "app._artifact_library_tab = artifact_library_tab" in text


def test_lazy_panel_keeps_retryable_placeholder_on_error() -> None:
    text = Path("onnx_splitpoint_tool/gui/app.py").read_text(encoding="utf-8")
    assert "without ever removing its visible tab on failure" in text
    assert "Der Tab bleibt verfügbar" in text
    assert "Erneut laden" in text
    assert "if frame is not None and str(frame) in self.main_notebook.tabs()" in text


def test_standard_run_with_one_case_warns_that_ranking_is_not_identifiable() -> None:
    plan = build_effective_execution_plan(_profile(1))
    assert plan["ranking_enabled"] is True
    assert plan["ranking_minimum_candidates"] == 3
    assert plan["ranking_candidate_shortfall"] == 2
    assert [w["id"] for w in plan["warnings"]] == ["ranking_candidate_shortfall"]
    rendered = execution_plan_text(plan)
    assert "insufficient_candidates" in rendered
    assert "at least 3 measured candidates" in rendered


def test_three_cases_remove_ranking_shortfall_warning() -> None:
    plan = build_effective_execution_plan(_profile(3))
    assert plan["ranking_candidate_shortfall"] == 0
    assert not plan["warnings"]


def test_detection_bootstrap_logs_standard_progress_and_eta() -> None:
    text = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert "[quality-gate][bootstrap] START" in text
    assert "[quality-gate][bootstrap] CACHE_READY" in text
    assert "[quality-gate][bootstrap] PROGRESS" in text
    assert "eta=" in text
    assert "[quality-gate][bootstrap] END" in text
    assert "if reps >= 1000" not in text
