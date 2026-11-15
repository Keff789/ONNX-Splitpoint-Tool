"""Focused regressions for the normal GUI's evaluation-profile defaults."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool.run_modes import (
    load_run_modes_config,
    save_run_modes_config,
)


ROOT = Path(__file__).resolve().parents[1]
SHIPPED_RUN_MODES = ROOT / "onnx_splitpoint_tool/resources/run_modes/default_run_modes.yaml"


def test_shipped_run_modes_yaml_is_complete_and_loadable(tmp_path: Path) -> None:
    """Every mode's statistics engine belongs inside its quality mapping.

    A misindented engine key made the distributed YAML fail to parse at the
    following statistics key, preventing profiles from displaying their run
    summary when this file was used as the Tool Config source.
    """
    shipped = yaml.safe_load(SHIPPED_RUN_MODES.read_text(encoding="utf-8"))
    assert set(shipped["modes"]) == {"smoke", "standard", "final"}
    for mode in shipped["modes"].values():
        assert mode["quality"]["statistics_engine"] == "legacy"
        assert "statistics_engine" not in mode
        assert mode["quality"]["statistics_block_repetitions"] == 256

    copied_path = tmp_path / "Tool Config with spaces" / "run_modes.yaml"
    copied_path.parent.mkdir()
    copied_path.write_text(SHIPPED_RUN_MODES.read_text(encoding="utf-8"), encoding="utf-8")
    loaded = load_run_modes_config(copied_path)
    assert set(loaded["modes"]) == {"smoke", "standard", "final"}
    for mode_id in shipped["modes"]:
        assert loaded["modes"][mode_id]["quality"]["statistics_engine"] == "legacy"


@pytest.mark.parametrize("mode_id", ["smoke", "standard", "final"])
def test_fresh_tool_config_persists_and_reloads_each_mode(tmp_path: Path, mode_id: str) -> None:
    """Use the same ensure/load/save path as an empty GUI installation."""
    path = tmp_path / "fresh config" / "run_modes.yaml"
    loaded = load_run_modes_config(path)
    assert path.is_file()
    loaded["default_mode"] = mode_id
    loaded["modes"][mode_id]["description"] = f"GUI acceptance {mode_id}"
    save_run_modes_config(path, loaded)
    reloaded = load_run_modes_config(path)
    assert reloaded["default_mode"] == mode_id
    assert reloaded["modes"][mode_id]["description"] == f"GUI acceptance {mode_id}"
    assert reloaded["modes"][mode_id]["quality"]["statistics_engine"] == "legacy"
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["default_mode"] == mode_id


def _gui_module_for_callback_tests(monkeypatch):
    import importlib
    import matplotlib

    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    return importlib.import_module("onnx_splitpoint_tool.gui.app")


@pytest.mark.parametrize("action", ["_evaluation_workflow_open_results_folder", "_evaluation_workflow_prepare_debug_pack"])
def test_workflow_actions_reject_existing_parent_without_run(tmp_path: Path, monkeypatch, action: str) -> None:
    """An existing Results parent is not an identified EvaluationRun."""
    from types import SimpleNamespace
    from onnx_splitpoint_tool.workflow import debug_pack

    module = _gui_module_for_callback_tests(monkeypatch)
    parent = tmp_path / "empty Results parent"
    parent.mkdir()
    messages = []
    monkeypatch.setattr(module.messagebox, "showinfo", lambda *args, **kwargs: messages.append(args))
    monkeypatch.setattr(debug_pack, "create_evaluation_debug_pack", lambda *args, **kwargs: pytest.fail("unidentified parent reached pack builder"))
    app = SimpleNamespace(
        _evaluation_workflow_last_run_dir=lambda **kwargs: parent,
        _evaluation_workflow_no_run_message=lambda path: f"No Results Bundle found yet: {path}",
        _open_path=lambda path: pytest.fail("unidentified parent opened as Results Bundle"),
    )
    getattr(module.SplitPointAnalyserGUI, action)(app)
    assert messages == [("Evaluation Workflow", f"No Results Bundle found yet: {parent}")]
    assert list(parent.iterdir()) == []


def test_workflow_actions_admit_identified_partial_run_with_controlled_pack_leaf(tmp_path: Path, monkeypatch) -> None:
    """Guard admission only; the replaced pack leaf is not archive validation."""
    import json
    from types import SimpleNamespace
    from onnx_splitpoint_tool.workflow import debug_pack
    from onnx_splitpoint_tool.workflow.run_discovery import is_evaluation_run_dir

    module = _gui_module_for_callback_tests(monkeypatch)
    run = tmp_path / "partial_run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run.name,
        "status": "partial",
    }), encoding="utf-8")
    (run / "evaluation_workflow.log").write_text("Controlled partial-run diagnostic.\n", encoding="utf-8")
    assert is_evaluation_run_dir(run, purpose="debug")
    opened = []
    packed = []
    messages = []
    monkeypatch.setattr(module.messagebox, "showinfo", lambda *args, **kwargs: messages.append(args))

    def controlled_pack_leaf(run_dir, out_zip, **kwargs):
        packed.append((run_dir, out_zip, kwargs))
        return {"archive_sha256": "controlled-leaf-response", "archive_size_bytes": 0}

    monkeypatch.setattr(debug_pack, "create_evaluation_debug_pack", controlled_pack_leaf)
    app = SimpleNamespace(
        _evaluation_workflow_last_run_dir=lambda **kwargs: run,
        _evaluation_workflow_no_run_message=lambda path: pytest.fail("identified partial run rejected"),
        _evaluation_workflow_pack_output_dir=lambda: tmp_path / "exports",
        _open_path=opened.append,
    )
    module.SplitPointAnalyserGUI._evaluation_workflow_open_results_folder(app)
    module.SplitPointAnalyserGUI._evaluation_workflow_prepare_debug_pack(app)
    expected_zip = tmp_path / "exports" / "partial_run_debug_pack.zip"
    assert opened == [str(run), str(expected_zip)]
    assert packed == [(run, expected_zip, {"source_selection_policy": "newest_identified_evaluation_run_for_debug"})]
    assert messages and messages[0][0] == "Evaluation Workflow"
