"""Keep the GUI's model count aligned with the saved dashboard contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from onnx_splitpoint_tool.gui.dashboard_projection import dashboard_summary_line


FIXTURE = (
    Path(__file__).parent / "fixtures" / "v2802_completsetdev_regression"
    / "reports" / "result_dashboard.json"
)


def _render_result(tmp_path, dashboard):
    report = tmp_path / "reports" / "result_dashboard.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps(dashboard).encode("utf-8")
    report.write_bytes(original)

    # Keep the GUI's TkAgg import separate from scientific tests using Agg.
    # The real result method runs without creating a window or hardware run.
    rendered = subprocess.run(
        [sys.executable, "-c", """
import json
import sys
from onnx_splitpoint_tool.gui.app import SplitPointAnalyserGUI

class Display:
    def _eval_workflow_text_set(self, text):
        self.text = text

display = Display()
SplitPointAnalyserGUI._eval_workflow_render_result(
    display, {"run_dir": sys.argv[1], "status": "failed"},
)
print(json.dumps(display.text))
""", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True,
        text=True, check=True, timeout=60,
    )
    assert report.read_bytes() == original
    return json.loads(rendered.stdout)


def test_completsetdev_original_dashboard_is_seven_models_not_77_rows(tmp_path):
    dashboard = json.loads(FIXTURE.read_text(encoding="utf-8"))
    original = copy.deepcopy(dashboard)
    assert dashboard["summary"]["model_count"] == 7
    assert dashboard["summary"]["row_count"] == 77
    assert "overview" not in dashboard
    assert "models" not in dashboard
    assert dashboard_summary_line(dashboard) == "models=7"
    text = _render_result(tmp_path, dashboard)
    assert "  - models=7\n" in text
    assert "models=0" not in text
    assert "models=77" not in text
    assert dashboard == original


def test_current_summary_wins_over_stale_legacy_projection():
    dashboard = {
        "summary": {"model_count": 7, "row_count": 77},
        "overview": {"model_count": 0},
        "model_count": 77,
        "models": [{"model_id": "same_model"}] * 77,
    }
    assert dashboard_summary_line(dashboard) == "models=7"


@pytest.mark.parametrize("value", [None, "", "7", -1, 7.5, True, [], {}])
def test_invalid_current_model_count_is_unavailable(value):
    assert dashboard_summary_line({
        "summary": {"model_count": value, "row_count": 77},
        "overview": {"model_count": 7},
    }) == "models=unavailable"


@pytest.mark.parametrize("dashboard", [
    {}, {"summary": {}}, {"summary": {"row_count": 77}},
    {"rows": [{"model_id": "m"}] * 77},
    {"models": [{"model_id": "m"}] * 7},
])
def test_missing_model_count_is_never_invented_from_rows_or_cards(tmp_path, dashboard):
    text = _render_result(tmp_path, dashboard)
    assert "models=unavailable" in text
    assert "models=0" not in text
    assert "could not read" not in text


def test_explicit_current_zero_is_preserved():
    assert dashboard_summary_line({"summary": {"model_count": 0}}) == "models=0"


@pytest.mark.parametrize("nested", [False, True])
def test_legacy_summary_still_displays_observed_counts(nested):
    overview = {
        "model_count": 2,
        "model_health_counts": {"ok": 1, "warn": 0, "partial": 1, "failed": 0},
    }
    dashboard = {"overview": overview} if nested else dict(overview)
    dashboard["models"] = [
        {"model_id": "a", "complete_split_count": 3,
         "hailo_evidence": {"runtime_verified": True}},
        {"model_id": "b", "complete_split_count": 0,
         "hailo_evidence": {"runtime_verified": False}},
    ]
    assert dashboard_summary_line(dashboard) == (
        "models=2, ok=1, warn=0, partial=1, failed=0, "
        "complete_split_models=1, hailo_runtime_models=1"
    )


def test_absent_legacy_health_and_cards_are_unavailable():
    assert dashboard_summary_line({"overview": {"model_count": 7}}) == (
        "models=7, ok=unavailable, warn=unavailable, partial=unavailable, "
        "failed=unavailable, complete_split_models=unavailable, "
        "hailo_runtime_models=unavailable"
    )


def test_malformed_dashboard_is_reported_as_read_failure(tmp_path):
    text = _render_result(tmp_path, ["invalid-dashboard-object"])
    assert "Result dashboard: could not read" in text
    assert "models=0" not in text
