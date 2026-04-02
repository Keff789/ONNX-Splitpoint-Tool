from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool.gui.hailo_parse_budget import (
    normalize_persisted_hailo_max_checks,
    resolve_hailo_max_checks,
)


def test_hailo_parse_budget_helper_supports_auto_and_explicit_values() -> None:
    budget, is_auto = resolve_hailo_max_checks("auto", topk=40)
    assert budget == 40
    assert is_auto is True

    budget, is_auto = resolve_hailo_max_checks("", topk=12)
    assert budget == 12
    assert is_auto is True

    budget, is_auto = resolve_hailo_max_checks("31", topk=12)
    assert budget == 31
    assert is_auto is False


def test_hailo_parse_budget_helper_normalizes_legacy_persisted_default() -> None:
    assert normalize_persisted_hailo_max_checks("") == "auto"
    assert normalize_persisted_hailo_max_checks(None) == "auto"
    assert normalize_persisted_hailo_max_checks("25") == "auto"
    assert normalize_persisted_hailo_max_checks("40") == "40"


def test_gui_exposes_auto_budget_and_clearer_hailo_max_label() -> None:
    gui_src = Path('onnx_splitpoint_tool/gui_app.py').read_text(encoding='utf-8')
    panel_src = Path('onnx_splitpoint_tool/gui/panels/panel_hardware.py').read_text(encoding='utf-8')
    app_src = Path('onnx_splitpoint_tool/gui/app.py').read_text(encoding='utf-8')

    assert 'from .gui.hailo_parse_budget import resolve_hailo_max_checks' in gui_src
    assert 'self.var_hailo_max_checks = tk.StringVar(value="auto")' in gui_src
    assert 'Max Hailo checks:' in gui_src
    assert 'resolve_hailo_max_checks(' in gui_src

    assert '_str_var(app, "var_hailo_max_checks", "auto")' in panel_src
    assert 'Max Hailo checks:' in panel_src
    assert 'Leer/auto = folgt Top-k.' in panel_src

    assert 'normalize_persisted_hailo_max_checks' in app_src
    assert 'if name == "var_hailo_max_checks":' in app_src


def test_select_picks_no_longer_breaks_when_hailo_budget_is_exhausted() -> None:
    src = Path('onnx_splitpoint_tool/gui_app.py').read_text(encoding='utf-8')
    start = src.index('        checks_budget = int(getattr(p, "hailo_max_checks", 0) or 0) if hailo_enabled else 0')
    end = src.index('        # Attach Hailo info to analysis', start)
    block = src[start:end]

    assert 'continued_without_hailo_checks = False' in block
    assert 'budget_notice_sent = False' in block
    assert 'Hailo parse-check {checks_done}/{checks_budget} | selected {len(picks)}/{p.topk}: boundary b={b}' in block
    assert 'continuing selection {len(picks)}/{p.topk} without further Hailo checks' in block
    assert 'unchecked_selected += 1' in block
    assert 'hailo_summary["budget_exhausted"] = True\n                    break' not in block


def test_hailo_summary_tracks_unchecked_selected_candidates() -> None:
    src = Path('onnx_splitpoint_tool/gui_app.py').read_text(encoding='utf-8')
    start = src.index('            hailo_summary.update(')
    end = src.index('            a["hailo_check"] = hailo_summary', start)
    block = src[start:end]

    assert '"unchecked_selected": int(unchecked_selected)' in block
    assert '"continued_without_checks": bool(continued_without_hailo_checks)' in block
    assert '"selection_target_topk": int(getattr(p, "topk", len(picks)) or len(picks))' in block
