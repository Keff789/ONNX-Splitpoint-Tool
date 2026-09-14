from __future__ import annotations

import ast
from pathlib import Path

from onnx_splitpoint_tool.energy import config as energy_config
from onnx_splitpoint_tool.gui.app import (
    SplitPointAnalyserGUI,
    _persist_as_generic_gui_var,
)
from onnx_splitpoint_tool.gui.panels.panel_hardware import (
    _PLATFORM_POWER_SETUP_CARDS,
    _deferred_platform_power_error_callback,
    _schedule_initial_platform_power_refresh,
)


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "onnx_splitpoint_tool" / "gui" / "panels" / "panel_hardware.py"
APP_PATH = ROOT / "onnx_splitpoint_tool" / "gui" / "app.py"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _called_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def test_setup_scoped_platform_power_vars_are_not_generic_gui_settings() -> None:
    assert _persist_as_generic_gui_var("var_platform_power_setup") is False
    assert _persist_as_generic_gui_var("var_platform_power_urecs_address") is False
    assert _persist_as_generic_gui_var("var_platform_power_udp_port") is False
    assert _persist_as_generic_gui_var("var_platform_power_enabled") is False
    assert _persist_as_generic_gui_var("var_platform_power_status_detail") is False
    assert _persist_as_generic_gui_var("var_platform_power_operation") is False
    assert _persist_as_generic_gui_var("var_hwsetup_orin_nx_hailo8_01_accel_idle_w") is False
    assert _persist_as_generic_gui_var("var_energy_enabled") is True


def test_gui_registry_save_delegates_to_central_atomic_writer(
    tmp_path, monkeypatch
) -> None:
    registry_path = tmp_path / "hardware_setups.yaml"
    captured: dict[str, object] = {}

    def central_save(payload, path):
        captured["payload"] = dict(payload)
        captured["path"] = Path(path)
        return Path(path)

    monkeypatch.setattr(energy_config, "save_hardware_registry", central_save)

    class FakeApp:
        @staticmethod
        def _hardware_setups_path() -> Path:
            return registry_path

    payload = {"schema_version": 2, "hardware_setups": [{"id": "setup-a"}]}
    SplitPointAnalyserGUI._hardware_registry_save(FakeApp(), payload)

    assert captured == {"payload": payload, "path": registry_path}
    save_source = ast.get_source_segment(
        APP_PATH.read_text(encoding="utf-8"),
        _function(ast.parse(APP_PATH.read_text(encoding="utf-8")), "_hardware_registry_save"),
    )
    assert save_source is not None
    assert "save_hardware_registry(payload, p)" in save_source
    assert "os.replace" not in save_source
    assert 'with_suffix(p.suffix + ".tmp")' not in save_source


def test_platform_power_cards_use_the_three_canonical_setup_ids() -> None:
    assert [setup_id for setup_id, _label in _PLATFORM_POWER_SETUP_CARDS] == [
        "orin_nx_hailo8_01",
        "orin_nx_hailo10_01",
        "orin_nx_deepx_m1_01",
    ]
    assert len({setup_id for setup_id, _label in _PLATFORM_POWER_SETUP_CARDS}) == 3


def test_platform_power_cards_are_registry_read_only_without_stale_auto_save() -> None:
    source = PANEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    outer = _function(tree, "_build_platform_power_ui")
    load = _function(outer, "_load_config")
    run_operation = _function(outer, "_run_operation")

    load_calls = [_called_name(node) for node in ast.walk(load) if isinstance(node, ast.Call)]
    operation_calls = [
        _called_name(node) for node in ast.walk(run_operation) if isinstance(node, ast.Call)
    ]
    assert "resolve_setup" in load_calls
    assert "save_hardware_registry" not in load_calls
    assert "_hardware_registry_save" not in operation_calls
    assert "save_hardware_registry" not in operation_calls
    assert "_save_selected_config" not in source
    assert "var_platform_power_urecs_address" not in source
    assert "var_platform_power_udp_port" not in source
    assert "var_platform_power_setup" not in source
    assert "Deliberately do not save GUI fields here" in source


def test_initial_platform_status_refresh_is_scheduled_exactly_once_without_polling() -> None:
    scheduled: list[tuple[int, object]] = []
    refresh_calls: list[dict[str, bool]] = []

    class Root:
        def after(self, delay_ms, callback):
            scheduled.append((int(delay_ms), callback))

    class App:
        root = Root()

    app = App()

    def refresh(**kwargs) -> None:
        refresh_calls.append(dict(kwargs))

    assert _schedule_initial_platform_power_refresh(app, refresh) is True
    assert _schedule_initial_platform_power_refresh(app, refresh) is False
    assert len(scheduled) == 1
    assert scheduled[0][0] == 300

    scheduled[0][1]()
    assert refresh_calls == [{"quiet": True}]
    # Executing the callback does not enqueue another callback.
    assert len(scheduled) == 1


def test_deferred_status_failure_keeps_exception_text_after_except_scope() -> None:
    received: list[str] = []
    try:
        raise RuntimeError("probe failed")
    except RuntimeError as exc:
        callback = _deferred_platform_power_error_callback(exc, received.append)

    # The exception binding has been cleared here; the deferred Tk callback must
    # nevertheless be self-contained.
    callback()
    assert received == ["Status refresh failed: RuntimeError: probe failed"]


def test_status_refresh_is_read_only_and_failure_releases_busy_state() -> None:
    tree = ast.parse(PANEL_PATH.read_text(encoding="utf-8"))
    refresh = _function(tree, "_refresh_status")
    calls = [_called_name(node) for node in ast.walk(refresh) if isinstance(node, ast.Call)]

    assert "probe_platform_status" in calls
    assert "_save_selected_config" not in calls
    # No status callback schedules itself; only the explicit one-shot scheduler
    # and user/actions can start another probe.
    assert "_refresh_status" not in calls

    fail = _function(refresh, "fail")
    releases_busy = False
    for call in (node for node in ast.walk(fail) if isinstance(node, ast.Call)):
        if _called_name(call) != "_set_card_busy" or not call.args:
            continue
        value = call.args[0]
        releases_busy = isinstance(value, ast.Constant) and value.value is False
    assert releases_busy is True


def test_urecs_badge_claims_only_icmp_host_reachability() -> None:
    tree = ast.parse(PANEL_PATH.read_text(encoding="utf-8"))
    apply_status = _function(tree, "_apply_status")
    labels = {
        node.value
        for node in ast.walk(apply_status)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "u.RECS host: reachable" in labels
    assert "u.RECS host: unreachable" in labels
    assert "u.RECS: online" not in labels
    assert "u.RECS: offline" not in labels


def test_manual_refreshes_are_explicit_and_power_operation_thread_is_non_daemon() -> None:
    source = PANEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    outer = _function(tree, "_build_platform_power_ui")

    manual_refresh_buttons = 0
    thread_daemon_by_name: dict[str, bool] = {}
    for call in (node for node in ast.walk(outer) if isinstance(node, ast.Call)):
        keywords = {item.arg: item.value for item in call.keywords if item.arg}
        if _called_name(call) == "Button":
            command = keywords.get("command")
            if isinstance(command, ast.Name) and command.id == "_refresh_status":
                manual_refresh_buttons += 1
        if _called_name(call) != "Thread":
            continue
        name_node = keywords.get("name")
        daemon_node = keywords.get("daemon")
        if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
            if isinstance(daemon_node, ast.Constant) and isinstance(daemon_node.value, bool):
                thread_daemon_by_name[name_node.value] = daemon_node.value

    # One button template is instantiated once per canonical setup card.
    assert manual_refresh_buttons == 1
    assert len(_PLATFORM_POWER_SETUP_CARDS) == 3
    assert thread_daemon_by_name["platform-power-operation"] is False
    # The dynamic status-thread name is separately asserted from source because
    # its AST node is a JoinedStr rather than a Constant.
    assert 'name=f"platform-status-{setup_id}"' in source


def test_lazy_hardware_build_reloads_all_registry_backed_cards() -> None:
    panel_source = PANEL_PATH.read_text(encoding="utf-8")
    app_source = APP_PATH.read_text(encoding="utf-8")
    assert "app._platform_power_reload_cards_callback = _reload_all_cards" in panel_source
    assert 'if key == "hardware":' in app_source
    assert "reload_platform_cards()" in app_source


def test_global_mutating_busy_disables_all_three_cards() -> None:
    tree = ast.parse(PANEL_PATH.read_text(encoding="utf-8"))
    outer = _function(tree, "_build_platform_power_ui")
    sync = _function(outer, "_sync_button_states")
    run_operation = _function(outer, "_run_operation")

    sync_source = ast.unparse(sync)
    operation_source = ast.unparse(run_operation)
    assert "shared['mutating']" in sync_source
    assert "for card_state in list(shared['cards'])" in sync_source
    assert "global_busy or card_busy" in sync_source
    assert "_set_global_mutating(True)" in operation_source
    assert "_set_global_mutating(False)" in operation_source


def test_no_global_platform_status_banner_and_three_equal_columns() -> None:
    source = PANEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    build = _function(tree, "build_panel")
    power_ui = _function(tree, "_build_platform_power_ui")

    build_source = ast.unparse(build)
    power_source = ast.unparse(power_ui)
    assert "platform_banner" not in build_source
    assert "LabelFrame(frame, text='Platform status')" not in build_source
    assert "_build_platform_power_ui(tab_platform, app=app)" in build_source
    assert "uniform='platform-power-card'" in power_source
    assert "enumerate(_PLATFORM_POWER_SETUP_CARDS)" in power_source


def test_refresh_all_is_scheduled_once_and_fans_out_to_every_card() -> None:
    tree = ast.parse(PANEL_PATH.read_text(encoding="utf-8"))
    outer = _function(tree, "_build_platform_power_ui")
    refresh_all = _function(outer, "_refresh_all")
    outer_source = ast.unparse(outer)
    refresh_source = ast.unparse(refresh_all)

    assert "for controller in controllers" in refresh_source
    assert "controller['refresh'](quiet=quiet)" in refresh_source
    assert outer_source.count("_schedule_initial_platform_power_refresh(app, _refresh_all)") == 1


def test_invalid_remote_energy_window_clears_host_normalized_energy() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    dispatch = _function(tree, "_run_remote_dispatch_with_energy")
    invalid_branch = next(
        node
        for node in ast.walk(dispatch)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "remote_invalid_reason"
    )
    cleared: set[str] = set()
    for assignment in (
        node for node in ast.walk(invalid_branch) if isinstance(node, ast.Assign)
    ):
        if not isinstance(assignment.value, ast.Constant) or assignment.value.value is not None:
            continue
        for target in assignment.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not isinstance(target.value, ast.Name) or target.value.id != "result":
                continue
            if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                cleared.add(target.slice.value)
    assert "avg_host_normalized_energy_est_j" in cleared
    assert "avg_host_normalized_energy_per_configured_work_unit_est_j" in cleared
