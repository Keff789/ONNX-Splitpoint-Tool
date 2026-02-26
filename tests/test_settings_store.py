import json
from pathlib import Path


def test_settings_defaults_load_when_missing(tmp_path: Path):
    from onnx_splitpoint_tool.settings import SettingsStore

    store = SettingsStore(home=tmp_path)
    data = store.load()
    assert isinstance(data, dict)
    assert data.get("schema_version") == 1
    assert "tk_vars" in data


def test_settings_roundtrip_save_load(tmp_path: Path):
    from onnx_splitpoint_tool.settings import SettingsStore

    store = SettingsStore(home=tmp_path)
    store.save(
        {
            "schema_version": 1,
            "output_dir": "/tmp/work",
            "tk_vars": {"var_x": "hello", "var_y": 123},
            "remote_hosts": [{"id": "nx", "label": "Jetson NX", "user": "u", "host": "h"}],
            "remote_selected_host_id": "nx",
        }
    )

    loaded = store.load()
    assert loaded["output_dir"] == "/tmp/work"
    assert loaded["tk_vars"]["var_x"] == "hello"
    assert loaded["remote_selected_host_id"] == "nx"


def test_settings_corrupt_json_is_backed_up(tmp_path: Path):
    from onnx_splitpoint_tool.settings import SettingsStore

    store = SettingsStore(home=tmp_path)
    p = store.path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json", encoding="utf-8")

    loaded = store.load()
    assert loaded.get("schema_version") == 1

    # A backup should exist
    baks = sorted(p.parent.glob(p.name + ".bak.*"))
    assert baks, "Expected a backup to be created for corrupt settings"
