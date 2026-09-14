from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import socket

import pytest
import yaml

from onnx_splitpoint_tool import platform_power as pp


SETUP_ID = "orin_nx_hailo8_01"


def _registry(
    path: Path,
    *,
    address: object = "192.0.2.10",
    power_updates: dict[str, object] | None = None,
) -> Path:
    power: dict[str, object] = {
        "enabled": True,
        "udp_port": 3000,
        "udp_terminator": "lf",
        "jetson_command": "jetson",
        "m2_command": "m.2",
        "require_ping_before_toggle": False,
        "shutdown_settle_s": 0,
        "power_toggle_settle_s": 0,
        "m2_toggle_settle_s": 0,
        "m2_post_boot_settle_s": 0,
        "m2_verify_interval_s": 0,
    }
    power.update(power_updates or {})
    payload = {
        "schema": "onnx-splitpoint/hardware-setups",
        "schema_version": 2,
        "hardware_setups": [
            {
                "id": SETUP_ID,
                "accelerator": "hailo8",
                "host": {"address": "192.0.2.20", "user": "nx", "port": 22},
                "energy": {"enabled": True, "urecs_address": address},
                "power_control": power,
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _status(*, ssh: bool = True, m2: bool = True) -> pp.PlatformStatus:
    return pp.PlatformStatus(
        setup_id=SETUP_ID,
        checked_at="2026-09-02T00:00:00+00:00",
        urecs_address="192.0.2.10",
        urecs_host="192.0.2.10",
        urecs_port=3000,
        urecs_configured=True,
        urecs_reachable=True,
        urecs_detail="reachable",
        jetson_host="nx@192.0.2.20:22",
        jetson_configured=True,
        jetson_ssh_ready=ssh,
        jetson_detail="ready" if ssh else "down",
        accelerator="hailo8",
        m2_present=m2,
        m2_detail="detected" if m2 else "not detected",
    )


class _MutationTransport:
    shutdown_runs: list[str] = []

    def __init__(self, _host: object):
        pass

    def run_read_only(self, command: str, timeout_s: int = 15):
        type(self).shutdown_runs.append(f"read:{command}")
        return 0, ""

    def run(self, command: str, timeout_s: int = 15):
        type(self).shutdown_runs.append(f"run:{command}")
        return 0, "__SPLITPOINT_SHUTDOWN_SCHEDULED__"

    def test_connection(self, timeout_s: int = 5):
        return True, "ready"


@contextmanager
def _unlocked(_setup_id: str, **_kwargs: object):
    yield


@pytest.mark.parametrize("operation", ["jetson", "m2"])
@pytest.mark.parametrize(
    ("address", "power_updates", "error"),
    [
        ("192.0.2.10", {"jetson_command": ""}, "jetson_command"),
        ("192.0.2.10", {"m2_command": ""}, "m2_command"),
        ("192.0.2.10", {"m2_command": "m2\nunsafe"}, "m2_command"),
        ("192.0.2.10", {"udp_terminator": "invalid"}, "udp_terminator"),
        ("192.0.2.10", {"udp_port": 65536}, "udp_port"),
        (
            "192.0.2.10",
            {"shutdown_preflight_command": ""},
            "shutdown_preflight_command",
        ),
        (
            "192.0.2.10",
            {"shutdown_preflight_command": ["sudo", "-n", "true"]},
            "shutdown_preflight_command",
        ),
        (
            "192.0.2.10",
            {"shutdown_command": ""},
            "shutdown_command",
        ),
        (
            "192.0.2.10",
            {"shutdown_command": ["systemctl", "poweroff"]},
            "shutdown_command",
        ),
        (
            "192.0.2.10",
            {"require_ping_before_toggle": "tru"},
            "require_ping_before_toggle",
        ),
        ("", {}, "urecs_address"),
    ],
)
def test_invalid_udp_configuration_fails_before_shutdown_or_send(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    address: object,
    power_updates: dict[str, object],
    error: str,
) -> None:
    path = _registry(
        tmp_path / "hardware_setups.yaml",
        address=address,
        power_updates=power_updates,
    )
    _MutationTransport.shutdown_runs = []
    sends: list[tuple[object, ...]] = []
    monkeypatch.setattr(pp, "platform_operation_lock", _unlocked)
    monkeypatch.setattr(pp, "probe_platform_status", lambda *_a, **_k: _status())
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *args, **kwargs: sends.append((*args, kwargs)) or {"ok": True},
    )
    monkeypatch.setattr(pp, "_wait_for_ssh", lambda *_a, **_k: (True, "ok"))
    monkeypatch.setattr(pp.time, "sleep", lambda _seconds: None)

    call = pp.set_jetson_state if operation == "jetson" else pp.set_m2_state
    with pytest.raises(pp.PlatformConfigurationError, match=error):
        call(
            SETUP_ID,
            False,
            registry_path=path,
            transport_factory=_MutationTransport,
        )

    assert _MutationTransport.shutdown_runs == []
    assert sends == []


@pytest.mark.parametrize(
    "field",
    [
        "enabled",
        "require_ping_before_toggle",
        "require_positive_calibration_delta",
        "jetson_command",
        "udp_terminator",
        "shutdown_preflight_command",
        "shutdown_command",
    ],
)
@pytest.mark.parametrize(
    "operation",
    ["set_jetson", "toggle_jetson", "set_m2", "toggle_m2", "calibrate"],
)
def test_explicit_null_power_field_blocks_every_public_mutator_before_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    operation: str,
) -> None:
    path = _registry(
        tmp_path / "hardware_setups.yaml",
        power_updates={field: None},
    )
    output = tmp_path / "must-not-exist"
    tools: list[str] = []

    monkeypatch.setattr(
        pp,
        "platform_operation_lock",
        lambda *_a, **_k: tools.append("lock"),
    )
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: tools.append("probe") or _status(),
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: tools.append("udp") or {"ok": True},
    )
    monkeypatch.setattr(
        pp.socket,
        "getaddrinfo",
        lambda *_a, **_k: tools.append("dns") or [],
    )

    kwargs = {"registry_path": path}
    if operation == "set_jetson":
        call = lambda: pp.set_jetson_state(SETUP_ID, False, **kwargs)
    elif operation == "toggle_jetson":
        call = lambda: pp.toggle_jetson(SETUP_ID, **kwargs)
    elif operation == "set_m2":
        call = lambda: pp.set_m2_state(SETUP_ID, False, **kwargs)
    elif operation == "toggle_m2":
        call = lambda: pp.toggle_m2(SETUP_ID, **kwargs)
    else:
        call = lambda: pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            output_dir=output,
            measurement_runner=lambda **_k: tools.append("measurement"),
            **kwargs,
        )

    with pytest.raises(pp.PlatformConfigurationError, match=field):
        call()

    assert tools == []
    assert not output.exists()


def test_explicit_null_power_control_mapping_fails_before_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["hardware_setups"][0]["power_control"] = None
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    tools: list[str] = []
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: tools.append("probe") or _status(),
    )
    monkeypatch.setattr(
        pp.socket,
        "getaddrinfo",
        lambda *_a, **_k: tools.append("dns") or [],
    )

    with pytest.raises(pp.PlatformConfigurationError, match="power_control"):
        pp.set_m2_state(SETUP_ID, False, registry_path=path)

    assert tools == []


def test_legacy_null_m2_command_is_migrated_to_punctuated_token(
    tmp_path: Path,
) -> None:
    path = _registry(
        tmp_path / "hardware_setups.yaml",
        power_updates={"m2_command": None},
    )
    _registry_snapshot, setup, cfg = pp.resolve_setup(
        SETUP_ID, registry_path=path
    )
    assert setup["power_control"]["m2_command"] == "m.2"
    assert cfg["power_control"]["m2_command"] == "m.2"


@pytest.mark.parametrize("registry_path", ["", 0, True, [], {}])
@pytest.mark.parametrize(
    "operation",
    ["set_jetson", "toggle_jetson", "set_m2", "toggle_m2", "calibrate"],
)
def test_invalid_registry_path_blocks_every_public_mutator_before_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry_path: object,
    operation: str,
) -> None:
    output = tmp_path / "must-not-exist"
    tools: list[str] = []
    monkeypatch.setattr(
        pp,
        "load_hardware_registry",
        lambda *_a, **_k: tools.append("registry") or {},
    )
    monkeypatch.setattr(
        pp,
        "platform_operation_lock",
        lambda *_a, **_k: tools.append("lock"),
    )
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: tools.append("probe") or _status(),
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: tools.append("udp") or {"ok": True},
    )

    kwargs = {"registry_path": registry_path}
    if operation == "set_jetson":
        call = lambda: pp.set_jetson_state(SETUP_ID, False, **kwargs)
    elif operation == "toggle_jetson":
        call = lambda: pp.toggle_jetson(SETUP_ID, **kwargs)
    elif operation == "set_m2":
        call = lambda: pp.set_m2_state(SETUP_ID, False, **kwargs)
    elif operation == "toggle_m2":
        call = lambda: pp.toggle_m2(SETUP_ID, **kwargs)
    else:
        call = lambda: pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            output_dir=output,
            measurement_runner=lambda **_k: tools.append("measurement"),
            **kwargs,
        )

    with pytest.raises(pp.PlatformConfigurationError, match="registry_path"):
        call()

    assert tools == []
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("udp_port", "3000"),
        ("status_ping_timeout_s", True),
        ("ssh_probe_timeout_s", float("nan")),
        ("m2_probe_timeout_s", float("inf")),
        ("shutdown_timeout_s", 9.99),
        ("boot_timeout_s", -1),
        ("ssh_poll_interval_s", 0.49),
        ("shutdown_settle_s", -0.01),
        ("power_toggle_settle_s", "3"),
        ("m2_toggle_settle_s", False),
        ("m2_post_boot_settle_s", float("nan")),
        ("m2_verify_observations", 2.0),
        ("m2_verify_interval_s", -0.01),
        ("calibration_stabilize_s", None),
        ("calibration_measure_s", 4.99),
        ("minimum_calibration_delta_w", -0.01),
    ],
)
def test_explicit_malformed_power_number_fails_before_any_hardware_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    path = _registry(
        tmp_path / "hardware_setups.yaml",
        power_updates={field: value},
    )
    tools: list[str] = []

    class ForbiddenTransport:
        def __init__(self, _host: object) -> None:
            tools.append("ssh_transport")

    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: tools.append("probe") or _status(),
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: tools.append("udp") or {"ok": True},
    )
    monkeypatch.setattr(
        pp.socket,
        "getaddrinfo",
        lambda *_a, **_k: tools.append("dns") or [],
    )

    with pytest.raises(pp.PlatformConfigurationError, match=field):
        pp.set_m2_state(
            SETUP_ID,
            False,
            registry_path=path,
            transport_factory=ForbiddenTransport,
        )

    assert tools == []


def test_missing_power_numbers_use_exact_release_defaults() -> None:
    merged = pp.merged_power_control({})
    for field, expected in pp.DEFAULT_POWER_CONTROL.items():
        if field in pp._POWER_CONTROL_NUMERIC_FIELDS:
            assert merged[field] == expected


def test_valid_numeric_boundaries_remain_admitted() -> None:
    merged = pp.merged_power_control(
        {
            "udp_port": 65535,
            "status_ping_timeout_s": 0.2,
            "ssh_probe_timeout_s": 1,
            "m2_probe_timeout_s": 1,
            "shutdown_timeout_s": 10,
            "boot_timeout_s": 10,
            "ssh_poll_interval_s": 0.5,
            "shutdown_settle_s": 0,
            "power_toggle_settle_s": 0,
            "m2_toggle_settle_s": 0,
            "m2_post_boot_settle_s": 0,
            "m2_verify_observations": 2,
            "m2_verify_interval_s": 0,
            "calibration_stabilize_s": 0,
            "calibration_measure_s": 5,
            "minimum_calibration_delta_w": 0,
        }
    )
    assert merged["udp_port"] == 65535
    assert merged["m2_verify_observations"] == 2
    assert merged["calibration_measure_s"] == 5.0


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("stabilize_s", "30"),
        ("stabilize_s", True),
        ("stabilize_s", float("nan")),
        ("stabilize_s", float("inf")),
        ("stabilize_s", -0.01),
        ("measure_s", "30"),
        ("measure_s", False),
        ("measure_s", float("nan")),
        ("measure_s", float("-inf")),
        ("measure_s", 4.99),
    ],
)
def test_invalid_calibration_window_override_fails_before_output_or_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: object,
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    output = tmp_path / "must-not-exist"
    tools: list[str] = []
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: tools.append("probe") or _status(),
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: tools.append("udp") or {"ok": True},
    )
    monkeypatch.setattr(
        pp,
        "_set_m2_state_locked",
        lambda *_a, **_k: tools.append("m2") or {"ok": True},
    )

    with pytest.raises(pp.PlatformConfigurationError, match=argument):
        pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            registry_path=path,
            output_dir=output,
            measurement_runner=lambda **_kwargs: tools.append("measurement"),
            **{argument: value},
        )

    assert tools == []
    assert not output.exists()


@pytest.mark.parametrize(
    "force_value",
    ["false", 1, None],
)
@pytest.mark.parametrize(
    "operation",
    ["set_jetson", "toggle_jetson", "set_m2", "toggle_m2", "calibrate"],
)
def test_nonboolean_force_active_workflow_rejected_by_every_public_mutator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    force_value: object,
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    output = tmp_path / "must-not-exist"
    tools: list[str] = []
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: tools.append("probe") or _status(),
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: tools.append("udp") or {"ok": True},
    )

    kwargs = {
        "registry_path": path,
        "force_active_workflow": force_value,
    }
    if operation == "set_jetson":
        call = lambda: pp.set_jetson_state(SETUP_ID, False, **kwargs)
    elif operation == "toggle_jetson":
        call = lambda: pp.toggle_jetson(SETUP_ID, **kwargs)
    elif operation == "set_m2":
        call = lambda: pp.set_m2_state(SETUP_ID, False, **kwargs)
    elif operation == "toggle_m2":
        call = lambda: pp.toggle_m2(SETUP_ID, **kwargs)
    else:
        call = lambda: pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            output_dir=output,
            measurement_runner=lambda **_kwargs: tools.append("measurement"),
            **kwargs,
        )

    with pytest.raises(
        pp.PlatformConfigurationError, match="force_active_workflow"
    ):
        call()

    assert tools == []
    assert not output.exists()


@pytest.mark.parametrize("force_value", ["false", 1, None])
def test_toggle_jetson_rejects_nonboolean_unknown_state_override_before_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force_value: object,
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    tools: list[str] = []
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: tools.append("probe") or _status(ssh=False),
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: tools.append("udp") or {"ok": True},
    )

    with pytest.raises(
        pp.PlatformConfigurationError, match="force_unknown_off_state"
    ):
        pp.toggle_jetson(
            SETUP_ID,
            registry_path=path,
            force_unknown_off_state=force_value,  # type: ignore[arg-type]
        )

    assert tools == []


def test_low_level_force_guard_rejects_truthy_string_before_lock_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scans: list[str] = []
    monkeypatch.setattr(
        pp,
        "held_workflow_locks",
        lambda: scans.append("scan") or [],
    )
    with pytest.raises(pp.PlatformConfigurationError, match="force"):
        pp.assert_no_active_workflow(force="false")  # type: ignore[arg-type]
    assert scans == []


def test_explicit_empty_registry_never_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[str] = []
    monkeypatch.setattr(
        pp,
        "load_hardware_registry",
        lambda *_a, **_k: loads.append("default") or _registry,
    )

    with pytest.raises(pp.PlatformConfigurationError, match="hardware_setups"):
        pp.resolve_setup(SETUP_ID, registry={})

    assert loads == []


@pytest.mark.parametrize(
    "operation",
    ["set_jetson", "toggle_jetson", "set_m2", "toggle_m2", "calibrate"],
)
def test_empty_registry_source_blocks_every_mutator_before_any_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    path = tmp_path / "empty-hardware-setups.yaml"
    output = tmp_path / "must-not-exist"
    tools: list[str] = []
    monkeypatch.setattr(
        pp,
        "platform_operation_lock",
        lambda *_a, **_k: tools.append("lock"),
    )
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: tools.append("probe"),
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: tools.append("udp"),
    )
    monkeypatch.setattr(
        pp.socket,
        "getaddrinfo",
        lambda *_a, **_k: tools.append("dns") or [],
    )
    monkeypatch.setattr(pp, "load_hardware_registry", lambda *_a, **_k: {})

    if operation == "set_jetson":
        call = lambda: pp.set_jetson_state(
            SETUP_ID, False, registry_path=path
        )
    elif operation == "toggle_jetson":
        call = lambda: pp.toggle_jetson(SETUP_ID, registry_path=path)
    elif operation == "set_m2":
        call = lambda: pp.set_m2_state(SETUP_ID, False, registry_path=path)
    elif operation == "toggle_m2":
        call = lambda: pp.toggle_m2(SETUP_ID, registry_path=path)
    else:
        call = lambda: pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            registry_path=path,
            output_dir=output,
        )

    with pytest.raises(pp.PlatformConfigurationError, match="hardware_setups"):
        call()

    assert tools == []
    assert not output.exists()


@pytest.mark.parametrize(
    ("host_value", "error"),
    [
        (["192.0.2.20"], "setup.host"),
        ({"address": " 192.0.2.20", "user": "nx", "port": 22}, "host_address"),
        ({"address": "192.0.2.20 ", "user": "nx", "port": 22}, "host_address"),
        ({"address": "192.0.2.20", "user": " nx", "port": 22}, "host_user"),
        ({"address": "192.0.2.20", "user": "nx\n", "port": 22}, "host_user"),
        ({"address": "192.0.2.20", "user": "", "port": 22}, "host.user"),
        ({"address": "192.0.2.20", "user": "nx", "port": True}, "host.port"),
        ({"address": "192.0.2.20", "user": "nx", "port": "22"}, "host.port"),
        ({"address": "192.0.2.20", "user": "nx", "port": 65536}, "host.port"),
        (
            {
                "address": "192.0.2.20",
                "user": "nx",
                "port": 22,
                "ssh_extra_args": ["-o", "BatchMode=yes"],
            },
            "ssh_extra_args",
        ),
    ],
)
def test_invalid_ssh_target_fails_before_probe_shutdown_or_udp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host_value: object,
    error: str,
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["hardware_setups"][0]["host"] = host_value
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    mutations: list[str] = []
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: mutations.append("probe") or _status(),
    )
    monkeypatch.setattr(
        pp,
        "_stop_jetson_and_cut_rail",
        lambda *_a, **_k: mutations.append("shutdown") or [],
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: mutations.append("udp") or {"ok": True},
    )

    with pytest.raises(pp.PlatformConfigurationError, match=error):
        pp.set_m2_state(SETUP_ID, False, registry_path=path)
    assert mutations == []


@pytest.mark.parametrize(
    ("host_update", "remote", "error"),
    [
        ({}, {"host": "192.0.2.21"}, "address_mismatch"),
        ({}, {"host": "192.0.2.20", "user": "root"}, "user_mismatch"),
        ({}, {"host": "192.0.2.20", "port": 2222}, "port_mismatch"),
        (
            {"ssh_extra_args": "-o BatchMode=yes"},
            {
                "host": "192.0.2.20",
                "ssh_extra_args": "-o ConnectTimeout=4",
            },
            "ssh_extra_args_mismatch",
        ),
        (
            {"host": "192.0.2.21"},
            {"host": "192.0.2.20"},
            "address_alias_mismatch",
        ),
    ],
)
def test_conflicting_host_and_remote_ssh_identity_fails_before_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host_update: dict[str, object],
    remote: dict[str, object],
    error: str,
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    setup = payload["hardware_setups"][0]
    setup["host"].update(host_update)
    setup["remote"] = remote
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    mutations: list[str] = []
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: mutations.append("ssh") or _status(),
    )
    monkeypatch.setattr(
        pp,
        "_stop_jetson_and_cut_rail",
        lambda *_a, **_k: mutations.append("shutdown") or [],
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: mutations.append("udp") or {"ok": True},
    )

    with pytest.raises(pp.PlatformConfigurationError, match=error):
        pp.set_m2_state(SETUP_ID, False, registry_path=path)

    assert mutations == []


@pytest.mark.parametrize("raw_id", [123, " orin_nx_hailo8_01", "orin_nx_hailo8_01\n"])
def test_noncanonical_registry_setup_id_fails_before_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_id: object,
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["hardware_setups"][0]["id"] = raw_id
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    mutations: list[str] = []
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: mutations.append("ssh") or _status(),
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: mutations.append("udp") or {"ok": True},
    )

    with pytest.raises(pp.PlatformConfigurationError, match="canonical"):
        pp.set_m2_state(SETUP_ID, False, registry_path=path)

    assert mutations == []


@pytest.mark.parametrize("requested", [123, " orin_nx_hailo8_01", "orin_nx_hailo8_01\t"])
def test_noncanonical_requested_setup_id_fails_before_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: object,
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    mutations: list[str] = []
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: mutations.append("ssh") or _status(),
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: mutations.append("udp") or {"ok": True},
    )

    with pytest.raises(pp.PlatformConfigurationError, match="canonical"):
        pp.set_m2_state(requested, False, registry_path=path)  # type: ignore[arg-type]

    assert mutations == []


def test_legacy_scalar_ssh_host_remains_supported(tmp_path: Path) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["hardware_setups"][0]["host"] = "192.0.2.20"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _reg, setup, cfg = pp.resolve_setup(SETUP_ID, registry_path=path)

    validated = pp._validate_platform_udp_configuration(setup, cfg)
    host = pp.host_config_from_setup(setup)
    assert validated["host"] == "192.0.2.10"
    assert host.host == "192.0.2.20"
    assert host.user == "nx"
    assert host.port == 22


@pytest.mark.parametrize("malformed", ["not-a-mapping", ["m.2"]])
def test_non_mapping_power_control_is_preserved_and_rejected_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformed: object,
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["hardware_setups"][0]["power_control"] = malformed
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    mutations: list[str] = []
    monkeypatch.setattr(
        pp,
        "_stop_jetson_and_cut_rail",
        lambda *_a, **_k: mutations.append("shutdown") or [],
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: mutations.append("udp") or {"ok": True},
    )

    with pytest.raises(pp.PlatformConfigurationError, match="must be a mapping"):
        pp.set_jetson_state(SETUP_ID, False, registry_path=path)

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    row = next(
        value for value in loaded["hardware_setups"] if value.get("id") == SETUP_ID
    )
    assert row["power_control"] == malformed
    assert mutations == []


def test_duplicate_setup_id_is_rejected_before_probe_or_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["hardware_setups"].append(dict(payload["hardware_setups"][0]))
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    mutations: list[str] = []
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: mutations.append("probe") or _status(),
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: mutations.append("udp") or {"ok": True},
    )

    with pytest.raises(pp.PlatformConfigurationError, match="Duplicate"):
        pp.set_m2_state(SETUP_ID, False, registry_path=path)
    assert mutations == []


def test_malformed_positive_delta_gate_blocks_calibration_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _registry(
        tmp_path / "hardware_setups.yaml",
        power_updates={"require_positive_calibration_delta": "fasle"},
    )
    mutations: list[str] = []
    monkeypatch.setattr(
        pp,
        "_set_m2_state_locked",
        lambda *_a, **_k: mutations.append("m2") or {"ok": True},
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: mutations.append("udp") or {"ok": True},
    )

    with pytest.raises(
        pp.PlatformConfigurationError,
        match="require_positive_calibration_delta",
    ):
        pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            registry_path=path,
            output_dir=tmp_path / "calibration",
        )
    assert mutations == []


def test_unresolvable_udp_address_fails_before_shutdown_when_ping_is_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _registry(
        tmp_path / "hardware_setups.yaml",
        address="does-not-resolve.invalid",
    )
    _MutationTransport.shutdown_runs = []
    sends: list[object] = []
    monkeypatch.setattr(
        pp.socket,
        "getaddrinfo",
        lambda *_a, **_k: (_ for _ in ()).throw(socket.gaierror("not found")),
    )
    monkeypatch.setattr(
        pp, "send_udp_command", lambda *args, **kwargs: sends.append((args, kwargs))
    )

    with pytest.raises(pp.PlatformConfigurationError, match="Cannot resolve"):
        pp.set_jetson_state(
            SETUP_ID,
            False,
            registry_path=path,
            transport_factory=_MutationTransport,
        )

    assert _MutationTransport.shutdown_runs == []
    assert sends == []


def test_custom_m2_token_is_preserved_exactly_and_empty_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _registry(
        tmp_path / "hardware_setups.yaml",
        power_updates={"m2_command": "custom.m2-token"},
    )
    _reg, setup, cfg = pp.resolve_setup(SETUP_ID, registry_path=path)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *args, **kwargs: calls.append((*args, kwargs)) or {"ok": True},
    )
    pp._send_configured_toggle(setup, cfg, "m2")
    assert calls[0][1] == "custom.m2-token"

    setup["power_control"]["m2_command"] = ""
    cfg["power_control"]["m2_command"] = ""
    calls.clear()
    with pytest.raises(pp.PlatformConfigurationError, match="m2_command"):
        pp._send_configured_toggle(setup, cfg, "m2")
    assert calls == []


@pytest.mark.parametrize("address", ["urecs.example", "192.0.2.10"])
def test_configured_toggle_pins_first_dns_resolution_through_udp_send(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    address: str,
) -> None:
    path = _registry(
        tmp_path / "hardware_setups.yaml",
        address=address,
    )
    _registry_snapshot, setup, cfg = pp.resolve_setup(
        SETUP_ID, registry_path=path
    )
    first = (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("192.0.2.31", 3000))
    second = (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("192.0.2.99", 3000))
    answers = [[first], [second]]
    resolver_calls: list[tuple[object, ...]] = []
    sent: list[tuple[bytes, tuple[str, int]]] = []

    def resolver(*args: object, **kwargs: object):
        resolver_calls.append((*args, kwargs))
        return answers.pop(0)

    class FakeSocket:
        def __init__(self, *_args: object) -> None:
            pass

        def settimeout(self, _timeout: float) -> None:
            pass

        def sendto(self, payload: bytes, sockaddr: tuple[str, int]) -> int:
            sent.append((payload, sockaddr))
            return len(payload)

        def close(self) -> None:
            pass

    original_send = pp.send_udp_command

    def intercepted_send(*args: object, **kwargs: object):
        kwargs["socket_factory"] = FakeSocket
        return original_send(*args, **kwargs)

    monkeypatch.setattr(pp.socket, "getaddrinfo", resolver)
    monkeypatch.setattr(pp, "send_udp_command", intercepted_send)

    result = pp._send_configured_toggle(setup, cfg, "m2")

    assert result["ok"] is True
    assert len(resolver_calls) == 1
    assert sent == [(b"m.2\n", ("192.0.2.31", 3000))]
    assert len(answers) == 1


def test_public_m2_transition_keeps_one_dns_pin_across_shutdown_and_boot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _registry(
        tmp_path / "hardware_setups.yaml",
        address="urecs.example",
    )
    first = (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("192.0.2.31", 3000))
    second = (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("192.0.2.99", 3000))
    answers = [[first], [second]]
    resolver_calls: list[str] = []
    sent: list[tuple[bytes, tuple[str, int]]] = []
    statuses = iter((_status(m2=True), _status(m2=False), _status(m2=False)))

    def resolver(*_args: object, **_kwargs: object):
        resolver_calls.append("resolve")
        return answers.pop(0)

    class FakeSocket:
        def __init__(self, *_args: object) -> None:
            pass

        def settimeout(self, _timeout: float) -> None:
            pass

        def sendto(self, payload: bytes, sockaddr: tuple[str, int]) -> int:
            sent.append((payload, sockaddr))
            return len(payload)

        def close(self) -> None:
            pass

    original_send = pp.send_udp_command

    def intercepted_send(*args: object, **kwargs: object):
        kwargs["socket_factory"] = FakeSocket
        return original_send(*args, **kwargs)

    monkeypatch.setattr(pp.socket, "getaddrinfo", resolver)
    monkeypatch.setattr(pp, "send_udp_command", intercepted_send)
    monkeypatch.setattr(pp, "platform_operation_lock", _unlocked)
    monkeypatch.setattr(
        pp, "probe_platform_status", lambda *_a, **_k: next(statuses)
    )
    monkeypatch.setattr(
        pp, "_wait_for_ssh", lambda *_a, **_k: (True, "observed")
    )
    monkeypatch.setattr(pp.time, "sleep", lambda _seconds: None)

    result = pp.set_m2_state(
        SETUP_ID,
        False,
        registry_path=path,
        transport_factory=_MutationTransport,
    )

    assert result["ok"] is True
    assert resolver_calls == ["resolve"]
    assert [sockaddr for _payload, sockaddr in sent] == [
        ("192.0.2.31", 3000),
        ("192.0.2.31", 3000),
        ("192.0.2.31", 3000),
    ]
    assert len(answers) == 1


@pytest.mark.parametrize("operation", ["jetson", "m2"])
@pytest.mark.parametrize("changed_field", ["host", "urecs", "command"])
def test_registry_change_while_waiting_for_lock_aborts_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    changed_field: str,
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    mutations: list[str] = []

    @contextmanager
    def change_while_waiting(_setup_id: str, **_kwargs: object):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        row = payload["hardware_setups"][0]
        if changed_field == "host":
            row["host"]["address"] = "192.0.2.21"
        elif changed_field == "urecs":
            row["energy"]["urecs_address"] = "192.0.2.11"
        else:
            row["power_control"]["jetson_command"] = "custom-jetson"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", change_while_waiting)
    monkeypatch.setattr(
        pp,
        "_stop_jetson_and_cut_rail",
        lambda *_a, **_k: mutations.append("shutdown") or [],
    )
    monkeypatch.setattr(
        pp,
        "_send_configured_toggle",
        lambda *_a, **_k: mutations.append("udp") or {"ok": True},
    )
    call = pp.set_jetson_state if operation == "jetson" else pp.set_m2_state

    with pytest.raises(pp.PlatformStateError, match="configuration changed"):
        call(SETUP_ID, False, registry_path=path)
    assert mutations == []


def test_direct_set_jetson_state_calls_locked_target_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(pp, "platform_operation_lock", _unlocked)

    def locked(setup_id: str, desired_up: bool, **_kwargs: object):
        calls.append((setup_id, desired_up))
        return {"ok": True, "changed": False}

    monkeypatch.setattr(pp, "_set_jetson_state_locked", locked)
    result = pp.set_jetson_state(SETUP_ID, False, registry_path=path)
    assert result["ok"] is True
    assert calls == [(SETUP_ID, False)]


@pytest.mark.parametrize("kind", ["jetson", "m2"])
def test_compatibility_toggle_selects_target_from_observation_under_same_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    inside_lock = {"value": False}

    @contextmanager
    def tracked_lock(_setup_id: str, **_kwargs: object):
        inside_lock["value"] = True
        try:
            yield
        finally:
            inside_lock["value"] = False

    monkeypatch.setattr(pp, "platform_operation_lock", tracked_lock)
    probes: list[bool] = []

    def probe(*_args: object, **_kwargs: object) -> pp.PlatformStatus:
        probes.append(inside_lock["value"])
        return _status(ssh=True, m2=True)

    monkeypatch.setattr(pp, "probe_platform_status", probe)
    targets: list[bool] = []
    if kind == "jetson":
        monkeypatch.setattr(
            pp,
            "_set_jetson_state_locked",
            lambda _sid, desired, **_kw: targets.append(bool(desired))
            or {"ok": True},
        )
        pp.toggle_jetson(SETUP_ID, registry_path=path)
    else:
        monkeypatch.setattr(
            pp,
            "_set_m2_state_locked",
            lambda _sid, desired, **_kw: targets.append(bool(desired))
            or {"ok": True},
        )
        pp.toggle_m2(SETUP_ID, registry_path=path)

    assert probes == [True]
    assert targets == [False]
