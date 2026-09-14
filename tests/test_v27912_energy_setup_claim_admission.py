from __future__ import annotations

import json
import hashlib
import threading
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool.benchmark import remote_run
from onnx_splitpoint_tool.benchmark.remote_run import RemoteBenchmarkArgs
from onnx_splitpoint_tool.energy import collector
from onnx_splitpoint_tool.energy.config import DuplicateEnergySetupIdError
from onnx_splitpoint_tool.energy.config import EnergyDefaults, EnergySetup
from onnx_splitpoint_tool.workflow.hardware_matrix import _setup_to_target
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig


SETUP_ID = "orin_nx_hailo8_01"


def _duplicate_registry() -> dict[str, object]:
    row = {
        "id": SETUP_ID,
        "accelerator": "hailo8",
        "host": {"address": "192.0.2.20", "user": "nx", "port": 22},
        "energy": {"enabled": True, "urecs_address": "192.0.2.10"},
    }
    return {
        "energy_defaults": {"data_port": 3000},
        "hardware_setups": [row, dict(row)],
    }


def test_collector_duplicate_setup_admission_starts_no_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[bool] = []
    monkeypatch.setattr(
        collector, "load_hardware_registry", lambda *_args, **_kwargs: _duplicate_registry()
    )
    monkeypatch.setattr(
        collector,
        "run_fast_firmware_measurement",
        lambda *_args, **_kwargs: started.append(True),
    )
    with pytest.raises(DuplicateEnergySetupIdError, match="duplicate hardware"):
        collector.test_fast_firmware_sleep(
            SETUP_ID,
            out_dir=tmp_path / "must-not-exist",
        )
    assert started == []
    assert not (tmp_path / "must-not-exist").exists()


def test_remote_benchmark_duplicate_setup_admission_starts_no_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    benchmark_set = suite / "benchmark_set.json"
    benchmark_set.write_text(
        json.dumps({"model_name": "resnet50"}), encoding="utf-8"
    )
    (suite / "benchmark_plan.json").write_text(
        json.dumps(
            {
                "model_suite": {"primary": [{"id": "resnet50"}]},
                "runs": [{"id": "ort_cpu", "type": "ort", "provider": "cpu"}],
            }
        ),
        encoding="utf-8",
    )
    from onnx_splitpoint_tool.energy import config as energy_config

    monkeypatch.setattr(
        energy_config,
        "load_hardware_registry",
        lambda *_args, **_kwargs: _duplicate_registry(),
    )
    transport_started: list[bool] = []

    class ForbiddenTransport:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            transport_started.append(True)
            raise AssertionError("transport must not start")

    monkeypatch.setattr(remote_run, "SSHTransport", ForbiddenTransport)
    with pytest.raises(DuplicateEnergySetupIdError, match="duplicate hardware"):
        remote_run.run_remote_benchmark(
            host=HostConfig(
                id=SETUP_ID,
                label="Hailo setup",
                host="192.0.2.20",
                user="nx",
            ),
            benchmark_set_json=benchmark_set,
            local_working_dir=tmp_path / "working",
            run_id="duplicate-energy-admission",
            args=RemoteBenchmarkArgs(
                resume=False,
                energy_enabled=True,
                energy_setup_id=SETUP_ID,
                energy_physical_scope="FS",
                energy_window_label="command",
            ),
            log=lambda _line: None,
            progress=lambda *_args: None,
            cancel_event=threading.Event(),
        )
    assert transport_started == []
    assert not (tmp_path / "working").exists()


def _valid_claim_setup() -> EnergySetup:
    return EnergySetup(
        setup_id=SETUP_ID,
        accelerator="hailo8",
        jetson_address="192.0.2.20",
        jetson_user="nx",
        jetson_port=22,
        jetson_ssh_extra_args="-o StrictHostKeyChecking=yes",
        jetson_identity_valid=True,
        enabled=True,
        urecs_address="192.0.2.10",
        data_port=3000,
        data_port_valid=True,
        calibration_manifest="/verified/method.json",
        calibration_sha256="a" * 64,
        expected_channel_bindings=(
            {
                "setup_id": SETUP_ID,
                "urecs_address": "192.0.2.10",
                "data_port": 3000,
                "channel": 0,
                "sample_rate_hz": 2000,
                "scope": "FS",
                "measurement_point": "complete_system_input",
            },
        ),
        expected_channel_bindings_valid=True,
        hardware_registry_path="/verified/hardware_setups.yaml",
        hardware_registry_snapshot_sha256="b" * 64,
        hardware_registry_provenance_valid=True,
    )


def _write_claim_registry(path: Path, *, address: str = "192.0.2.20") -> dict:
    method = path.parent / "energy_calibration_manifest.json"
    method.parent.mkdir(parents=True, exist_ok=True)
    method.write_text("{}\n", encoding="utf-8")
    method_sha = hashlib.sha256(method.read_bytes()).hexdigest()
    payload = {
        "schema": "onnx-splitpoint/hardware-setups",
        "schema_version": 2,
        "energy_defaults": {
            "enabled": True,
            "collector_binary": "urecs-data-collector",
            "power_calculations_binary": "power_calculations",
            "mode": "fast_firmware",
            "data_port": 3000,
            "channel": 0,
            "sample_rate": 2000,
            "physical_scope": "FS",
            "window_label": "command",
        },
        "hardware_setups": [
            {
                "id": SETUP_ID,
                "accelerator": "hailo8",
                "host": {
                    "address": address,
                    "user": "nx",
                    "port": 22,
                    "ssh_extra_args": "-o StrictHostKeyChecking=yes",
                },
                "energy": {
                    "enabled": True,
                    "urecs_address": "192.0.2.10",
                    "calibration_manifest": str(method.resolve()),
                    "calibration_sha256": method_sha,
                },
            }
        ],
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return payload


def _patch_valid_energy_admission(
    monkeypatch: pytest.MonkeyPatch,
    *,
    method_verified: bool = True,
) -> None:
    from onnx_splitpoint_tool.energy import config as energy_config
    from onnx_splitpoint_tool.energy import method_manifest

    registry = {"fresh": True}
    monkeypatch.setattr(
        energy_config, "load_hardware_registry", lambda *_args, **_kwargs: registry
    )
    monkeypatch.setattr(
        energy_config,
        "hardware_registry_snapshot_sha256",
        lambda *_args, **_kwargs: "b" * 64,
    )
    monkeypatch.setattr(
        energy_config,
        "energy_setup_from_registry",
        lambda *_args, **_kwargs: _valid_claim_setup(),
    )
    monkeypatch.setattr(
        energy_config,
        "energy_defaults_from_registry",
        lambda *_args, **_kwargs: EnergyDefaults(
            registry_contract_valid=True,
            physical_scope="FS",
            window_label="command",
        ),
    )
    monkeypatch.setattr(
        method_manifest,
        "verify_configured_energy_method",
        lambda *_args, **_kwargs: {
            "verified": method_verified,
            "status": (
                "inherited_validated_method_verified"
                if method_verified
                else "inherited_method_manifest_invalid"
            ),
            "runtime_binding_id": (
                "runtime-binding-test" if method_verified else ""
            ),
            "source_integrity_verification": {
                "ok": method_verified,
                "status": (
                    "verified"
                    if method_verified
                    else "source_integrity_binding_failed"
                ),
                "errors": [],
            },
        },
    )


@pytest.mark.parametrize(
    "host",
    [
        HostConfig(id=SETUP_ID, label="wrong address", host="192.0.2.21", user="nx", port=22, ssh_extra_args="-o StrictHostKeyChecking=yes"),
        HostConfig(id=SETUP_ID, label="wrong user", host="192.0.2.20", user="root", port=22, ssh_extra_args="-o StrictHostKeyChecking=yes"),
        HostConfig(id=SETUP_ID, label="wrong port", host="192.0.2.20", user="nx", port=2222, ssh_extra_args="-o StrictHostKeyChecking=yes"),
        HostConfig(id=SETUP_ID, label="wrong args", host="192.0.2.20", user="nx", port=22, ssh_extra_args="-o StrictHostKeyChecking=no"),
        HostConfig(id=SETUP_ID, label="padded host", host="192.0.2.20 ", user="nx", port=22, ssh_extra_args="-o StrictHostKeyChecking=yes"),
        HostConfig(id=SETUP_ID, label="spaced user", host="192.0.2.20", user="n x", port=22, ssh_extra_args="-o StrictHostKeyChecking=yes"),
    ],
)
def test_remote_energy_host_mismatch_starts_no_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: HostConfig,
) -> None:
    _patch_valid_energy_admission(monkeypatch)
    started: list[bool] = []

    class ForbiddenTransport:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            started.append(True)
            raise AssertionError("transport must not start")

    monkeypatch.setattr(remote_run, "SSHTransport", ForbiddenTransport)
    suite = tmp_path / "suite"
    suite.mkdir()
    benchmark_set = suite / "benchmark_set.json"
    benchmark_set.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="remote energy admission failed"):
        remote_run.run_remote_benchmark(
            host=host,
            benchmark_set_json=benchmark_set,
            local_working_dir=tmp_path / "working",
            run_id="wrong-host",
            args=RemoteBenchmarkArgs(
                resume=False,
                energy_enabled=True,
                energy_setup_id=SETUP_ID,
                energy_physical_scope="FS",
                energy_window_label="command",
            ),
            log=lambda _line: None,
            progress=lambda *_args: None,
            cancel_event=threading.Event(),
        )
    assert started == []
    assert not (tmp_path / "working").exists()


def test_remote_energy_unverified_method_starts_no_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_valid_energy_admission(monkeypatch, method_verified=False)
    monkeypatch.setattr(
        remote_run,
        "SSHTransport",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("transport must not start")
        ),
    )
    suite = tmp_path / "suite"
    suite.mkdir()
    benchmark_set = suite / "benchmark_set.json"
    benchmark_set.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="configured_energy_method_not_verified"):
        remote_run.run_remote_benchmark(
            host=HostConfig(
                id=SETUP_ID,
                label="valid",
                host="192.0.2.20",
                user="nx",
                port=22,
                ssh_extra_args="-o StrictHostKeyChecking=yes",
            ),
            benchmark_set_json=benchmark_set,
            local_working_dir=tmp_path / "working",
            run_id="unverified-method",
            args=RemoteBenchmarkArgs(
                resume=False,
                energy_enabled=True,
                energy_setup_id=SETUP_ID,
                energy_physical_scope="FS",
                energy_window_label="command",
            ),
            log=lambda _line: None,
            progress=lambda *_args: None,
            cancel_event=threading.Event(),
        )


@pytest.mark.parametrize("bad_enabled", [0, 1, "", "true", None])
def test_remote_energy_enabled_requires_literal_boolean_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_enabled: object,
) -> None:
    monkeypatch.setattr(
        remote_run,
        "SSHTransport",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("transport must not start")
        ),
    )
    suite = tmp_path / "suite"
    suite.mkdir()
    benchmark_set = suite / "benchmark_set.json"
    benchmark_set.write_text("{}", encoding="utf-8")
    args = RemoteBenchmarkArgs(resume=False)
    args.energy_enabled = bad_enabled  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="energy_enabled_not_literal_boolean"):
        remote_run.run_remote_benchmark(
            host=HostConfig(id="setup", label="setup", host="192.0.2.20"),
            benchmark_set_json=benchmark_set,
            local_working_dir=tmp_path / "working",
            run_id="malformed-enable",
            args=args,
            log=lambda _line: None,
            progress=lambda *_args: None,
            cancel_event=threading.Event(),
        )


@pytest.mark.parametrize(
    ("scope", "window", "reason"),
    [
        ("", "command", "energy_physical_scope_not_exact_full_system"),
        (False, "command", "energy_physical_scope_not_exact_full_system"),
        (0, "command", "energy_physical_scope_not_exact_full_system"),
        ("MB", "command", "energy_physical_scope_not_exact_full_system"),
        ("FS ", "command", "energy_physical_scope_not_exact_full_system"),
        ("FS", False, "energy_window_label_not_exact_command"),
        ("FS", 0, "energy_window_label_not_exact_command"),
        ("FS", "", "energy_window_label_not_exact_command"),
        ("FS", "command_window", "energy_window_label_not_exact_command"),
        ("FS", " command", "energy_window_label_not_exact_command"),
    ],
)
def test_remote_inherited_claim_scope_and_window_fail_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: object,
    window: object,
    reason: str,
) -> None:
    _patch_valid_energy_admission(monkeypatch)
    monkeypatch.setattr(
        remote_run,
        "SSHTransport",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("transport must not start")
        ),
    )
    suite = tmp_path / "suite"
    suite.mkdir()
    benchmark_set = suite / "benchmark_set.json"
    benchmark_set.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match=reason):
        remote_run.run_remote_benchmark(
            host=HostConfig(
                id=SETUP_ID,
                label="valid",
                host="192.0.2.20",
                user="nx",
                port=22,
                ssh_extra_args="-o StrictHostKeyChecking=yes",
            ),
            benchmark_set_json=benchmark_set,
            local_working_dir=tmp_path / "working",
            run_id="bad-method-identity",
            args=RemoteBenchmarkArgs(
                resume=False,
                energy_enabled=True,
                energy_setup_id=SETUP_ID,
                energy_physical_scope=scope,
                energy_window_label=window,
            ),
            log=lambda _line: None,
            progress=lambda *_args: None,
            cancel_event=threading.Event(),
        )


def test_remote_energy_omitted_scope_and_window_derive_from_registry_then_reach_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_valid_energy_admission(monkeypatch)
    reached: list[HostConfig] = []

    class ReachedTransport:
        def __init__(self, host: HostConfig, **_kwargs: object) -> None:
            reached.append(host)
            raise RuntimeError("TEST_TRANSPORT_REACHED")

    monkeypatch.setattr(remote_run, "SSHTransport", ReachedTransport)
    suite = tmp_path / "suite"
    suite.mkdir()
    benchmark_set = suite / "benchmark_set.json"
    benchmark_set.write_text(
        json.dumps({"model_name": "resnet50"}), encoding="utf-8"
    )
    (suite / "benchmark_plan.json").write_text(
        json.dumps(
            {
                "model_suite": {"primary": [{"id": "resnet50"}]},
                "runs": [
                    {"id": "ort_cpu", "type": "ort", "provider": "cpu"}
                ],
            }
        ),
        encoding="utf-8",
    )
    args = RemoteBenchmarkArgs(
        resume=False,
        energy_enabled=True,
        energy_setup_id=SETUP_ID,
    )
    with pytest.raises(RuntimeError, match="TEST_TRANSPORT_REACHED"):
        remote_run.run_remote_benchmark(
            host=HostConfig(
                id=SETUP_ID,
                label="valid",
                host="192.0.2.20",
                user="nx",
                port=22,
                ssh_extra_args="-o StrictHostKeyChecking=yes",
            ),
            benchmark_set_json=benchmark_set,
            local_working_dir=tmp_path / "working",
            run_id="valid-default-method-identity",
            args=args,
            log=lambda _line: None,
            progress=lambda *_args: None,
            cancel_event=threading.Event(),
        )
    assert len(reached) == 1
    assert args.energy_physical_scope == "FS"
    assert args.energy_window_label == "command"


def test_remote_energy_custom_registry_reaches_transport_without_default_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onnx_splitpoint_tool.energy import config as energy_config
    from onnx_splitpoint_tool.energy import method_manifest

    custom_path = tmp_path / "custom" / "hardware_setups.yaml"
    custom_path.parent.mkdir(parents=True)
    _write_claim_registry(custom_path)
    default_path = tmp_path / "default" / "hardware_setups.yaml"
    default_path.parent.mkdir(parents=True)
    _write_claim_registry(default_path, address="192.0.2.99")
    real_load = energy_config.load_hardware_registry
    loaded: list[Path] = []

    def load_recorded(path: str | Path | None = None) -> dict:
        selected = Path(path or default_path).resolve()
        loaded.append(selected)
        return real_load(selected)

    monkeypatch.setattr(energy_config, "load_hardware_registry", load_recorded)
    monkeypatch.setattr(energy_config, "default_registry_path", lambda: default_path)
    monkeypatch.setattr(
        method_manifest,
        "verify_configured_energy_method",
        lambda _setup_id, **_kwargs: {
            "verified": True,
            "status": "inherited_validated_method_verified",
            "runtime_binding_id": SETUP_ID,
            "source_integrity_verification": {
                "ok": True,
                "status": "verified",
                "errors": [],
            },
        },
    )
    reached: list[bool] = []

    class ReachedTransport:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            reached.append(True)
            raise RuntimeError("CUSTOM_TRANSPORT_REACHED")

    monkeypatch.setattr(remote_run, "SSHTransport", ReachedTransport)
    suite = tmp_path / "suite"
    suite.mkdir()
    benchmark_set = suite / "benchmark_set.json"
    benchmark_set.write_text("{}", encoding="utf-8")
    args = RemoteBenchmarkArgs(
        resume=False,
        energy_enabled=True,
        energy_setup_id=SETUP_ID,
        energy_registry_path=str(custom_path),
        energy_physical_scope="FS",
        energy_window_label="command",
    )
    with pytest.raises(RuntimeError, match="CUSTOM_TRANSPORT_REACHED"):
        remote_run.run_remote_benchmark(
            host=HostConfig(
                id=SETUP_ID,
                label="custom",
                host="192.0.2.20",
                user="nx",
                port=22,
                ssh_extra_args="-o StrictHostKeyChecking=yes",
            ),
            benchmark_set_json=benchmark_set,
            local_working_dir=tmp_path / "working",
            run_id="custom-registry",
            args=args,
            log=lambda _line: None,
            progress=lambda *_args: None,
            cancel_event=threading.Event(),
        )
    assert reached == [True]
    assert loaded and set(loaded) == {custom_path.resolve()}
    assert default_path.resolve() not in loaded
    assert args.energy_registry_path == str(custom_path.resolve())
    assert args.energy_registry_snapshot_sha256 == (
        energy_config.hardware_registry_snapshot_sha256(
            real_load(custom_path)
        )
    )


def _write_remote_energy_resume_candidate(
    *,
    root: Path,
    suite: Path,
    benchmark_set: Path,
    host: HostConfig,
    args: RemoteBenchmarkArgs,
    registry_path: str,
    registry_sha256: str,
) -> Path:
    candidate = root / "Results" / suite.name / "1" / "energy-partial"
    results = candidate / "results"
    results.mkdir(parents=True)
    (candidate / "run_meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": candidate.name,
                "repeat_idx": "1",
                "started_at": "2026-09-01T00:00:00Z",
                "benchmark_set_json": str(benchmark_set.resolve()),
                "suite_semantic_cache_key": (
                    remote_run._stable_suite_cache_key(suite)
                ),
                "host": {
                    "user": host.user,
                    "host": host.host,
                    "port": host.port,
                },
                "args": {
                    "provider": args.provider,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "repeats": args.repeats,
                    "add_args": args.add_args,
                    "energy_enabled": True,
                    "energy_setup_id": args.energy_setup_id,
                    "energy_registry_path": registry_path,
                    "energy_registry_snapshot_sha256": registry_sha256,
                },
                "energy_registry_binding": {
                    "schema": (
                        "onnx-splitpoint/remote-energy-registry-binding"
                    ),
                    "schema_version": 1,
                    "energy_enabled": True,
                    "setup_id": args.energy_setup_id,
                    "path": registry_path,
                    "snapshot_sha256": registry_sha256,
                    "verified": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (candidate / "run_status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "partial",
                "ended_at": "2026-09-01T00:01:00Z",
            }
        ),
        encoding="utf-8",
    )
    (results / "benchmark_results_energy.json").write_text(
        json.dumps({"results": [{"status": "partial"}]}),
        encoding="utf-8",
    )
    return candidate


def _remote_energy_resume_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, HostConfig, RemoteBenchmarkArgs]:
    suite = tmp_path / "suite-resume"
    suite.mkdir()
    benchmark_set = suite / "benchmark_set.json"
    benchmark_set.write_text("{}\n", encoding="utf-8")
    (suite / "benchmark_plan.json").write_text(
        json.dumps({"runs": []}), encoding="utf-8"
    )
    working = tmp_path / "working-resume"
    host = HostConfig(
        id=SETUP_ID,
        label="resume setup",
        host="192.0.2.20",
        user="nx",
        port=22,
    )
    args = RemoteBenchmarkArgs(
        provider="auto",
        repeats=1,
        warmup=10,
        iters=100,
        add_args="",
        resume=True,
        energy_enabled=True,
        energy_setup_id=SETUP_ID,
        energy_registry_path="/registry/current.yaml",
        energy_registry_snapshot_sha256="b" * 64,
    )
    return suite, benchmark_set, working, host, args


@pytest.mark.parametrize(
    ("candidate_path", "candidate_sha256"),
    [
        ("/registry/previous.yaml", "b" * 64),
        ("/registry/current.yaml", "a" * 64),
    ],
)
def test_remote_energy_resume_rejects_registry_path_or_snapshot_drift(
    tmp_path: Path,
    candidate_path: str,
    candidate_sha256: str,
) -> None:
    suite, benchmark_set, working, host, args = (
        _remote_energy_resume_fixture(tmp_path)
    )
    _write_remote_energy_resume_candidate(
        root=working,
        suite=suite,
        benchmark_set=benchmark_set,
        host=host,
        args=args,
        registry_path=candidate_path,
        registry_sha256=candidate_sha256,
    )
    assert remote_run._find_resumable_local_run(
        local_working_dir=working,
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        repeat_dir="1",
        host=host,
        args=args,
    ) is None


def test_remote_energy_resume_accepts_identical_registry_binding(
    tmp_path: Path,
) -> None:
    suite, benchmark_set, working, host, args = (
        _remote_energy_resume_fixture(tmp_path)
    )
    candidate = _write_remote_energy_resume_candidate(
        root=working,
        suite=suite,
        benchmark_set=benchmark_set,
        host=host,
        args=args,
        registry_path=args.energy_registry_path,
        registry_sha256=args.energy_registry_snapshot_sha256,
    )
    matched = remote_run._find_resumable_local_run(
        local_working_dir=working,
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        repeat_dir="1",
        host=host,
        args=args,
    )
    assert matched is not None
    assert matched[0] == candidate
    assert matched[1]["args"]["energy_registry_path"] == (
        args.energy_registry_path
    )
    assert matched[1]["args"]["energy_registry_snapshot_sha256"] == (
        args.energy_registry_snapshot_sha256
    )


def test_remote_energy_resume_rejects_inconsistent_registry_metadata_alias(
    tmp_path: Path,
) -> None:
    suite, benchmark_set, working, host, args = (
        _remote_energy_resume_fixture(tmp_path)
    )
    candidate = _write_remote_energy_resume_candidate(
        root=working,
        suite=suite,
        benchmark_set=benchmark_set,
        host=host,
        args=args,
        registry_path=args.energy_registry_path,
        registry_sha256=args.energy_registry_snapshot_sha256,
    )
    metadata = json.loads(
        (candidate / "run_meta.json").read_text(encoding="utf-8")
    )
    metadata["energy_registry_binding"]["snapshot_sha256"] = "a" * 64
    (candidate / "run_meta.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    assert remote_run._find_resumable_local_run(
        local_working_dir=working,
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        repeat_dir="1",
        host=host,
        args=args,
    ) is None


def test_remote_energy_stale_registry_snapshot_starts_no_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onnx_splitpoint_tool.energy import config as energy_config

    custom_path = tmp_path / "custom" / "hardware_setups.yaml"
    custom_path.parent.mkdir(parents=True)
    current = _write_claim_registry(custom_path)
    real_load = energy_config.load_hardware_registry
    stale = real_load(custom_path)
    current["operator_note"] = "changed-after-first-snapshot"
    custom_path.write_text(
        yaml.safe_dump(current, sort_keys=False), encoding="utf-8"
    )
    calls = {"count": 0}

    def load_stale_then_fresh(path: str | Path | None = None) -> dict:
        calls["count"] += 1
        if calls["count"] == 1:
            return stale
        return real_load(path)

    monkeypatch.setattr(
        energy_config, "load_hardware_registry", load_stale_then_fresh
    )
    monkeypatch.setattr(
        remote_run,
        "SSHTransport",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("transport must not start")
        ),
    )
    suite = tmp_path / "suite-stale"
    suite.mkdir()
    benchmark_set = suite / "benchmark_set.json"
    benchmark_set.write_text("{}", encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="hardware_registry_snapshot_stale_or_mismatched",
    ):
        remote_run.run_remote_benchmark(
            host=HostConfig(
                id=SETUP_ID,
                label="stale",
                host="192.0.2.20",
                user="nx",
                port=22,
                ssh_extra_args="-o StrictHostKeyChecking=yes",
            ),
            benchmark_set_json=benchmark_set,
            local_working_dir=tmp_path / "working-stale",
            run_id="stale-registry",
            args=RemoteBenchmarkArgs(
                resume=False,
                energy_enabled=True,
                energy_setup_id=SETUP_ID,
                energy_registry_path=str(custom_path),
                energy_physical_scope="FS",
                energy_window_label="command",
            ),
            log=lambda _line: None,
            progress=lambda *_args: None,
            cancel_event=threading.Event(),
        )
    assert calls["count"] >= 2
    assert not (tmp_path / "working-stale").exists()


@pytest.mark.parametrize(
    "conflict",
    [
        {"remote": {"host": "192.0.2.21"}},
        {"remote": {"user": "root"}},
        {"runtime": {"port": 2222}},
        {"remote_execution": {"ssh_extra_args": "-o BatchMode=no"}},
    ],
)
def test_hardware_matrix_rejects_conflicting_remote_identity_aliases(
    conflict: dict[str, object],
) -> None:
    setup: dict[str, object] = {
        "id": SETUP_ID,
        "accelerator": "hailo8",
        "host": {
            "address": "192.0.2.20",
            "user": "nx",
            "port": 22,
            "ssh_extra_args": "-o BatchMode=yes",
        },
    }
    setup.update(conflict)
    with pytest.raises(ValueError, match="conflicting remote identity aliases"):
        _setup_to_target(setup, build_envs=[], registry_source="test")


def test_gui_energy_matrix_preflight_blocks_primary_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GUI's energy-disabled primary run must stay behind the shared gate."""

    from onnx_splitpoint_tool.gui import app as gui_app

    benchmark_dir = tmp_path / "suite"
    benchmark_dir.mkdir()
    benchmark_set = benchmark_dir / "benchmark_set.json"
    benchmark_set.write_text("{}", encoding="utf-8")
    registry_path = tmp_path / "selected-hardware-setups.yaml"
    registry_path.write_text("hardware_setups: []\n", encoding="utf-8")
    workload_calls: list[str] = []
    preflight_calls: list[tuple[str, Path]] = []

    class ImmediateThread:
        def __init__(self, *, target, **_kwargs: object) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

    class ImmediateRoot:
        def after(self, _delay: int, callback) -> None:
            callback()

    class ForbiddenService:
        def run(self, **_kwargs: object) -> dict[str, object]:
            workload_calls.append("primary")
            raise AssertionError("primary workload must not run")

    def blocked_preflight(*, host, args, registry_path=None, **_kwargs):
        preflight_calls.append((str(args.energy_setup_id), Path(registry_path)))
        raise RuntimeError(
            "remote energy admission failed before transport: "
            "configured_energy_method_source_integrity_not_verified"
        )

    monkeypatch.setattr(gui_app.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(gui_app, "RemoteBenchmarkService", ForbiddenService)
    monkeypatch.setattr(
        gui_app, "preflight_remote_energy_dispatch", blocked_preflight
    )

    gui = object.__new__(gui_app.SplitPointAnalyserGUI)
    gui.root = ImmediateRoot()
    gui._remote_result_group_name = lambda _run_id: "group"
    gui._hardware_setup_remote_payload = lambda _sid: {"provider": "hailo8"}
    gui._hardware_setup_host_config = lambda _sid: HostConfig(
        id=SETUP_ID,
        label="Hailo setup",
        host="192.0.2.20",
        user="nx",
        port=22,
        ssh_extra_args="-o StrictHostKeyChecking=yes",
    )
    gui._hardware_setups_path = lambda: registry_path
    # Simulate the checkbox changing after the immutable dispatch arguments
    # were captured.  Admission must follow ``base_args``, not live GUI state.
    gui._remote_energy_enabled = lambda: False
    gui._jobs_register = lambda **_kwargs: None
    gui._jobs_append_log = lambda *_args, **_kwargs: None
    gui._jobs_set_progress = lambda *_args, **_kwargs: None
    gui._jobs_update_paths = lambda *_args, **_kwargs: None
    gui._jobs_finish = lambda *_args, **_kwargs: None
    gui._set_background_job_active = lambda *_args, **_kwargs: None

    job_id = gui._start_remote_benchmark_matrix_job(
        bench_path=benchmark_set,
        local_working_dir=tmp_path / "working",
        suite_name="gate-test",
        run_id="energy-gate-test",
        dispatches=[{"setup_id": SETUP_ID, "run_id": "hailo8"}],
        base_args=RemoteBenchmarkArgs(
            energy_enabled=True,
            energy_setup_id=SETUP_ID,
            energy_physical_scope="FS",
            energy_window_label="command",
        ),
        show_result_dialogs=False,
    )

    assert job_id == "remote-matrix-energy-gate-test"
    assert preflight_calls == [(SETUP_ID, registry_path)]
    assert workload_calls == []
