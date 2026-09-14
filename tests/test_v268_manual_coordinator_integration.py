from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from onnx_splitpoint_tool.remote_runtime_closure import (
    native_remote_package_closure,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_legacy_run_manifest(run_dir: Path) -> None:
    """Bind historical coordinator fixtures to their pre-2.69f policy."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run_dir.name,
        "workflow_version": "v2.68-level-playing-field",
        "tool_version": "2.68.0",
    }), encoding="utf-8")


def test_manual_cli_config_keeps_performance_repeats_separate_from_energy() -> None:
    module = _load_script("update_evalset_native_producers.py")
    args = Namespace(
        hailo8_ssh="", hailo10_ssh="", deepx_ssh="",
        hailo8_env="", hailo10_env="", deepx_env="",
        backends="hailo8", case_policy="all_accepted", precision="fp16",
        frames=100, warmup=10, repetitions=3, queue_depth=3, inflight=8,
        hailo_format="uint8", native_letterbox_pad_value=0,
        native_force_rebuild_engines=False, native_dequant_scale=0.0,
        native_dequant_zero_point=0.0, native_boundary_layout="as_input",
        remote_root="/remote/results", remote_tool_dir="/remote/tool",
        dump_outputs=False, native_boundary_debug=False,
        no_build_missing_engines=False, engine_build_python="auto", no_copy=True,
        native_full_baselines=False, native_full_backends="",
        native_full_backends_by_producer="", native_energy=True,
        native_energy_mode="plan", native_energy_timeout=900,
        native_energy_runs=1, native_energy_allow_unpaired=False,
        native_energy_duration_s=60.0, native_validation=False,
        native_validation_mode="dump_and_visual", native_validation_topk=5,
        case_map="", native_telemetry_label="standard_01",
    )

    cfg = module._native_cfg_from_args(args)

    assert cfg["repetitions"] == 3
    assert cfg["energy"]["repeat_override"] == 1
    assert cfg["telemetry_label"] == "standard_01"
    assert module._scaled_performance_timeout(7200, 3) == 21600


def test_variant_command_forwards_repetitions_and_telemetry_label(tmp_path: Path) -> None:
    module = _load_script("run_evalrun_native_producer_variants.py")
    # Current coordinators expand every variant to an exact, frozen case map
    # before constructing the child command.  Keep this command-forwarding
    # fixture runnable under that fail-closed selection contract.
    (tmp_path / "models" / "yolo26s" / "benchmark_set" / "b038").mkdir(
        parents=True,
    )
    variant = {"id": "yolo_raw", "models": ["yolo26s"], "cases": ["b038"]}
    cmd = module._build_update_cmd(
        tmp_path,
        {"backends": ["deepx"], "repetitions": 5},
        variant,
        refresh_suites=False,
        timeout_s=120,
    )

    assert cmd[cmd.index("--repetitions") + 1] == "5"
    assert cmd[cmd.index("--native-telemetry-label") + 1] == "yolo_raw"
    assert cmd[cmd.index("--timeout") + 1] == "120"
    assert "--native-energy-runs" not in cmd


def test_manual_performance_commands_repeat_and_never_inherit_energy_duration(
    tmp_path: Path, monkeypatch,
) -> None:
    module = _load_script("update_evalset_native_producers.py")
    run = tmp_path / "eval"
    _write_legacy_run_manifest(run)
    benchmark_set = run / "models" / "resnet50" / "benchmark_set"
    case = benchmark_set / "b001"
    case.mkdir(parents=True)
    (benchmark_set / "benchmark_set.json").write_text("{}", encoding="utf-8")
    (case / "split_manifest.json").write_text("{}", encoding="utf-8")

    calls: list[tuple[list[str], dict[str, Any]]] = []
    telemetry_calls: list[dict[str, Any]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append((list(cmd), dict(kwargs)))
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_sync_remote_script_v60i", lambda *a, **k: [])
    monkeypatch.setattr(module, "_sync_remote_package_asset_v263", lambda *a, **k: [])
    monkeypatch.setattr(module, "_verify_remote_module_binding_v263", lambda *a, **k: {"rc": 0})
    def fake_telemetry(**kwargs: Any) -> dict[str, Any]:
        telemetry_calls.append(dict(kwargs))
        return {"name": f"capture_{kwargs['phase']}", "rc": 0}

    monkeypatch.setattr(module, "_capture_remote_host_telemetry", fake_telemetry)
    monkeypatch.setattr(
        module, "_summarize_host_telemetry",
        lambda *a, **k: {"name": "summary", "rc": 0},
    )

    stage = module._run_native_producers(run, {
        "backends": ["deepx"],
        "remotes": {"deepx": {"ssh": "jetson", "setup_id": "lab_deepx_custom"}},
        "remote_root": "/remote/results",
        "remote_tool_dir": "/remote/tool",
        "copy_benchmarksets": False,
        "build_missing_engines": False,
        "frames": 100,
        "warmup": 10,
        "repetitions": 3,
        # This test exercises repeat/timeout separation for the DeepX Full
        # row only. TensorRT Full is covered by the strict Quality-FIRST tests
        # and may never run without its central producer set.
        "full_baselines": {
            "enabled": True,
            "backends_by_producer": {"deepx": ["deepx"]},
        },
        # This belongs only to the later energy replay and must never change
        # the exact-frame Full performance command.
        "energy": {"enabled": False, "duration_s": 60.0, "repeat_override": 1},
    }, timeout=10)

    labelled = {str(kwargs.get("label")): (cmd, kwargs) for cmd, kwargs in calls if kwargs.get("label")}
    split_cmd, split_kwargs = labelled["split:deepx"]
    full_cmd, full_kwargs = labelled["full:deepx"]
    split_shell = split_cmd[-1]
    full_shell = full_cmd[-1]
    assert "--repetitions 3" in split_shell
    assert "--repetitions 3" in full_shell
    assert "--duration-s" not in full_shell
    assert "--setup-id lab_deepx_custom" in full_shell
    assert split_kwargs["timeout"] == 30
    assert full_kwargs["timeout"] == 30
    assert stage["performance_repetitions"] == 3
    assert stage["energy_repetitions_independent"] is True
    assert stage["native_validation"]["status"] == "not_applicable"
    assert stage["quality_gated_final_report"]["status"] == "not_applicable"
    assert stage["backend_results"][0]["setup_id"] == "lab_deepx_custom"
    assert telemetry_calls
    assert {call["setup_id"] for call in telemetry_calls} == {"lab_deepx_custom"}


def test_telemetry_capture_and_summary_are_non_blocking(tmp_path: Path, monkeypatch) -> None:
    module = _load_script("update_evalset_native_producers.py")

    def raise_transport(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("ssh unavailable")

    monkeypatch.setattr(module, "_run", raise_transport)
    capture = module._capture_remote_host_telemetry(
        ssh="jetson", env="", remote_tool_dir="/tool", remote_root="/results",
        backend="hailo8", setup_id="jetson_h8", run_id="run", phase="pre",
        label="standard / 1", timeout=60,
    )
    summary = module._summarize_host_telemetry(tmp_path, tmp_path / "reports")

    assert capture["rc"] == -1
    assert capture["non_blocking"] is True
    assert capture["remote_output"].endswith("/host_telemetry/standard___1_pre.json")
    assert "--capture-group standard___1" in capture["cmd_shell"]
    assert capture["capture_group"] == "standard___1"
    assert summary["rc"] == -1
    assert summary["non_blocking"] is True


def test_failure_host_telemetry_is_recovered_to_the_evalrun(tmp_path: Path, monkeypatch) -> None:
    module = _load_script("update_evalset_native_producers.py")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append((list(cmd), dict(kwargs)))
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(module, "_run", fake_run)
    result = module._collect_failure_host_telemetry(
        run_dir=tmp_path,
        ssh="nx@jetson",
        remote_root="/remote/eval/run",
        backend="deepx",
        timeout=600,
    )

    assert result["rc"] == 0
    assert result["name"] == "collect_failure_host_telemetry"
    assert result["non_blocking"] is True
    cmd, kwargs = calls[0]
    assert cmd[:2] == ["rsync", "-a"]
    assert cmd[2] == "nx@jetson:/remote/eval/run/host_telemetry/"
    assert cmd[3].endswith("/native_producers/deepx/host_telemetry/")
    assert kwargs["label"] == "collect-failure-host-telemetry:deepx"


def test_manual_runner_failure_invokes_telemetry_recovery(tmp_path: Path, monkeypatch) -> None:
    module = _load_script("update_evalset_native_producers.py")
    run = tmp_path / "eval"
    _write_legacy_run_manifest(run)
    benchmark_set = run / "models" / "resnet50" / "benchmark_set"
    case = benchmark_set / "b001"
    case.mkdir(parents=True)
    (benchmark_set / "benchmark_set.json").write_text("{}", encoding="utf-8")
    (case / "split_manifest.json").write_text("{}", encoding="utf-8")
    recovered: list[dict[str, Any]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("label") == "split:deepx":
            return {"rc": 9, "stdout_tail": "", "stderr_tail": "runner failed"}
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    def fake_recovery(**kwargs: Any) -> dict[str, Any]:
        recovered.append(dict(kwargs))
        return {"name": "collect_failure_host_telemetry", "rc": 0, "non_blocking": True}

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_sync_remote_script_v60i", lambda *a, **k: [])
    monkeypatch.setattr(module, "_sync_remote_package_asset_v263", lambda *a, **k: [])
    monkeypatch.setattr(module, "_verify_remote_module_binding_v263", lambda *a, **k: {"rc": 0})
    monkeypatch.setattr(
        module, "_capture_remote_host_telemetry",
        lambda **kwargs: {"name": f"capture_{kwargs['phase']}", "rc": 0},
    )
    monkeypatch.setattr(module, "_collect_failure_host_telemetry", fake_recovery)
    monkeypatch.setattr(module, "_summarize_host_telemetry", lambda *a, **k: {"name": "summary", "rc": 0})

    stage = module._run_native_producers(run, {
        "backends": ["deepx"],
        "remotes": {"deepx": {"ssh": "jetson", "setup_id": "deepx_lab"}},
        "remote_root": "/remote/results",
        "remote_tool_dir": "/remote/tool",
        "copy_benchmarksets": False,
        "build_missing_engines": False,
        "frames": 10,
        "warmup": 1,
        "repetitions": 1,
        "full_baselines": {"enabled": False},
        "energy": {"enabled": False},
        "validation": {"enabled": False},
    }, timeout=10)

    assert recovered == [{
        "run_dir": run,
        "ssh": "jetson",
        "remote_root": f"/remote/results/{run.name}",
        "backend": "deepx",
        "timeout": 10,
    }]
    backend = stage["backend_results"][0]
    assert backend["status"] == "partial"
    assert any(step.get("name") == "collect_failure_host_telemetry" for step in backend["steps"])


def test_manual_and_variant_coordinators_rerun_report_with_quality_summary() -> None:
    manual = (ROOT / "scripts" / "update_evalset_native_producers.py").read_text(encoding="utf-8")
    variants = (ROOT / "scripts" / "run_evalrun_native_producer_variants.py").read_text(encoding="utf-8")

    for source in (manual, variants):
        assert '"--quality-summary", str(quality_summary)' in source
        assert '"skipped_fail_closed"' in source
        assert '"not_applicable"' in source
        assert 'native_producer_summary.{ext}' in source


def test_manual_coordinator_stages_runtime_endpoint_attestor() -> None:
    source = (ROOT / "scripts" / "update_evalset_native_producers.py").read_text(encoding="utf-8")
    closure = {path: (module, tokens) for path, module, tokens in native_remote_package_closure()}
    module, tokens = closure["onnx_splitpoint_tool/native_output_endpoint.py"]
    assert module == "onnx_splitpoint_tool.native_output_endpoint"
    assert {
        "def attest_decoded_nms",
        "def runtime_output_contract",
        "def load_manifest_outputs",
    } <= set(tokens)
    assert "native_remote_package_closure()" in source


def test_legacy_evalrun_entry_delegates_with_standard_contract_and_model_filter(
    tmp_path: Path,
) -> None:
    module = _load_script("update_evalrun_native_producers.py")
    run = tmp_path / "eval"
    for model, cases in {"wanted": ["b001", "b002"], "other": ["b003"]}.items():
        benchmark_set = run / "models" / model / "benchmark_set"
        benchmark_set.mkdir(parents=True)
        (benchmark_set / "benchmark_set.json").write_text("{}", encoding="utf-8")
        for case in cases:
            (benchmark_set / case).mkdir()

    ns = Namespace(
        eval_run_dir=str(run), backend=["hailo8"], models="wanted",
        case_map="", case_policy="all_accepted",
        remote_root="/remote/results", remote_tool_dir="/remote/tool",
        hailo8_ssh="jetson", hailo10_ssh="", deepx_ssh="",
        hailo8_env="", hailo10_env="", deepx_env="",
        precision="uint8_cast_fp16", frames=1000, warmup=100,
        repetitions=3, queue_depth=3, inflight=8, hailo_format="uint8",
        native_letterbox_pad_value=0, timeout=7200,
        native_telemetry_label="legacy_standard", no_copy=False,
        no_build_missing_engines=False, dump_outputs=False,
    )

    cmd = module._canonical_delegate_command(ns, run)

    assert Path(cmd[1]).name == "update_evalset_native_producers.py"
    assert "--run-native-producers" in cmd
    assert cmd[cmd.index("--repetitions") + 1] == "3"
    assert cmd[cmd.index("--native-telemetry-label") + 1] == "legacy_standard"
    assert cmd[cmd.index("--case-policy") + 1] == "case_map_only"
    case_map = json.loads(cmd[cmd.index("--case-map") + 1])
    assert case_map == {"wanted": ["b001", "b002"]}


def test_generated_evalset_entry_forwards_repeats_and_telemetry(
    tmp_path: Path, monkeypatch,
) -> None:
    module = _load_script("update_generated_evalset.py")
    run = tmp_path / "eval"
    benchmark_set = run / "models" / "resnet50" / "benchmark_set"
    benchmark_set.mkdir(parents=True)
    (benchmark_set / "benchmark_set.json").write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "update_generated_evalset.py", "--eval-run-dir", str(run),
        "--run-native-producers", "--native-backend", "hailo8",
        "--hailo8-ssh", "jetson",
    ])

    assert module.main() == 0
    cmd = calls[0]
    assert Path(cmd[1]).name == "update_evalrun_native_producers.py"
    assert cmd[cmd.index("--repetitions") + 1] == "3"
    assert cmd[cmd.index("--native-telemetry-label") + 1] == "legacy_generated_standard"


def test_profile_helper_writes_repeated_telemetered_measurement_contract(
    tmp_path: Path, monkeypatch,
) -> None:
    yaml = __import__("yaml")
    module = _load_script("configure_native_producer_profile.py")
    profile = tmp_path / "profile.yaml"
    profile.write_text("models: []\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "configure_native_producer_profile.py", "--profile", str(profile),
        "--backends", "hailo8", "--hailo8-ssh", "jetson",
    ])

    assert module.main() == 0
    cfg = yaml.safe_load(profile.read_text(encoding="utf-8"))["native_producers"]
    assert cfg["repetitions"] == 3
    assert cfg["performance_aggregation"] == "median_ci95"
    assert cfg["telemetry_label"] == "profile_standard"
    assert cfg["host_telemetry"] == {"enabled": True, "mode": "pre_post_non_blocking"}
