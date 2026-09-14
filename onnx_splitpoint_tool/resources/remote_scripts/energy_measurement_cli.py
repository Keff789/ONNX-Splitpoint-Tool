#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import os
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.energy.config import EnergyDefaults, apply_energy_ab_config, default_registry_path, energy_defaults_from_registry, energy_measurements_root, energy_setup_from_registry, get_setup_energy, load_energy_defaults, load_hardware_registry, save_energy_defaults  # noqa: E402
from onnx_splitpoint_tool.energy.collector import check_energy_tools, run_fast_firmware_measurement, select_host_normalization_role  # noqa: E402
from onnx_splitpoint_tool.workflow.hardware_matrix import ensure_hardware_setups_file  # noqa: E402


def _print(data: dict) -> int:
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0 if data.get("ok", False) else 1


def _bind_window_method_ab(defaults: EnergyDefaults, raw_json: str) -> dict:
    """Apply the profile-frozen A/B contract inside the measurement process."""
    text = str(raw_json or "").strip()
    if not text:
        return {}
    try:
        raw = json.loads(text)
    except Exception as exc:
        raise SystemExit(f"--window-method-ab-json is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit("--window-method-ab-json must encode an object")
    resolved = apply_energy_ab_config(defaults, raw)
    if resolved.get("enabled") and not resolved.get("valid"):
        errors = ", ".join(str(item) for item in resolved.get("validation_errors") or [])
        raise SystemExit(f"invalid frozen energy A/B contract: {errors or 'unknown error'}")
    return resolved


def _measurement_context(
    setup_id: str,
    registry_path: str | Path | None = None,
) -> tuple[EnergyDefaults, object]:
    """Resolve a claim-bearing setup and defaults from one registry snapshot.

    ``energy_config.yaml`` remains a legacy/UI mirror.  Once a setup owns any
    configured method identity, it must not be possible for that second file
    to override the registry snapshot used for the method and its channel
    bindings.  Unconfigured legacy/manual calls retain their historical
    standalone-default behaviour.
    """

    explicit_registry_path = registry_path not in (None, "")
    selected_registry_input = Path(
        registry_path or default_registry_path()
    ).expanduser()
    selected_registry_path = selected_registry_input.resolve(strict=False)
    registry = (
        load_hardware_registry(selected_registry_path)
        if explicit_registry_path
        else load_hardware_registry()
    )
    registry_setup = energy_setup_from_registry(
        registry,
        setup_id,
        registry_path=selected_registry_input,
    )
    configured_method_identity = bool(
        getattr(registry_setup, "calibration_manifest", "")
        or getattr(registry_setup, "calibration_sha256", "")
        or getattr(registry_setup, "expected_channel_bindings", ())
    )
    if configured_method_identity:
        return energy_defaults_from_registry(registry), registry_setup
    return (
        load_energy_defaults(),
        get_setup_energy(setup_id, selected_registry_path)
        if explicit_registry_path
        else get_setup_energy(setup_id),
    )




def _hardware_setup(
    setup_id: str,
    registry_path: str | Path | None = None,
) -> dict:
    try:
        reg = load_energy_defaults  # keep lint quiet
        from onnx_splitpoint_tool.energy.config import load_hardware_registry
        data = load_hardware_registry(
            registry_path or default_registry_path()
        )
        for row in list(data.get("hardware_setups") or []):
            if isinstance(row, dict) and str(row.get("id") or "") == str(setup_id):
                return row
    except Exception:
        pass
    return {}


def _remote_from_setup(setup_id: str, explicit: str = "", remote_host: str = "", remote_user: str = "", remote_port: int | None = None, registry_path: str | Path | None = None) -> tuple[str, int]:
    if explicit:
        if "@" in explicit:
            target = explicit
        else:
            target = explicit
        port = int(remote_port or 22)
        if target.count(":") == 1 and not target.startswith("["):
            left, right = target.rsplit(":", 1)
            if right.isdigit():
                target, port = left, int(right)
        return target, port
    row = _hardware_setup(setup_id, registry_path)
    host = dict(row.get("host") or {}) if isinstance(row.get("host"), dict) else {}
    addr = str(remote_host or host.get("address") or "").strip()
    user = str(remote_user or host.get("user") or "nx").strip()
    port = int(remote_port or host.get("port") or 22)
    if not addr:
        raise SystemExit("Remote host is missing. Pass --remote nx@host or configure host.address in hardware_setups.yaml.")
    if "@" in addr:
        target = addr
    else:
        target = f"{user}@{addr}"
    return target, port


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


def _preflight_command_from_args(args: argparse.Namespace) -> str:
    command = str(getattr(args, "preflight_command", "") or "").strip()
    command_file = str(getattr(args, "preflight_command_file", "") or "").strip()
    if command_file:
        path = Path(command_file).expanduser()
        if not path.is_file():
            raise SystemExit(f"--preflight-command-file not found: {path}")
        command = path.read_text(encoding="utf-8").strip()
    return command


def _native_fifo_remote_command(args: argparse.Namespace) -> str:
    target, port = _remote_from_setup(
        args.setup_id,
        args.remote,
        args.remote_host,
        args.remote_user,
        args.remote_port,
        getattr(args, "hardware_setups_file", "") or None,
    )
    remote_tool_dir = str(args.remote_tool_dir or "~/ONNX-Splitpoint-Tool")
    remote_bs = str(args.remote_benchmark_set or args.benchmark_set or "")
    if not remote_bs:
        raise SystemExit("--remote-benchmark-set is required for native FIFO energy measurement")
    remote_cmd_parts = [
        "cd", remote_tool_dir, "&&",
        "python", "scripts/native_hailo_trt_fifo_from_benchmarkset.py",
        "--benchmark-set", remote_bs,
        "--case", args.case,
        "--hw-arch", args.hw_arch,
        "--precision", args.precision,
        "--frames", str(args.frames),
        "--warmup", str(args.warmup),
        "--queue-depth", str(args.queue_depth),
        "--hailo-format", args.hailo_format,
    ]
    if args.dump_outputs:
        remote_cmd_parts.append("--dump-outputs")
    if not getattr(args, "build_during_measurement", False):
        remote_cmd_parts.append("--no-build")
    if args.extra_args:
        remote_cmd_parts.extend(args.extra_args)
    # Use bash -lc remotely so ~ expansion, exports, and optional activation work as users expect.
    remote_body = _shell_join(remote_cmd_parts)
    if args.remote_env:
        remote_body = str(args.remote_env).strip() + " && " + remote_body
    ssh = ["ssh", "-p", str(port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", target, "bash", "-lc", remote_body]
    return _shell_join(ssh)


def cmd_native_fifo_command(args: argparse.Namespace) -> int:
    cmd = _native_fifo_remote_command(args)
    out = Path(args.out).expanduser().resolve() if args.out else Path.cwd() / f"{args.run_id or 'native_fifo'}_command.sh"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + cmd + "\n", encoding="utf-8")
    try:
        out.chmod(0o755)
    except Exception:
        pass
    return _print({"ok": True, "command_file": str(out), "command": cmd})


def cmd_measure_native_fifo(args: argparse.Namespace) -> int:
    defaults, setup = _measurement_context(
        args.setup_id,
        getattr(args, "hardware_setups_file", "") or None,
    )
    _bind_window_method_ab(defaults, getattr(args, "window_method_ab_json", ""))
    if args.out:
        out = Path(args.out).expanduser().resolve()
    else:
        import time as _time
        out = (energy_measurements_root(args.workdir) / "Manual" / args.setup_id / (args.run_id or "native_fifo") / _time.strftime("%Y%m%d_%H%M%S")).resolve()
    command = _native_fifo_remote_command(args)
    # Write the exact command next to the measurement for reproducibility.
    out.mkdir(parents=True, exist_ok=True)
    (out / "native_fifo_remote_command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n", encoding="utf-8")
    try:
        (out / "native_fifo_remote_command.sh").chmod(0o755)
    except Exception:
        pass
    data = run_fast_firmware_measurement(
        command=command,
        out_dir=out,
        setup=setup,
        defaults=defaults,
        setup_id=args.setup_id,
        run_id=args.run_id or "native_fifo",
        cwd=args.cwd,
        duration_s=args.duration,
        run_count=args.runs or defaults.run_count,
        exact_run_count=bool(getattr(args, "exact_run_count", False)),
        timeout_s=args.timeout,
        inference_count=args.inference_count or args.frames,
        pipeline_fps_selected=args.pipeline_fps,
        physical_scope=(
            None if args.physical_scope == "" else args.physical_scope
        ),
        window_label=args.window_label,
        require_runtime_work_units=bool(args.require_runtime_work_units),
        require_command_window_alignment=bool(args.require_command_window_alignment),
        compare_legacy_window=args.compare_legacy_window,
        calibration_manifest=args.calibration_manifest,
        calibration_sha256=args.calibration_sha256,
        preflight_command=_preflight_command_from_args(args),
        preflight_timeout_s=args.preflight_timeout_s,
        preflight_attestation_max_age_s=args.preflight_attestation_max_age_s,
        preflight_runtime_attestation_path=args.preflight_runtime_attestation_path,
        preflight_expected_command_contract_sha256=args.preflight_expected_command_contract_sha256,
        invalid_repeat_max_retries=getattr(args, "invalid_repeat_max_retries", None),
        diagnostic_only=bool(getattr(args, "diagnostic_only", False)),
        claim_exclusion_reason=str(
            getattr(args, "claim_exclusion_reason", "") or ""
        ),
    )
    data.setdefault("native_fifo", {})
    data["native_fifo"].update({
        "remote_command": command,
        "remote_benchmark_set": args.remote_benchmark_set or args.benchmark_set,
        "case": args.case,
        "precision": args.precision,
        "frames": args.frames,
        "warmup": args.warmup,
        "queue_depth": args.queue_depth,
        "hailo_format": args.hailo_format,
        "build_during_measurement": bool(getattr(args, "build_during_measurement", False)),
    })
    try:
        (out / "energy_summary.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return _print(data)

def cmd_init_config(args: argparse.Namespace) -> int:
    p = save_energy_defaults(EnergyDefaults())
    hp = ensure_hardware_setups_file()
    return _print({"ok": True, "energy_config_file": str(p), "hardware_setups_file": str(hp), "energy_measurements_root": str(energy_measurements_root())})


def cmd_check_tools(args: argparse.Namespace) -> int:
    defaults = load_energy_defaults()
    data = check_energy_tools(defaults)
    return _print(data)


def cmd_test_setup(args: argparse.Namespace) -> int:
    defaults, setup = _measurement_context(
        args.setup_id,
        getattr(args, "hardware_setups_file", "") or None,
    )
    data = {
        "setup_id": args.setup_id,
        "energy_enabled": setup.enabled,
        "urecs_address": setup.urecs_address,
        "defaults": defaults.to_dict(),
        "tools": check_energy_tools(defaults),
    }
    if not setup.urecs_address:
        data.update({"ok": False, "reason": "uRECS address missing for this hardware setup"})
        return _print(data)
    if args.acquire:
        out_dir = Path(args.out).expanduser() if args.out else (energy_measurements_root(args.workdir) / "Tests" / args.setup_id / __import__('time').strftime("%Y%m%d_%H%M%S"))
        meas = run_fast_firmware_measurement(
            command=f"sleep {float(args.sleep_s):g}",
            out_dir=out_dir,
            setup=setup,
            defaults=defaults,
            setup_id=args.setup_id,
            run_id="energy_setup_test",
            duration_s=float(args.sleep_s),
            run_count=1,
            exact_run_count=True,
            compare_legacy_window=False,
            timeout_s=float(args.sleep_s) + 120,
        )
        data["measurement"] = meas
        data["ok"] = bool(meas.get("ok"))
    else:
        data["ok"] = bool(data["tools"].get("collector_found") and setup.urecs_address)
        data["note"] = "Acquisition not run. Pass --acquire to perform a short fast-firmware measurement."
    return _print(data)



def cmd_inspect(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser().resolve()
    files = []
    if root.exists():
        for p in sorted(root.rglob("*")):
            if p.is_file():
                files.append({"path": str(p), "relative_path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size})
    interesting = {}
    for name in ("energy_summary.json", "energy_aggregate.json", "artifact_index.json", "duration_probe.json"):
        matches = [str(p) for p in root.rglob(name)] if root.exists() else []
        interesting[name] = matches
    return _print({"ok": root.exists(), "path": str(root), "file_count": len(files), "interesting": interesting, "files": files[-200:]})

def cmd_measure(args: argparse.Namespace) -> int:
    defaults, setup = _measurement_context(
        args.setup_id,
        getattr(args, "hardware_setups_file", "") or None,
    )
    _bind_window_method_ab(defaults, getattr(args, "window_method_ab_json", ""))
    if args.out:
        out = Path(args.out).expanduser().resolve()
    else:
        import time as _time
        out = (energy_measurements_root(args.workdir) / "Manual" / args.setup_id / (args.run_id or "manual_energy_measurement") / _time.strftime("%Y%m%d_%H%M%S")).resolve()
    if args.command_file:
        _cf = Path(args.command_file).expanduser()
        if not _cf.exists():
            raise SystemExit(f"--command-file not found on this host: {_cf}. If the workload runs on a remote NX, create the command file on this measurement host or use the measure-native-fifo subcommand.")
        command = _cf.read_text(encoding="utf-8").strip()
    else:
        command = args.command
    if not command:
        raise SystemExit("--command or --command-file is required")
    normalization_source_run_id = str(
        getattr(args, "host_normalization_source_run_id", "") or ""
    ).strip()
    normalization_target_variant = str(
        getattr(args, "host_normalization_target_variant", "") or ""
    ).strip()
    host_normalization_role = select_host_normalization_role(
        run_id=normalization_source_run_id,
        target_variant=normalization_target_variant,
    )
    data = run_fast_firmware_measurement(
        command=command,
        out_dir=out,
        setup=setup,
        defaults=defaults,
        setup_id=args.setup_id,
        run_id=args.run_id or "manual_energy_measurement",
        cwd=args.cwd,
        duration_s=args.duration,
        run_count=args.runs or defaults.run_count,
        exact_run_count=bool(getattr(args, "exact_run_count", False)),
        timeout_s=args.timeout,
        inference_count=args.inference_count,
        pipeline_fps_selected=args.pipeline_fps,
        physical_scope=(
            None if args.physical_scope == "" else args.physical_scope
        ),
        window_label=args.window_label,
        require_runtime_work_units=bool(args.require_runtime_work_units),
        require_command_window_alignment=bool(args.require_command_window_alignment),
        compare_legacy_window=args.compare_legacy_window,
        calibration_manifest=args.calibration_manifest,
        calibration_sha256=args.calibration_sha256,
        preflight_command=_preflight_command_from_args(args),
        preflight_timeout_s=args.preflight_timeout_s,
        preflight_attestation_max_age_s=args.preflight_attestation_max_age_s,
        preflight_runtime_attestation_path=args.preflight_runtime_attestation_path,
        preflight_expected_command_contract_sha256=args.preflight_expected_command_contract_sha256,
        invalid_repeat_max_retries=getattr(args, "invalid_repeat_max_retries", None),
        diagnostic_only=bool(getattr(args, "diagnostic_only", False)),
        claim_exclusion_reason=str(
            getattr(args, "claim_exclusion_reason", "") or ""
        ),
        host_normalization_role=host_normalization_role,
        host_normalization_source_run_id=normalization_source_run_id,
        host_normalization_target_variant=normalization_target_variant,
    )
    return _print(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="u.RECS fast-firmware energy measurement helper for ONNX Splitpoint Tool")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-config"); p.set_defaults(func=cmd_init_config)
    p = sub.add_parser("check-tools"); p.set_defaults(func=cmd_check_tools)
    p = sub.add_parser("test-setup")
    p.add_argument("--setup-id", required=True)
    p.add_argument("--hardware-setups-file", default="")
    p.add_argument("--acquire", action="store_true")
    p.add_argument("--sleep-s", type=float, default=2.0)
    p.add_argument("--out", default="")
    p.add_argument("--workdir", default="", help="Optional working-directory root; default uses saved GUI Working Dir and writes under EnergyMeasurements/")
    p.set_defaults(func=cmd_test_setup)
    p = sub.add_parser("inspect")
    p.add_argument("path")
    p.set_defaults(func=cmd_inspect)
    p = sub.add_parser("measure")
    p.add_argument("--setup-id", required=True)
    p.add_argument("--hardware-setups-file", default="")
    p.add_argument("--out", default="", help="Output directory. If omitted, writes under <WorkingDir>/EnergyMeasurements/Manual/<setup>/<run>/<timestamp>.")
    p.add_argument("--workdir", default="", help="Optional working-directory root; default uses saved GUI Working Dir")
    p.add_argument("--command", default="")
    p.add_argument("--command-file", default="")
    preflight = p.add_mutually_exclusive_group()
    preflight.add_argument("--preflight-command", default="", help="Command run before every repeat and before u.RECS starts. It and the workload must contain __ONNX_SPLITPOINT_PREFLIGHT_NONCE__.")
    preflight.add_argument("--preflight-command-file", default="", help="File containing the per-repeat preflight command template.")
    p.add_argument("--preflight-timeout-s", type=float, default=300.0)
    p.add_argument("--preflight-attestation-max-age-s", type=float, default=60.0)
    p.add_argument("--preflight-runtime-attestation-path", default="", help="Attestation path as seen by the workload host. Required for SSH/remote workloads when using the ATTESTATION placeholder; may include NONCE/REPEAT tokens.")
    p.add_argument("--preflight-expected-command-contract-sha256", default="", help="Required with preflight; binds its sealed attestation to the exact successful workload contract.")
    p.add_argument("--invalid-repeat-max-retries", type=int, default=None, help="Bounded retries for a logical repeat invalidated by collector marker/first-sample transport (default from Energy Config; normally 1).")
    p.add_argument("--run-id", default="")
    p.add_argument("--cwd", default=None)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--runs", type=int, default=None)
    p.add_argument(
        "--exact-run-count", action="store_true",
        help="Use exactly --runs acquisitions. Intended for callers that manage an outer repeat loop.",
    )
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--inference-count", type=int, default=None)
    p.add_argument("--pipeline-fps", type=float, default=None)
    p.add_argument("--require-runtime-work-units", action="store_true", help="Fail unless every measured window contains an exact runtime completion counter.")
    p.add_argument("--require-command-window-alignment", action="store_true", help="Use/require command-window postprocessing and reject duration-misaligned windows.")
    legacy_group = p.add_mutually_exclusive_group()
    legacy_group.add_argument("--compare-legacy-window", dest="compare_legacy_window", action="store_true", help="Also run the historical power-edge/estimated-duration window on the same trace (diagnostic only).")
    legacy_group.add_argument("--no-compare-legacy-window", dest="compare_legacy_window", action="store_false", help="Disable the diagnostic historical-window comparison.")
    p.set_defaults(compare_legacy_window=None)
    p.add_argument("--calibration-manifest", default="")
    p.add_argument("--calibration-sha256", default="")
    p.add_argument("--physical-scope", default="")
    p.add_argument("--window-label", default="command")
    p.add_argument("--window-method-ab-json", default="", help="Frozen energy.window_method_ab object propagated by the campaign runner.")
    p.add_argument("--diagnostic-only", action="store_true", help="Keep technical energy results but make every scientific claim field ineligible.")
    p.add_argument("--claim-exclusion-reason", default="", help="Stable reason recorded when --diagnostic-only suppresses claims.")
    p.add_argument("--host-normalization-source-run-id", default="", help="Canonical source run identity. Only an exact TensorRT Full identity can select M.2-idle normalization.")
    p.add_argument("--host-normalization-target-variant", default="", help="Canonical target variant. Must be exactly 'full' for TensorRT Full normalization.")
    p.set_defaults(func=cmd_measure)

    # Native FIFO energy helper: builds an SSH command for running the native HailoRT->TRT FIFO fastpath on the NX while measuring from the u.RECS/SM2 host.
    def _add_native_fifo_args(pnf: argparse.ArgumentParser) -> None:
        pnf.add_argument("--setup-id", required=True)
        pnf.add_argument("--hardware-setups-file", default="")
        pnf.add_argument("--remote", default="", help="SSH target such as nx@192.168.0.104 or nx@host:22. If omitted, hardware_setups.yaml host for --setup-id is used.")
        pnf.add_argument("--remote-host", default="")
        pnf.add_argument("--remote-user", default="")
        pnf.add_argument("--remote-port", type=int, default=None)
        pnf.add_argument("--remote-env", default="", help="Optional remote shell prefix, e.g. export PYTHONNOUSERSITE=1; source ~/venvs/hailo10/bin/activate")
        pnf.add_argument("--remote-tool-dir", default="~/ONNX-Splitpoint-Tool")
        pnf.add_argument("--benchmark-set", dest="benchmark_set", default="", help="Alias for --remote-benchmark-set")
        pnf.add_argument("--remote-benchmark-set", default="", help="BenchmarkSet path on the remote NX host")
        pnf.add_argument("--case", default="b066")
        pnf.add_argument("--hw-arch", default="hailo8")
        pnf.add_argument("--precision", default="uint8_cast_fp16")
        pnf.add_argument("--frames", type=int, default=5000)
        pnf.add_argument("--warmup", type=int, default=200)
        pnf.add_argument("--queue-depth", type=int, default=3)
        pnf.add_argument("--hailo-format", default="uint8")
        pnf.add_argument("--dump-outputs", action="store_true")
        pnf.add_argument("--build-during-measurement", action="store_true", help="Allow cmake/build inside measured workload. Default is off and appends --no-build for cleaner inference-only energy windows.")
        pnf.add_argument("--extra-args", nargs="*", default=[])
        preflight = pnf.add_mutually_exclusive_group()
        preflight.add_argument("--preflight-command", default="", help="Command run before every repeat and before u.RECS starts. It and the workload must contain __ONNX_SPLITPOINT_PREFLIGHT_NONCE__.")
        preflight.add_argument("--preflight-command-file", default="", help="File containing the per-repeat preflight command template.")
        pnf.add_argument("--preflight-timeout-s", type=float, default=300.0)
        pnf.add_argument("--preflight-attestation-max-age-s", type=float, default=60.0)
        pnf.add_argument("--preflight-runtime-attestation-path", default="", help="Attestation path as seen by the workload host. Required for SSH/remote workloads when using the ATTESTATION placeholder; may include NONCE/REPEAT tokens.")
        pnf.add_argument("--preflight-expected-command-contract-sha256", default="", help="Required with preflight; binds its sealed attestation to the exact successful workload contract.")
        pnf.add_argument("--invalid-repeat-max-retries", type=int, default=None, help="Bounded retries for a logical repeat invalidated by collector marker/first-sample transport.")
        pnf.add_argument("--run-id", default="yolov7_b066_native_fifo")
        pnf.add_argument("--cwd", default=None)
        pnf.add_argument("--duration", type=float, default=None, help="Known workload duration in seconds. If omitted, a duration probe is run first.")
        pnf.add_argument("--runs", type=int, default=None)
        pnf.add_argument(
            "--exact-run-count", action="store_true",
            help="Use exactly --runs acquisitions when an outer caller owns repetitions.",
        )
        pnf.add_argument("--timeout", type=float, default=None)
        pnf.add_argument("--inference-count", type=int, default=None)
        pnf.add_argument("--pipeline-fps", type=float, default=None)
        pnf.add_argument("--require-runtime-work-units", action="store_true")
        pnf.add_argument("--require-command-window-alignment", action="store_true")
        legacy_group = pnf.add_mutually_exclusive_group()
        legacy_group.add_argument("--compare-legacy-window", dest="compare_legacy_window", action="store_true", help="Also run the historical power-edge/estimated-duration window on the same trace (diagnostic only).")
        legacy_group.add_argument("--no-compare-legacy-window", dest="compare_legacy_window", action="store_false", help="Disable the diagnostic historical-window comparison.")
        pnf.set_defaults(compare_legacy_window=None)
        pnf.add_argument("--calibration-manifest", default="")
        pnf.add_argument("--calibration-sha256", default="")
        pnf.add_argument("--physical-scope", default="")
        pnf.add_argument("--window-label", default="command")
        pnf.add_argument("--window-method-ab-json", default="", help="Frozen energy.window_method_ab object propagated by the campaign runner.")
        pnf.add_argument("--diagnostic-only", action="store_true", help="Keep technical energy results but make every scientific claim field ineligible.")
        pnf.add_argument("--claim-exclusion-reason", default="", help="Stable reason recorded when --diagnostic-only suppresses claims.")
        pnf.add_argument("--out", default="")
        pnf.add_argument("--workdir", default="")

    p = sub.add_parser("make-native-fifo-command", help="Write a local shell command file that SSHes to the NX and runs the native FIFO fastpath.")
    _add_native_fifo_args(p)
    p.set_defaults(func=cmd_native_fifo_command)
    p = sub.add_parser("measure-native-fifo", help="Measure a remote native FIFO run with u.RECS energy collection from this host.")
    _add_native_fifo_args(p)
    p.set_defaults(func=cmd_measure_native_fifo)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
