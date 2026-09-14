#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.energy.config import load_hardware_registry, energy_defaults_from_registry, energy_setup_from_registry, energy_measurements_root
from onnx_splitpoint_tool.energy.collector import check_energy_tools, test_fast_firmware_sleep, run_fast_firmware_measurement, run_duration_probe


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="Check/test u.RECS fast-firmware energy measurement integration.")
    ap.add_argument("--hardware-setups-file", default="", help="Path to hardware_setups.yaml. Default: ~/.onnx_splitpoint_tool/hardware_setups.yaml")
    ap.add_argument("--setup-id", default="", help="Hardware setup id, e.g. orin_nx_deepx_m1_01")
    ap.add_argument("--test", choices=["binaries", "sleep", "command", "probe"], default="binaries")
    ap.add_argument("--sleep", type=float, default=None, help="Run a sleep measurement; implies --test sleep unless --test was set")
    ap.add_argument("--command", default="", help="Command to probe/measure when --test command/probe")
    ap.add_argument("--output-dir", "--out", dest="output_dir", default="", help="Output directory for measurements")
    ap.add_argument("--workdir", default="", help="Optional working-directory root; default uses saved GUI Working Dir")
    ap.add_argument("--known-duration", type=float, default=None)
    ap.add_argument("--run-count", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--collector-binary", default="")
    ap.add_argument("--power-calculations-binary", default="")
    args = ap.parse_args()

    if args.sleep is not None and args.test == "binaries":
        args.test = "sleep"
    if args.command and args.test == "binaries":
        args.test = "command"

    payload = {
        "schema": "onnx-splitpoint/energy-check",
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.executable,
        "cwd": os.getcwd(),
        "test": args.test,
    }
    try:
        registry = load_hardware_registry(args.hardware_setups_file or None)
        defaults = energy_defaults_from_registry(registry)
        if args.collector_binary:
            defaults.collector_binary = args.collector_binary
        if args.power_calculations_binary:
            defaults.power_calculations_binary = args.power_calculations_binary
        tools = check_energy_tools(defaults)
        payload["defaults"] = defaults.to_dict()
        payload["tools"] = tools
        if args.test == "binaries":
            payload["ok"] = bool(tools.get("collector_found"))
            if not tools.get("power_calculations_found"):
                payload["postprocess_warning"] = "power_calculations not found; raw parquet measurements can still be collected"
            _print(payload)
            return 0 if payload["ok"] else 2

        if not args.setup_id:
            raise SystemExit("--setup-id is required for sleep/command/probe tests")
        setup = energy_setup_from_registry(registry, args.setup_id)
        payload["setup"] = setup.to_dict() | {"setup_id": setup.setup_id}
        out_dir = Path(os.path.expandvars(os.path.expanduser(args.output_dir))) if args.output_dir else energy_measurements_root(args.workdir) / "Tests" / args.setup_id / time.strftime("%Y%m%d_%H%M%S")

        if args.test == "probe":
            if not args.command:
                raise SystemExit("--command is required for --test probe")
            res = run_duration_probe(args.command, out_dir, timeout_s=args.timeout, margin_s=defaults.duration_margin_s)
            payload["duration_probe"] = res
            payload["ok"] = bool(res.get("ok"))
            _print(payload)
            return 0 if payload["ok"] else 3
        if args.test == "sleep":
            res = test_fast_firmware_sleep(args.setup_id, out_dir=out_dir, sleep_s=float(args.sleep if args.sleep is not None else 2.0), registry_path=args.hardware_setups_file or None)
        else:
            if not args.command:
                raise SystemExit("--command is required for --test command")
            res = run_fast_firmware_measurement(
                args.command,
                out_dir,
                setup=setup,
                defaults=defaults,
                duration_s=args.known_duration,
                run_count=args.run_count,
                timeout_s=args.timeout,
            )
        payload["measurement"] = res
        payload["ok"] = bool(res.get("ok")) if isinstance(res, dict) else False
        _print(payload)
        return 0 if payload["ok"] else 4
    except Exception as exc:
        payload["ok"] = False
        payload["error"] = f"{type(exc).__name__}: {exc}"
        _print(payload)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
