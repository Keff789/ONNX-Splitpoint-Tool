#!/usr/bin/env python3
from __future__ import annotations

# This is a CLI helper, not a pytest module.  The imported collector function
# is named test_fast_firmware_sleep and would otherwise be collected as a test.
__test__ = False
import argparse, json, shlex, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from onnx_splitpoint_tool.energy.config import load_hardware_registry, energy_defaults_from_registry, energy_setup_from_registry
from onnx_splitpoint_tool.energy.collector import check_energy_tools, run_fast_firmware_measurement, test_fast_firmware_sleep


def main() -> int:
    ap = argparse.ArgumentParser(description="u.RECS fast-firmware energy measurement smoke/helper")
    ap.add_argument("--hardware-setups-file", default="", help="Path to hardware_setups.yaml; default ~/.onnx_splitpoint_tool/hardware_setups.yaml")
    ap.add_argument("--setup-id", default="", help="Hardware setup id, e.g. orin_nx_deepx_m1_01")
    ap.add_argument("--out", default="", help="Output directory for measurement files")
    ap.add_argument("--sleep", type=float, default=None, help="Run a built-in sleep command of this many seconds")
    ap.add_argument("--command", default="", help="Command to measure")
    ap.add_argument("--duration", type=float, default=None, help="Known measurement duration in seconds; otherwise probe command first")
    ap.add_argument("--runs", type=int, default=None, help="Number of repeated measurements")
    ap.add_argument("--check-tools", action="store_true", help="Only check collector/power_calculations availability")
    args = ap.parse_args()
    reg = load_hardware_registry(args.hardware_setups_file or None)
    defaults = energy_defaults_from_registry(reg)
    if args.check_tools:
        print(json.dumps(check_energy_tools(defaults), indent=2, ensure_ascii=False))
        return 0
    if not args.setup_id:
        raise SystemExit("--setup-id is required unless --check-tools is used")
    setup = energy_setup_from_registry(reg, args.setup_id)
    out = Path(args.out).expanduser() if args.out else None
    if args.sleep is not None and not args.command:
        result = test_fast_firmware_sleep(args.setup_id, out_dir=out, sleep_s=args.sleep, registry_path=args.hardware_setups_file or None)
    else:
        if not args.command:
            raise SystemExit("--command or --sleep is required")
        result = run_fast_firmware_measurement(args.command, out or Path("energy_measurement"), setup=setup, defaults=defaults, duration_s=args.duration, run_count=args.runs)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 2

if __name__ == "__main__":
    raise SystemExit(main())
