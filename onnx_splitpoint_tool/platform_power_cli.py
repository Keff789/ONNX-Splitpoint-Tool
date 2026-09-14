"""Command-line interface for u.RECS/Jetson platform power control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .energy.method_manifest import (
    EnergyMethodManifestError,
    prepare_configured_energy_method,
)
from .platform_power import (
    PlatformPowerError,
    calibrate_m2_accelerator_idle_power,
    probe_platform_status,
    registry_setup_choices,
    set_jetson_state,
    set_m2_state,
    toggle_jetson,
    toggle_m2,
)


_FORCE_WORKFLOW_HELP = (
    "DANGEROUS: bypass the legacy workflow-lock scan; the global "
    "EvaluationRun/platform-power interlock remains enforced"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onnx-splitpoint-platform-power",
        description=(
            "Inspect and safely control a configured u.RECS + Jetson platform. "
            "State-changing commands are blocked while a workflow lock is held."
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="hardware_setups.yaml path (default: ~/.onnx_splitpoint_tool/hardware_setups.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List configured hardware setup IDs")
    p_list.set_defaults(action="list")

    p_status = sub.add_parser("status", help="One bounded u.RECS/SSH/M.2 status refresh")
    p_status.add_argument("setup_id")

    p_jetson = sub.add_parser("toggle-jetson", help="Safely toggle the Jetson rail")
    p_jetson.add_argument("setup_id")
    p_jetson.add_argument(
        "--confirm-unknown-off-state",
        action="store_true",
        help="Allow one power-on toggle when SSH is down/ambiguous",
    )
    p_jetson.add_argument(
        "--force-active-workflow",
        action="store_true",
        help=_FORCE_WORKFLOW_HELP,
    )

    p_set_jetson = sub.add_parser(
        "set-jetson",
        help="Set an explicit SSH-observable Jetson target without stale inversion",
    )
    p_set_jetson.add_argument("setup_id")
    p_set_jetson.add_argument("state", choices=("on", "off"))
    p_set_jetson.add_argument(
        "--force-active-workflow", action="store_true", help=_FORCE_WORKFLOW_HELP
    )

    p_m2 = sub.add_parser("toggle-m2", help="Safely shutdown, toggle M.2, reboot and verify")
    p_m2.add_argument("setup_id")
    p_m2.add_argument(
        "--force-active-workflow", action="store_true", help=_FORCE_WORKFLOW_HELP
    )

    p_set = sub.add_parser("set-m2", help="Set M.2 to a verified on/off state")
    p_set.add_argument("setup_id")
    p_set.add_argument("state", choices=("on", "off"))
    p_set.add_argument(
        "--force-active-workflow", action="store_true", help=_FORCE_WORKFLOW_HELP
    )

    p_cal = sub.add_parser(
        "calibrate-m2-idle",
        help="Measure Jetson idle power with M.2 off/on and save accelerator_idle_w",
    )
    p_cal.add_argument("setup_id")
    p_cal.add_argument("--stabilize-seconds", type=float, default=None)
    p_cal.add_argument("--measure-seconds", type=float, default=None)
    p_cal.add_argument("--output-dir", type=Path, default=None)
    p_cal.add_argument(
        "--force-active-workflow", action="store_true", help=_FORCE_WORKFLOW_HELP
    )

    p_method = sub.add_parser(
        "prepare-energy-method",
        help=(
            "Hash, verify and configure the inherited full-system u.RECS "
            "measurement method without touching hardware"
        ),
    )
    p_method.add_argument(
        "--attested-by",
        required=True,
        help="Name of the person accepting reuse of the validated method",
    )
    p_method.add_argument(
        "--accept-validated-method-reuse",
        action="store_true",
        required=True,
        help=(
            "Explicitly attest that the independently validated u.RECS method "
            "and exact hashed implementation are reused without claiming a new calibration"
        ),
    )
    p_method.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New directory for the immutable method spec and manifest",
    )
    p_method.add_argument(
        "--setup-id",
        action="append",
        default=None,
        help=(
            "Target setup ID (repeatable); default: hardware group all_accelerators"
        ),
    )
    return parser


def _progress(text: str) -> None:
    print(str(text), file=sys.stderr, flush=True)


def _dump(payload: Any) -> None:
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = args.registry
    try:
        if args.command == "list":
            _dump(
                [
                    {"setup_id": setup_id, "label": label}
                    for setup_id, label in registry_setup_choices(registry_path=registry)
                ]
            )
        elif args.command == "status":
            _dump(probe_platform_status(args.setup_id, registry_path=registry))
        elif args.command == "toggle-jetson":
            _dump(
                toggle_jetson(
                    args.setup_id,
                    registry_path=registry,
                    force_unknown_off_state=bool(args.confirm_unknown_off_state),
                    force_active_workflow=bool(args.force_active_workflow),
                    callback=_progress,
                )
            )
        elif args.command == "set-jetson":
            _dump(
                set_jetson_state(
                    args.setup_id,
                    args.state == "on",
                    registry_path=registry,
                    force_active_workflow=bool(args.force_active_workflow),
                    callback=_progress,
                )
            )
        elif args.command == "toggle-m2":
            _dump(
                toggle_m2(
                    args.setup_id,
                    registry_path=registry,
                    force_active_workflow=bool(args.force_active_workflow),
                    callback=_progress,
                )
            )
        elif args.command == "set-m2":
            _dump(
                set_m2_state(
                    args.setup_id,
                    args.state == "on",
                    registry_path=registry,
                    force_active_workflow=bool(args.force_active_workflow),
                    callback=_progress,
                )
            )
        elif args.command == "calibrate-m2-idle":
            _dump(
                calibrate_m2_accelerator_idle_power(
                    args.setup_id,
                    registry_path=registry,
                    stabilize_s=args.stabilize_seconds,
                    measure_s=args.measure_seconds,
                    output_dir=args.output_dir,
                    force_active_workflow=bool(args.force_active_workflow),
                    callback=_progress,
                )
            )
        elif args.command == "prepare-energy-method":
            _dump(
                prepare_configured_energy_method(
                    attested_by=args.attested_by,
                    accepted_validated_method_reuse=bool(
                        args.accept_validated_method_reuse
                    ),
                    registry_path=registry,
                    output_dir=args.output_dir,
                    setup_ids=args.setup_id,
                )
            )
        else:  # pragma: no cover - argparse prevents this
            raise AssertionError(args.command)
        return 0
    except (PlatformPowerError, EnergyMethodManifestError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print('{"ok": false, "error": "interrupted"}', file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
