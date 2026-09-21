from __future__ import annotations

"""Command-line access to the central Smoke / Standard / Final run modes."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .run_modes import (
    default_run_modes_path,
    ensure_run_modes_file,
    load_run_modes_config,
    mode_summary,
    normalize_mode_id,
    reset_run_mode,
    run_mode_display_rows,
    run_modes_revision,
    save_run_modes_config,
    validate_run_modes_config,
)


def _print_status(config: Mapping[str, Any], path: Path, *, as_json: bool = False) -> None:
    payload = {
        "status": "ok",
        "path": str(path),
        "default_mode": str(config.get("default_mode") or "standard"),
        "modes": [],
    }
    for row in run_mode_display_rows(config):
        item = dict(row)
        item["summary"] = mode_summary(str(row["id"]), config)
        payload["modes"].append(item)
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    print(f"Run-mode registry: {path}")
    print(f"Default: {payload['default_mode']}")
    for item in payload["modes"]:
        print()
        print(item["summary"])
        print(f"Recommended for: {item['recommended_for']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onnx-splitpoint-run-modes",
        description="Inspect and maintain the central Smoke / Standard / Final run-mode registry.",
    )
    parser.add_argument("--file", default="", help="Override the central run_modes.yaml path.")
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="Show all effective run modes.")
    status.add_argument("--json", action="store_true")

    show = sub.add_parser("show", help="Show one complete mode definition.")
    show.add_argument("mode", choices=["smoke", "standard", "final"])
    show.add_argument("--json", action="store_true")

    validate = sub.add_parser("validate", help="Validate the central registry.")
    validate.add_argument("--json", action="store_true")

    init = sub.add_parser("init", help="Create the registry when it does not exist.")
    init.add_argument("--force", action="store_true", help="Replace the file with packaged defaults.")

    reset = sub.add_parser("reset", help="Reset one mode to packaged defaults.")
    reset.add_argument("mode", choices=["smoke", "standard", "final"])

    default = sub.add_parser("set-default", help="Select the default mode for new profiles.")
    default.add_argument("mode", choices=["smoke", "standard", "final"])

    sub.add_parser("path", help="Print the active run-mode registry path.")
    integrate = sub.add_parser("integrate-runtime", help="Install reviewed collector and persist narrow Native Standard migration.")
    integrate.add_argument("--collector-source", required=True)
    integrate.add_argument("--backup-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command or "status"
    path = Path(args.file).expanduser().resolve() if str(args.file or "").strip() else default_run_modes_path().resolve()

    try:
        if command == "integrate-runtime":
            import shutil
            from .energy.config import install_reviewed_collector
            from .run_modes import _run_modes_write_lock
            backup = Path(args.backup_dir).expanduser().resolve()
            backup.mkdir(parents=True, exist_ok=True)
            # Save validates/migrates under the existing lock; retain original
            # bytes and ACLs before the first durable configuration write.
            with _run_modes_write_lock(path):
                raw = yaml.safe_load(path.read_text())
                normalized = validate_run_modes_config(raw)
                if raw != normalized:
                    saved = backup / path.name
                    if saved.exists():
                        raise ValueError("Run-mode backup already exists; choose a fresh backup directory")
                    shutil.copy2(path, saved)
            result = install_reviewed_collector(args.collector_source, backup)
            if raw != normalized:
                save_run_modes_config(path, normalized, expected_revision=run_modes_revision(raw), baseline=raw)
            result["run_modes"] = {"path": str(path), "before": raw, "after": normalized}
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if command == "path":
            print(path)
            return 0
        if command == "init":
            if path.exists() and not args.force:
                ensure_run_modes_file(path)
            else:
                from .run_modes import default_run_modes_config
                baseline = load_run_modes_config(path) if path.exists() else None
                save_run_modes_config(path, default_run_modes_config(), expected_revision=run_modes_revision(baseline) if baseline is not None else "", baseline=baseline)
            print(path)
            return 0
        if command == "reset":
            reset_run_mode(normalize_mode_id(args.mode), path)
            print(f"reset {normalize_mode_id(args.mode)}: {path}")
            return 0
        config = load_run_modes_config(path)
        if command == "set-default":
            baseline = copy.deepcopy(config)
            revision = run_modes_revision(config)
            config["default_mode"] = normalize_mode_id(args.mode)
            save_run_modes_config(path, config, expected_revision=revision, baseline=baseline)
            print(f"default_mode={config['default_mode']}")
            return 0
        if command == "validate":
            validated = validate_run_modes_config(config)
            payload = {"status": "ok", "path": str(path), "default_mode": validated["default_mode"], "mode_count": len(validated["modes"])}
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                print(f"OK: {path} ({payload['mode_count']} modes, default={payload['default_mode']})")
            return 0
        if command == "show":
            mid = normalize_mode_id(args.mode)
            mode = config["modes"][mid]
            if args.json:
                print(json.dumps(mode, indent=2, ensure_ascii=False))
            else:
                print(yaml.safe_dump(mode, sort_keys=False, allow_unicode=True).rstrip())
            return 0
        _print_status(config, path, as_json=bool(getattr(args, "json", False)))
        return 0
    except Exception as exc:
        parser.exit(2, f"ERROR: {type(exc).__name__}: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
