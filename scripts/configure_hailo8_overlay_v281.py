#!/usr/bin/env python3
"""Persist the already reviewed H8 GPU overlay, preserving explicit choices."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onnx_splitpoint_tool.hailo_overlay_migration import migrate_configs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tool", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--evidence", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        result = migrate_configs(args.tool, args.output_dir, evidence=args.evidence)
        print("HAILO8_CONFIG_MIGRATION=" + ("PASS" if result["status"] == "pass" else "NOT_REQUIRED"))
        print("DEPENDENCY_MANIFEST=" + (result["dependency_manifest"] or ""))
        print("MIGRATION_REPORT=" + str(args.output_dir / "hailo8_overlay_migration.json"))
        return 0
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "hailo8_overlay_migration.json").write_text(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        print("HAILO8_CONFIG_MIGRATION=FAIL")
        print("ERROR=" + f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
