#!/usr/bin/env python3
"""Prepare or execute the ordinary leased Native dataset Quality action."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onnx_splitpoint_tool.workflow.native_dataset_quality import prepare_native_quality_action, execute_native_quality_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path, help="Preparation JSON with config_path and normal target/budget fields")
    group.add_argument("--execute", type=Path, help="Prepared ACTION.json")
    args = parser.parse_args()
    result = (prepare_native_quality_action(**json.loads(args.prepare.read_text())) if args.prepare
              else execute_native_quality_action(args.execute))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
