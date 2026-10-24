#!/usr/bin/env python3
"""Verify/copy one closed Generic run; cleanup needs --apply --cleanup + bound authorization."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onnx_splitpoint_tool.remote.collection_recovery import recover_generic_closed_collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'session-id', 'setup-id', 'model-id', 'evidence-path',
                 'authorization-path', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--registry-path')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--cleanup', action='store_true')
    result = recover_generic_closed_collection(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
