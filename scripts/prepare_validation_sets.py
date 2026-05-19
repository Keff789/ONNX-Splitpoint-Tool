#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.benchmark.validation_assets import prepare_all_validation_assets, validation_assets_summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prepare/download validation datasets for ONNX Split-Point Tool clean releases.")
    ap.add_argument("--coco50", action="store_true", help="Prepare the COCO-50 detection validation subset")
    ap.add_argument("--imagenette200", action="store_true", help="Prepare downloadable Imagenette mini-200 classification preset")
    ap.add_argument("--imagenette500", action="store_true", help="Prepare downloadable Imagenette mini-500 classification preset")
    ap.add_argument("--all", action="store_true", help="Prepare all built-in downloadable validation sets")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--status", action="store_true", help="Print status only")
    ns = ap.parse_args(argv)
    if ns.status and not ns.coco50 and not ns.imagenette200 and not ns.imagenette500 and not ns.all:
        print(json.dumps(validation_assets_summary(), indent=2))
        return 0
    def _log(msg: str) -> None:
        print(msg, flush=True)
    any_explicit = ns.coco50 or ns.imagenette200 or ns.imagenette500
    result = prepare_all_validation_assets(
        include_coco50=ns.all or ns.coco50 or not any_explicit,
        include_imagenette200=ns.all or ns.imagenette200 or not any_explicit,
        include_imagenette500=ns.all or ns.imagenette500,
        include_test_images=ns.all or not any_explicit,
        overwrite=bool(ns.overwrite),
        log=_log,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
