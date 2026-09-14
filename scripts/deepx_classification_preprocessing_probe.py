#!/usr/bin/env python3
"""Run the paired DeepX classification preprocessing control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.deepx.preprocessing_probe import run_paired_probe


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare scale-only and ImageNet mean/std preprocessing on the same "
            "float ONNX and the exact same ordered validation samples."
        )
    )
    parser.add_argument("--model", required=True, help="Source classification ONNX")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--validation-manifest",
        help=(
            "Prefer the suite-local 500-image manifest.json from an EvaluationRun; "
            "this guarantees exact sample parity with the hardware canary."
        ),
    )
    source.add_argument(
        "--run-dir",
        help=(
            "EvaluationRun containing one suite-local model-bound classification "
            "validation manifest; it is resolved fail-closed."
        ),
    )
    parser.add_argument("--model-id", default="resnet50", help="Exact model ID inside the run")
    parser.add_argument("--limit", type=int, default=32, help="First N samples in manifest order; distinct from expected count")
    parser.add_argument("--output-name", help="Bound ONNX classification output")
    parser.add_argument("--class-count", type=int)
    parser.add_argument("--raw-logits", help="New bounded NPZ, no model artifacts")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--expected-images", type=int, default=500)
    parser.add_argument(
        "--verify-content",
        action="store_true",
        help="Re-hash every image that carries a declared SHA-256",
    )
    args = parser.parse_args()

    if not args.model_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.model_id):
        parser.error("invalid model-id")
    if not 1 <= args.limit <= 32: parser.error("limit must be between 1 and 32")
    validation_manifest = args.validation_manifest
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        manifests = sorted({
            path.resolve()
            for path in run_dir.glob(
                f"models/{args.model_id}/benchmark_set/legacy_suite/resources/"
                "validation/classification/*/manifest.json"
            )
            if path.is_file()
        })
        if len(manifests) != 1:
            parser.error(
                "--run-dir must contain exactly one suite-local model-bound "
                f"classification manifest; found {len(manifests)}"
            )
        validation_manifest = str(manifests[0])
    payload = run_paired_probe(
        model_path=args.model,
        manifest_path=str(validation_manifest),
        expected_images=int(args.expected_images),
        verify_content=bool(args.verify_content), limit=args.limit, model_id=args.model_id,
        output_name=args.output_name, class_count=args.class_count, raw_logits_path=args.raw_logits,
    )
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "output": str(output.resolve()),
        "dataset": payload["dataset"],
        "modes": payload["modes"],
        "paired": payload["paired"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
