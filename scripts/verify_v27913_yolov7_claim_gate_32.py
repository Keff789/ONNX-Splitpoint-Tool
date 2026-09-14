#!/usr/bin/env python3
"""Fail-closed v2.79.13 verifier for retained YOLOv7 claim-gate evidence.

The scientific validation contract and the admitted v2.79.6 measurement are
the frozen v2.79.10 implementation.  This compatibility verifier first pins
that implementation byte-for-byte, then applies only the v2.79.13 release
identity and the exact current snapshot-member identities.  The complete
v2.79.13 source manifest, the retained source snapshot, the prior receipt and
all runtime/evidence identities remain mandatory; inference is never rerun.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


PACKAGE_VERSION = "2.79.13"
BUILD_ID = "v2.79.13-platform-power-calibration-operational-repair"
BASE_VERIFIER = "verify_v27910_yolov7_claim_gate_32.py"
BASE_VERIFIER_SHA256 = (
    "cf9b130e7ab62600d0b1daa933d8b529d460aac178997626a34f6c0ccf79fcd9"
)
VERIFIER_FILENAME = "verify_v27913_yolov7_claim_gate_32.py"


def _load_frozen_base() -> Any:
    path = Path(__file__).resolve().with_name(BASE_VERIFIER)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != BASE_VERIFIER_SHA256:
        raise RuntimeError("v27910_claim_gate_base_sha256_mismatch")
    spec = importlib.util.spec_from_file_location(
        "_onnx_splitpoint_v27913_yolov7_claim_gate_base", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("v27910_claim_gate_base_import_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_frozen_base()

# Preserve the exact admitted measurement identity.  Only the verifying
# release and these hash-pinned release-metadata members differ from v2.79.10.
_origin = dict(_base.CARRY_FORWARD_ORIGIN)
_origin["current_snapshot_member_sha256"] = {
    "onnx_splitpoint_tool/__init__.py": (
        "58e9d054ba7d14c673e2567da50648a7b254021e8ec105b2249de0c3dfbe2ecb"
    ),
    "onnx_splitpoint_tool/release_identity.py": (
        "1e03debac7c0544f476242e7546419e3232bd7d64c3caf39338a5d66ba2bac6e"
    ),
    "onnx_splitpoint_tool/native_three_stage.py": (
        "6ab076454a231f5a7939c05489a885dba92785e2711aa0595125a79aca25311f"
    ),
    "scripts/native_hailo_trt_fifo_from_benchmarkset.py": (
        "da285602e2fb6373b6ee7a3c5eafd7ececc5d8728cc560ad78bcdd38ae674800"
    ),
    "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py": (
        "ba187ca8bd36cc58c24775477fc18877b324f30bda776bcf83d9f08794de6ecf"
    ),
}

_base.PACKAGE_VERSION = PACKAGE_VERSION
_base.BUILD_ID = BUILD_ID
_base.CARRY_FORWARD_ORIGIN = _origin
_base.REQUIRED_SOURCE_FILES = tuple(_base.REQUIRED_SOURCE_FILES) + (
    "scripts/verify_v27910_yolov7_claim_gate_32.py",
    "scripts/verify_v27913_yolov7_claim_gate_32.py",
)

# Export the frozen validator surface so focused tests can exercise the same
# private fail-closed helpers as the historical release did.  Functions retain
# the configured base module as their globals.
for _name in dir(_base):
    if not _name.startswith("__") and _name not in {
        "BUILD_ID",
        "PACKAGE_VERSION",
        "CARRY_FORWARD_ORIGIN",
        "REQUIRED_SOURCE_FILES",
        "main",
        "verify_claim_gate",
    }:
        globals()[_name] = getattr(_base, _name)

CARRY_FORWARD_ORIGIN: Mapping[str, Any] = _origin
REQUIRED_SOURCE_FILES = _base.REQUIRED_SOURCE_FILES


def verify_claim_gate(
    *,
    result_json: Path,
    evidence_root: Path,
    source_root: Path,
    expected_source_manifest_sha256: str,
) -> Dict[str, Any]:
    """Run the frozen validator and identify this compatibility verifier."""

    receipt = _base.verify_claim_gate(
        result_json=result_json,
        evidence_root=evidence_root,
        source_root=source_root,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    verification = receipt.get("verification_identity")
    if not isinstance(verification, dict):
        raise _base.ClaimGateError("verification_identity_missing")
    verification["verifier"] = VERIFIER_FILENAME
    verification["frozen_base_verifier"] = {
        "path": BASE_VERIFIER,
        "sha256": BASE_VERIFIER_SHA256,
    }
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _base.build_parser().parse_args(argv)
    output = Path(os.path.abspath(args.output))
    try:
        receipt = verify_claim_gate(
            result_json=Path(args.result_json),
            evidence_root=Path(args.evidence_root),
            source_root=Path(args.source_root),
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        )
        _base._write_receipt(output, receipt)
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        print("V27913_YOLOV7_CLAIM_GATE=PASS")
        return 0
    except Exception as exc:
        failure = {
            "schema": _base.VERDICT_SCHEMA,
            "schema_version": 1,
            "status": "FAIL",
            "ok": False,
            "valid_claim": False,
            "model_id": _base.MODEL_ID,
            "case_id": _base.CASE_ID,
            "claim_gate": _base.CLAIM_SCOPE,
            "required_item_count": _base.CLAIM_ITEM_COUNT,
            "passed_item_count": 0,
            "claim_eligible": False,
            "failure_class": type(exc).__name__,
            "failure_reason": str(exc),
        }
        try:
            _base._write_receipt(output, failure)
        except Exception as write_exc:
            print(
                "ERROR: unable to write fail receipt: %s" % write_exc,
                file=sys.stderr,
            )
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        print("V27913_YOLOV7_CLAIM_GATE=FAIL", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
