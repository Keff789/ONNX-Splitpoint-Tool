#!/usr/bin/env python3
"""Task-level validator for classification output dumps.

Input dumps can be native FIFO manifests, generic output_dump manifests, or NPZ
files.  Without a reference, the script reports top-k classes and finite-value
checks.  With a reference, it compares top-1/top-k agreement and cosine
similarity.  This keeps native/generic runner validation on a shared tensor-dump
contract while leaving dataset-level accuracy to the Evaluation workflow.
"""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.validate_output_dumps import load_dump, summarize  # type: ignore
except Exception:
    # direct execution from scripts/ directory fallback
    from validate_output_dumps import load_dump, summarize  # type: ignore


def _select_logits(tensors: dict[str, np.ndarray], name: str = "") -> tuple[str, np.ndarray]:
    if name:
        if name not in tensors:
            raise KeyError(f"output '{name}' not found; available={list(tensors)}")
        return name, np.asarray(tensors[name])
    # Prefer the smallest 1D/2D tensor with at least two classes.
    candidates = []
    for k, v in tensors.items():
        arr = np.asarray(v)
        flat = arr.reshape(-1) if arr.ndim <= 2 else arr.reshape(-1)
        if flat.size >= 2:
            candidates.append((flat.size, k, arr))
    if not candidates:
        raise ValueError("no classification-like output tensor found")
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def _topk(arr: np.ndarray, k: int) -> list[dict[str, Any]]:
    x = np.asarray(arr).astype(np.float32).reshape(-1)
    if x.size == 0:
        return []
    k = min(int(k), int(x.size))
    idx = np.argpartition(-x, kth=k-1)[:k]
    idx = idx[np.argsort(-x[idx])]
    return [{"index": int(i), "score": float(x[i])} for i in idx]


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = np.asarray(a).astype(np.float32).reshape(-1)
    bb = np.asarray(b).astype(np.float32).reshape(-1)
    if aa.shape != bb.shape or aa.size == 0:
        return None
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom == 0.0:
        return None
    return float(np.dot(aa, bb) / denom)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate classification output dump")
    ap.add_argument("--candidate", required=True, help="Candidate dump manifest or NPZ")
    ap.add_argument("--reference", default="", help="Optional reference dump manifest or NPZ")
    ap.add_argument("--output-name", default="", help="Specific candidate/reference output tensor to use")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cand, cand_meta = load_dump(args.candidate)
    cname, carr = _select_logits(cand, args.output_name)
    ctop = _topk(carr, args.topk)
    ok = bool(np.isfinite(carr.astype(np.float32, copy=False)).all())
    report: dict[str, Any] = {
        "schema": "onnx-splitpoint/classification-dump-validation",
        "schema_version": 1,
        "ok": ok,
        "candidate": str(Path(args.candidate).expanduser().resolve()),
        "candidate_output": cname,
        "candidate_summary": summarize(carr),
        "candidate_topk": ctop,
        "reference": str(Path(args.reference).expanduser().resolve()) if args.reference else "",
    }
    if args.reference:
        ref, ref_meta = load_dump(args.reference)
        rname, rarr = _select_logits(ref, args.output_name if args.output_name else cname if cname in ref else "")
        rtop = _topk(rarr, args.topk)
        report["reference_output"] = rname
        report["reference_summary"] = summarize(rarr)
        report["reference_topk"] = rtop
        cidx = [x["index"] for x in ctop]
        ridx = [x["index"] for x in rtop]
        report["top1_match"] = bool(cidx and ridx and cidx[0] == ridx[0])
        report["topk_overlap"] = int(len(set(cidx) & set(ridx)))
        report["topk_overlap_ratio"] = float(len(set(cidx) & set(ridx)) / max(1, min(len(cidx), len(ridx))))
        report["cosine_similarity"] = _cosine(carr, rarr)
        if not report["top1_match"]:
            ok = False
    report["ok"] = ok
    out = Path(args.out).expanduser().resolve() if args.out else Path(args.candidate).expanduser().resolve().parent / "classification_output_validation.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": ok, "out": str(out), "candidate_top1": ctop[0] if ctop else None}, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
