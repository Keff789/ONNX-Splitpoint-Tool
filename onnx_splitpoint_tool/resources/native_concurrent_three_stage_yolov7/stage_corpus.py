#!/usr/bin/env python3
"""Select and stage a deterministic, annotation-diverse COCO image corpus.

The selection is score/prediction independent. It uses only the frozen COCO
validation manifest, ground-truth annotation counts and image geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_sha(value: str) -> str:
    token = str(value or "")
    return token[7:] if token.startswith("sha256:") else token


def quantile_pick(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for pos in range(count):
        index = int(round(pos * (len(rows) - 1) / max(1, count - 1)))
        while index in used and index + 1 < len(rows):
            index += 1
        while index in used and index - 1 >= 0:
            index -= 1
        if index in used:
            continue
        used.add(index)
        selected.append(rows[index])
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--images-root", default="")
    ap.add_argument("--annotations", default="")
    ap.add_argument("--force-image-id", action="append", default=["632"])
    args = ap.parse_args()

    manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    images_out = out / "images"
    if out.exists():
        shutil.rmtree(out)
    images_out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != "coco2017-val" or int(manifest.get("item_count") or 0) != 5000:
        raise RuntimeError("unexpected_detection_validation_manifest")
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != 5000:
        raise RuntimeError("dataset_manifest_items_invalid")

    images_root = Path(args.images_root or manifest["root"]).expanduser().resolve()
    annotations_spec = manifest.get("annotations") or {}
    annotations_path = Path(args.annotations or annotations_spec.get("path") or "").expanduser().resolve()
    if not images_root.is_dir():
        raise RuntimeError(f"images_root_missing:{images_root}")
    if not annotations_path.is_file():
        raise RuntimeError(f"annotations_missing:{annotations_path}")
    expected_ann_sha = strip_sha(str(annotations_spec.get("sha256") or ""))
    actual_ann_sha = sha256_file(annotations_path)
    if expected_ann_sha and actual_ann_sha != expected_ann_sha:
        raise RuntimeError(
            f"annotations_sha256_mismatch:expected={expected_ann_sha}:actual={actual_ann_sha}"
        )

    coco = json.loads(annotations_path.read_text(encoding="utf-8"))
    all_counts: dict[int, int] = {}
    noncrowd_counts: dict[int, int] = {}
    category_counts: dict[int, set[int]] = {}
    for ann in coco.get("annotations") or []:
        image_id = int(ann["image_id"])
        all_counts[image_id] = all_counts.get(image_id, 0) + 1
        if not int(ann.get("iscrowd") or 0):
            noncrowd_counts[image_id] = noncrowd_counts.get(image_id, 0) + 1
        category_counts.setdefault(image_id, set()).add(int(ann.get("category_id") or -1))

    rows: list[dict[str, Any]] = []
    by_id: dict[int, dict[str, Any]] = {}
    for item in items:
        image_id = int(item["image_id"])
        width = int(item["width"])
        height = int(item["height"])
        row = {
            "image_id": image_id,
            "relative_path": str(item["relative_path"]),
            "width": width,
            "height": height,
            "aspect_ratio": width / float(height),
            "annotation_count": int(all_counts.get(image_id, 0)),
            "noncrowd_annotation_count": int(noncrowd_counts.get(image_id, 0)),
            "category_count": len(category_counts.get(image_id, set())),
            "sha256": strip_sha(str(item["sha256"])),
            "size_bytes": int(item["size_bytes"]),
        }
        rows.append(row)
        by_id[image_id] = row

    bins = [
        ("sparse_0_2", lambda n: n <= 2),
        ("light_3_5", lambda n: 3 <= n <= 5),
        ("medium_6_10", lambda n: 6 <= n <= 10),
        ("dense_11_plus", lambda n: n >= 11),
    ]
    target_count = max(4, int(args.count))
    base_quota = target_count // len(bins)
    remainder = target_count % len(bins)
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    # Keep the already validated sentinel in the corpus for continuity.
    for raw in args.force_image_id:
        try:
            image_id = int(raw)
        except ValueError:
            continue
        if image_id in by_id and image_id not in selected_ids:
            row = dict(by_id[image_id])
            row["selection_bin"] = "forced_anchor"
            row["selection_reason"] = "previous_v2782_b066_sentinel"
            selected.append(row)
            selected_ids.add(image_id)

    bin_diagnostics: list[dict[str, Any]] = []
    for idx, (name, predicate) in enumerate(bins):
        quota = base_quota + (1 if idx < remainder else 0)
        already = sum(1 for row in selected if predicate(int(row["noncrowd_annotation_count"])))
        needed = max(0, quota - already)
        pool = [
            row for row in rows
            if row["image_id"] not in selected_ids
            and predicate(int(row["noncrowd_annotation_count"]))
        ]
        # Quantile selection over count, aspect ratio, category diversity and ID.
        pool.sort(key=lambda row: (
            int(row["noncrowd_annotation_count"]),
            float(row["aspect_ratio"]),
            int(row["category_count"]),
            int(row["image_id"]),
        ))
        picks = quantile_pick(pool, needed)
        for row in picks:
            copied = dict(row)
            copied["selection_bin"] = name
            copied["selection_reason"] = "annotation_count_and_geometry_quantile"
            selected.append(copied)
            selected_ids.add(int(row["image_id"]))
        bin_diagnostics.append({
            "name": name,
            "population": len(pool) + already,
            "quota": quota,
            "already_forced": already,
            "selected_new": len(picks),
        })

    if len(selected) < target_count:
        remaining = [row for row in rows if row["image_id"] not in selected_ids]
        remaining.sort(key=lambda row: (
            int(row["noncrowd_annotation_count"]),
            float(row["aspect_ratio"]),
            int(row["image_id"]),
        ))
        for row in quantile_pick(remaining, target_count - len(selected)):
            copied = dict(row)
            copied["selection_bin"] = "deterministic_backfill"
            copied["selection_reason"] = "quota_backfill_without_model_predictions"
            selected.append(copied)
            selected_ids.add(int(row["image_id"]))

    selected = selected[:target_count]
    if len(selected) != target_count:
        raise RuntimeError(f"corpus_selection_shortfall:{len(selected)}:{target_count}")

    staged: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        source = images_root / str(row["relative_path"])
        if not source.is_file():
            raise RuntimeError(f"selected_image_missing:{source}")
        actual_sha = sha256_file(source)
        if actual_sha != row["sha256"]:
            raise RuntimeError(
                f"selected_image_sha256_mismatch:{source}:expected={row['sha256']}:actual={actual_sha}"
            )
        if source.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"selected_image_size_mismatch:{source}")
        # Prefix the deterministic selection index so a plain lexical directory
        # scan in the native C++ runtime preserves corpus-manifest order.
        target = images_out / f"{index:02d}_{source.name}"
        shutil.copy2(source, target)
        copied = dict(row)
        copied.update({
            "index": index,
            "staged_file": f"images/{target.name}",
            "staged_sha256": sha256_file(target),
        })
        staged.append(copied)

    corpus = {
        "schema": "onnx-splitpoint/yolov7-fast-decode-parity-corpus",
        "schema_version": 1,
        "selection_scope": "coco2017_val_ground_truth_annotation_and_geometry_only",
        "selection_uses_model_predictions": False,
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_id": manifest["dataset_id"],
        "images_root": str(images_root),
        "annotations": str(annotations_path),
        "annotations_sha256": actual_ann_sha,
        "requested_count": target_count,
        "selected_count": len(staged),
        "bins": bin_diagnostics,
        "items": staged,
    }
    (out / "corpus_manifest.json").write_text(
        json.dumps(corpus, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "selected_count": len(staged),
        "out": str(out),
        "manifest": str(out / "corpus_manifest.json"),
        "bin_counts": {
            name: sum(1 for row in staged if row["selection_bin"] == name)
            for name, _ in bins
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
