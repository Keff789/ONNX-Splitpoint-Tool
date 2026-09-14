"""Official COCO bbox evaluation and immutable artefact export.

This is intentionally separate from the internal paired-bootstrap proxy.  The
internal metric supports the non-inferiority gate; this module runs the official
``pycocotools.cocoeval.COCOeval`` implementation and archives inputs, parameters,
summary output, tensors and hashes for the final detector claim.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from ..workflow.artifacts import now_iso, sha256_file, sha256_json, write_json, write_text

METRIC_NAMES = [
    "AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
]

_SHA256_RE = re.compile(r"(?:sha256:)?([0-9a-fA-F]{64})\Z")


def normalize_sha256(value: Any, *, label: str = "sha256") -> str:
    """Return a bare, lower-case SHA-256 digest from an accepted encoding.

    Accepted values are exactly 64 hexadecimal characters, optionally
    preceded by the exact ``sha256:`` algorithm prefix.  Keeping this helper
    public lets preflight and evaluation apply the same fail-closed parsing
    without depending on the serialization convention of ``sha256_file``.
    """
    if not isinstance(value, str):
        raise ValueError(f"{label}_invalid_sha256")
    match = _SHA256_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"{label}_invalid_sha256")
    return match.group(1).lower()


def pycocotools_status() -> dict[str, Any]:
    try:
        import pycocotools  # type: ignore
        from pycocotools.coco import COCO  # noqa: F401
        from pycocotools.cocoeval import COCOeval  # noqa: F401
        try:
            version = importlib.metadata.version("pycocotools")
        except Exception:
            version = "unknown"
        return {"available": True, "version": version, "module": str(getattr(pycocotools, "__file__", ""))}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _normalise_predictions(predictions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in predictions:
        try:
            bbox = [float(x) for x in list(raw.get("bbox") or [])[:4]]
            if len(bbox) != 4 or bbox[2] <= 0.0 or bbox[3] <= 0.0:
                continue
            rows.append({
                "image_id": int(raw.get("image_id")),
                "category_id": int(raw.get("category_id")),
                "bbox": bbox,
                "score": float(raw.get("score")),
            })
        except Exception:
            continue
    return rows


def evaluate_coco_bbox(
    *, annotations: str | Path, predictions: Sequence[Mapping[str, Any]], output_dir: str | Path,
    image_ids: Sequence[int] | None = None, category_ids: Sequence[int] | None = None,
    max_detections: Sequence[int] = (1, 10, 100), archive_tensors: bool = True,
    required: bool = False, variant: str = "composed",
    annotations_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    status = pycocotools_status()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ann = Path(annotations).expanduser().resolve()
    provenance = (
        dict(annotations_provenance)
        if isinstance(annotations_provenance, Mapping) else {}
    )
    annotations_evidence_path = str(
        provenance.get("source_path") or ann
    )
    if not ann.is_file():
        payload = {"status": "unavailable", "reason": "annotations_missing", "annotations": str(ann), "pycocotools": status}
        write_json(out / "coco_official_metrics.json", payload)
        if required:
            raise FileNotFoundError(ann)
        return payload
    annotations_sha256 = normalize_sha256(
        sha256_file(ann), label="annotations_actual_sha256",
    )
    for key in ("source_sha256", "exact_use_snapshot_sha256"):
        if key not in provenance or provenance.get(key) in (None, ""):
            continue
        declared = normalize_sha256(
            provenance.get(key),
            label=f"annotations_provenance_{key}",
        )
        if declared != annotations_sha256:
            raise ValueError(
                f"annotations_provenance_sha256_mismatch:{key}:"
                f"{declared}!={annotations_sha256}"
            )
    if not status.get("available"):
        payload = {"status": "unavailable", "reason": "pycocotools_missing", "annotations": str(ann), "pycocotools": status}
        write_json(out / "coco_official_metrics.json", payload)
        if required:
            raise RuntimeError(str(status.get("error") or "pycocotools missing"))
        return payload

    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore

    preds = _normalise_predictions(predictions)
    pred_path = write_json(out / "coco_predictions.json", preds)
    if not preds:
        payload = {
            "status": "unavailable", "reason": "no_valid_predictions", "variant": variant,
            "annotations": annotations_evidence_path,
            "annotations_sha256": annotations_sha256,
            "annotations_exact_use": provenance or None,
            "predictions": str(pred_path), "predictions_sha256": sha256_file(pred_path), "pycocotools": status,
        }
        write_json(out / "coco_official_metrics.json", payload)
        if required:
            raise RuntimeError("no valid COCO predictions")
        return payload

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        coco_gt = COCO(str(ann))
        coco_dt = coco_gt.loadRes(str(pred_path))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        if image_ids:
            evaluator.params.imgIds = sorted({int(x) for x in image_ids})
        else:
            evaluator.params.imgIds = sorted({int(x["image_id"]) for x in preds})
        if category_ids:
            evaluator.params.catIds = sorted({int(x) for x in category_ids})
        max_dets = sorted({max(1, int(x)) for x in max_detections})
        while len(max_dets) < 3:
            max_dets.insert(0, 1 if not max_dets else max_dets[0])
        evaluator.params.maxDets = max_dets[-3:]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    summary_text = capture.getvalue()
    write_text(out / "coco_eval_summary.txt", summary_text)
    stats = [float(x) for x in list(evaluator.stats)]
    metrics = {name: (stats[i] if i < len(stats) else None) for i, name in enumerate(METRIC_NAMES)}
    params_payload = {
        "iou_type": "bbox",
        "image_ids": [int(x) for x in list(evaluator.params.imgIds)],
        "category_ids": [int(x) for x in list(evaluator.params.catIds)],
        "iou_thresholds": [float(x) for x in evaluator.params.iouThrs.tolist()],
        "recall_thresholds": [float(x) for x in evaluator.params.recThrs.tolist()],
        "area_ranges": [[float(y) for y in x] for x in evaluator.params.areaRng],
        "area_range_labels": [str(x) for x in evaluator.params.areaRngLbl],
        "max_detections": [int(x) for x in evaluator.params.maxDets],
        "use_categories": int(evaluator.params.useCats),
    }
    params_path = write_json(out / "coco_eval_params.json", params_payload)
    tensor_path: Optional[Path] = None
    if archive_tensors:
        tensor_path = out / "coco_eval_tensors.npz"
        np.savez_compressed(
            tensor_path,
            precision=np.asarray(evaluator.eval.get("precision")),
            recall=np.asarray(evaluator.eval.get("recall")),
            scores=np.asarray(evaluator.eval.get("scores")),
            counts=np.asarray(evaluator.eval.get("counts")),
        )
    payload = {
        "schema": "onnx-splitpoint/official-coco-evaluation",
        "schema_version": 1,
        "status": "ok",
        "created_at": now_iso(),
        "variant": variant,
        "implementation": "pycocotools.cocoeval.COCOeval",
        "pycocotools": status,
        "numpy_version": str(np.__version__),
        "annotations": annotations_evidence_path,
        "annotations_sha256": annotations_sha256,
        "annotations_exact_use": provenance or None,
        "predictions": str(pred_path.name),
        "predictions_sha256": sha256_file(pred_path),
        "prediction_count": len(preds),
        "metrics": metrics,
        "stats": stats,
        "parameters": params_payload,
        "parameters_file": str(params_path.name),
        "summary_file": "coco_eval_summary.txt",
        "tensors_file": str(tensor_path.name) if tensor_path else "",
    }
    payload["evaluation_payload_sha256"] = sha256_json({k: v for k, v in payload.items() if k != "evaluation_payload_sha256"})
    metrics_path = write_json(out / "coco_official_metrics.json", payload)
    manifest = {
        "schema": "onnx-splitpoint/official-coco-artifact-manifest",
        "schema_version": 1,
        "created_at": now_iso(),
        "files": [],
    }
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "coco_eval_manifest.json":
            manifest["files"].append({"name": p.name, "size_bytes": int(p.stat().st_size), "sha256": sha256_file(p)})
    manifest["manifest_payload_sha256"] = sha256_json(manifest)
    write_json(out / "coco_eval_manifest.json", manifest)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run official pycocotools COCO bbox evaluation and archive evidence")
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-id", action="append", type=int, default=[])
    ap.add_argument("--category-id", action="append", type=int, default=[])
    ap.add_argument("--max-detections", default="1,10,100")
    ap.add_argument("--required", action="store_true")
    ap.add_argument("--no-tensors", action="store_true")
    ap.add_argument("--variant", default="composed")
    ns = ap.parse_args(argv)
    try:
        raw = json.loads(Path(ns.predictions).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("predictions JSON must be a list")
        max_dets = [int(x.strip()) for x in str(ns.max_detections).split(",") if x.strip()]
        payload = evaluate_coco_bbox(
            annotations=ns.annotations, predictions=raw, output_dir=ns.out,
            image_ids=ns.image_id or None, category_ids=ns.category_id or None,
            max_detections=max_dets or [1, 10, 100], archive_tensors=not ns.no_tensors,
            required=bool(ns.required), variant=ns.variant,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if payload.get("status") == "ok" or not ns.required else 2
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
