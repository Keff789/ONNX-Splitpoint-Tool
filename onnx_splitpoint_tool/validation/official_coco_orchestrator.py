from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Mapping
import hashlib
import io
import json
import platform


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prediction_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(x) for x in payload if isinstance(x, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("predictions", "detections", "results", "coco_predictions"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(x) for x in value if isinstance(x, Mapping)]
    raise ValueError("Prediction JSON does not contain a COCO result list")


def evaluate_official_coco(
    *,
    annotations: str | Path,
    predictions: str | Path,
    output_dir: str | Path,
    row_metadata: Mapping[str, Any] | None = None,
    max_detections: tuple[int, int, int] = (1, 10, 100),
) -> dict[str, Any]:
    """Run official ``pycocotools.COCOeval`` and archive all claim artefacts."""
    try:
        import numpy as np
        import pycocotools
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:  # pragma: no cover - dependency is optional outside final runs
        return {"status": "unavailable", "reason": f"pycocotools_import_failed: {exc}"}

    ann = Path(annotations).expanduser().resolve()
    pred = Path(predictions).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not ann.is_file():
        return {"status": "unavailable", "reason": "annotations_missing", "path": str(ann)}
    if not pred.is_file():
        return {"status": "unavailable", "reason": "predictions_missing", "path": str(pred)}

    raw = json.loads(pred.read_text(encoding="utf-8"))
    rows = _prediction_list(raw)
    canonical_predictions = out / "predictions.json"
    canonical_predictions.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    coco_gt = COCO(str(ann))
    # COCO.loadRes handles an empty list poorly in some versions.  Preserve an
    # explicit unavailable result rather than fabricating a zero AP claim.
    if not rows:
        result = {"status": "unavailable", "reason": "empty_predictions"}
        (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    coco_dt = coco_gt.loadRes(rows)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = list(max_detections)
    capture = io.StringIO()
    with redirect_stdout(capture):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    text = capture.getvalue()
    (out / "summary.txt").write_text(text, encoding="utf-8")

    stats = [float(x) for x in evaluator.stats.tolist()]
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    metrics = dict(zip(names, stats))
    np.save(out / "precision.npy", evaluator.eval.get("precision"))
    np.save(out / "recall.npy", evaluator.eval.get("recall"))
    np.save(out / "scores.npy", evaluator.eval.get("scores"))
    parameters = {
        "iou_type": "bbox",
        "max_detections": list(max_detections),
        "image_ids": [int(x) for x in evaluator.params.imgIds],
        "category_ids": [int(x) for x in evaluator.params.catIds],
        "area_ranges": evaluator.params.areaRng,
        "iou_thresholds": [float(x) for x in evaluator.params.iouThrs.tolist()],
        "recall_thresholds": [float(x) for x in evaluator.params.recThrs.tolist()],
    }
    (out / "parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    result = {
        "status": "ok",
        "metrics": metrics,
        "prediction_count": len(rows),
        "image_count": len(set(int(x["image_id"]) for x in rows if "image_id" in x)),
        "row_metadata": dict(row_metadata or {}),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    files = [canonical_predictions, out / "summary.json", out / "summary.txt", out / "parameters.json",
             out / "precision.npy", out / "recall.npy", out / "scores.npy"]
    manifest = {
        "schema": "onnx-splitpoint.official-coco-artifact.v1",
        "annotations": str(ann),
        "annotations_sha256": _sha256(ann),
        "source_predictions": str(pred),
        "source_predictions_sha256": _sha256(pred),
        "python": platform.python_version(),
        "pycocotools_version": getattr(pycocotools, "__version__", "unknown"),
        "files": {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in files},
        "row_metadata": dict(row_metadata or {}),
    }
    (out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return result


def discover_prediction_files(run_dir: str | Path) -> list[Path]:
    root = Path(run_dir)
    names = {"coco_predictions.json", "predictions_coco.json", "official_coco_predictions.json"}
    found: list[Path] = []
    for p in root.rglob("*.json"):
        if p.name in names or ("prediction" in p.name.lower() and "coco" in p.name.lower()):
            if "official_coco" not in p.parts:
                found.append(p)
    return sorted(set(found))


def run_official_coco_for_run(
    run_dir: str | Path,
    *,
    annotations: str | Path,
    enabled: bool,
    required: bool,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    out_root = Path(output_dir) if output_dir is not None else root / "reports" / "scientific" / "official_coco"
    summary: dict[str, Any] = {"enabled": enabled, "required": required, "rows": []}
    if not enabled:
        summary["status"] = "disabled"
    else:
        preds = discover_prediction_files(root)
        for index, pred in enumerate(preds):
            try:
                rel = pred.relative_to(root)
            except Exception:
                rel = Path(pred.name)
            row_id = "__".join(rel.with_suffix("").parts[-5:])
            row_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in row_id)
            result = evaluate_official_coco(
                annotations=annotations,
                predictions=pred,
                output_dir=out_root / row_id,
                row_metadata={"prediction_file": str(rel), "row_id": row_id},
            )
            summary["rows"].append({"row_id": row_id, "prediction_file": str(rel), **result})
        ok_count = sum(1 for x in summary["rows"] if x.get("status") == "ok")
        summary["status"] = "ok" if ok_count else ("required_missing" if required else "unavailable")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "official_coco_index.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
