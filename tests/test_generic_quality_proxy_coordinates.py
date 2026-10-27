"""Device mini-AP compares completed detections in their original pixel frame."""
from __future__ import annotations

import ast
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


TEMPLATE = Path(__file__).parents[1] / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"


def test_completed_generic_mini_ap_uses_original_gt_without_changing_legacy_proxy(tmp_path):
    tree = ast.parse(TEMPLATE.read_text())
    names = {"_boxes_iou_xyxy", "_mini_coco_ap50_101", "_mini_coco_ap50_for_variant",
             "_semantic_annotation_sidecar_path", "_original_image_geometry",
             "_load_semantic_reference_detections_original"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    branch, = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and isinstance(node.test, ast.Name) and node.test.id == "mini_coco_ap50_enabled"
               and any(isinstance(child, ast.Name) and child.id == "dets_gt_for_ap50"
                       for child in ast.walk(node))]
    namespace = {"np": np, "Image": Image, "json": json, "Mapping": Mapping,
                 # Class mapping is fixed for this coordinate-only fixture.
                 "_map_semantic_category_to_model_class": lambda _category, _labels: (0, "person")}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(TEMPLATE), "exec"), namespace)
    proxy_branch = compile(ast.Module(body=[branch], type_ignores=[]), str(TEMPLATE), "exec")

    # A 640x426 source letterboxed to 640x640 has 107 pixels of top padding.
    img_file = tmp_path / "source.jpg"
    Image.new("RGB", (640, 426)).save(img_file)
    annotation = {"category_id": 1, "bbox": [100.0, 142.51, 80.0, 69.5]}
    img_file.with_suffix(".json").write_text(json.dumps({"annotations": [annotation, {**annotation, "iscrowd": 1}]}))
    original_gt = [{"class_id": 0, "class_name": "person", "score": 1.0,
                    "x1": 100.0, "y1": 142.51, "x2": 180.0, "y2": 212.01}]
    model_gt = [{**original_gt[0], "y1": 249.51, "y2": 319.01}]
    completed_prediction = [{**original_gt[0], "score": 0.9}]
    metric = namespace["_mini_coco_ap50_for_variant"]
    assert metric(gt_by_image={"source.jpg": model_gt}, pred_by_image={"source.jpg": completed_prediction})["ap50"] == 0.0

    for completed, detections in ((True, completed_prediction), (False, [{**model_gt[0], "score": 0.9}])):
        original_snapshot = deepcopy(original_gt)
        values = {"mini_coco_ap50_enabled": True, "ann_exists": True,
                  "_generic_quality_completion": object() if completed else None,
                  "quality_cache": {"gt_original": original_gt},
                  "dets_gt_for_ap50": deepcopy(model_gt), "dets_out": detections,
                  "img_file": img_file, "labels": ["person"], "mini_coco_gt_by_image": {},
                  "mini_coco_pred_by_image": {}, "mini_coco_warnings_variant": []}
        exec(proxy_branch, namespace, values)
        result = metric(gt_by_image=values["mini_coco_gt_by_image"], pred_by_image=values["mini_coco_pred_by_image"])
        assert result["ap50"] == pytest.approx(1.0)
        assert values["mini_coco_gt_by_image"]["source.jpg"] == (original_gt if completed else model_gt)
        assert original_gt == original_snapshot
        assert values["mini_coco_pred_by_image"]["source.jpg"] == detections
