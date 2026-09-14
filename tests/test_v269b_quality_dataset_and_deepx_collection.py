from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from onnx_splitpoint_tool.benchmark.remote_run import _remote_result_collect_script
from onnx_splitpoint_tool.benchmark.suite_refresh import _normalize_suite_validation_payloads
from onnx_splitpoint_tool.workflow.execution_binding import _copy_remote_result_files


def _write_materialized_detection_subset(
    root: Path,
    *,
    dataset: str,
    source: str,
    image_ids: list[int],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    samples = []
    images = []
    annotations = []
    for index, image_id in enumerate(image_ids):
        name = f"{image_id:012d}.jpg"
        image_bytes = f"image-{image_id}".encode()
        (root / name).write_bytes(image_bytes)
        ground_truth = [{
            "id": index + 1,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [1, 2, 3, 4],
            "area": 12,
            "iscrowd": 0,
        }]
        (root / Path(name).with_suffix(".json")).write_text(
            json.dumps(ground_truth, sort_keys=True), encoding="utf-8",
        )
        samples.append({
            "image": name,
            "sample_id": str(image_id),
            "image_id": image_id,
            "annotation_count": 1,
            "source_image": f"/frozen/coco/val2017/{name}",
            "source_sha256": "sha256:" + hashlib.sha256(image_bytes).hexdigest(),
            "source_size_bytes": len(image_bytes),
        })
        images.append({"id": image_id, "file_name": name, "width": 64, "height": 64})
        annotations.extend(ground_truth)
    payload = {
        "schema": "onnx-splitpoint/detection-validation-manifest",
        "schema_version": 2,
        "dataset": dataset,
        "source_type": "materialized_run_mode_subset",
        "source": source,
        "source_manifest": "/frozen/manifests/coco2017_val_manifest.json",
        "source_annotations": "/frozen/coco/annotations/instances_val2017.json",
        "selection": {
            "type": "deterministic_hash",
            "seed": 20260710,
            "requested_images": len(image_ids),
            "selected_images": len(image_ids),
            "source_population": 5000,
        },
        "samples": samples,
        "filtered_coco_annotations": {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "object"}],
        },
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest


def test_authoritative_run_keeps_exact_manifest_when_remote_preset_alias_differs(
    tmp_path: Path,
) -> None:
    """Run32 regression: different aliases must not create two contracts.

    The two fixture manifests deliberately have different raw hashes, like the
    observed c89f... and 8f39... files, while Image IDs and ground truth are
    identical.  The generated run has already selected the ``val2017`` artefact;
    a remote ``coco_50`` compatibility override must keep that exact file.
    """

    suite = tmp_path / "suite"
    selected_root = suite / "resources/validation/detection/val2017_n2_s20260710"
    alias_root = suite / "resources/validation/detection/coco_50_n2_s20260710"
    selected_manifest = _write_materialized_detection_subset(
        selected_root, dataset="val2017_n2_s20260710", source="/frozen/coco/val2017",
        image_ids=[212226, 369037],
    )
    alias_manifest = _write_materialized_detection_subset(
        alias_root, dataset="coco_50_n2_s20260710", source="/legacy/coco_50_data",
        image_ids=[212226, 369037],
    )
    selected_sha = hashlib.sha256(selected_manifest.read_bytes()).hexdigest()
    alias_sha = hashlib.sha256(alias_manifest.read_bytes()).hexdigest()
    assert selected_sha != alias_sha
    selected_payload = json.loads(selected_manifest.read_text(encoding="utf-8"))
    alias_payload = json.loads(alias_manifest.read_text(encoding="utf-8"))
    assert selected_payload["samples"] == alias_payload["samples"]
    assert selected_payload["filtered_coco_annotations"] == alias_payload["filtered_coco_annotations"]

    exact_rel = selected_root.relative_to(suite).as_posix()
    alias_rel = alias_root.relative_to(suite).as_posix()
    run = {
        "id": "ort_cpu",
        "benchmark_task": "detection",
        "validation_images": exact_rel,
        "validation_manifest": "/frozen/manifests/coco2017_val_manifest.json",
        "validation_items_requested": 2,
        "validation_budget_authoritative": True,
        "validation_max_images": 2,
        "mini_coco_ap50": True,
    }
    plan = {"runs": [run]}
    (suite / "benchmark_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (suite / "benchmark_set.json").write_text(
        json.dumps({"model_name": "yolo26s", "plan": plan}), encoding="utf-8",
    )

    result = _normalize_suite_validation_payloads(
        suite,
        benchmark_set_json=suite / "benchmark_set.json",
        validation_images=alias_rel,
        validation_max_images=50,
        validation_reference_mode="auto",
        mini_coco_ap50=True,
        benchmark_task="detection",
        mini_classification_eval=False,
        log=None,
    )
    updated = json.loads((suite / "benchmark_plan.json").read_text(encoding="utf-8"))["runs"][0]
    assert updated["validation_images"] == exact_rel
    assert updated["validation_max_images"] == 2
    assert result["validation_images"] == exact_rel
    assert hashlib.sha256(selected_manifest.read_bytes()).hexdigest() == selected_sha


def test_remote_collector_transports_and_discovers_top_level_deepx_full_quality(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "remote_suite"
    quality = suite / "results/deepx_m1_full/task_quality_inputs"
    quality.mkdir(parents=True)
    request = {
        "schema": "onnx-splitpoint/task-quality-request",
        "schema_version": 1,
        "task": "classification",
        "variant": "full",
        "producer_identity": {
            "source_run_id": "deepx_m1_full",
            "case_id": "full",
            "variant": "full",
        },
        "candidate_artifact": "full_candidate.json",
    }
    (quality / "full_request.json").write_text(json.dumps(request), encoding="utf-8")
    (quality / "full_candidate.json").write_text(
        json.dumps({"records": [{"image_id": "i1", "candidate": {"top1": 1}}]}),
        encoding="utf-8",
    )
    compiled = suite / "results/deepx_m1_full/model.dxnn"
    compiled.write_bytes(b"must-not-be-collected")

    local_remote_run = tmp_path / "local_remote_run"
    downloaded_results = local_remote_run / "results"
    script = _remote_result_collect_script(
        remote_results_dir=str(downloaded_results), remote_suite_dir=str(suite),
    )
    subprocess.run(["bash", "-lc", script], check=True)
    copied_quality = downloaded_results / "deepx_m1_full/task_quality_inputs"
    assert (copied_quality / "full_request.json").is_file()
    assert (copied_quality / "full_candidate.json").is_file()
    assert not (downloaded_results / "deepx_m1_full/model.dxnn").exists()

    evaluation_results = tmp_path / "evaluation_results"
    copied = _copy_remote_result_files(
        local_remote_run, evaluation_results, flat_prefix="deepx_setup",
    )
    queued = list(evaluation_results.rglob("task_quality_inputs/*_request.json"))
    assert len(queued) == 1
    assert queued[0].name == "full_request.json"
    assert any(
        row.get("kind") == "central_quality_input"
        and str(row.get("destination") or "").endswith("full_request.json")
        for row in copied
    )
    queued_request = json.loads(queued[0].read_text(encoding="utf-8"))
    producer = queued_request["producer_identity"]
    assert producer["case_id"] == "full"
    assert producer["source_run_id"] == "deepx_m1_full"
    assert producer["variant"] == "full"
