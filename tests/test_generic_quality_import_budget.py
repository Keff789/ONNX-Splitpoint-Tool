"""Preserve imported validation bytes and budgets without evaluating a CPU reference."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from onnx_splitpoint_tool.benchmark.suite_refresh import refresh_suite_harness
from test_generic_quality_suite_import import _fixture, _import, _write


SUBSET = "resources/validation/detection/val2017_n5000_s20260710"
MARKERS = ("validation_items_requested", "validation_budget_authoritative")


@pytest.fixture(scope="module")
def frozen_subset(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_frozen_n5000")
    image = root / "placeholder_image"
    annotation = root / "placeholder_annotation"
    image.write_bytes(b"synthetic image placeholder; never decoded")
    annotation.write_bytes(b"[]")
    dataset = root / "dataset"
    dataset.mkdir()
    samples = []
    for index in range(5000):
        name = f"{index:012d}.jpg"
        os.link(image, dataset / name)
        os.link(annotation, (dataset / name).with_suffix(".json"))
        samples.append({"image": name, "image_id": index, "sample_id": str(index)})
    _write(dataset / "manifest.json", {
        "schema": "onnx-splitpoint/detection-validation-manifest", "schema_version": 2,
        "dataset": "val2017_n5000_s20260710", "source_type": "materialized_run_mode_subset",
        "selection": {"requested_images": 5000, "selected_images": 5000, "seed": 20260710},
        "samples": samples,
    })
    return dataset


def _budget_fixture(tmp_path, frozen_subset):
    data = _fixture(tmp_path)
    shutil.copytree(frozen_subset, data["source"] / SUBSET, copy_function=os.link)
    source_set = data["source"] / "benchmark_set.json"
    payload = json.loads(source_set.read_text())
    payload["task"] = "detection"
    _write(source_set, payload)
    profile = data["profile"]
    profile["execution_preset"] = {
        "effective": {"validation_items": {"detection": 5000}},
        "snapshot": {"data": {"use_final_dataset_registry": False}},
    }
    profile["validation_execution"] = {"max_items": {"detection": 5000}}
    manifest = data["source"] / SUBSET / "manifest.json"
    profile["quality_gate"]["management_reference"] = {"existing_reference": {
        "path": str(tmp_path / "metadata_only_cpu_reference_not_opened.json"),
        "quality_contract": {"dataset": {
            "image_count": 5000,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }},
    }}
    for run in profile["run_profiles"]:
        run.update(validation_images=SUBSET, validation_max_images=5000, image_scale="norm")
    return data


def _import_detection(data):
    return _import(data, row={"benchmark_set_dir": str(data["source"]), "task": "detection"})


def _rows(plan_path):
    return json.loads(plan_path.read_text())["runs"]


def _assert_bound_rows(rows):
    assert len(rows) == 3
    assert sum(row.get("quality_source_only") is True for row in rows) == 2
    cpu = [row for row in rows if row.get("semantic_reference_only") is True]
    assert len(cpu) == 1 and cpu[0]["automatic_reference"] is True
    assert cpu[0]["variants"] == ["full"]
    for key in ("performance_eligible", "energy_eligible", "ranking_eligible", "pareto_eligible"):
        assert cpu[0][key] is False
    for row in rows:
        assert row["validation_images"] == SUBSET
        assert row["validation_max_images"] == row["validation_items_requested"] == 5000
        assert row["validation_budget_authoritative"] is True
        assert row["benchmark_task"] == "detection"
        assert row["image_scale"] == "norm"


def test_normal_import_and_harness_refresh_preserve_frozen_n5000_for_generic_and_auto_cpu(
        tmp_path, frozen_subset, monkeypatch):
    from onnx_splitpoint_tool.benchmark import validation_assets

    data = _budget_fixture(tmp_path, frozen_subset)
    profile_before = copy.deepcopy(data["profile"])
    source_manifest = data["source"] / SUBSET / "manifest.json"
    original = source_manifest.read_bytes()
    result = _import_detection(data)
    current_manifest = result.suite_dir / SUBSET / "manifest.json"
    assert current_manifest.read_bytes() == original
    for plan in (result.artifacts["benchmark_plan_json"],
                 result.artifacts["legacy_suite_benchmark_plan_json"]):
        _assert_bound_rows(_rows(plan))
    monkeypatch.setattr(validation_assets, "_materialize_detection_subset",
                        lambda *a, **k: pytest.fail("frozen N5000 was rematerialized"))
    refresh_suite_harness(result.suite_dir, validation_images="coco_50",
                          validation_max_images=50, benchmark_task="detection")
    _assert_bound_rows(_rows(result.artifacts["legacy_suite_benchmark_plan_json"]))
    assert current_manifest.read_bytes() == source_manifest.read_bytes() == original
    assert data["profile"] == profile_before
    assert not Path(data["profile"]["quality_gate"]["management_reference"][
        "existing_reference"]["path"]).exists()


@pytest.mark.parametrize("fault", ["mode_budget", "max_items_mismatch", "per_run_budget", "ambiguous_source"])
def test_import_rejects_conflicting_frozen_validation_budget_or_source(tmp_path, frozen_subset, fault):
    data = _budget_fixture(tmp_path, frozen_subset)
    if fault == "mode_budget":
        data["profile"]["execution_preset"]["effective"]["validation_items"]["detection"] = 50
        data["profile"]["validation_execution"]["max_items"]["detection"] = 50
    elif fault == "max_items_mismatch":
        data["profile"]["validation_execution"]["max_items"]["detection"] = 50
    elif fault == "per_run_budget":
        data["profile"]["run_profiles"][1]["validation_max_images"] = 50
    else:
        data["profile"]["run_profiles"][1]["validation_images"] = SUBSET + "_other"
    manifest = data["source"] / SUBSET / "manifest.json"
    original = manifest.read_bytes()
    with pytest.raises(ValueError):
        _import_detection(data)
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("kind", ["no_existing_reference", "non_generic"])
def test_unrelated_import_keeps_existing_validation_bytes_and_absent_markers(tmp_path, frozen_subset, kind):
    data = _budget_fixture(tmp_path, frozen_subset)
    if kind == "no_existing_reference":
        data["profile"]["quality_gate"].pop("management_reference")
    else:
        for row in data["profile"]["run_profiles"]:
            row.pop("quality_source_only")
            for key in ("generic_quality_completion_configs", "generic_quality_completion_config_sha256",
                        "generic_quality_completion_sources"):
                row.pop(key)
    original = (data["source"] / SUBSET / "manifest.json").read_bytes()
    profile_before = copy.deepcopy(data["profile"])
    result = _import_detection(data)
    assert (result.suite_dir / SUBSET / "manifest.json").read_bytes() == original
    assert (data["source"] / SUBSET / "manifest.json").read_bytes() == original
    assert all(key not in row for row in _rows(result.artifacts["benchmark_plan_json"])
               for key in MARKERS)
    assert data["profile"] == profile_before
