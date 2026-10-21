from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from onnx_splitpoint_tool.remote.bundle import (
    build_suite_bundle,
    remote_minimal_bundle_patterns,
)
from onnx_splitpoint_tool.workflow.benchmark_binding import (
    _benchmark_runs_from_profile,
)
from onnx_splitpoint_tool.workflow.execution_binding import (
    _prepare_suite_for_runtime,
)
from onnx_splitpoint_tool.workflow.generic_quality_sidecars import (
    materialize_generic_quality_sidecars,
)


def _profile_inputs(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    code = suite / "splitpoint_runners" / "completion.py"
    code.parent.mkdir()
    code.write_bytes(b"controlled shipped source, no inference\n")
    origins = tmp_path / "source_configs"
    origins.mkdir()
    sidecars, hashes, sources = {}, {}, {}
    for case_id in ("b039", "b071"):
        target = origins / f"{case_id}.json"
        target.write_text(json.dumps({
            "schema": "onnx-splitpoint/generic-composed-quality-completion-config",
            "schema_version": 1, "path_root": "suite_root",
            "identity": {"model_id": "yolo26s", "case_id": case_id},
            "expected_image_ids": ["000000000139.jpg", "000000000285.jpg"],
            "runtime_source_files": [{
                "path": "splitpoint_runners/completion.py",
                "sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
                "size_bytes": code.stat().st_size,
            }],
        }))
        sidecars[case_id] = f"quality_runtime/{case_id}.json"
        hashes[case_id] = hashlib.sha256(target.read_bytes()).hexdigest()
        sources[case_id] = {"path": str(target), "sha256": hashes[case_id],
                            "size_bytes": target.stat().st_size}
    profile = {"run_profiles": [{
        "id": "hailo10_to_tensorrt", "type": "mixed_backend",
        "stage1": "hailo10", "stage2": "tensorrt",
        "variants": ["composed"], "case_ids": list(sidecars),
        "quality_source_only": True,
        "generic_quality_completion_configs": sidecars,
        "generic_quality_completion_config_sha256": hashes,
        "generic_quality_completion_sources": sources,
    }]}
    return suite, profile


def test_normal_profile_sidecars_survive_runtime_plan_and_remote_bundle(tmp_path):
    suite, profile = _profile_inputs(tmp_path)
    sidecars = profile["run_profiles"][0]["generic_quality_completion_configs"]
    runs = _benchmark_runs_from_profile(profile, ["hailo10", "tensorrt"])
    plan = {"model_id": "yolo26s", "runs": runs}
    (suite / "benchmark_plan.json").write_text(json.dumps(plan))
    (suite / "benchmark_set.json").write_text(json.dumps({"cases": []}))
    preflight = _prepare_suite_for_runtime(
        suite_dir=suite, run_root=tmp_path, model_id="yolo26s",
        suite_payload={"cases": []}, benchmark_plan=plan,
        model_task="detection",
    )
    assert preflight["status"] == "ok"
    assert len(preflight["generic_quality_completion_sidecars"]) == 2
    assert {r["status"] for r in preflight["generic_quality_completion_sidecars"]} == {
        "copied_exact_bytes",
    }
    before = {path: hashlib.sha256((suite / path).read_bytes()).hexdigest()
              for path in sidecars.values()}
    (suite / "unrelated_analysis.json").write_text("{}")
    includes, excludes = remote_minimal_bundle_patterns()
    archive = tmp_path / "suite.tar.gz"
    build_suite_bundle(suite, archive, includes=includes, excludes=excludes)
    extracted = tmp_path / "remote"
    with tarfile.open(archive) as bundle:
        assert "unrelated_analysis.json" not in bundle.getnames()
        bundle.extractall(extracted, filter="data")
    remote_plan = json.loads((extracted / "benchmark_plan.json").read_text())
    run = remote_plan["runs"][0]
    assert run["quality_source_only"] is True
    assert set(run["case_ids"]) == set(sidecars)
    assert run["generic_quality_completion_configs"] == sidecars
    for case_id, relative in run["generic_quality_completion_configs"].items():
        actual = extracted / relative
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == before[relative]
        assert json.loads(actual.read_text())["identity"]["case_id"] == case_id


@pytest.mark.parametrize("fault", ["source_bytes", "runtime_source", "existing_target",
                                   "path_escape", "scope", "hash_scope"])
def test_sidecar_materialization_rejects_drift_and_preserves_existing_bytes(tmp_path, fault):
    suite, profile = _profile_inputs(tmp_path)
    run = profile["run_profiles"][0]
    source = Path(run["generic_quality_completion_sources"]["b039"]["path"])
    target = suite / "quality_runtime/b039.json"
    if fault == "source_bytes":
        source.write_text(source.read_text() + " ")
    elif fault == "runtime_source":
        (suite / "splitpoint_runners/completion.py").write_text("changed source")
    elif fault == "existing_target":
        target.parent.mkdir()
        target.write_bytes(b"keep existing unequal bytes")
    elif fault == "path_escape":
        run["generic_quality_completion_configs"]["b039"] = "../escape.json"
    elif fault == "scope":
        run["case_ids"] = ["b039"]
    else:
        run["generic_quality_completion_config_sha256"].pop("b071")
    with pytest.raises(ValueError, match="generic_quality_sidecar_"):
        materialize_generic_quality_sidecars(
            suite_dir=suite, model_id="yolo26s", benchmark_plan={"runs": [run]},
        )
    if fault == "existing_target":
        assert target.read_bytes() == b"keep existing unequal bytes"
    else:
        assert not target.exists()


def test_ordinary_profile_does_not_request_sidecar_materialization(tmp_path):
    assert materialize_generic_quality_sidecars(
        suite_dir=tmp_path, model_id="yolo26s",
        benchmark_plan={"runs": [{"id": "ordinary"}]},
    ) == []


def _add_asset(tmp_path, profile):
    run = profile["run_profiles"][0]
    source = Path(run["generic_quality_completion_sources"]["b039"]["path"])
    artifact = tmp_path / "preserved.hef"
    artifact.write_bytes(b"controlled exact artifact bytes\x00\x01")
    config = json.loads(source.read_text())
    config["expected_runtime_artifacts"] = {"part1_runtime": {
        "path": "quality_runtime/assets/b039/part1_runtime/preserved.hef",
        "local_path": str(artifact),
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "size_bytes": artifact.stat().st_size,
    }}
    source.write_text(json.dumps(config))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    run["generic_quality_completion_sources"]["b039"].update(
        sha256=digest, size_bytes=source.stat().st_size,
    )
    run["generic_quality_completion_config_sha256"]["b039"] = digest
    return artifact, source, config


def test_bound_quality_assets_survive_minimal_normal_bundle(tmp_path):
    suite, profile = _profile_inputs(tmp_path)
    artifact, _, config = _add_asset(tmp_path, profile)
    records = materialize_generic_quality_sidecars(
        suite_dir=suite, model_id="yolo26s", benchmark_plan={"runs": profile["run_profiles"]},
    )
    bound = config["expected_runtime_artifacts"]["part1_runtime"]
    assert records[0]["assets"][0]["sha256"] == bound["sha256"]
    includes, excludes = remote_minimal_bundle_patterns()
    archive = tmp_path / "suite.tar.gz"
    build_suite_bundle(suite, archive, includes=includes, excludes=excludes)
    with tarfile.open(archive) as bundle:
        assert bundle.extractfile(bound["path"]).read() == artifact.read_bytes()
    reused = materialize_generic_quality_sidecars(
        suite_dir=suite, model_id="yolo26s", benchmark_plan={"runs": profile["run_profiles"]},
    )
    assert reused[0]["assets"][0]["status"] == "reused_exact_bytes"


@pytest.mark.parametrize("fault", ["source_drift", "existing_bytes", "other_case", "escape"])
def test_quality_asset_copy_rejects_drift_and_scope_escape(tmp_path, fault):
    suite, profile = _profile_inputs(tmp_path)
    artifact, source, config = _add_asset(tmp_path, profile)
    descriptor = config["expected_runtime_artifacts"]["part1_runtime"]
    target = suite / descriptor["path"]
    if fault == "source_drift":
        artifact.write_bytes(b"changed source")
    elif fault == "existing_bytes":
        target.parent.mkdir(parents=True)
        target.write_bytes(b"preserve existing bytes")
    else:
        descriptor["path"] = (
            "quality_runtime/assets/b071/part1.hef" if fault == "other_case"
            else "quality_runtime/assets/b039/../../escape.hef"
        )
        source.write_text(json.dumps(config))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        run = profile["run_profiles"][0]
        run["generic_quality_completion_sources"]["b039"].update(
            sha256=digest, size_bytes=source.stat().st_size,
        )
        run["generic_quality_completion_config_sha256"]["b039"] = digest
    with pytest.raises(ValueError, match="generic_quality_sidecar_"):
        materialize_generic_quality_sidecars(
            suite_dir=suite, model_id="yolo26s", benchmark_plan={"runs": profile["run_profiles"]},
        )
    assert not (suite / "quality_runtime/b039.json").exists()
    if fault == "existing_bytes":
        assert target.read_bytes() == b"preserve existing bytes"
