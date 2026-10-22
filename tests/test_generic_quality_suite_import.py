from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from onnx_splitpoint_tool.benchmark.evaluation_profiles import profile_model_entries
from onnx_splitpoint_tool.gui.controller import _copy_runner_lib
from onnx_splitpoint_tool.split_export_runners import write_runner_skeleton_onnxruntime
from onnx_splitpoint_tool.workflow.generator_binding import import_existing_suite
from onnx_splitpoint_tool.workflow.generic_quality_sidecars import materialize_generic_quality_sidecars


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path):
    source = tmp_path / "source_suite"
    source.mkdir()
    catalogue = [{"case_id": f"b{i:03d}", "boundary": i,
                  "case_dir": f"b{i:03d}"} for i in range(1, 21)]
    _write(source / "benchmark_set.json", {"cases": catalogue,
           "model_id": "model", "planned_runs": [{"id": "stale_run"}],
           "plan": {"run_id": "old", "runs": [{"id": "stale_run"}]}})
    _write(source / "benchmark_plan.json", {"run_id": "old", "eval_run_id": "old",
           "profile_id": "old", "runs": [{"id": "stale_run"}],
           "management_cpu_reference": {"reference_path": "/stale/reference.json"}})
    (source / "benchmark_suite.py").write_text("pass\n")
    code = source / "splitpoint_runners/completion.py"
    code.parent.mkdir()
    code.write_bytes(b"controlled generated source\n")
    current = tmp_path / "current_runtime_source"
    _copy_runner_lib(current)
    marker = source / "b020/artifact.bin"
    marker.parent.mkdir()
    marker.write_bytes(b"unchanged scientific catalogue artifact\x00")
    runs = []
    for backend, case, run_id in [("hailo8", "b001", "hailo8_to_trt"),
                                  ("deepx_m1", "b002", "deepx_m1_to_tensorrt")]:
        write_runner_skeleton_onnxruntime(str(current / case))
        declared_sources = []
        for relative in [f"{case}/run_split_onnxruntime.py",
                         "splitpoint_runners/generic_quality_completion.py"]:
            target = current / relative
            declared_sources.append({"path": relative,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "size_bytes": target.stat().st_size})
        config = tmp_path / "configs" / f"{case}.json"
        _write(config, {
            "schema": "onnx-splitpoint/generic-composed-quality-completion-config",
            "schema_version": 1, "path_root": "suite_root",
            "identity": {"model_id": "model", "case_id": case,
                         "setup_id": backend + "_setup"},
            "runtime_source_files": declared_sources,
        })
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        runs.append({"id": run_id, "type": "mixed_backend", "stage1": backend,
            "stage2": "tensorrt", "variants": ["composed"],
            "quality_source_only": True, "hardware_setup_id": backend + "_setup",
            "case_ids": [case], "generic_quality_completion_configs": {
                case: f"quality_runtime/{case}.json"},
            "generic_quality_completion_config_sha256": {case: digest},
            "generic_quality_completion_sources": {case: {"path": str(config),
                "sha256": digest, "size_bytes": config.stat().st_size}}})
    run_root = tmp_path / "new_run"
    return {"source": source, "catalogue": catalogue, "profile": {
        "run_profiles": runs,
        "quality_gate": {"statistics": {"execution_location": "central_management"}}},
        "candidate_plan": {"model_id": "model", "selected_candidates": [
            {"case_id": row["case_id"], "split_index": row["boundary"]}
            for row in catalogue]}, "run_root": run_root}


def _import(data, **overrides):
    args = dict(model_id="model", row={"benchmark_set_dir": str(data["source"])},
        model_dir=data["run_root"] / "models/model", run_dir=data["run_root"],
        profile_id="current_profile", run_id="new_run", prediction={},
        candidate_plan=data["candidate_plan"], targets=["hailo8", "deepx_m1", "tensorrt"],
        profile_payload=data["profile"])
    args.update(overrides)
    return import_existing_suite(**args)


def test_generic_import_keeps_twenty_cases_and_current_exact_runtime_bindings(tmp_path):
    data = _fixture(tmp_path)
    source_before = (data["source"] / "benchmark_set.json").read_bytes()
    result = _import(data)
    formal = json.loads(result.artifacts["benchmark_plan_json"].read_text())
    runtime = json.loads(result.artifacts["legacy_suite_benchmark_plan_json"].read_text())
    assert runtime == formal
    assert runtime["run_id"] == "new_run"
    assert runtime["profile_id"] == "current_profile"
    assert "eval_run_id" not in runtime
    assert "management_cpu_reference" not in runtime
    assert len(runtime["cases"]) == result.metrics["accepted_cases"] == 20
    quality = [row for row in runtime["runs"] if row.get("quality_source_only")]
    assert {row["id"] for row in quality} == {row["id"] for row in data["profile"]["run_profiles"]}
    for actual, original in zip(quality, data["profile"]["run_profiles"]):
        for key in ["variants", "case_ids", "hardware_setup_id",
                    "generic_quality_completion_configs",
                    "generic_quality_completion_config_sha256",
                    "generic_quality_completion_sources"]:
            assert actual[key] == original[key]
    runtime_set = json.loads((result.suite_dir / "benchmark_set.json").read_text())
    assert runtime_set["cases"] == data["catalogue"]
    assert runtime_set["planned_runs"] == runtime["runs"]
    assert runtime_set["plan"]["run_id"] == "new_run"
    assert "generic_quality_completion_configs" in (result.suite_dir / "benchmark_suite.py").read_text()
    assert (data["source"] / "benchmark_suite.py").read_text() == "pass\n"
    assert (result.suite_dir / "b020/artifact.bin").read_bytes() == b"unchanged scientific catalogue artifact\x00"
    records = materialize_generic_quality_sidecars(suite_dir=result.suite_dir,
        model_id="model", benchmark_plan=runtime)
    assert len(records) == 2
    for row in quality:
        for case, source in row["generic_quality_completion_sources"].items():
            assert (result.suite_dir / f"quality_runtime/{case}.json").read_bytes() == Path(source["path"]).read_bytes()
    assert (data["source"] / "benchmark_set.json").read_bytes() == source_before


@pytest.mark.parametrize("fault", ["missing", "extra", "changed_boundary", "duplicate",
                                   "outside_runtime_scope", "config_scope", "variants",
                                   "stale_run", "source_model", "missing_source", "unsafe_case"])
def test_generic_import_rejects_scope_or_run_drift_before_copy(tmp_path, fault):
    data = _fixture(tmp_path)
    contract_path = data["source"] / "benchmark_set.json"
    contract = json.loads(contract_path.read_text())
    overrides = {}
    if fault == "missing": contract["cases"].pop()
    elif fault == "extra": contract["cases"].append({"case_id": "b021", "boundary": 21})
    elif fault == "changed_boundary": contract["cases"][0]["boundary"] = 99
    elif fault == "duplicate": contract["cases"][1] = copy.deepcopy(contract["cases"][0])
    elif fault == "outside_runtime_scope": data["profile"]["run_profiles"][0]["case_ids"] = ["b999"]
    elif fault == "config_scope": data["profile"]["run_profiles"][0]["generic_quality_completion_sources"] = {}
    elif fault == "variants": data["profile"]["run_profiles"][0]["variants"] = ["part1", "composed"]
    elif fault == "stale_run": overrides["run_id"] = "old_run"
    elif fault == "missing_source": overrides["row"] = {"benchmark_set_dir": str(tmp_path / "missing_suite")}
    elif fault == "unsafe_case": contract["cases"][0]["case_id"] = "../outside"
    else: contract["model_id"] = "other_model"
    _write(contract_path, contract)
    with pytest.raises(ValueError, match="generic_quality_suite_import_"):
        _import(data, **overrides)
    assert not (data["run_root"] / "models/model/benchmark_set/legacy_suite").exists()


def test_import_does_not_weaken_existing_sidecar_source_guard(tmp_path):
    data = _fixture(tmp_path)
    result = _import(data)
    runtime = json.loads(result.artifacts["legacy_suite_benchmark_plan_json"].read_text())
    source = Path(data["profile"]["run_profiles"][0]["generic_quality_completion_sources"]["b001"]["path"])
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="generic_quality_sidecar_source_bytes_changed"):
        materialize_generic_quality_sidecars(suite_dir=result.suite_dir,
            model_id="model", benchmark_plan=runtime)


def test_model_entry_accepts_existing_suite_without_opening_other_schema_fields():
    schema = json.loads((Path(__file__).parents[1] /
        "onnx_splitpoint_tool/resources/schemas/evaluation_profile.schema.json").read_text())
    model = schema["$defs"]["modelEntryObject"]
    payload = {"id": "model", "onnx": "/model.onnx", "benchmark_set_dir": "/existing/suite"}
    jsonschema.validate(payload, model)
    assert profile_model_entries({"model_suite": {"primary": [payload]}})[0]["benchmark_set_dir"] == "/existing/suite"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**payload, "unknown_suite_bypass": True}, model)
