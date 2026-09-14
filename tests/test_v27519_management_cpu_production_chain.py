from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService
from onnx_splitpoint_tool.management_reference import (
    _cpu_reference_run,
    _publish_immutable_reference,
    _source_contract,
    bind_management_cpu_reference_runs,
    finalize_management_cpu_reference_plan_aliases,
    generate_management_cpu_reference,
)
from onnx_splitpoint_tool.workflow.execution_binding import (
    _performance_run_ids_v263,
)
from onnx_splitpoint_tool.workflow.generator_binding import (
    import_existing_suite,
    materialize_suite_from_candidate_plan,
)
from onnx_splitpoint_tool.workflow.runner import (
    _model_row_suite_import_allowed_v27519,
    _performance_plan_v263,
    expected_profile_measurements_v60r,
)
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions


def _central_profile() -> dict[str, object]:
    return {
        "run_profiles": [{
            "id": "ort_tensorrt",
            "full": "tensorrt",
            "stage1": "tensorrt",
            "stage2": "tensorrt",
        }],
        "quality_gate": {
            "statistics": {"execution_location": "central_management"},
        },
    }


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _classification_reference(*, top1_hit: bool) -> dict[str, object]:
    return {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "task": "classification",
        "records": [{
            "image_id": "sample-1",
            "reference": {"top1_hit": top1_hit, "top5_hit": True},
        }],
    }


def _write_minimal_reference_suite(
    suite: Path,
    *,
    contract_marker: str,
    top1_hit: bool,
) -> None:
    suite.mkdir(parents=True, exist_ok=True)
    (suite / "benchmark_plan.json").write_text(json.dumps({
        "runs": [{
            "id": "ort_cpu",
            "provider": "cpu",
            "semantic_reference_only": True,
        }],
    }), encoding="utf-8")
    (suite / "benchmark_set.json").write_text(json.dumps({
        "cases": [{"case_id": "b001", "boundary": 1}],
        "contract_marker": contract_marker,
    }), encoding="utf-8")
    payload = repr(_classification_reference(top1_hit=top1_hit))
    (suite / "benchmark_suite.py").write_text(
        "from pathlib import Path\n"
        "import json\n"
        f"payload = {payload}\n"
        "out = Path('task_quality_inputs')\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'canonical_classification_reference.json').write_text(\n"
        "    json.dumps(payload), encoding='utf-8')\n",
        encoding="utf-8",
    )


def _assert_semantic_aliases(
    executable_path: Path,
    formal_path: Path,
) -> dict[str, object]:
    executable = _read(executable_path)
    formal = _read(formal_path)
    assert executable["runs"] == formal["runs"]
    cpu_rows = [
        row for row in executable["runs"]
        if row.get("id") == "ort_cpu"
    ]
    assert len(cpu_rows) == 1
    cpu = cpu_rows[0]
    assert cpu["semantic_reference_only"] is True
    assert cpu["canonical_cpu_reference"] is True
    assert cpu["automatic_reference"] is True
    assert cpu["performance_eligible"] is False
    assert cpu["energy_eligible"] is False
    assert cpu["ranking_eligible"] is False
    assert cpu["pareto_eligible"] is False
    assert cpu["execution_location"] == "central_management"
    assert executable["management_cpu_reference_invariant"]["status"] == "verified"
    assert formal["management_cpu_reference_invariant"]["status"] == "verified"
    return executable


def test_real_service_writer_round_trips_into_management_consumer(
    tmp_path: Path,
) -> None:
    """Exercise service writer -> both files -> isolated consumer process."""

    profile = _central_profile()
    suite = tmp_path / "legacy_suite"
    suite.mkdir()
    (suite / "b001").mkdir()
    model = suite / "model.onnx"
    model.write_bytes(b"fixture-model")
    log_path = suite / "benchmark_generation.log"
    log_path.write_text("", encoding="utf-8")

    run_plan = BenchmarkGenerationService().build_run_plan(
        acc_cpu=True,
        acc_cuda=False,
        acc_trt=True,
        acc_h8=False,
        acc_h10=False,
        acc_deepx=False,
        hailo_custom_full=False,
        hailo_custom_composed=False,
        hailo_custom_part1=False,
        hailo_custom_part2=False,
        matrix_trt_to_hailo=False,
        matrix_hailo_to_trt=False,
    )
    rows = bind_management_cpu_reference_runs(
        run_plan.bench_plan_runs,
        automatic=True,
        require_existing=True,
    )

    def _write_fake_harness(out_dir: str, _bench_name: str) -> str:
        harness = Path(out_dir) / "benchmark_suite.py"
        harness.write_text(
            "from pathlib import Path\n"
            "import json\n"
            "out = Path('task_quality_inputs')\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'canonical_classification_reference.json').write_text(\n"
            "    json.dumps({\n"
            "        'schema': 'onnx-splitpoint/task-quality-reference-input',\n"
            "        'schema_version': 1,\n"
            "        'reference_role': 'canonical_cpu_ort',\n"
            "        'semantic_reference_only': True,\n"
            "        'task': 'classification',\n"
            "        'records': [{'image_id': 'sample-1', 'reference': {\n"
            "            'top1_hit': True, 'top5_hit': True}}],\n"
            "    }), encoding='utf-8')\n",
            encoding="utf-8",
        )
        return str(harness)

    BenchmarkGenerationService().finalize_generation_outputs(
        out_dir=suite,
        base="model",
        full_model_src=str(model),
        full_model_dst=str(model),
        analysis_params={},
        system_spec=None,
        cases=[{"case_id": "b001", "folder": "b001", "boundary": 1}],
        errors=[],
        discarded_cases=[],
        requested_cases=1,
        preferred_shortlist_original=[1],
        ranked_candidates=[1],
        shortlist_prefiltered_boundaries=[],
        candidate_search_pool=[1],
        bench_log_path=log_path,
        analysis_payload={},
        bench_plan_runs=rows,
        hef_targets=[],
        hef_full=False,
        hef_part1=False,
        hef_part2=False,
        hef_backend="local",
        hef_wsl_distro=None,
        hef_wsl_venv="",
        hef_opt_level=0,
        hef_calib_count=0,
        hef_calib_bs=1,
        hef_calib_dir=None,
        hef_fixup=False,
        hef_force=False,
        hef_keep=False,
        suite_hailo_hefs=None,
        write_harness_script=_write_fake_harness,
    )

    formal_plan = tmp_path / "benchmark_set" / "benchmark_plan.json"
    invariant = finalize_management_cpu_reference_plan_aliases(
        executable_plan_path=suite / "benchmark_plan.json",
        formal_plan_path=formal_plan,
        profile=profile,
        cache_verify_enabled=False,
        automatic=True,
        require_existing=True,
        benchmark_set_paths=(suite / "benchmark_set.json",),
    )
    assert invariant["status"] == "verified"
    serialized = _assert_semantic_aliases(
        suite / "benchmark_plan.json", formal_plan,
    )
    suite_contract = _read(suite / "benchmark_set.json")
    assert suite_contract["planned_runs"] == serialized["runs"]
    assert suite_contract["plan"]["runs"] == serialized["runs"]
    assert _cpu_reference_run(serialized) is not None
    assert _performance_run_ids_v263(serialized) == ["ort_tensorrt"]
    assert _performance_plan_v263(serialized, profile)["runs"] == [
        next(row for row in serialized["runs"] if row["id"] == "ort_tensorrt")
    ]
    required = expected_profile_measurements_v60r(
        model_id="model",
        benchmark_plan=serialized,
        benchmark_set_contract=suite_contract,
    )
    assert required
    assert not any(row["backend"] == "cpu_ort" for row in required)

    status = generate_management_cpu_reference(
        suite_dir=suite,
        output_dir=tmp_path / "quality_reference",
        model_id="model",
        workers=1,
        timeout_s=10,
    )
    assert status["status"] == "completed"
    assert status["semantic_reference_only"] is True
    assert status["record_count"] == 1


def test_imported_suite_gets_one_recipe_in_both_plan_aliases(
    tmp_path: Path,
) -> None:
    profile = _central_profile()
    source = tmp_path / "source_suite"
    source.mkdir()
    (source / "benchmark_set.json").write_text(json.dumps({
        "cases": [{"case_id": "b001", "boundary": 1}],
        "planned_runs": [{"id": "stale_hailo8_to_trt"}],
    }), encoding="utf-8")
    (source / "benchmark_plan.json").write_text(json.dumps({
        "runs": [
            {
                "id": "ort_tensorrt", "provider": "tensorrt",
                "stage1": {"type": "onnxruntime", "provider": "tensorrt"},
                "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            },
            {
                "id": "stale_hailo8_to_trt", "provider": "hailo8",
                "stage1": {"type": "hailo", "hw_arch": "hailo8"},
                "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            },
        ],
    }), encoding="utf-8")
    (source / "benchmark_suite.py").write_text("pass\n", encoding="utf-8")

    model_dir = tmp_path / "models" / "model"
    result = import_existing_suite(
        model_id="model",
        row={"benchmark_set_dir": str(source)},
        model_dir=model_dir,
        run_dir=tmp_path,
        profile_id="profile",
        run_id="run",
        prediction={},
        candidate_plan={
            "selected_candidates": [{"case_id": "b001", "split_index": 1}],
        },
        targets=["tensorrt", "hailo8"],
        profile_payload=profile,
    )
    assert result is not None
    serialized = _assert_semantic_aliases(
        result.suite_dir / "benchmark_plan.json",
        model_dir / "benchmark_set" / "benchmark_plan.json",
    )
    assert _performance_run_ids_v263(serialized) == ["ort_tensorrt"]
    assert "stale_hailo8_to_trt" not in {
        row["id"] for row in serialized["runs"]
    }
    assert _read(result.suite_dir / "benchmark_set.json")["planned_runs"] == serialized["runs"]
    assert _read(model_dir / "benchmark_set" / "benchmark_set.json")["planned_runs"] == serialized["runs"]
    assert result.metrics["management_cpu_reference_invariant"]["status"] == "verified"


def test_direct_materializer_adds_internal_recipe_after_target_expansion(
    tmp_path: Path,
) -> None:
    profile = _central_profile()
    model_dir = tmp_path / "models" / "model"
    result = materialize_suite_from_candidate_plan(
        model_id="model",
        model_path=str(tmp_path / "missing.onnx"),
        model_dir=model_dir,
        run_dir=tmp_path,
        profile_id="profile",
        run_id="run",
        prediction={},
        candidate_plan={
            "selected_candidates": [{"case_id": "b001", "split_index": 1}],
        },
        targets=["tensorrt"],
        full_baseline_plan={},
        output_contracts={},
        profile_payload=profile,
        model_entry={"id": "model", "task": "classification"},
        dry_run=True,
    )
    serialized = _assert_semantic_aliases(
        result.suite_dir / "benchmark_plan.json",
        model_dir / "benchmark_set" / "benchmark_plan.json",
    )
    assert [row["id"] for row in serialized["runs"]] == [
        "ort_tensorrt", "ort_cpu",
    ]
    assert serialized["targets"] == ["tensorrt"]
    assert _performance_run_ids_v263(serialized) == ["ort_tensorrt"]
    assert _read(result.suite_dir / "benchmark_set.json")["planned_runs"] == serialized["runs"]
    assert _read(model_dir / "benchmark_set" / "benchmark_set.json")["planned_runs"] == serialized["runs"]
    assert result.metrics["management_cpu_reference_invariant"]["status"] == "verified"


def test_direct_full_only_plan_is_exact_four_plus_cpu_without_splits(
    tmp_path: Path,
) -> None:
    profile = {
        "run_profiles": [
            {
                "id": "ort_tensorrt", "type": "same_backend_reference",
                "full": "tensorrt", "stage1": "tensorrt",
                "stage2": "tensorrt",
            },
            {
                "id": "hailo8", "type": "same_backend_reference",
                "full": "hailo8", "stage1": "hailo8", "stage2": "hailo8",
            },
            {
                "id": "hailo10", "type": "same_backend_reference",
                "full": "hailo10", "stage1": "hailo10", "stage2": "hailo10",
            },
            {
                "id": "deepx_m1_full", "type": "same_backend_reference",
                "full": "deepx_m1", "stage1": "deepx_m1",
                "stage2": "deepx_m1",
            },
        ],
        "quality_gate": {
            "statistics": {"execution_location": "central_management"},
        },
    }
    model_dir = tmp_path / "models" / "model"
    result = materialize_suite_from_candidate_plan(
        model_id="model",
        model_path=str(tmp_path / "missing.onnx"),
        model_dir=model_dir,
        run_dir=tmp_path,
        profile_id="profile",
        run_id="run",
        prediction={},
        candidate_plan={
            "selected_candidates": [{"case_id": "b001", "split_index": 1}],
        },
        targets=["tensorrt", "hailo8", "hailo10", "deepx_m1"],
        full_baseline_plan={},
        output_contracts={},
        profile_payload=profile,
        model_entry={"id": "model", "task": "classification"},
        dry_run=True,
    )
    expected_ids = [
        "ort_tensorrt", "hailo8", "hailo10", "deepx_m1_full", "ort_cpu",
    ]
    serialized = _assert_semantic_aliases(
        result.suite_dir / "benchmark_plan.json",
        model_dir / "benchmark_set" / "benchmark_plan.json",
    )
    assert [row["id"] for row in serialized["runs"]] == expected_ids
    assert [row["type"] for row in serialized["runs"]] == [
        "onnxruntime", "hailo", "hailo", "deepx", "onnxruntime",
    ]
    assert all(row["variants"] == ["full"] for row in serialized["runs"])
    assert not any(
        "_to_" in str(row.get("id") or "")
        or str(row.get("variant") or "").lower() == "split"
        for row in serialized["runs"]
    )
    for path in (
        result.suite_dir / "benchmark_set.json",
        model_dir / "benchmark_set" / "benchmark_set.json",
    ):
        assert [row["id"] for row in _read(path)["planned_runs"]] == expected_ids
    assert _performance_run_ids_v263(serialized) == expected_ids[:-1]


def test_fresh_field_run_prohibits_model_row_suite_import() -> None:
    assert _model_row_suite_import_allowed_v27519(
        WorkflowOptions(profile="", out="", require_fresh_run=False),
    ) is True
    assert _model_row_suite_import_allowed_v27519(
        WorkflowOptions(profile="", out="", require_fresh_run=True),
    ) is False


def test_management_reference_resume_is_immutable_per_source_contract(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    output = tmp_path / "quality_management" / "references" / "model"
    _write_minimal_reference_suite(
        suite,
        contract_marker="contract-a",
        top1_hit=True,
    )

    first = generate_management_cpu_reference(
        suite_dir=suite,
        output_dir=output,
        model_id="model",
        workers=1,
        timeout_s=10,
    )
    assert first["status"] == "completed"
    assert first["reference_storage"] == "immutable_source_contract"
    assert first["reference_immutable"] is True
    first_contract = str(first["source_contract_sha256"]).removeprefix("sha256:")
    first_path = Path(str(first["reference_path"]))
    assert first_path == (
        output / "by_source_contract" / first_contract
        / "canonical_cpu_reference.json"
    )
    assert first["reference_sha256"] == _sha256(first_path)
    assert first["reference_size_bytes"] == first_path.stat().st_size
    assert not (output / "canonical_cpu_reference.json").exists()

    first_bytes = first_path.read_bytes()
    first_stat = first_path.stat()
    cached = generate_management_cpu_reference(
        suite_dir=suite,
        output_dir=output,
        model_id="model",
        workers=1,
        timeout_s=10,
    )
    assert cached["status"] == "cache_hit"
    assert cached["reference_path"] == first["reference_path"]
    assert cached["reference_sha256"] == first["reference_sha256"]
    assert first_path.read_bytes() == first_bytes
    assert first_path.stat().st_ino == first_stat.st_ino
    assert first_path.stat().st_mtime_ns == first_stat.st_mtime_ns

    _write_minimal_reference_suite(
        suite,
        contract_marker="contract-b",
        top1_hit=False,
    )
    second = generate_management_cpu_reference(
        suite_dir=suite,
        output_dir=output,
        model_id="model",
        workers=1,
        timeout_s=10,
    )
    assert second["status"] == "completed"
    second_contract = str(second["source_contract_sha256"]).removeprefix("sha256:")
    second_path = Path(str(second["reference_path"]))
    assert second_path == (
        output / "by_source_contract" / second_contract
        / "canonical_cpu_reference.json"
    )
    assert second_path != first_path
    assert second["reference_sha256"] == _sha256(second_path)
    assert first_path.read_bytes() == first_bytes
    assert first_path.stat().st_ino == first_stat.st_ino
    assert first_path.stat().st_mtime_ns == first_stat.st_mtime_ns
    assert _read(first_path)["records"][0]["reference"]["top1_hit"] is True
    assert _read(second_path)["records"][0]["reference"]["top1_hit"] is False

    second_cached = generate_management_cpu_reference(
        suite_dir=suite,
        output_dir=output,
        model_id="model",
        workers=1,
        timeout_s=10,
    )
    assert second_cached["status"] == "cache_hit"
    assert second_cached["reference_path"] == second["reference_path"]
    assert second_cached["reference_sha256"] == second["reference_sha256"]
    second_bytes = second_path.read_bytes()
    second_stat = second_path.stat()

    # A real resumed campaign may return to an earlier suite/contract after a
    # newer invocation.  The old immutable artifact must be selected directly;
    # neither A nor B may be rewritten through a mutable "latest" file.
    _write_minimal_reference_suite(
        suite,
        contract_marker="contract-a",
        top1_hit=True,
    )
    resumed_first = generate_management_cpu_reference(
        suite_dir=suite,
        output_dir=output,
        model_id="model",
        workers=1,
        timeout_s=10,
    )
    assert resumed_first["status"] == "cache_hit"
    assert resumed_first["reference_path"] == first["reference_path"]
    assert resumed_first["reference_sha256"] == first["reference_sha256"]
    assert first_path.read_bytes() == first_bytes
    assert first_path.stat().st_ino == first_stat.st_ino
    assert first_path.stat().st_mtime_ns == first_stat.st_mtime_ns
    assert second_path.read_bytes() == second_bytes
    assert second_path.stat().st_ino == second_stat.st_ino
    assert second_path.stat().st_mtime_ns == second_stat.st_mtime_ns


def test_management_reference_same_contract_different_bytes_fails_closed(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first.json"
    second_source = tmp_path / "second.json"
    first_source.write_text(
        json.dumps(_classification_reference(top1_hit=True)),
        encoding="utf-8",
    )
    second_source.write_text(
        json.dumps(_classification_reference(top1_hit=False)),
        encoding="utf-8",
    )
    target = (
        tmp_path / "references" / "model" / "by_source_contract"
        / ("a" * 64) / "canonical_cpu_reference.json"
    )

    _metadata, first_sha256, cache_hit = _publish_immutable_reference(
        first_source,
        target,
    )
    assert cache_hit is False
    original_bytes = target.read_bytes()
    original_stat = target.stat()

    with pytest.raises(RuntimeError, match="source-contract collision"):
        _publish_immutable_reference(second_source, target)

    assert target.read_bytes() == original_bytes
    assert _sha256(target) == first_sha256
    assert target.stat().st_ino == original_stat.st_ino
    assert target.stat().st_mtime_ns == original_stat.st_mtime_ns


def test_management_reference_migrates_bound_legacy_file_without_using_it_as_evidence(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    output = tmp_path / "quality_management" / "references" / "model"
    _write_minimal_reference_suite(
        suite,
        contract_marker="legacy-contract",
        top1_hit=True,
    )
    plan = _read(suite / "benchmark_plan.json")
    contract = _read(suite / "benchmark_set.json")
    source_contract_sha256 = _source_contract(suite, plan, contract)
    legacy = output / "canonical_cpu_reference.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(_classification_reference(top1_hit=True)),
        encoding="utf-8",
    )
    legacy_bytes = legacy.read_bytes()
    legacy_stat = legacy.stat()
    status_path = output / "management_cpu_reference_status.json"
    status_path.write_text(json.dumps({
        "schema": "onnx-splitpoint/management-cpu-reference-job",
        "schema_version": 1,
        "status": "completed",
        "model_id": "model",
        "source_contract_sha256": source_contract_sha256,
        "reference_path": str(legacy),
        "reference_sha256": _sha256(legacy),
    }), encoding="utf-8")

    migrated = generate_management_cpu_reference(
        suite_dir=suite,
        output_dir=output,
        model_id="model",
        workers=1,
        timeout_s=10,
    )

    assert migrated["status"] == "cache_hit"
    assert migrated["legacy_reference_migrated"] is True
    immutable = Path(str(migrated["reference_path"]))
    assert immutable != legacy
    assert immutable == (
        output / "by_source_contract"
        / str(source_contract_sha256).removeprefix("sha256:")
        / "canonical_cpu_reference.json"
    )
    assert immutable.read_bytes() == legacy_bytes
    assert migrated["reference_sha256"] == _sha256(immutable)
    assert migrated["reference_size_bytes"] == len(legacy_bytes)
    assert legacy.read_bytes() == legacy_bytes
    assert legacy.stat().st_ino == legacy_stat.st_ino
    assert legacy.stat().st_mtime_ns == legacy_stat.st_mtime_ns
    persisted_status = _read(status_path)
    assert persisted_status["reference_path"] == str(immutable)
    assert persisted_status["reference_path"] != str(legacy)


@pytest.mark.parametrize("legacy_problem", ["wrong_hash", "symlink"])
def test_management_reference_rejects_unbound_or_unsafe_legacy_file(
    tmp_path: Path,
    legacy_problem: str,
) -> None:
    suite = tmp_path / "suite"
    output = tmp_path / "quality_management" / "references" / "model"
    _write_minimal_reference_suite(
        suite,
        contract_marker="legacy-contract",
        top1_hit=True,
    )
    source_contract_sha256 = _source_contract(
        suite,
        _read(suite / "benchmark_plan.json"),
        _read(suite / "benchmark_set.json"),
    )
    legacy = output / "canonical_cpu_reference.json"
    legacy.parent.mkdir(parents=True)
    external = tmp_path / "external-reference.json"
    external.write_text(
        json.dumps(_classification_reference(top1_hit=True)),
        encoding="utf-8",
    )
    if legacy_problem == "symlink":
        legacy.symlink_to(external)
        expected_hash = _sha256(external)
    else:
        legacy.write_bytes(external.read_bytes())
        expected_hash = "sha256:" + ("f" * 64)
        assert expected_hash != _sha256(legacy)
    before_external = external.read_bytes()
    status_path = output / "management_cpu_reference_status.json"
    status_path.write_text(json.dumps({
        "schema": "onnx-splitpoint/management-cpu-reference-job",
        "schema_version": 1,
        "status": "completed",
        "model_id": "model",
        "source_contract_sha256": source_contract_sha256,
        "reference_path": str(legacy),
        "reference_sha256": expected_hash,
    }), encoding="utf-8")

    result = generate_management_cpu_reference(
        suite_dir=suite,
        output_dir=output,
        model_id="model",
        workers=1,
        timeout_s=10,
    )

    assert result["status"] == "failed"
    assert "reference_path" not in result
    immutable = (
        output / "by_source_contract"
        / str(source_contract_sha256).removeprefix("sha256:")
        / "canonical_cpu_reference.json"
    )
    assert not immutable.exists()
    assert external.read_bytes() == before_external
    if legacy_problem == "symlink":
        assert legacy.is_symlink()
        assert legacy.resolve() == external.resolve()
    else:
        assert legacy.read_bytes() == before_external


def test_management_reference_rejects_symlinked_diagnostic_status(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    output = tmp_path / "quality_management" / "references" / "model"
    _write_minimal_reference_suite(
        suite,
        contract_marker="contract",
        top1_hit=True,
    )
    output.mkdir(parents=True)
    external = tmp_path / "external-status.json"
    external.write_text('{"must_remain": true}\n', encoding="utf-8")
    external_bytes = external.read_bytes()
    status_path = output / "management_cpu_reference_status.json"
    status_path.symlink_to(external)

    result = generate_management_cpu_reference(
        suite_dir=suite,
        output_dir=output,
        model_id="../unsafe-model-id",
        workers=1,
        timeout_s=10,
    )

    assert result["status"] == "failed"
    assert "status is not a regular file" in str(result["error"])
    assert status_path.is_symlink()
    assert external.read_bytes() == external_bytes
    assert not (output / "by_source_contract").exists()
