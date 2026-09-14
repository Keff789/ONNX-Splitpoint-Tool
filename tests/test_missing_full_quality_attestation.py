from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy
from onnx_splitpoint_tool.workflow.artifacts import sha256_json
from onnx_splitpoint_tool.workflow.missing_full_quality_attestation import (
    MissingFullQualityAttestationError,
    attest_current_index_binding,
    attest_current_path,
    attest_generate_benchmark_set_supersession,
    attest_prepare_full_baselines_supersession,
)


MODEL = "mobilenet_v3_large"
TASK = "classification"
RUN_ID = "phase5_hailo8_b500_20260825_170314"
QUALITY_GATE = AccuracyGatePolicy().as_dict()
QUALITY_POLICY_SHA256 = AccuracyGatePolicy.from_mapping(QUALITY_GATE).sha256()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _record(
    run_dir: Path,
    path: Path,
    *,
    stage: str,
    current: bool = True,
) -> dict[str, object]:
    if current:
        raw = path.read_bytes()
        size = len(raw)
        digest = hashlib.sha256(raw).hexdigest()
    else:
        size = 1
        digest = "1" * 64
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "kind": "test",
        "producer_stage": stage,
        "model_id": MODEL,
        "size_bytes": size,
        "sha256": "sha256:" + digest,
    }


def _index(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "onnx-splitpoint/artifact-index",
        "schema_version": 1,
        "run_id": RUN_ID,
        "artifacts": rows,
    }


def _prepare_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], Path, Path]:
    run_dir = tmp_path / RUN_ID
    formal = (
        run_dir / "models" / MODEL / "full_baselines" / "output_contracts.json"
    )
    suite = (
        run_dir / "models" / MODEL / "benchmark_set" / "legacy_suite"
        / "output_contracts.json"
    )
    payload = {
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 3,
        "model_id": MODEL,
        "task": TASK,
        "outputs": [{"name": "logits", "shape": [1, 1000]}],
    }
    _write_json(formal, payload)
    _write_json(suite, payload)
    artifact_index = _index([
        _record(
            run_dir, formal, stage="prepare_full_baselines", current=False,
        ),
        _record(
            run_dir, suite, stage="build_backend_artifacts", current=True,
        ),
    ])
    return run_dir, artifact_index, formal, suite


def _final_cpu_run() -> dict[str, object]:
    return {
        "id": "ort_cpu",
        "type": "onnxruntime",
        "provider": "cpu",
        "backend": "ort_cpu",
        "variant": "full",
        "variants": ["full"],
        "stage1": {"type": "onnxruntime", "provider": "cpu"},
        "stage2": {"type": "onnxruntime", "provider": "cpu"},
        "semantic_reference_only": True,
        "canonical_cpu_reference": True,
        "automatic_reference": True,
        "performance_eligible": False,
        "energy_eligible": False,
        "ranking_eligible": False,
        "pareto_eligible": False,
        "execution_location": "central_management",
        "task": TASK,
        "benchmark_task": TASK,
        "task_quality_gate": copy.deepcopy(QUALITY_GATE),
        "quality_gate_policy_sha256": QUALITY_POLICY_SHA256,
    }


def _generate_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], dict[str, Path]]:
    run_dir = tmp_path / RUN_ID
    base = run_dir / "models" / MODEL / "benchmark_set"
    paths = {
        "immutable_set": base / "legacy_benchmark_set.json",
        "immutable_plan": base / "legacy_benchmark_plan.json",
        "formal_set": base / "benchmark_set.json",
        "formal_plan": base / "benchmark_plan.json",
        "suite_set": base / "legacy_suite" / "benchmark_set.json",
        "suite_plan": base / "legacy_suite" / "benchmark_plan.json",
    }
    immutable_runs = [
        {
            "id": "cpu_ort",
            "backend": "cpu_ort",
            "variant": "full",
            "stage1": "cpu",
            "stage2": "cpu",
        },
        {
            "id": "hailo8",
            "type": "matrix",
            "backend": "hailo8_to_tensorrt",
            "variant": "full",
            "setup_id": "orin_nx_hailo8_01",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
            "stage1": "hailo8",
            "stage2": "tensorrt",
        },
    ]
    final_runs = [
        _final_cpu_run(),
        {
            **immutable_runs[1],
            "stage1": {"type": "hailo", "hw_arch": "hailo8"},
            "stage2": {"type": "tensorrt"},
            "task": TASK,
            "benchmark_task": TASK,
            "task_quality_gate": copy.deepcopy(QUALITY_GATE),
            "quality_gate_policy_sha256": QUALITY_POLICY_SHA256,
        },
    ]
    cases = [{"case_id": "b001", "boundary": 42}]
    immutable_plan = {
        "schema": "onnx-splitpoint/benchmark-plan",
        "schema_version": 1,
        "runs": immutable_runs,
        "objective": "latency",
    }
    immutable_set = {
        "schema": "onnx-splitpoint/benchmark-set",
        "schema_version": 2,
        "model_name": MODEL,
        "cases": cases,
        "plan": immutable_plan,
    }
    invariant = {
        "status": "verified",
        "recipe_count": 1,
        "execution_location": "central_management",
        "performance_dispatch_allowed": False,
        "run_plan_sha256": sha256_json(final_runs),
    }
    final_plan = {
        "schema": "onnx-splitpoint/benchmark-plan",
        "schema_version": 1,
        "task": TASK,
        "model_task": TASK,
        "benchmark_task": TASK,
        "quality_gate_policy_sha256": QUALITY_POLICY_SHA256,
        "quality_gate": copy.deepcopy(QUALITY_GATE),
        "runs": final_runs,
        "planned_runs": final_runs,
        "management_cpu_reference_invariant": invariant,
    }
    suite_set = {
        "schema": "onnx-splitpoint/benchmark-set",
        "schema_version": 2,
        "task": TASK,
        "model_task": TASK,
        "benchmark_task": TASK,
        "quality_gate_policy_sha256": QUALITY_POLICY_SHA256,
        "quality_gate": copy.deepcopy(QUALITY_GATE),
        "cases": cases,
        "planned_runs": final_runs,
        "plan": copy.deepcopy(final_plan),
        "management_cpu_reference_invariant": invariant,
    }
    suite_rel = f"models/{MODEL}/benchmark_set/legacy_suite"
    formal_set = {
        "schema": "onnx-splitpoint/benchmark-set-contract",
        "schema_version": 4,
        "model_id": MODEL,
        "run_id": RUN_ID,
        "task": TASK,
        "model_task": TASK,
        "benchmark_task": TASK,
        "quality_gate_policy_sha256": QUALITY_POLICY_SHA256,
        "quality_gate": copy.deepcopy(QUALITY_GATE),
        "suite_dir": suite_rel,
        "legacy_suite_dir": suite_rel,
        "legacy_suite_benchmark_set": suite_rel + "/benchmark_set.json",
        "benchmark_plan": suite_rel + "/benchmark_plan.json",
        "cases": cases,
        "planned_runs": final_runs,
        "management_cpu_reference_invariant": invariant,
    }
    payloads = {
        "immutable_set": immutable_set,
        "immutable_plan": immutable_plan,
        "formal_set": formal_set,
        "formal_plan": final_plan,
        "suite_set": suite_set,
        "suite_plan": final_plan,
    }
    for key, path in paths.items():
        _write_json(path, payloads[key])
    rows = [
        _record(run_dir, paths[key], stage="generate_benchmark_set", current=True)
        for key in ("immutable_set", "immutable_plan")
    ]
    rows.extend(
        _record(run_dir, paths[key], stage="generate_benchmark_set", current=False)
        for key in ("formal_set", "formal_plan", "suite_set", "suite_plan")
    )
    return run_dir, _index(rows), paths


def test_current_file_and_index_binding_attest_exact_current_bytes(
    tmp_path: Path,
) -> None:
    run_dir, artifact_index, _formal, suite = _prepare_fixture(tmp_path)
    logical = suite.relative_to(run_dir).as_posix()
    raw = suite.read_bytes()

    direct = attest_current_path(
        run_dir=run_dir,
        path=logical,
        expected_size_bytes=len(raw),
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    bound = attest_current_index_binding(
        run_dir=run_dir,
        path=logical,
        artifact_index=artifact_index,
        expected_model_id=MODEL,
        allowed_producer_stages=("build_backend_artifacts",),
    )
    assert direct["ok"] is True
    assert bound["sha256"] == hashlib.sha256(raw).hexdigest()

    suite.write_bytes(raw + b" ")
    with pytest.raises(
        MissingFullQualityAttestationError,
        match="current_path_not_index_bound",
    ):
        attest_current_index_binding(
            run_dir=run_dir,
            path=logical,
            artifact_index=artifact_index,
            expected_model_id=MODEL,
        )


def test_current_file_attestation_rejects_symlinks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    target = run_dir / "target.json"
    _write_json(target, {"ok": True})
    alias = run_dir / "alias.json"
    alias.symlink_to(target)

    with pytest.raises(
        MissingFullQualityAttestationError,
        match="attested_path_symlink_forbidden",
    ):
        attest_current_path(run_dir=run_dir, path="alias.json")


def test_prepare_supersession_requires_byte_identical_current_suite_authority(
    tmp_path: Path,
) -> None:
    run_dir, artifact_index, formal, suite = _prepare_fixture(tmp_path)
    before = {path: path.read_bytes() for path in (formal, suite)}

    result = attest_prepare_full_baselines_supersession(
        run_dir=run_dir,
        model_id=MODEL,
        artifact_index=artifact_index,
    )

    assert result["ok"] is True
    assert result["scope"] == "prepare_full_baselines_output_contract_supersession"
    assert result["byte_identical"] is True
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("mutation", ["formal_bytes", "suite_index", "schema"])
def test_prepare_supersession_fails_closed_on_unattested_aliases(
    tmp_path: Path,
    mutation: str,
) -> None:
    run_dir, artifact_index, formal, suite = _prepare_fixture(tmp_path)
    if mutation == "formal_bytes":
        payload = json.loads(formal.read_text(encoding="utf-8"))
        payload["outputs"][0]["shape"] = [1, 999]
        _write_json(formal, payload)
        expected = "output_contract_alias_bytes_mismatch"
    elif mutation == "suite_index":
        artifact_index["artifacts"][1]["sha256"] = "sha256:" + "2" * 64
        expected = "suite_output_contract_not_currently_index_bound"
    else:
        payload = json.loads(formal.read_text(encoding="utf-8"))
        payload["schema"] = "wrong/schema"
        _write_json(formal, payload)
        _write_json(suite, payload)
        artifact_index["artifacts"][1] = _record(
            run_dir, suite, stage="build_backend_artifacts", current=True,
        )
        expected = "artifact_schema_mismatch"

    with pytest.raises(MissingFullQualityAttestationError, match=expected):
        attest_prepare_full_baselines_supersession(
            run_dir=run_dir,
            model_id=MODEL,
            artifact_index=artifact_index,
        )


@pytest.mark.parametrize(
    "record_mutation",
    ["non_mapping", "backslash_path", "coerced_size", "invalid_historical_sha"],
)
def test_artifact_index_rejects_dirty_records_without_filter_or_coercion(
    tmp_path: Path,
    record_mutation: str,
) -> None:
    run_dir, artifact_index, _formal, _suite = _prepare_fixture(tmp_path)
    rows = artifact_index["artifacts"]
    assert isinstance(rows, list)
    if record_mutation == "non_mapping":
        rows.append("silently ignored before v2.77.10")
    elif record_mutation == "backslash_path":
        rows[1]["path"] = str(rows[1]["path"]).replace("/", "\\")
    elif record_mutation == "coerced_size":
        rows[1]["size_bytes"] = str(rows[1]["size_bytes"])
    else:
        rows[0]["sha256"] = "not-an-old-sha"

    with pytest.raises(
        MissingFullQualityAttestationError,
        match="artifact_index_record_invalid",
    ):
        attest_prepare_full_baselines_supersession(
            run_dir=run_dir,
            model_id=MODEL,
            artifact_index=artifact_index,
        )


def test_generate_supersession_accepts_only_anchored_finalized_alias_family(
    tmp_path: Path,
) -> None:
    run_dir, artifact_index, paths = _generate_fixture(tmp_path)
    before = {path: path.read_bytes() for path in paths.values()}

    result = attest_generate_benchmark_set_supersession(
        run_dir=run_dir,
        model_id=MODEL,
        task=TASK,
        run_id=RUN_ID,
        expected_quality_gate_policy_sha256=QUALITY_POLICY_SHA256,
        artifact_index=artifact_index,
    )

    assert result["ok"] is True
    assert result["ordered_run_ids"] == ["ort_cpu", "hailo8"]
    assert result["case_count"] == 1
    assert result["run_count"] == 2
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("immutable_tamper", "immutable_legacy_alias_not_index_bound"),
        ("task_drift", "benchmark_task_mismatch"),
        ("plan_alias_drift", "final_benchmark_plan_alias_mismatch"),
        ("management_hash", "management_cpu_reference_invariant_invalid"),
        ("management_missing", "management_cpu_reference_invariant_alias_missing"),
        ("quality_hash", "quality_gate_policy_sha256_mismatch"),
        ("quality_semantics", "quality_gate_policy_semantic_mismatch"),
        ("run_quality_hash", "quality_gate_policy_sha256_mismatch"),
        ("run_quality_semantics", "quality_gate_policy_semantic_mismatch"),
        ("artifact_run", "artifact_index_run_id_mismatch"),
    ],
)
def test_generate_supersession_rejects_anchor_or_finalization_drift(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    run_dir, artifact_index, paths = _generate_fixture(tmp_path)
    if mutation == "immutable_tamper":
        payload = json.loads(paths["immutable_set"].read_text(encoding="utf-8"))
        payload["cases"].append({"case_id": "b999"})
        _write_json(paths["immutable_set"], payload)
    elif mutation == "task_drift":
        payload = json.loads(paths["formal_set"].read_text(encoding="utf-8"))
        payload["task"] = "detection"
        _write_json(paths["formal_set"], payload)
    elif mutation == "plan_alias_drift":
        payload = json.loads(paths["suite_plan"].read_text(encoding="utf-8"))
        payload["runs"][1]["backend"] = "deepx_to_tensorrt"
        payload["planned_runs"] = copy.deepcopy(payload["runs"])
        _write_json(paths["suite_plan"], payload)
    elif mutation == "management_hash":
        for key in ("formal_set", "formal_plan", "suite_set", "suite_plan"):
            payload = json.loads(paths[key].read_text(encoding="utf-8"))
            payload["management_cpu_reference_invariant"]["run_plan_sha256"] = (
                "sha256:" + "3" * 64
            )
            _write_json(paths[key], payload)
    elif mutation == "management_missing":
        for key in ("formal_set", "formal_plan", "suite_set", "suite_plan"):
            payload = json.loads(paths[key].read_text(encoding="utf-8"))
            payload.pop("management_cpu_reference_invariant")
            _write_json(paths[key], payload)
    elif mutation == "quality_hash":
        payload = json.loads(paths["suite_set"].read_text(encoding="utf-8"))
        payload["quality_gate_policy_sha256"] = "b" * 64
        _write_json(paths["suite_set"], payload)
    elif mutation == "quality_semantics":
        payload = json.loads(paths["formal_plan"].read_text(encoding="utf-8"))
        payload["quality_gate"]["bootstrap_repetitions"] += 1
        _write_json(paths["formal_plan"], payload)
    elif mutation in {"run_quality_hash", "run_quality_semantics"}:
        for key in ("formal_set", "formal_plan", "suite_set", "suite_plan"):
            payload = json.loads(paths[key].read_text(encoding="utf-8"))
            run_lists = []
            for field in ("runs", "planned_runs"):
                if isinstance(payload.get(field), list):
                    run_lists.append(payload[field])
            embedded = payload.get("plan")
            if isinstance(embedded, dict):
                for field in ("runs", "planned_runs"):
                    if isinstance(embedded.get(field), list):
                        run_lists.append(embedded[field])
            for run_list in run_lists:
                if mutation == "run_quality_hash":
                    run_list[1]["quality_gate_policy_sha256"] = "b" * 64
                else:
                    run_list[1]["task_quality_gate"][
                        "bootstrap_repetitions"
                    ] += 1
            _write_json(paths[key], payload)
    else:
        artifact_index["run_id"] = "another_run"

    with pytest.raises(MissingFullQualityAttestationError, match=error):
        attest_generate_benchmark_set_supersession(
            run_dir=run_dir,
            model_id=MODEL,
            task=TASK,
            run_id=RUN_ID,
            expected_quality_gate_policy_sha256=QUALITY_POLICY_SHA256,
            artifact_index=artifact_index,
        )
