"""Probe metadata may recover only its exact historical byte binding."""
import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow import supplement
from onnx_splitpoint_tool.workflow.artifacts import sha256_file, write_json


PROBE_FILES = {
    "build_backend_artifacts_request.json": "backend-build-request",
    "backend_artifact_decisions.json": "backend-artifact-decisions",
    "artifact_reuse_manifest.json": "artifact-reuse-manifest",
    "hailo_artifact_service_plan.json": "hailo-artifact-service-plan",
    "hailo_artifact_status.json": "hailo-artifact-status",
    "hailo_expected_unsupported.json": "hailo-expected-unsupported",
    "legacy_suite/hailo_artifacts_manifest.json": "hailo-artifacts-manifest",
    "hailo_build_attempts.json": "hailo-build-attempts",
    "hailo_build_reuse_plan.json": "hailo-build-reuse-plan",
    "hailo_build_queue.json": "hailo-build-queue",
    "hailo_service_dispatch.json": "hailo-service-dispatch",
    "hailo_build_service_dispatch.json": "hailo-build-service-dispatch",
    "hailo_build_service_status.json": "hailo-build-service-status",
    "deepx/deepx_artifact_status.json": "deepx-artifact-status",
    "deepx/deepx_build_queue.json": "deepx-build-queue",
    "deepx/deepx_output_contracts.json": "deepx-output-contracts",
}
INITIAL = "2026-09-26T23:32:29+02:00"
FINISHED = "2026-09-26T23:32:31+02:00"
FRESH = "2026-09-29T19:13:30+02:00"
PROFILE = {"deepx_build": {"mode": "reuse_and_build_missing"}}


def fixture_binding(tmp_path, filename, *, preserved=True):
    run = tmp_path / "run"
    original = run / "jobs/supplement_attempts/first/original"
    raw = f"models/yolo26m/benchmark_set/{filename}"
    payload = {
        "schema": "onnx-splitpoint/" + PROBE_FILES.get(filename, "original-source"),
        "schema_version": 1,
        "model_id": "yolo26m",
        "created_at": FINISHED if filename in {
            "deepx/deepx_build_queue.json", "deepx/deepx_output_contracts.json"
        } else INITIAL,
        "binding": {"boundary": "b038", "precision": "fp16", "artifact_sha256": "a" * 64},
    }
    if filename == "hailo_artifact_service_plan.json":
        payload["source_build_backend_artifacts_request"] = {"created_at": INITIAL, "model_id": "yolo26m"}
    if filename == "deepx/deepx_artifact_status.json":
        payload["mode"] = PROFILE["deepx_build"]["mode"]
    source = write_json(run / raw, payload)
    descriptor = {"path": raw, "sha256": sha256_file(source), "size_bytes": source.stat().st_size,
                  "producer_stage": "build_backend_artifacts", "model_id": "yolo26m"}
    write_json(original / "artifact_index.json", {"artifacts": [descriptor]})
    if preserved:
        write_json(original / raw, payload)
    else:
        write_json(original / "models/yolo26m/benchmark_set/hailo_artifact_service_plan.json",
                   {"created_at": INITIAL})
        write_json(original / "models/yolo26m/stages/build_backend_artifacts/stage_result.json",
                   {"finished_at": FINISHED})
    current = copy.deepcopy(payload)
    current["created_at"] = FRESH
    if filename == "hailo_artifact_service_plan.json":
        current["source_build_backend_artifacts_request"]["created_at"] = FRESH
    if filename == "deepx/deepx_artifact_status.json":
        current["mode"] = "cache_verify_only"
    write_json(source, current)
    return run, original, raw, descriptor


def check(binding, profile=PROFILE):
    return supplement.probe_metadata_binding(*binding, profile)


@pytest.mark.parametrize("filename", PROBE_FILES)
def test_exact_probe_fields_match_existing_descriptor_without_rewriting(tmp_path, filename):
    binding = fixture_binding(tmp_path, filename)
    run, original, raw, _ = binding
    before = {(root / raw): (root / raw).read_bytes() for root in (run, original)}
    assert check(binding)
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("filename", ["legacy_suite/hailo_artifacts_manifest.json",
                                    "deepx/deepx_artifact_status.json",
                                    "deepx/deepx_build_queue.json",
                                    "deepx/deepx_output_contracts.json"])
def test_missing_legacy_backup_requires_exact_historical_hash(tmp_path, filename):
    binding = fixture_binding(tmp_path, filename, preserved=False)
    assert check(binding)
    _, original, raw, descriptor = binding
    assert not (original / raw).exists()
    descriptor["sha256"] = "sha256:" + "0" * 64
    assert check(binding) is None


@pytest.mark.parametrize("field,value", [("boundary", "b043"), ("precision", "int8"),
                                        ("artifact_sha256", "b" * 64)])
def test_contract_changes_cannot_be_hidden_behind_probe_timestamp(tmp_path, field, value):
    binding = fixture_binding(tmp_path, "hailo_artifact_service_plan.json")
    run, _, raw, _ = binding
    current = json.loads((run / raw).read_text())
    current["binding"][field] = value
    write_json(run / raw, current)
    assert check(binding) is None


def test_other_created_at_fields_are_not_volatile(tmp_path):
    binding = fixture_binding(tmp_path, "hailo_artifact_service_plan.json")
    run, _, raw, _ = binding
    payload = json.loads((run / raw).read_text())
    payload["source_build_backend_artifacts_request"]["model_id"] = "yolo26s"
    write_json(run / raw, payload)
    assert check(binding) is None


@pytest.mark.parametrize("damage", ["wrong_profile", "unsafe_mode", "missing_current", "original_tampered"])
def test_deepx_override_and_original_proof_are_fail_closed(tmp_path, damage):
    binding = fixture_binding(tmp_path, "deepx/deepx_artifact_status.json")
    run, original, raw, _ = binding
    if damage == "wrong_profile":
        assert check(binding, {"deepx_build": {"mode": "force_build"}}) is None
        return
    if damage == "missing_current":
        (run / raw).unlink()
        with pytest.raises(ValueError, match="supplement_original_evidence_missing"):
            check(binding)
        return
    else:
        path = (original if damage == "original_tampered" else run) / raw
        payload = json.loads(path.read_text())
        payload["mode"] = "force_build"
        write_json(path, payload)
    assert check(binding) is None


@pytest.mark.parametrize("filename", ["required_run_scope.json", "legacy_suite/split_manifest.json",
                                    "arbitrary_status.json", "deepx/full/deepx_cache_receipt.json"])
def test_no_general_json_or_timestamp_exemption(tmp_path, filename):
    assert check(fixture_binding(tmp_path, filename)) is None


def test_reentry_cannot_rebaseline_a_changed_probe_contract(tmp_path):
    binding = fixture_binding(tmp_path, "hailo_build_queue.json")
    run, original, raw, descriptor = binding
    assert check(binding)
    payload = json.loads((run / raw).read_text())
    payload["binding"]["boundary"] = "b136"
    write_json(run / raw, payload)
    second = run / "jobs/supplement_attempts/second/original"
    write_json(second / raw, payload)
    write_json(second / "artifact_index.json", {"artifacts": [descriptor]})
    assert check((run, original, raw, descriptor)) is None
    assert check((run, second, raw, descriptor)) is None


def original_attempts(tmp_path):
    run = tmp_path / "run"
    write_json(run / "run_manifest.json", {
        "run_id": "run", "execution_sessions": [{"session_id": "z-first"}, {"session_id": "a-second"}]
    })
    for session in ("z-first", "a-second"):
        original = run / "jobs/supplement_attempts" / session / "original"
        write_json(original / "run_manifest.json", {"run_id": "run"})
        write_json(original / "artifact_index.json", {"artifacts": [{"path": session}]})
    write_json(run / "artifact_index.json", {"artifacts": [{"path": "current"}]})
    return run


def test_source_evidence_uses_first_execution_session_without_rebaselining(tmp_path):
    run = original_attempts(tmp_path)
    original, index = supplement.prepare_source_evidence(run)
    assert original == run / "jobs/supplement_attempts/z-first/original"
    assert index == {"artifacts": [{"path": "z-first"}]}


@pytest.mark.parametrize("damage", ["index_missing", "other_run", "symlink"])
def test_later_snapshot_cannot_replace_broken_original_evidence(tmp_path, damage):
    run = original_attempts(tmp_path)
    original = run / "jobs/supplement_attempts/z-first/original"
    if damage == "index_missing":
        (original / "artifact_index.json").unlink()
    elif damage == "other_run":
        write_json(original / "run_manifest.json", {"run_id": "unrelated"})
    else:
        (original / "artifact_index.json").unlink()
        (original / "artifact_index.json").symlink_to(run / "artifact_index.json")
    with pytest.raises(ValueError, match="supplement_"):
        supplement.prepare_source_evidence(run)


def test_unpublished_admission_original_requires_unambiguous_history(tmp_path):
    run = original_attempts(tmp_path)
    write_json(run / "run_manifest.json", {"run_id": "run", "execution_sessions": []})
    with pytest.raises(ValueError, match="supplement_original_attempt_ambiguous"):
        supplement.prepare_source_evidence(run)


@pytest.mark.parametrize("legacy", [False, True])
def test_missing_original_directory_cannot_silently_select_later_snapshot(tmp_path, legacy):
    run = original_attempts(tmp_path)
    manifest = json.loads((run / "run_manifest.json").read_text())
    if legacy:
        manifest["execution_sessions"][0]["error_detail"] = "supplement_source_contract_changed:yolo26m:build_backend_artifacts"
    else:
        manifest["execution_sessions"][0]["supplement_source_root"] = "jobs/supplement_attempts/z-first/original"
    write_json(run / "run_manifest.json", manifest)
    original = run / "jobs/supplement_attempts/z-first/original"
    original.rename(original.with_name("original-unavailable"))
    with pytest.raises(ValueError, match="supplement_original_evidence_missing"):
        supplement.prepare_source_evidence(run)


def test_new_supplement_preserves_the_four_previously_omitted_probe_outputs(tmp_path):
    run = tmp_path / "run"
    filenames = ["legacy_suite/hailo_artifacts_manifest.json", "deepx/deepx_artifact_status.json",
                 "deepx/deepx_build_queue.json", "deepx/deepx_output_contracts.json"]
    for name in filenames:
        write_json(run / "models/yolo26s/benchmark_set" / name, {"original": name})
    original = supplement.preserve_metadata(run, "new-attempt")
    for name in filenames:
        relative = Path("models/yolo26s/benchmark_set") / name
        assert (run / relative).read_bytes() == (original / relative).read_bytes()
        assert (run / relative).stat().st_ino != (original / relative).stat().st_ino
