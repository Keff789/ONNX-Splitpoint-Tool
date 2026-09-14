from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from onnx_splitpoint_tool.build_evidence import (
    ABORTED_UNKNOWN,
    ARTIFACT_PASS,
    BUILD_CONTEXT_SCHEMA,
    COMPILE_INFEASIBLE,
    HAILO_CACHE_SCHEMA_V3,
    HAILO_RECEIPT_SCHEMA,
    PARSER_UNSUPPORTED,
    TRANSIENT_INFRASTRUCTURE,
    BuildEvidenceError,
    boundary_endpoint_contract_sha256,
    build_evidence_index,
    build_key_sha256,
    canonical_build_key,
    canonical_build_key_from_hailo_v3,
    canonical_build_key_from_hailo_v3_payload,
    canonical_sha256,
    classify_build_outcome,
    create_build_evidence_index,
    harvest_b5_run,
    lookup_build_evidence,
    make_build_evidence_record,
    materialize_verified_artifact,
    verify_hailo_artifact,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _base_key_kwargs() -> dict[str, object]:
    return {
        "full_source_onnx_sha256": "1" * 64,
        "builder_source_onnx_sha256": "2" * 64,
        "compiler_onnx_sha256": "3" * 64,
        "boundary_endpoint_contract_sha256": "4" * 64,
        "backend": "hailo_dfc",
        "hw_arch": "hailo8",
        "compiler_version": "hailo-dataflow-compiler:3.33.1",
        "recipe": {
            "optimization_level": 1,
            "model_script_sha256": "5" * 64,
            "start_nodes": [],
            "end_nodes": ["head"],
        },
        "calibration": {
            "identity": "manifest:" + "6" * 64,
            "effective_count": 5,
            "requested_count": 5,
            "batch_size": 1,
        },
        "preprocessing_contract_sha256": "7" * 64,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(full_source_onnx_sha256="8" * 64),
        lambda value: value.update(builder_source_onnx_sha256="8" * 64),
        lambda value: value.update(compiler_onnx_sha256="8" * 64),
        lambda value: value.update(boundary_endpoint_contract_sha256="8" * 64),
        lambda value: value.update(backend="deepx_compiler"),
        lambda value: value.update(hw_arch="hailo10h"),
        lambda value: value.update(compiler_version="hailo-dataflow-compiler:5.3.0"),
        lambda value: value["recipe"].update(optimization_level=2),
        lambda value: value["recipe"].update(model_script_sha256="8" * 64),
        lambda value: value["calibration"].update(identity="manifest:" + "8" * 64),
        lambda value: value["calibration"].update(
            effective_count=4, requested_count=4
        ),
        lambda value: value.update(preprocessing_contract_sha256="8" * 64),
    ],
)
def test_v2783_exact_build_key_changes_for_every_required_axis(mutate) -> None:
    baseline_args = _base_key_kwargs()
    baseline = build_key_sha256(canonical_build_key(**baseline_args))
    changed_args = copy.deepcopy(baseline_args)
    mutate(changed_args)
    changed = build_key_sha256(canonical_build_key(**changed_args))
    assert changed != baseline


@pytest.mark.parametrize(
    ("result", "log_text", "terminal", "expected"),
    [
        ({"ok": True}, "", True, ARTIFACT_PASS),
        (
            {"ok": False, "error": "UnsupportedShuffleLayerError: node /x"},
            "",
            True,
            PARSER_UNSUPPORTED,
        ),
        (
            {"ok": False, "error": "Mapping Failed: No successful assignments; Agent infeasible"},
            "Unable to register cuDNN factory (startup warning)",
            True,
            COMPILE_INFEASIBLE,
        ),
        (
            {"ok": False, "timed_out": True, "timeout_kind": "hard"},
            "",
            True,
            TRANSIENT_INFRASTRUCTURE,
        ),
        (
            {"ok": False, "error": "CUDA_DNN.CC host failure"},
            "",
            True,
            TRANSIENT_INFRASTRUCTURE,
        ),
        ({"ok": False}, "bias correction running", False, ABORTED_UNKNOWN),
        ({"ok": False, "returncode": 130}, "", True, ABORTED_UNKNOWN),
    ],
)
def test_v2783_build_state_classification(
    result: dict[str, object],
    log_text: str,
    terminal: bool,
    expected: str,
) -> None:
    assert classify_build_outcome(
        result, log_text=log_text, terminal=terminal
    ) == expected


def _create_positive_b5_fixture(tmp_path: Path) -> dict[str, object]:
    run = tmp_path / "source-run"
    suite = run / "models/demo/benchmark_set/legacy_suite"
    case = suite / "b7"
    artifact_dir = case / "hailo/hailo8/part1"
    artifact_dir.mkdir(parents=True)
    full_model = suite / "models/demo.onnx"
    builder_model = case / "part1.onnx"
    compiler_model = artifact_dir / "demo_part1_b7_hailo_fixed.onnx"
    hef = artifact_dir / "compiled.hef"
    full_model.parent.mkdir(parents=True)
    full_model.write_bytes(b"full-source-onnx\n")
    builder_model.write_bytes(b"builder-part1-onnx\n")
    compiler_model.write_bytes(b"fixed-compiler-onnx\n")
    hef.write_bytes(b"compiled-hef-bytes\n")
    manifest = {
        "schema": "onnx-splitpoint/split-manifest",
        "schema_version": 2,
        "boundary": 7,
        "strict_boundary": True,
        "full_model": "../models/demo.onnx",
        "part1": "part1.onnx",
        "part2": "part2.onnx",
        "cut_tensors": ["/head/Conv_output_0"],
        "io": {"part1_outputs": ["/head/Conv_output_0"]},
    }
    _write_json(case / "split_manifest.json", manifest)
    preprocessing = {
        "schema": "onnx-splitpoint/image-preprocessing-contract",
        "schema_version": 2,
        "task": "detection",
        "target_hw": [640, 640],
        "image_scale": "norm",
    }
    preprocessing_sha = canonical_sha256(preprocessing)
    calibration_identity = "manifest:" + "9" * 64
    prepared_identity = canonical_sha256({
        "calibration_identity": calibration_identity,
        "preprocessing_contract_sha256": preprocessing_sha,
    })
    payload = {
        "schema": HAILO_CACHE_SCHEMA_V3,
        "model_sha256": _sha_file(compiler_model),
        "activation_part1_sha256": "",
        "hw_arch": "hailo8",
        "hailo_sdk_version": "hailo-dataflow-compiler:3.33.1",
        "optimization_level": 1,
        "calibration_identity": calibration_identity,
        "prepared_calibration_identity_sha256": prepared_identity,
        "calibration_count": 5,
        "requested_calibration_count": 5,
        "calibration_storage": "memory",
        "calibration_memory_cap_bytes": 1048576,
        "calibration_batch_size": 1,
        "extra_model_script": "normalization1 = normalization([0,0,0], [255,255,255])",
        "start_nodes": [],
        "end_nodes": ["/head/Conv"],
        "integrity": "strict",
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha,
        "net_name": "demo_part1_b7",
        "net_input_shapes": [1, 3, 640, 640],
        "disable_rt_metadata_extraction": True,
    }
    cache_key = canonical_sha256(payload)
    receipt = {
        "schema": HAILO_RECEIPT_SCHEMA,
        "source_onnx_sha256": _sha_file(builder_model),
        "compiler_onnx_sha256": _sha_file(compiler_model),
        "compiler_onnx_filename": compiler_model.name,
        "hef_sha256": _sha_file(hef),
        "hef_size_bytes": hef.stat().st_size,
        "hw_arch": "hailo8",
        "net_name": "demo_part1_b7",
        "hailo_sdk_version": "hailo-dataflow-compiler:3.33.1",
        "calibration_identity": calibration_identity,
        "prepared_calibration_identity_sha256": prepared_identity,
        "calibration_count": 5,
        "requested_calibration_count": 5,
        "calibration_storage": "memory",
        "calibration_memory_cap_bytes": 1048576,
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha,
        "cache_key": cache_key,
        "cache_payload": payload,
    }
    receipt_path = artifact_dir / "hailo_hef_build_receipt.json"
    _write_json(receipt_path, receipt)
    endpoint_sha = boundary_endpoint_contract_sha256(
        stage="part1", cache_payload=payload, split_manifest=manifest
    )
    key = canonical_build_key_from_hailo_v3(
        receipt,
        full_source_onnx_sha256=_sha_file(full_model),
        boundary_endpoint_contract_sha256=endpoint_sha,
        expected_cache_key=cache_key,
    )
    assert canonical_build_key_from_hailo_v3_payload(
        payload,
        builder_source_onnx_sha256=_sha_file(builder_model),
        full_source_onnx_sha256=_sha_file(full_model),
        boundary_endpoint_contract_sha256=endpoint_sha,
        expected_cache_key=cache_key,
    ) == key
    return {
        "run": run,
        "hef": hef,
        "receipt": receipt_path,
        "compiler": compiler_model,
        "payload": payload,
        "key": key,
    }


def test_v2783_positive_artifact_harvest_lookup_materialize_and_tamper(
    tmp_path: Path,
) -> None:
    fixture = _create_positive_b5_fixture(tmp_path)
    verified = verify_hailo_artifact(
        fixture["hef"], receipt_path=fixture["receipt"]
    )
    assert verified.hef_sha256 == _sha_file(fixture["hef"])
    index = harvest_b5_run(fixture["run"])
    assert index["state_counts"][ARTIFACT_PASS] == 1
    decision = lookup_build_evidence(
        index, fixture["key"], artifact_root=fixture["run"]
    )
    assert decision.status == "HIT"
    assert decision.state == ARTIFACT_PASS

    destination = tmp_path / "materialized"
    outcome = materialize_verified_artifact(
        decision,
        destination,
        fixture["key"],
        artifact_root=fixture["run"],
    )
    assert outcome["ok"] is True
    assert outcome["details"]["exact_build_evidence"][
        "verified_after_materialization"
    ] is True
    verify_hailo_artifact(
        destination / "compiled.hef",
        receipt_path=destination / "hailo_hef_build_receipt.json",
    )

    fixture["hef"].write_bytes(b"tampered-hef\n")
    tampered = lookup_build_evidence(
        index, fixture["key"], artifact_root=fixture["run"]
    )
    assert tampered.status == "MISS"
    assert "unverified" in tampered.reason


def test_v2783_materializer_admits_only_identical_preexisting_fixed_onnx(
    tmp_path: Path,
) -> None:
    fixture = _create_positive_b5_fixture(tmp_path)
    index = harvest_b5_run(fixture["run"])
    decision = lookup_build_evidence(
        index, fixture["key"], artifact_root=fixture["run"]
    )
    compiler_name = fixture["compiler"].name

    identical_dir = tmp_path / "identical-probe-output"
    identical_dir.mkdir()
    identical_compiler = identical_dir / compiler_name
    identical_compiler.write_bytes(fixture["compiler"].read_bytes())
    inode_before = identical_compiler.stat().st_ino
    result = materialize_verified_artifact(
        decision,
        identical_dir,
        fixture["key"],
        artifact_root=fixture["run"],
    )
    assert identical_compiler.stat().st_ino == inode_before
    assert result["details"]["exact_build_evidence"][
        "compiler_onnx_materialization"
    ] == "preexisting_identical_probe_output"
    assert (identical_dir / "compiled.hef").is_file()
    assert (identical_dir / "hailo_hef_build_receipt.json").is_file()

    conflict_dir = tmp_path / "conflicting-probe-output"
    conflict_dir.mkdir()
    (conflict_dir / compiler_name).write_bytes(b"different-fixed-onnx\n")
    with pytest.raises(
        BuildEvidenceError, match="materialize_compiler_identity_conflict"
    ):
        materialize_verified_artifact(
            decision,
            conflict_dir,
            fixture["key"],
            artifact_root=fixture["run"],
        )
    assert not (conflict_dir / "compiled.hef").exists()
    assert not (conflict_dir / "hailo_hef_build_receipt.json").exists()

    symlink_dir = tmp_path / "symlinked-probe-output"
    symlink_dir.mkdir()
    symlink_target = tmp_path / "unrelated.onnx"
    symlink_target.write_bytes(fixture["compiler"].read_bytes())
    (symlink_dir / compiler_name).symlink_to(symlink_target)
    with pytest.raises(
        BuildEvidenceError,
        match="materialize_compiler_destination_symlink",
    ):
        materialize_verified_artifact(
            decision,
            symlink_dir,
            fixture["key"],
            artifact_root=fixture["run"],
        )
    assert not (symlink_dir / "compiled.hef").exists()
    assert not (symlink_dir / "hailo_hef_build_receipt.json").exists()

    partial_dir = tmp_path / "partial-compiler-hef"
    partial_dir.mkdir()
    partial_compiler = partial_dir / compiler_name
    partial_hef = partial_dir / "compiled.hef"
    partial_compiler.write_bytes(fixture["compiler"].read_bytes())
    partial_hef.write_bytes(fixture["hef"].read_bytes())
    partial_inodes = (partial_compiler.stat().st_ino, partial_hef.stat().st_ino)
    partial_result = materialize_verified_artifact(
        decision,
        partial_dir,
        fixture["key"],
        artifact_root=fixture["run"],
    )
    assert partial_inodes == (
        partial_compiler.stat().st_ino,
        partial_hef.stat().st_ino,
    )
    assert partial_result["details"]["exact_build_evidence"][
        "file_materialization"
    ] == {
        "compiler_onnx": "preexisting_identical_probe_output",
        "hef": "preexisting_identical",
        "receipt": "exclusive_copy",
    }

    complete_dir = tmp_path / "complete-identical-set"
    complete_dir.mkdir()
    complete_paths = (
        complete_dir / compiler_name,
        complete_dir / "compiled.hef",
        complete_dir / "hailo_hef_build_receipt.json",
    )
    for source, target in zip(
        (fixture["compiler"], fixture["hef"], fixture["receipt"]),
        complete_paths,
    ):
        target.write_bytes(source.read_bytes())
    complete_inodes = tuple(path.stat().st_ino for path in complete_paths)
    complete_result = materialize_verified_artifact(
        decision,
        complete_dir,
        fixture["key"],
        artifact_root=fixture["run"],
    )
    assert complete_inodes == tuple(path.stat().st_ino for path in complete_paths)
    assert complete_result["details"]["exact_build_evidence"][
        "file_materialization"
    ] == {
        "compiler_onnx": "preexisting_identical_probe_output",
        "hef": "preexisting_identical",
        "receipt": "preexisting_identical",
    }

    hef_conflict_dir = tmp_path / "conflicting-hef"
    hef_conflict_dir.mkdir()
    (hef_conflict_dir / compiler_name).write_bytes(fixture["compiler"].read_bytes())
    (hef_conflict_dir / "compiled.hef").write_bytes(b"conflicting-hef\n")
    with pytest.raises(BuildEvidenceError, match="materialize_hef_identity_conflict"):
        materialize_verified_artifact(
            decision,
            hef_conflict_dir,
            fixture["key"],
            artifact_root=fixture["run"],
        )
    assert not (hef_conflict_dir / "hailo_hef_build_receipt.json").exists()

    hef_symlink_dir = tmp_path / "symlinked-hef"
    hef_symlink_dir.mkdir()
    (hef_symlink_dir / compiler_name).write_bytes(fixture["compiler"].read_bytes())
    (hef_symlink_dir / "compiled.hef").symlink_to(fixture["hef"])
    with pytest.raises(BuildEvidenceError, match="materialize_hef_destination_symlink"):
        materialize_verified_artifact(
            decision,
            hef_symlink_dir,
            fixture["key"],
            artifact_root=fixture["run"],
        )

    receipt_conflict_dir = tmp_path / "conflicting-receipt"
    receipt_conflict_dir.mkdir()
    (receipt_conflict_dir / compiler_name).write_bytes(
        fixture["compiler"].read_bytes()
    )
    (receipt_conflict_dir / "compiled.hef").write_bytes(fixture["hef"].read_bytes())
    (receipt_conflict_dir / "hailo_hef_build_receipt.json").write_bytes(
        b'{"different":true}\n'
    )
    with pytest.raises(
        BuildEvidenceError, match="materialize_receipt_identity_conflict"
    ):
        materialize_verified_artifact(
            decision,
            receipt_conflict_dir,
            fixture["key"],
            artifact_root=fixture["run"],
        )

    receipt_symlink_dir = tmp_path / "symlinked-receipt"
    receipt_symlink_dir.mkdir()
    (receipt_symlink_dir / compiler_name).write_bytes(
        fixture["compiler"].read_bytes()
    )
    (receipt_symlink_dir / "compiled.hef").write_bytes(fixture["hef"].read_bytes())
    (receipt_symlink_dir / "hailo_hef_build_receipt.json").symlink_to(
        fixture["receipt"]
    )
    with pytest.raises(
        BuildEvidenceError, match="materialize_receipt_destination_symlink"
    ):
        materialize_verified_artifact(
            decision,
            receipt_symlink_dir,
            fixture["key"],
            artifact_root=fixture["run"],
        )


def test_v2783_conflicting_deterministic_negatives_fail_closed() -> None:
    key = canonical_build_key(**_base_key_kwargs())
    parser_record = make_build_evidence_record(
        key,
        PARSER_UNSUPPORTED,
        evidence_origin={"result": "parser.json"},
    )
    compile_record = make_build_evidence_record(
        key,
        COMPILE_INFEASIBLE,
        evidence_origin={"result": "compile.json"},
    )
    index = build_evidence_index(
        [parser_record, compile_record], source_run_name="b5"
    )
    decision = lookup_build_evidence(index, key)
    assert decision.status == "CONFLICT"
    assert decision.reusable is False


def _tree_snapshot(root: Path) -> list[tuple[str, str, int, int]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        info = path.stat()
        rows.append((
            path.relative_to(root).as_posix(),
            _sha_file(path),
            info.st_size,
            info.st_mtime_ns,
        ))
    return rows


def test_v2783_recovery_is_read_only_atomic_and_requires_external_output(
    tmp_path: Path,
) -> None:
    fixture = _create_positive_b5_fixture(tmp_path)
    before = _tree_snapshot(fixture["run"])
    output = tmp_path / "recovery/build_evidence_index.json"
    payload = create_build_evidence_index(
        run_dir=fixture["run"], output=output
    )
    assert output.is_file() and not output.is_symlink()
    assert payload["source_run_mutated"] is False
    assert _tree_snapshot(fixture["run"]) == before
    with pytest.raises(BuildEvidenceError, match="output_inside_source_run"):
        create_build_evidence_index(
            run_dir=fixture["run"],
            output=fixture["run"] / "forbidden.json",
        )
    with pytest.raises(BuildEvidenceError, match="output_already_exists"):
        create_build_evidence_index(
            run_dir=fixture["run"], output=output
        )


def test_v2783_writes_never_create_through_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    fixture = _create_positive_b5_fixture(tmp_path)
    index = harvest_b5_run(fixture["run"])
    decision = lookup_build_evidence(
        index, fixture["key"], artifact_root=fixture["run"]
    )
    link = tmp_path / "source-link"
    link.symlink_to(fixture["run"], target_is_directory=True)
    before = _tree_snapshot(fixture["run"])

    with pytest.raises(BuildEvidenceError, match="unsafe_symlink_component"):
        create_build_evidence_index(
            run_dir=fixture["run"],
            output=link / "new-index-dir/build_evidence.json",
        )
    assert not (fixture["run"] / "new-index-dir").exists()
    assert _tree_snapshot(fixture["run"]) == before

    with pytest.raises(BuildEvidenceError, match="unsafe_symlink_component"):
        materialize_verified_artifact(
            decision,
            link / "new-materialization-dir",
            fixture["key"],
            artifact_root=fixture["run"],
        )
    assert not (fixture["run"] / "new-materialization-dir").exists()
    assert _tree_snapshot(fixture["run"]) == before


def test_v2783_cli_preflights_runtime_output_before_any_write(
    tmp_path: Path,
) -> None:
    fixture = _create_positive_b5_fixture(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/recover_b5_build_evidence.py"
    build_output = tmp_path / "recovery/build.json"
    inner_runtime = fixture["run"] / "forbidden/runtime.json"
    before = _tree_snapshot(fixture["run"])
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "create",
            "--run-dir",
            str(fixture["run"]),
            "--out",
            str(build_output),
            "--runtime-out",
            str(inner_runtime),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    result = json.loads(process.stdout)
    assert process.returncode == 2
    assert result["reason_code"] == "runtime_output_inside_source_run"
    assert not build_output.exists()
    assert not inner_runtime.exists()
    assert not inner_runtime.parent.exists()
    assert _tree_snapshot(fixture["run"]) == before

    same_output = tmp_path / "same-output.json"
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "create",
            "--run-dir",
            str(fixture["run"]),
            "--out",
            str(same_output),
            "--runtime-out",
            str(same_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    result = json.loads(process.stdout)
    assert process.returncode == 2
    assert result["reason_code"] == "output_paths_not_distinct"
    assert not same_output.exists()
    assert _tree_snapshot(fixture["run"]) == before

    runtime_link = tmp_path / "runtime-source-link"
    runtime_link.symlink_to(fixture["run"], target_is_directory=True)
    linked_build_output = tmp_path / "linked-recovery/build.json"
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "create",
            "--run-dir",
            str(fixture["run"]),
            "--out",
            str(linked_build_output),
            "--runtime-out",
            str(runtime_link / "forbidden/runtime.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    result = json.loads(process.stdout)
    assert process.returncode == 2
    assert result["reason_code"] == "unsafe_symlink_component"
    assert not linked_build_output.exists()
    assert not (fixture["run"] / "forbidden").exists()
    assert _tree_snapshot(fixture["run"]) == before


def test_v2783_harvest_refuses_symlinked_evidence(tmp_path: Path) -> None:
    fixture = _create_positive_b5_fixture(tmp_path)
    receipt = fixture["receipt"]
    real = receipt.with_name("real-receipt.json")
    receipt.rename(real)
    receipt.symlink_to(real.name)
    with pytest.raises(BuildEvidenceError, match="unsafe_evidence_symlink"):
        harvest_b5_run(fixture["run"])


def test_v2783_attempt_without_terminal_result_is_kept_nonreusable(
    tmp_path: Path,
) -> None:
    run = tmp_path / "source-run"
    directory = run / "models/demo/benchmark_set/legacy_suite/b9/hailo/hailo8/part1"
    directory.mkdir(parents=True)
    key = canonical_build_key(**_base_key_kwargs())
    attempt = {
        "schema": "onnx-splitpoint/hailo-hef-build-attempt/v1",
        "schema_version": 1,
        "key": key,
        "cache_key": key["backend_cache_contract_sha256"],
    }
    attempt["attempt_sha256"] = canonical_sha256(attempt)
    _write_json(directory / "hailo_hef_build_attempt.json", attempt)
    index = harvest_b5_run(run)
    assert index["state_counts"][ABORTED_UNKNOWN] == 1
    assert index["records"][0]["reusable"] is False
    assert lookup_build_evidence(index, key).status == "MISS"


def test_v2783_logged_b5_abort_preserves_partial_compiler_artifact(
    tmp_path: Path,
) -> None:
    run = tmp_path / "source-run"
    directory = (
        run
        / "models/demo/benchmark_set/legacy_suite/b574/hailo/hailo10/part1"
    )
    directory.mkdir(parents=True)
    fixed = directory / "demo_part1_b574_hailo_fixed.onnx"
    fixed.write_bytes(b"partial-fixed-onnx\n")
    log = tmp_path / "workflow.log"
    log.write_text(
        "[benchmarkset:demo] (b574 hailo10) "
        "[hailo][cache] miss key=abcdef012345 "
        "net=demo_part1_b574 hw_arch=hailo10h\n"
        "[benchmarkset:demo] (b574 hailo10) "
        "[hailo][optimize] start\n",
        encoding="utf-8",
    )
    index = harvest_b5_run(run, workflow_log=log)
    assert index["state_counts"][ABORTED_UNKNOWN] == 1
    row = index["unresolved_observations"][0]
    assert row["reason_code"] == "logged_attempt_without_terminal_exact_identity"
    assert row["partial_artifacts"][0]["sha256"] == _sha_file(fixed)
    assert row["reusable"] is False


def test_v27920_structured_hailo_cache_miss_remains_harvestable(
    tmp_path: Path,
) -> None:
    run = tmp_path / "source-run"
    directory = (
        run
        / "models/demo/benchmark_set/legacy_suite/b574/hailo/hailo10/part1"
    )
    directory.mkdir(parents=True)
    fixed = directory / "demo_part1_b574_hailo_fixed.onnx"
    fixed.write_bytes(b"partial-fixed-onnx\n")
    log = tmp_path / "workflow.log"
    log.write_text(
        "[benchmarkset:demo] (b574 hailo10) "
        "[hailo-cache] MISS role=hef model=demo_part1_b574 "
        "identity=abcdef012345abcdef012345abcdef012345abcdef012345abcdef012345abcd "
        "reason=not_found_or_receipt_invalid artifact=/tmp/compiled.hef "
        "hw_arch=hailo10h\n"
        "[benchmarkset:demo] (b574 hailo10) [hailo][optimize] start\n",
        encoding="utf-8",
    )

    index = harvest_b5_run(run, workflow_log=log)

    assert index["state_counts"][ABORTED_UNKNOWN] == 1
    row = index["unresolved_observations"][0]
    assert row["reason_code"] == "logged_attempt_without_terminal_exact_identity"
    assert row["partial_artifacts"][0]["sha256"] == _sha_file(fixed)
    assert row["reusable"] is False
