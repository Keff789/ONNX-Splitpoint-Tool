from __future__ import annotations

import copy
import hashlib
import importlib.util
import ast
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_v2798_yolov7_claim_gate_32.py"
SPEC = importlib.util.spec_from_file_location("v2798_yolov7_claim_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    elif isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(str(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    source = tmp_path / "source"
    evidence = tmp_path / "evidence"
    source.mkdir()
    evidence.mkdir()

    required = list(gate.REQUIRED_SOURCE_FILES)
    for index, relative in enumerate(required):
        content = ("source-%02d-%s\n" % (index, relative)).encode()
        if relative.endswith("native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"):
            content = b"identical-helper\n"
        _write(source / relative, content)
    rows = []
    for relative in sorted(required):
        path = source / relative
        rows.append({"path": relative, "size": path.stat().st_size, "sha256": _sha(path)})
    source_manifest = {
        "schema": "onnx-splitpoint/source-manifest-v1",
        "package_version": gate.PACKAGE_VERSION,
        "workflow_version": gate.BUILD_ID,
        "file_count": len(rows),
        "files": rows,
    }
    source_manifest_path = _write(source / "SOURCE_MANIFEST.json", source_manifest)
    source_manifest_sha = _sha(source_manifest_path)

    image_ids = list(range(1000, 1000 + gate.CLAIM_ITEM_COUNT))
    dataset_items = []
    image_payloads = {}
    for image_id in image_ids:
        payload = ("image-%d\n" % image_id).encode()
        image_payloads[image_id] = payload
        dataset_items.append(
            {
                "image_id": image_id,
                "relative_path": "%012d.jpg" % image_id,
                "width": 640,
                "height": 480,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    for image_id in range(2000, 2000 + (5000 - len(dataset_items))):
        payload = ("unused-%d" % image_id).encode()
        dataset_items.append(
            {
                "image_id": image_id,
                "relative_path": "%012d.jpg" % image_id,
                "width": 640,
                "height": 480,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    dataset = {
        "schema": gate.DATASET_SCHEMA,
        "schema_version": 1,
        "dataset_id": "coco2017-val",
        "item_count": 5000,
        "items": dataset_items,
    }
    dataset_path = _write(evidence / "reference" / "dataset.json", dataset)
    dataset_sha = _sha(dataset_path)
    monkeypatch.setattr(gate, "DATASET_MANIFEST_SHA256", dataset_sha)

    reference_rows = [{"image_id": image_id} for image_id in image_ids]
    reference = {
        "schema": gate.REFERENCE_SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "corpus": {"dataset_manifest_sha256": dataset_sha},
        "rows": reference_rows,
    }
    reference_path = _write(evidence / "reference" / "reference.json", reference)
    reference_sha = _sha(reference_path)
    monkeypatch.setattr(gate, "REFERENCE_REPORT_SHA256", reference_sha)

    corpus_items = []
    for index, image_id in enumerate(image_ids):
        staged = _write(
            evidence / "work" / "corpus" / "images" / ("%02d_%012d.jpg" % (index, image_id)),
            image_payloads[image_id],
        )
        source_row = dataset_items[index]
        corpus_items.append(
            {
                "index": index,
                "image_id": image_id,
                "relative_path": source_row["relative_path"],
                "staged_file": "images/%s" % staged.name,
                "staged_sha256": source_row["sha256"],
            }
        )
    corpus = {
        "schema": gate.CORPUS_SCHEMA,
        "schema_version": 1,
        "selection_scope": "ground_truth_only",
        "selection_uses_model_predictions": False,
        "dataset_id": "coco2017-val",
        "dataset_manifest_sha256": dataset_sha,
        "requested_count": 32,
        "selected_count": 32,
        "items": corpus_items,
    }
    corpus_path = _write(evidence / "work" / "corpus" / "corpus_manifest.json", corpus)

    hef_sha = "1" * 64
    engine_sha = "2" * 64
    binding = {
        "schema": gate.QUALITY_BINDING_SCHEMA,
        "quality_completed": True,
        "artifacts": {
            "part1_runtime": {"path": "/cache/part1.hef", "sha256": hef_sha, "size_bytes": 111},
            "engine": {"path": "/cache/part2.engine", "sha256": engine_sha, "size_bytes": 222},
        },
    }
    binding_path = _write(evidence / "reference" / "binding.json", binding)

    out_root = evidence / "three_stage"
    repetitions = []
    for rep in (1, 2):
        runtime = {
            "schema": gate.RUNTIME_SCHEMA,
            "schema_version": 1,
            "ok": True,
            "frames": 64,
            "warmup": 8,
            "warmup_fully_drained_before_measurement": True,
            "image_count": 32,
            "warmup_result": {"requested": 8, "raw_completed": 8, "completed": 8, "callback_failures": 0},
            "raw_model_outputs": {"completed_frames": 64, "throughput_fps": 97.1},
            "completed_detection": {"completed_frames": 64, "throughput_fps": 97.0},
            "callback_failures": 0,
        }
        runtime_path = _write(
            out_root / "run" / "repetitions" / ("rep_%02d" % rep) / "native_three_stage_runtime.json",
            runtime,
        )
        repetitions.append(
            {
                "repetition": rep,
                "return_code": 0,
                "runtime_ok": True,
                "runtime_report_path": str(runtime_path),
                "callback_errors": [],
                "measurement_result_parity": {
                    "requested_images": 32,
                    "exact_images": 32,
                    "all_exact": True,
                    "rows": [{"image_id": image_id, "exact": True} for image_id in image_ids],
                },
            }
        )
    postflight_rows = [
        {
            "sequence": index,
            "image_index": index,
            "image_id": image_id,
            "raw_head_sha256_exact": True,
            "oracle_fast_exact": True,
            "current_oracle_equals_prior_oracle": True,
            "exact": True,
        }
        for index, image_id in enumerate(image_ids)
    ]
    report = {
        "schema": gate.CANARY_SCHEMA,
        "schema_version": 1,
        "status": "PASS_THREE_STAGE_TARGET_MET",
        "quality_oracle_outside_performance_timing": True,
        "performance_hotloop_contains_crypto_hashing": False,
        "source_ok": True,
        "reference_metadata_checks": [
            {"passed": True, "expected_sha256": "3" * 64, "actual_sha256": "3" * 64}
        ],
        "source_checks": [
            {"passed": True, "expected_sha256": "4" * 64, "actual_sha256": "4" * 64}
        ],
        "corpus": corpus,
        "runtime_artifacts": {
            "hef_sha256": hef_sha,
            "engine_sha256": engine_sha,
            "source_mutated": False,
        },
        "measurement_contract": {
            "images_cycled": 32,
            "completed_endpoint": "completed_detection",
            "quality_oracle": "separate_32_image_postflight_outside_timing",
        },
        "repetitions": repetitions,
        "postflight_quality_oracle": {
            "performance_interpretation": "forbidden_verification_only",
            "return_code": 0,
            "callback_errors": [],
            "requested_images": 32,
            "verified_images": 32,
            "exact_images": 32,
            "all_exact": True,
            "rows": postflight_rows,
        },
        "aggregate": {"repetitions_requested": 2, "repetitions_valid": 2},
    }
    report_path = _write(out_root / "run" / "three_stage_canary_report.json", report)

    snapshot_path = evidence / "work" / "source_snapshot.zip"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(snapshot_path, "w") as archive:
        prefix = "ONNX-Splitpoint-Tool_v%s" % gate.PACKAGE_VERSION
        for relative in gate.SNAPSHOT_MEMBERS:
            archive.write(source / relative, "%s/%s" % (prefix, relative))

    out_binding = gate._canonical_sha256(
        {
            "out_root": str(out_root),
            "execution_scope": gate.CLAIM_SCOPE,
            "model_id": gate.MODEL_ID,
            "case_id": gate.CASE_ID,
            "setup_id": gate.SETUP_ID,
        }
    )
    invocation = {
        "schema": gate.INVOCATION_SCHEMA,
        "schema_version": 1,
        "benchmark_set": "/benchmark-set",
        "case_id": gate.CASE_ID,
        "setup_id": gate.SETUP_ID,
        "corpus_manifest": str(corpus_path),
        "dataset_manifest": str(dataset_path),
        "reference_report": str(reference_path),
        "quality_binding": str(binding_path),
        "out_root": str(out_root),
        "expected_item_count": 32,
        "model_id": gate.MODEL_ID,
        "precision": gate.PRECISION,
        "execution_scope": gate.CLAIM_SCOPE,
        "claim_eligible": True,
        "corpus_manifest_sha256": _sha(corpus_path),
        "dataset_manifest_sha256": dataset_sha,
        "reference_report_sha256": reference_sha,
        "quality_binding_sha256": _sha(binding_path),
        "out_root_binding_sha256": out_binding,
    }
    invocation["invocation_sha256"] = gate._canonical_sha256(invocation)
    invocation_path = _write(out_root / "three_stage_invocation_receipt.json", invocation)
    command = {
        "schema": "onnx-splitpoint/three-stage-remote-command-receipt",
        "schema_version": 1,
        "invocation_sha256": invocation["invocation_sha256"],
        "argv": [
            "python", "canary.py", "--corpus", str(corpus_path),
            "--reference-report", str(reference_path), "--out-root", str(out_root),
            "--expected-corpus-count", "32",
        ],
    }
    command_path = _write(out_root / "three_stage_remote_command_receipt.json", command)
    console_path = _write(out_root / "vendored_canary_console.log", "PASS\n")

    run_id = "v2798_yolov7_claim_gate_fixture"
    result = {
        "ok": True,
        "schema": gate.RESULT_SCHEMA,
        "schema_version": 2,
        "build_id": gate.BUILD_ID,
        "performance_endpoint": "p2_output",
        "application_performance_endpoint": "completed_detection",
        "endpoint_execution_policy": "concurrent_three_stage_single_invocation",
        "three_stage_concurrency_directly_measured": True,
        "three_stage_hardware_integration_status": "passed",
        "quality_oracle_location": "outside_performance_timing",
        "quality_oracle_status": "passed",
        "p2_output_fps": 97.1,
        "completed_detection_fps": 97.0,
        "completed_to_p2_ratio": 0.999,
        "endpoint_relation_verified": True,
        "directly_measured": True,
        "execution_scope": gate.CLAIM_SCOPE,
        "claim_eligible": True,
        "expected_item_count": 32,
        "child_returncode": 0,
        "vendored_runtime_rc": 0,
        "canary_status": "PASS_THREE_STAGE_TARGET_MET",
        "failure_class": "",
        "failure_reason": "",
        "out_root": str(out_root),
        "out_root_binding_sha256": out_binding,
        "canary_report": str(report_path),
        "stage_timings": {
            "schema": "onnx-splitpoint/native-three-stage-timing-projection",
            "schema_version": 1,
            "P1": {"mean_ms_by_repetition": [10.0, 10.1]},
            "P2": {"mean_ms_by_repetition": [8.0, 8.1]},
            "Post": {"mean_ms_by_repetition": [2.0, 2.1]},
        },
        "identity": {
            "model_id": gate.MODEL_ID,
            "case_id": gate.CASE_ID,
            "setup_id": gate.SETUP_ID,
            "precision": gate.PRECISION,
            "eval_run_id": run_id,
            "source_run_id": "hailo8_to_trt",
            "binding_sha256": _sha(binding_path),
        },
        "invocation": invocation,
        "product_execution_context": {
            "source_snapshot": str(snapshot_path),
            "source_snapshot_sha256": _sha(snapshot_path),
            "source_snapshot_prefix": "ONNX-Splitpoint-Tool_v2.79.8",
            "performance_corpus_manifest": str(corpus_path),
            "performance_corpus_manifest_sha256": _sha(corpus_path),
            "performance_corpus_count": 32,
            "dataset_manifest": str(dataset_path),
            "dataset_manifest_sha256": dataset_sha,
            "postflight_reference_report": str(reference_path),
            "postflight_reference_report_sha256": reference_sha,
            "quality_binding": str(binding_path),
            "quality_binding_sha256": _sha(binding_path),
            "canary_out_root": str(out_root),
            "canary_out_root_binding_sha256": out_binding,
            "execution_scope": gate.CLAIM_SCOPE,
            "claim_eligible": True,
            "quality_oracle_inside_performance_timing": False,
        },
        "oracle_parity": {
            "schema": "onnx-splitpoint/native-three-stage-oracle-parity",
            "schema_version": 1,
            "status": "passed",
            "inside_performance_timing": False,
            "postflight": report["postflight_quality_oracle"],
            "measurement_repetitions": [
                {
                    "repetition": repetition,
                    "requested_images": 32,
                    "exact_images": 32,
                    "all_exact": True,
                }
                for repetition in (1, 2)
            ],
        },
    }
    result_path = _write(evidence / "native_three_stage_result.json", result)

    def artifact(path: Path) -> dict[str, Any]:
        return {"path": str(path), "sha256": _sha(path), "size_bytes": path.stat().st_size}

    artifact_index = {
        "schema": gate.ARTIFACT_INDEX_SCHEMA,
        "schema_version": 1,
        "out_root": str(out_root),
        "out_root_binding_sha256": out_binding,
        "invocation_sha256": invocation["invocation_sha256"],
        "artifacts": {
            "invocation_receipt": artifact(invocation_path),
            "remote_command_receipt": artifact(command_path),
            "canonical_result": artifact(result_path),
            "child_report": artifact(report_path),
            "child_console": artifact(console_path),
            "failure_receipt": None,
        },
    }
    artifact_index_path = _write(out_root / "three_stage_artifact_index.json", artifact_index)

    capture = {
        "schema": gate.CAPTURE_SCHEMA,
        "schema_version": 1,
        "original_evidence_root": str(evidence),
        "run_id": run_id,
        "source_release_sha256": "5" * 64,
        "source_manifest_sha256": source_manifest_sha,
        "external_runtime_artifacts": {
            "part1_runtime": {"sha256": hef_sha, "size_bytes": 111, "verified_at_capture": True},
            "engine": {"sha256": engine_sha, "size_bytes": 222, "verified_at_capture": True},
        },
    }
    capture_path = _write(evidence / "capture_manifest.json", capture)
    return {
        "source": source,
        "source_manifest_sha": source_manifest_sha,
        "evidence": evidence,
        "result": result_path,
        "report": report_path,
        "corpus": corpus_path,
        "artifact_index": artifact_index_path,
        "capture": capture_path,
    }


def _verify(paths: dict[str, Any]) -> dict[str, Any]:
    return gate.verify_claim_gate(
        result_json=paths["result"],
        evidence_root=paths["evidence"],
        source_root=paths["source"],
        expected_source_manifest_sha256=paths["source_manifest_sha"],
    )


def test_exact_claim_gate_fixture_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = _verify(paths)
    assert receipt["status"] == "PASS"
    assert receipt["valid_claim"] is True
    assert receipt["passed_item_count"] == 32
    assert receipt["output_identity"]["requested_work_units"] == 128
    assert receipt["output_identity"]["completed_work_units"] == 128
    assert len(receipt["invocation_identity"]["ordered_image_ids"]) == 32


@pytest.mark.parametrize(
    "target",
    ["corpus_image", "result_claim", "postflight", "runtime_count", "artifact_index"],
)
def test_tampered_or_downgraded_evidence_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    if target == "corpus_image":
        image = next((paths["corpus"].parent / "images").iterdir())
        image.write_bytes(b"tampered")
    elif target == "result_claim":
        payload = json.loads(paths["result"].read_text())
        payload["claim_eligible"] = False
        _write(paths["result"], payload)
    elif target == "postflight":
        payload = json.loads(paths["report"].read_text())
        payload["postflight_quality_oracle"]["exact_images"] = 31
        _write(paths["report"], payload)
    elif target == "runtime_count":
        runtime = next(paths["report"].parent.rglob("native_three_stage_runtime.json"))
        payload = json.loads(runtime.read_text())
        payload["completed_detection"]["completed_frames"] = 63
        _write(runtime, payload)
    else:
        payload = json.loads(paths["artifact_index"].read_text())
        payload["artifacts"]["canonical_result"]["sha256"] = "0" * 64
        _write(paths["artifact_index"], payload)
    with pytest.raises(gate.ClaimGateError):
        _verify(paths)


def test_symlinked_evidence_member_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    original = paths["result"]
    real = original.with_name("real_result.json")
    original.rename(real)
    original.symlink_to(real.name)
    with pytest.raises(gate.ClaimGateError, match="symlink"):
        _verify(paths)


def test_source_tree_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    helper = paths["source"] / "scripts/native_hailo_trt_fifo_from_benchmarkset.py"
    helper.write_text("drift\n", encoding="utf-8")
    with pytest.raises(gate.ClaimGateError, match="source_file_"):
        _verify(paths)


def test_cli_writes_fail_receipt_and_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payload = json.loads(paths["result"].read_text())
    payload["expected_item_count"] = 1
    _write(paths["result"], payload)
    output = tmp_path / "verdict.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result-json",
            str(paths["result"]),
            "--evidence-root",
            str(paths["evidence"]),
            "--source-root",
            str(paths["source"]),
            "--expected-source-manifest-sha256",
            paths["source_manifest_sha"],
            "--output",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 1
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "FAIL"
    assert receipt["valid_claim"] is False
    assert receipt["claim_eligible"] is False


def test_verifier_source_stays_python38_compatible() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'print("V2798_YOLOV7_CLAIM_GATE=PASS")' in text
    assert "V2796_YOLOV7_CLAIM_GATE=PASS" not in text
    assert ".removeprefix(" not in text
    assert ".removesuffix(" not in text
    assert ".is_relative_to(" not in text
    subprocess.run([sys.executable, "-m", "py_compile", str(SCRIPT)], check=True)


def test_claim_runtime_sources_stay_python38_compatible_and_identical() -> None:
    helper = ROOT / "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"
    vendored = (
        ROOT
        / "onnx_splitpoint_tool/resources/remote_scripts"
        / "native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"
    )
    stager = (
        ROOT
        / "onnx_splitpoint_tool/resources/native_concurrent_three_stage_yolov7"
        / "stage_corpus.py"
    )
    assert helper.read_bytes() == vendored.read_bytes()
    for path in (helper, vendored, stager, SCRIPT):
        text = path.read_text(encoding="utf-8")
        assert ".removeprefix(" not in text
        assert ".removesuffix(" not in text
        assert ".is_relative_to(" not in text
        ast.parse(text, filename=str(path), feature_version=(3, 8))
    helper_text = helper.read_text(encoding="utf-8")
    assert 'from __future__ import annotations' in helper_text

    helper_spec = importlib.util.spec_from_file_location("claim_helper_py38", helper)
    assert helper_spec and helper_spec.loader
    helper_module = importlib.util.module_from_spec(helper_spec)
    sys.modules[helper_spec.name] = helper_module
    try:
        helper_spec.loader.exec_module(helper_module)
    finally:
        sys.modules.pop(helper_spec.name, None)
    assert helper_module._strip_sha256("sha256:" + "a" * 64) == "a" * 64

    stager_spec = importlib.util.spec_from_file_location("claim_stager_py38", stager)
    assert stager_spec and stager_spec.loader
    stager_module = importlib.util.module_from_spec(stager_spec)
    stager_spec.loader.exec_module(stager_module)
    assert stager_module.strip_sha("sha256:" + "b" * 64) == "b" * 64


def _source_manifest_rows() -> dict[str, dict[str, Any]]:
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    return {str(row["path"]): dict(row) for row in manifest["files"]}


def test_v2796_carry_forward_computational_surface_is_byte_identical() -> None:
    attestation = gate._validate_carry_forward_computational_source(
        ROOT, _source_manifest_rows()
    )
    assert attestation == {
        "schema": "onnx-splitpoint/cross-release-computational-source/v1",
        "file_count": 21,
        "inventory_sha256": gate.CARRY_FORWARD_COMPUTATIONAL_TREE_SHA256,
        "byte_identical_to_measurement_release": True,
    }


def _replace_once(payload: bytes, old: bytes, new: bytes, label: str) -> bytes:
    assert payload.count(old) == 1, label
    return payload.replace(old, new, 1)


def _historical_snapshot_member(
    relative: str, *, version: str, build_id: str
) -> bytes:
    """Recreate an exact frozen source member from the cumulative tree.

    The executable Three-Stage surface is byte-identical across these releases.
    Only cumulative feature metadata and the explicitly audited release/build
    literals changed.  Each transformation is cardinality-checked and the
    resulting complete member set is checked against the verifier's immutable
    SHA-256 pins before it is used by a test.
    """

    payload = (ROOT / relative).read_bytes()
    if relative == "onnx_splitpoint_tool/__init__.py":
        for added_import in (
            b"    BYTECODE_ISOLATION_CONTRACT as _bytecode_isolation_contract,\n",
            b"    REMOTE_ENERGY_PRIMARY_ADMISSION_CONTRACT as _remote_energy_primary_admission_contract,\n",
            b"    SOURCE_INTEGRITY_CONTRACT as _source_integrity_contract,\n",
        ):
            assert payload.count(added_import) == 1
            payload = payload.replace(added_import, b"", 1)
        first_feature = {
            "2.79.8": b'    "yolo11_gate_profile_schema_closure",\n',
            "2.79.6": b'    "remaining_changes_release_closure",\n',
        }[version]
        opener = b"__build_features__ = (\n"
        assert payload.count(opener) == 1
        assert payload.count(first_feature) == 1
        start = payload.index(opener) + len(opener)
        retained = payload.index(first_feature, start)
        assert retained > start
        payload = payload[:start] + payload[retained:]
    elif relative == "onnx_splitpoint_tool/release_identity.py":
        payload = _replace_once(
            payload,
            b'VERSION = "2.79.13"',
            ('VERSION = "%s"' % version).encode("utf-8"),
            "current release version literal cardinality",
        )
        payload = _replace_once(
            payload,
            b'BUILD_ID = "v2.79.13-platform-power-calibration-operational-repair"',
            ('BUILD_ID = "%s"' % build_id).encode("utf-8"),
            "current release build literal cardinality",
        )
        added_contracts = (
            b'SOURCE_INTEGRITY_CONTRACT = "installed_release_source_manifest_runtime_binding"\n'
            b'BYTECODE_ISOLATION_CONTRACT = "source_local_bytecode_cache_isolation"\n'
            b'REMOTE_ENERGY_PRIMARY_ADMISSION_CONTRACT = (\n'
            b'    "benchmark_primary_remote_energy_admission_before_transport"\n'
            b')\n'
        )
        assert payload.count(added_contracts) == 1
        payload = payload.replace(added_contracts, b"", 1)
    elif relative == (
        "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"
    ):
        payload = _replace_once(
            payload,
            b"from onnx_splitpoint_tool.release_identity import BUILD_ID",
            ('BUILD_ID = "%s"' % build_id).encode("utf-8"),
            "current helper central BUILD_ID import cardinality",
        )
    return payload


def _historical_v2798_source(tmp_path: Path) -> Path:
    source = tmp_path / "frozen_v2798_source"
    observed: dict[str, str] = {}
    for relative in gate.SNAPSHOT_MEMBERS:
        payload = _historical_snapshot_member(
            relative, version=gate.PACKAGE_VERSION, build_id=gate.BUILD_ID
        )
        _write(source / relative, payload)
        observed[relative] = hashlib.sha256(payload).hexdigest()
    assert observed == dict(
        gate.CARRY_FORWARD_ORIGIN["current_snapshot_member_sha256"]
    )
    return source


def _carry_forward_snapshot(
    tmp_path: Path, *, tamper_helper: bool = False
) -> tuple[Path, dict[str, Any], Path]:
    origin = copy.deepcopy(gate.CARRY_FORWARD_ORIGIN)
    origin_version = str(origin["package_version"])
    origin_build = str(origin["build_id"])
    prefix = str(origin["source_snapshot_prefix"])
    source = _historical_v2798_source(tmp_path)
    captured_hashes: dict[str, str] = {}
    current_hashes: dict[str, str] = {}
    archive_path = tmp_path / "source_snapshot.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for relative in gate.SNAPSHOT_MEMBERS:
            current = (source / relative).read_bytes()
            current_hashes[relative] = hashlib.sha256(current).hexdigest()
            captured = _historical_snapshot_member(
                relative, version=origin_version, build_id=origin_build
            )
            if tamper_helper and relative == (
                "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"
            ):
                captured = _replace_once(
                    captured,
                    b"CLAIM_ITEM_COUNT = 32",
                    b"CLAIM_ITEM_COUNT = 31",
                    "historical helper claim-count literal cardinality",
                )
            captured_hashes[relative] = hashlib.sha256(captured).hexdigest()
            archive.writestr("%s/%s" % (prefix, relative), captured)

    assert current_hashes == dict(origin["current_snapshot_member_sha256"])
    if tamper_helper:
        # Admit the deliberately modified synthetic capture by its own exact
        # identity so the test reaches and exercises the computational-drift
        # comparison rather than failing earlier at the hash pin.
        origin["snapshot_member_sha256"] = captured_hashes
    else:
        assert captured_hashes == dict(origin["snapshot_member_sha256"])
    origin["mode"] = "retained_v2796_byte_identical_computation"
    return archive_path, origin, source


def test_carry_forward_snapshot_accepts_only_explicit_release_identity_drift(
    tmp_path: Path,
) -> None:
    archive, policy, historical_source = _carry_forward_snapshot(tmp_path)
    attestation = gate._validate_snapshot(
        archive, historical_source, _sha(archive), policy
    )
    comparisons = {
        row["path"]: row["comparison"] for row in attestation["source_comparison"]
    }
    assert comparisons["onnx_splitpoint_tool/native_three_stage.py"] == "byte_identical_computation"
    assert comparisons["onnx_splitpoint_tool/release_identity.py"] == "release_version_and_build_id_only"
    assert comparisons[
        "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"
    ] == "byte_identical_after_build_id_normalization"


def test_carry_forward_snapshot_rejects_non_identity_helper_drift(
    tmp_path: Path,
) -> None:
    archive, policy, historical_source = _carry_forward_snapshot(
        tmp_path, tamper_helper=True
    )
    with pytest.raises(
        gate.ClaimGateError, match="carry_forward_snapshot_computational_drift"
    ):
        gate._validate_snapshot(
            archive, historical_source, _sha(archive), policy
        )


def test_carry_forward_current_computational_drift_fails_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    rows: dict[str, dict[str, Any]] = {}
    for relative in gate.CARRY_FORWARD_COMPUTATIONAL_SOURCE_FILES:
        current = ROOT / relative
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(current, target)
    tampered = (
        source
        / "onnx_splitpoint_tool/resources/native_concurrent_three_stage_yolov7/stage_corpus.py"
    )
    tampered.write_bytes(tampered.read_bytes() + b"\n# computational drift\n")
    for relative in gate.CARRY_FORWARD_COMPUTATIONAL_SOURCE_FILES:
        path = source / relative
        rows[relative] = {
            "path": relative,
            "size": path.stat().st_size,
            "sha256": _sha(path),
        }
    with pytest.raises(
        gate.ClaimGateError, match="carry_forward_computational_source_drift"
    ):
        gate._validate_carry_forward_computational_source(source, rows)
