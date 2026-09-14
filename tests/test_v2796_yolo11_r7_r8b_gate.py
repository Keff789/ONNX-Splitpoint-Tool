from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool.hailo_backend import _resolve_hef_timeout_policy
from onnx_splitpoint_tool.hailo_attempt_receipts import (
    begin_hailo_attempt,
    finalize_hailo_attempt,
)
from onnx_splitpoint_tool.hailo_timeout_policy import (
    canonical_hailo_disable_tokens,
    parse_hailo_timeout_seconds,
)
from onnx_splitpoint_tool.run_modes import apply_run_mode
from onnx_splitpoint_tool.runners.backends.hailo_backend import _timeout_seconds


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/yolo11l_v2796_r8b_full_b067_gate.yaml"
LONG_PROFILE = ROOT / "profiles/complete_set_7models_v2796_b500_audit20.yaml"
HISTORICAL_PROFILE = ROOT / "profiles/complete_set_7models_v2792_b500_audit20.yaml"
MODEL_SHA = "f0fcdf56a4ac24d87ec30c627170492ccad9db80486ec5694df6de65c1b3d147"
BUILD_ID = "v2.79.6-remaining-changes-yolo11-admission-closure"
SESSION_ID = "1" * 32


def _load_verifier(*, isolate_source_manifest: bool = True):
    path = ROOT / "scripts/verify_v2796_yolo11_r8b_gate.py"
    spec = importlib.util.spec_from_file_location("verify_v2796_yolo11_r8b_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if isolate_source_manifest:
        source_manifest = ROOT / "SOURCE_MANIFEST.json"
        module._release_source_identity = lambda _profile: {
            "sha256": _sha(source_manifest),
            "size_bytes": source_manifest.stat().st_size,
            "package_version": "2.79.6",
            "workflow_version": BUILD_ID,
            "verification_contract": (
                "wrapper_installed_scope_verified_before_hardware"
            ),
        }
    return module


def test_r8b_verifier_has_no_python39_only_removeprefix_call() -> None:
    source = (ROOT / "scripts/verify_v2796_yolo11_r8b_gate.py").read_text(
        encoding="utf-8"
    )
    assert ".removeprefix(" not in source


def test_r8b_wrapper_verifies_release_and_installed_source_before_hardware() -> None:
    path = ROOT / "scripts/run_v2796_yolo11_r8b_gate.sh"
    source = path.read_text(encoding="utf-8")
    assert 'expected_version = "2.79.6"' in source
    assert f'expected_build = "{BUILD_ID}"' in source
    assert 'importlib.metadata.version("onnx-splitpoint-tool")' in source
    assert "onnx_splitpoint_tool.__release__" in source
    assert "onnx_splitpoint_tool.__build_id__" in source
    assert "WORKFLOW_VERSION" in source
    assert 'SOURCE_MANIFEST_VERIFIER="$TOOL/scripts/build_source_manifest.py"' in source
    assert "--verify \\\n  --scope installed" in source
    identity = source.index("V2796_R8B_RELEASE_IDENTITY=PASS")
    manifest = source.index("V2796_R8B_SOURCE_MANIFEST=PASS")
    workflow = source.index('"$TOOL_PYTHON" -B "$WORKFLOW"')
    assert identity < manifest < workflow


def test_source_manifest_identity_is_exact_and_version_bound(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier(isolate_source_manifest=False)
    tool = tmp_path / "tool"
    profile = tool / "profiles/gate.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text("name: gate\n", encoding="utf-8")
    manifest = tool / "SOURCE_MANIFEST.json"
    _write_json(manifest, {
        "schema": "onnx-splitpoint/source-manifest-v1",
        "package_version": "2.79.6",
        "workflow_version": BUILD_ID,
        "file_count": 1,
        "files": [],
    })
    identity = verifier._release_source_identity(profile)
    assert identity["sha256"] == _sha(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["package_version"] = "2.79.5"
    _write_json(manifest, payload)
    with pytest.raises(verifier.GateError, match="source_manifest_package_version"):
        verifier._release_source_identity(profile)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("token", [0, "0", "off", "none", "unlimited", "disabled"])
def test_every_documented_hailo_unlimited_token_is_canonical(token) -> None:
    assert parse_hailo_timeout_seconds(token, default=9000) == 0
    assert _resolve_hef_timeout_policy(token)[0] == 0
    assert _timeout_seconds(token, 9000) == 0
    assert canonical_hailo_disable_tokens() == [0, "off", "none", "unlimited", "disabled"]


def test_zero_timeout_means_unlimited() -> None:
    assert parse_hailo_timeout_seconds(0, default=9000) == 0
    assert _resolve_hef_timeout_policy(0)[0] == 0


@pytest.mark.parametrize("token", ["1.5", -1, "banana", float("inf")])
def test_invalid_hailo_timeout_tokens_fail_closed(token) -> None:
    with pytest.raises(ValueError):
        parse_hailo_timeout_seconds(token, default=9000)


def test_retry_preserves_previous_timeout_receipt(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    first = begin_hailo_attempt(
        outdir=tmp_path,
        bound={"onnx_path": str(model), "hw_arch": "hailo8", "endpoint": "raw_head_fallback"},
    )
    timed_out = finalize_hailo_attempt(
        attempt=first,
        result={
            "ok": False,
            "timed_out": True,
            "timeout_kind": "hard_timeout",
            "last_stage": "compile_prep",
            "error": "timeout",
        },
    )
    timeout_bytes = timed_out.read_bytes()
    second = begin_hailo_attempt(
        outdir=tmp_path,
        bound={"onnx_path": str(model), "hw_arch": "hailo8", "endpoint": "raw_head_fallback"},
    )
    succeeded = finalize_hailo_attempt(
        attempt=second,
        result={"ok": True, "status": "success", "last_stage": "compile"},
    )
    terminal = json.loads(
        (tmp_path / "hailo_attempt_receipts/terminal_attempt.json").read_text(encoding="utf-8")
    )
    assert timed_out.read_bytes() == timeout_bytes
    assert timed_out != succeeded
    immutable = [
        path
        for path in (tmp_path / "hailo_attempt_receipts").glob("attempt_*.json")
        if not path.name.endswith(".started.json")
    ]
    assert len(immutable) == 2
    assert terminal["semantic_status"] == "success"
    assert terminal["immutable_receipt"] == str(succeeded)


def test_long_profile_does_not_use_legacy_9000_second_cold_build_cap() -> None:
    assert _sha(HISTORICAL_PROFILE) == "f609ea640d268bca1a72fd4eb66e58a3dee494624577f3272bfd8bc08b5d6467"
    profile = yaml.safe_load(LONG_PROFILE.read_text(encoding="utf-8"))
    assert profile["name"] == "complete_set_7models_v2796_b500_audit20"
    assert profile["campaign"]["id"] == profile["name"]
    assert profile["hailo_build"]["timeout_s"] == 9000
    assert profile["hailo_build"]["cold_build_timeout_s"] == 0
    assert profile["hailo_build"]["hard_timeout_disable_tokens"] == [
        0, "off", "none", "unlimited", "disabled",
    ]
    nested = profile["execution_preset"]["snapshot"]["build"]["hailo"]
    assert nested["timeout_s"] == 9000
    assert nested["cold_build_timeout_s"] == 0
    resolved, _audit = apply_run_mode(profile)
    assert resolved["hailo_build"]["timeout_s"] == 9000
    assert resolved["hailo_build"]["cold_build_timeout_s"] == 0
    assert resolved["hailo_build"]["immutable_attempt_receipts"] is True
    assert resolved["hailo_build"]["terminal_attempt_selection"] == "last_attempt_even_on_failure"


def _result(
    case_id: str,
    variant: str,
    *,
    run_id: str,
    success: bool = True,
) -> dict:
    timings = {
        "full": {"mean_ms": None},
        "part1": {"mean_ms": None},
        "part2": {"mean_ms": None},
        "composed": {"mean_ms": None},
    }
    timings[variant]["mean_ms"] = 4.25 if success else None
    return {
        "run_id": run_id,
        "case_id": case_id,
        "runtime_ok": success,
        "returncode": 0,
        "variant_status": {variant: "ok" if success else "error"},
        "measured_variants": [variant] if success else [],
        "timings": timings,
        # Deliberately false: technical success must not be hidden by Quality.
        "final_pass": False,
        "task_quality_pass": False,
        "quality_gate_status": "failed",
    }


def _hef_receipt(
    root: Path,
    relative: str,
    arch: str,
    *,
    full: bool,
    source_sha256: str,
) -> None:
    directory = root / Path(relative).parent
    directory.mkdir(parents=True, exist_ok=True)
    hef = directory / "compiled.hef"
    hef.write_bytes((arch + relative).encode("utf-8"))
    receipt = {
        "schema": "onnx-splitpoint/hailo-hef-build-receipt/v2",
        "source_onnx_sha256": source_sha256,
        "compiler_onnx_sha256": "2" * 64,
        "hef_sha256": _sha(hef),
        "hef_size_bytes": hef.stat().st_size,
        "hw_arch": arch,
        "cache_key": "3" * 64,
        "cache_payload": {
            "hw_arch": arch,
            "model_sha256": "2" * 64,
            "end_nodes": (
                [
                    "/model.23/cv2.0/cv2.0.2/Conv",
                    "/model.23/cv3.0/cv3.0.2/Conv",
                    "/model.23/cv2.1/cv2.1.2/Conv",
                    "/model.23/cv3.1/cv3.1.2/Conv",
                    "/model.23/cv2.2/cv2.2.2/Conv",
                    "/model.23/cv3.2/cv3.2.2/Conv",
                ] if full else []
            ),
        },
    }
    _write_json(root / relative, receipt)


def _deepx_receipts(root: Path, *, part1_sha256: str) -> None:
    full_status = root / "models/yolo11l/benchmark_set/deepx/deepx_artifact_status.json"
    full_dxnn = root / "models/yolo11l/benchmark_set/legacy_suite/deepx/deepx_m1/full/model.dxnn"
    full_dxnn.parent.mkdir(parents=True, exist_ok=True)
    full_dxnn.write_bytes(b"current-full-dxnn")
    full_dxnn_sha256 = _sha(full_dxnn)
    full_dxnn_size = full_dxnn.stat().st_size
    _write_json(full_status, {
        "schema": "onnx-splitpoint/deepx-artifact-status",
        "status": "ok",
        "source_onnx_sha256": MODEL_SHA,
        "current_full_artifact_identity": {
            "schema": "onnx-splitpoint/deepx-current-artifact-identity",
            "schema_version": 1,
            "mode": "v2796_current_bytes_sha256",
            "status": "verified",
            "source_onnx_sha256": MODEL_SHA,
            "artifact_sha256": full_dxnn_sha256,
            "artifact_size_bytes": full_dxnn_size,
            "suite_artifact_sha256": full_dxnn_sha256,
            "suite_artifact_size_bytes": full_dxnn_size,
        },
        "contracts": [{
            "variant": "full",
            "full_cache_contract_mode": "legacy_implicit",
            "source_onnx_sha256": MODEL_SHA,
            "build_onnx_sha256": "4" * 64,
            "artifact_sha256": full_dxnn_sha256,
            "artifact_size_bytes": full_dxnn_size,
            "suite_artifact_sha256": full_dxnn_sha256,
            "suite_artifact_size_bytes": full_dxnn_size,
            "endpoint_semantic_attestation": {"pass": True},
        }],
    })
    case = root / "models/yolo11l/benchmark_set/legacy_suite/b067"
    part = case / "deepx/deepx_m1/part1"
    part.mkdir(parents=True, exist_ok=True)
    dxnn = part / "model.dxnn"
    dxnn.write_bytes(b"current-b067-dxnn")
    dxnn_sha256 = _sha(dxnn)
    dxnn_size = dxnn.stat().st_size
    contract_path = part / "output_contract.json"
    _write_json(contract_path, {
        "schema": "test",
        "case_id": "b067",
        "source_onnx_sha256": part1_sha256,
        "artifact_sha256": dxnn_sha256,
        "artifact_size_bytes": dxnn_size,
    })
    _write_json(part / "deepx_part1_artifact_status.json", {
        "ok": True,
        "status": "ok",
        "build_status": "ready_built",
        "source_onnx_sha256": part1_sha256,
        "dxnn_sha256": dxnn_sha256,
        "dxnn_size_bytes": dxnn_size,
        "dxnn_path": "deepx/deepx_m1/part1/model.dxnn",
        "output_contract": "deepx/deepx_m1/part1/output_contract.json",
        "output_contract_sha256": _sha(contract_path),
        "cache_key": "deepx-current-b067",
        "cache_contract": {
            "case_id": "b067",
            "compiler_identity": "dx_com:current",
            "calibration_manifest_identity": "sha256:" + "7" * 64,
        },
    })


def _seal_artifact_index(run: Path) -> None:
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    closure_path = run / "reports/artifact_index_closure.json"
    files_before_closure = [
        path
        for path in run.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(run).as_posix() != "artifact_index.json"
    ]
    record_count = len(files_before_closure) + (
        0 if closure_path in files_before_closure else 1
    )
    _write_json(closure_path, {
        "schema": "onnx-splitpoint/artifact-index-terminal-closure",
        "schema_version": 1,
        "status": "pass",
        "run_id": run.name,
        "session_id": SESSION_ID,
        "workflow_status": manifest["status"],
        "artifact_index_path": "artifact_index.json",
        "self_hash_excluded": True,
        "artifact_record_count": record_count,
        "required_coverage_path_count": record_count,
        "verification_error_count": 0,
        "missing_file_count": 0,
        "hash_mismatch_count": 0,
        "unindexed_required_path_count": 0,
    })
    files = sorted(
        (
            path for path in run.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.relative_to(run).as_posix() != "artifact_index.json"
        ),
        key=lambda path: path.relative_to(run).as_posix(),
    )
    assert len(files) == record_count
    _write_json(run / "artifact_index.json", {
        "schema": "onnx-splitpoint/artifact-index",
        "schema_version": 2,
        "run_id": run.name,
        "artifacts": [
            {
                "path": path.relative_to(run).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
            for path in files
        ],
        "terminal_closure": {
            "schema": (
                "onnx-splitpoint/artifact-index-terminal-closure-binding"
            ),
            "schema_version": 1,
            "status": "pass",
            "run_id": run.name,
            "session_id": SESSION_ID,
            "report_path": "reports/artifact_index_closure.json",
            "self_hash_excluded": True,
            "artifact_record_count": record_count,
            "required_coverage_path_count": record_count,
            "verification_error_count": 0,
        },
    })


def _fixture(tmp_path: Path) -> Path:
    run = tmp_path / "yolo11l_v2796_r8b_fixture"
    run.mkdir()
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    (run / "profile_source.yaml").write_text(PROFILE.read_text(encoding="utf-8"), encoding="utf-8")
    (run / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    _write_json(run / "run_manifest.json", {
        "profile_id": profile["name"],
        "run_id": run.name,
        "status": "ok",
        "tool_version": "2.79.6",
        "current_tool_version": "2.79.6",
        "workflow_version": BUILD_ID,
        "current_workflow_version": BUILD_ID,
        "current_session_id": SESSION_ID,
        "current_tool_build": {
            "package_version": "2.79.6",
            "build_id": BUILD_ID,
        },
        "execution_sessions": [{
            "session_id": SESSION_ID,
            "tool_version": "2.79.6",
            "workflow_version": BUILD_ID,
            "tool_build": {
                "package_version": "2.79.6",
                "build_id": BUILD_ID,
            },
        }],
    })
    _write_json(run / "models/yolo11l/model_manifest.json", {
        "model_id": "yolo11l", "observed_model_sha256": MODEL_SHA,
    })
    _write_json(run / "hardware_matrix.json", {
        "hardware_targets": [
            {"id": "orin_nx_hailo8_01", "accelerator": "hailo8", "enabled": True},
            {"id": "orin_nx_hailo10_01", "accelerator": "hailo10", "enabled": True},
            {"id": "orin_nx_deepx_m1_01", "accelerator": "deepx_m1", "enabled": True},
        ],
    })
    case = run / "models/yolo11l/benchmark_set/legacy_suite/b067"
    part1 = case / "yolo11l_part1_b67.onnx"
    part1.parent.mkdir(parents=True, exist_ok=True)
    part1.write_bytes(b"current-yolo11l-b067-part1-onnx")
    part1_sha256 = _sha(part1)
    _write_json(case / "split_manifest.json", {
        "boundary": 67,
        "full_model": "../models/yolo11l.onnx",
        "part1_model": part1.name,
    })
    result_dir = run / "models/yolo11l/benchmark_results"
    for filename, run_id, case_id, variant in (
        ("benchmark_results_hailo8_auto.json", "hailo8", "full", "full"),
        ("benchmark_results_hailo10_auto.json", "hailo10", "full", "full"),
        ("benchmark_results_deepx_m1_full_auto.json", "deepx_m1_full", "full", "full"),
        ("benchmark_results_hailo8_to_trt_auto.json", "hailo8_to_trt", "b067", "composed"),
        ("benchmark_results_hailo10_to_trt_auto.json", "hailo10_to_tensorrt", "b067", "composed"),
        ("benchmark_results_deepx_m1_to_tensorrt_auto.json", "deepx_m1_to_tensorrt", "b067", "composed"),
    ):
        _write_json(
            result_dir / filename,
            [_result(case_id, variant, run_id=run_id)],
        )
    for relative, arch, full in (
        ("models/yolo11l/benchmark_set/legacy_suite/hailo/hailo8/full/hailo_hef_build_receipt.json", "hailo8", True),
        ("models/yolo11l/benchmark_set/legacy_suite/hailo/hailo10/full/hailo_hef_build_receipt.json", "hailo10h", True),
        ("models/yolo11l/benchmark_set/legacy_suite/b067/hailo/hailo8/part1/hailo_hef_build_receipt.json", "hailo8", False),
        ("models/yolo11l/benchmark_set/legacy_suite/b067/hailo/hailo10/part1/hailo_hef_build_receipt.json", "hailo10h", False),
    ):
        _hef_receipt(
            run,
            relative,
            arch,
            full=full,
            source_sha256=MODEL_SHA if full else part1_sha256,
        )
    _deepx_receipts(run, part1_sha256=part1_sha256)
    _seal_artifact_index(run)
    return run


def test_r8b_gate_accepts_six_receipt_bound_technical_successes(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    verdict = verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)
    assert verdict["status"] == "PASS"
    assert verdict["required_path_count"] == 6
    assert verdict["success_count"] == 6
    assert verdict["blocked_count"] == 0
    assert verdict["version"] == "2.79.6"
    assert verdict["build_id"] == BUILD_ID
    assert verdict["workflow_version"] == BUILD_ID
    assert verdict["release_identity"]["current_session_id"] == SESSION_ID
    assert verdict["source_manifest_sha256"] == _sha(ROOT / "SOURCE_MANIFEST.json")
    assert verdict["artifact_index_closure"]["status"] == "pass"
    assert verdict["artifact_index"]["sha256"] == _sha(run / "artifact_index.json")
    assert verdict["source_bindings"]["full_sha256"] == MODEL_SHA
    assert verdict["source_bindings"]["full_binding_mode"] == (
        "portable_profile_pin_plus_model_manifest_observed_sha256"
    )
    assert verdict["source_bindings"]["model_manifest"]["sha256"] == _sha(
        run / "models/yolo11l/model_manifest.json"
    )
    assert verdict["quality_decision_used_for_terminal"] is False
    assert all(row["quality_annotation"]["final_pass"] is False for row in verdict["paths"].values())


def test_quality_pass_cannot_hide_missing_hailo10_composed(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    path = run / "models/yolo11l/benchmark_results/benchmark_results_hailo10_to_trt_auto.json"
    row = _result(
        "b067", "part2", run_id="hailo10_to_tensorrt"
    )
    row["final_pass"] = True
    _write_json(path, [row])
    _seal_artifact_index(run)
    with pytest.raises(verifier.GateError, match="result_block_class_not_allowed:missing"):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)


def test_exact_runtime_block_is_terminal_even_when_quality_says_pass(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    path = run / "models/yolo11l/benchmark_results/benchmark_results_deepx_m1_full_auto.json"
    _write_json(path, [{
        **_result(
            "full", "full", run_id="deepx_m1_full", success=False
        ),
        "error_class": "deepx_runtime_failed",
        "error_detail": "RuntimeError: native_full_raw_detection_model_identity_missing",
        "final_pass": True,
    }])
    _seal_artifact_index(run)
    verdict = verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=1)
    row = verdict["paths"]["deepx_full"]
    assert row["terminal_status"] == "blocked"
    assert row["evidence"]["reason"].endswith("native_full_raw_detection_model_identity_missing")
    assert row["quality_annotation"]["final_pass"] is True


def test_exact_immutable_hailo_block_is_a_valid_terminal(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    result = run / "models/yolo11l/benchmark_results/benchmark_results_hailo8_auto.json"
    _write_json(result, [{
        **_result("full", "full", run_id="hailo8", success=False),
        "error_class": "hailo_compiler_timeout",
        "timed_out": True,
        "final_pass": True,
    }])
    full = run / "models/yolo11l/benchmark_set/legacy_suite/hailo/hailo8/full"
    (full / "hailo_hef_build_receipt.json").unlink()
    (full / "compiled.hef").unlink()
    receipts = full / "hailo_attempt_receipts"
    immutable = receipts / "attempt_raw.json"
    payload = {
        "schema": "onnx-splitpoint/hailo-build-attempt-receipt",
        "schema_version": 2,
        "attempt_id": "raw",
        "terminal": True,
        "semantic_status": "timeout",
        "hw_arch": "hailo8",
        "source_onnx_sha256": MODEL_SHA,
        "ended_at_epoch_s": 2.0,
        "compiler_phase": "compile_prep",
        "last_active_stage": "compile_prep",
        "error_class": "hard_timeout",
        "error": "compiler heartbeat continued beyond policy window",
        "end_nodes": verifier.YOLO11_RAW_END_NODES,
    }
    _write_json(immutable, payload)
    pointer = dict(payload)
    pointer.update({
        "immutable_receipt": str(immutable),
        "immutable_receipt_sha256": _sha(immutable),
    })
    _write_json(receipts / "terminal_attempt.json", pointer)
    _seal_artifact_index(run)
    verdict = verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=1)
    assert verdict["status"] == "PASS"
    assert verdict["blocked_count"] == 1
    assert verdict["paths"]["hailo8_full"]["terminal_status"] == "blocked"
    assert verdict["paths"]["hailo10h_full"]["terminal_status"] == "success"
    assert verdict["paths"]["hailo8_full"]["quality_annotation"]["final_pass"] is True


@pytest.mark.parametrize("workflow_rc", [2, 97, -1, True])
def test_unknown_workflow_rc_is_rejected(
    tmp_path: Path, workflow_rc: object
) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    with pytest.raises(verifier.GateError, match="workflow_rc_not_allowed"):
        verifier.verify(
            gate_output=run,
            expected_profile=PROFILE,
            workflow_rc=workflow_rc,
        )


def test_workflow_rc_one_requires_an_exact_technical_block(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    with pytest.raises(
        verifier.GateError,
        match="workflow_rc_1_without_exact_technical_block",
    ):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=1)


def test_nonzero_result_rc_without_allowed_class_and_reason_is_not_a_block(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    results = run / "models/yolo11l/benchmark_results"
    for path in results.glob("*.json"):
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert len(rows) == 1
        row = rows[0]
        row.update({
            "runtime_ok": False,
            "returncode": 1,
            "measured_variants": [],
            "variant_status": {
                key: "error" for key in row.get("variant_status", {})
            },
        })
        for key in (
            "error_class", "failure_kind", "timeout_kind", "error_detail",
            "failure_reason", "status_detail", "error", "unsupported_reason",
        ):
            row.pop(key, None)
        _write_json(path, [row])
    _seal_artifact_index(run)
    with pytest.raises(
        verifier.GateError, match="result_block_class_not_allowed:missing"
    ):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=1)


def test_result_run_id_must_match_the_exact_terminal_path(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    path = run / "models/yolo11l/benchmark_results/benchmark_results_hailo8_auto.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0]["run_id"] = "hailo10"
    _write_json(path, rows)
    _seal_artifact_index(run)
    with pytest.raises(
        verifier.GateError, match="result_run_id_mismatch:hailo8_full"
    ):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)


def test_each_terminal_result_file_rejects_out_of_scope_rows(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    path = (
        run / "models/yolo11l/benchmark_results/"
        "benchmark_results_hailo8_to_trt_auto.json"
    )
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        _result("b068", "composed", run_id="hailo8_to_trt")
    )
    _write_json(path, rows)
    _seal_artifact_index(run)
    with pytest.raises(verifier.GateError, match="unexpected_result_rows"):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)


def test_full_binding_requires_pinned_observed_model_sha(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    path = run / "models/yolo11l/model_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["observed_model_sha256"] = "e" * 64
    _write_json(path, manifest)
    with pytest.raises(verifier.GateError, match="observed_model_sha_mismatch"):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)


def test_stale_v2795_run_identity_is_rejected_even_after_reseal(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    path = run / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stale_build = "v2.79.5-remaining-changes-release-closure"
    manifest.update({
        "tool_version": "2.79.5",
        "current_tool_version": "2.79.5",
        "workflow_version": stale_build,
        "current_workflow_version": stale_build,
    })
    manifest["current_tool_build"].update({
        "package_version": "2.79.5",
        "build_id": stale_build,
    })
    manifest["execution_sessions"][0].update({
        "tool_version": "2.79.5",
        "workflow_version": stale_build,
    })
    manifest["execution_sessions"][0]["tool_build"].update({
        "package_version": "2.79.5",
        "build_id": stale_build,
    })
    _write_json(path, manifest)
    _seal_artifact_index(run)
    with pytest.raises(verifier.GateError, match="run_manifest_tool_version"):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)


def test_any_ancestor_symlink_in_evidence_path_is_rejected(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    result_dir = run / "models/yolo11l/benchmark_results"
    outside = tmp_path / "outside_benchmark_results"
    result_dir.rename(outside)
    result_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(verifier.GateError, match="symlink_component"):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)


@pytest.mark.parametrize(
    "relative,expected",
    [
        (
            "models/yolo11l/benchmark_set/legacy_suite/hailo/hailo8/full/"
            "hailo_hef_build_receipt.json",
            "hailo8_source_artifact_mismatch",
        ),
        (
            "models/yolo11l/benchmark_set/legacy_suite/b067/hailo/hailo10/"
            "part1/hailo_hef_build_receipt.json",
            "hailo10h_source_artifact_mismatch",
        ),
    ],
)
def test_hailo_success_receipt_must_bind_current_source_bytes(
    tmp_path: Path, relative: str, expected: str
) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    path = run / relative
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["source_onnx_sha256"] = "a" * 64
    _write_json(path, receipt)
    _seal_artifact_index(run)
    with pytest.raises(verifier.GateError, match=expected):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)


def test_hailo_cache_payload_must_bind_compiler_onnx(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    path = (
        run / "models/yolo11l/benchmark_set/legacy_suite/hailo/hailo8/full/"
        "hailo_hef_build_receipt.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["cache_payload"]["model_sha256"] = "b" * 64
    _write_json(path, receipt)
    _seal_artifact_index(run)
    with pytest.raises(verifier.GateError, match="hailo8_cache_compiler_mismatch"):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)


def test_hailo_terminal_block_must_bind_current_source_bytes(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    result = run / "models/yolo11l/benchmark_results/benchmark_results_hailo8_auto.json"
    _write_json(result, [{
        **_result("full", "full", run_id="hailo8", success=False),
        "error_class": "hailo_compiler_timeout",
        "error_detail": "compiler heartbeat continued beyond policy window",
    }])
    full = run / "models/yolo11l/benchmark_set/legacy_suite/hailo/hailo8/full"
    (full / "hailo_hef_build_receipt.json").unlink()
    (full / "compiled.hef").unlink()
    receipts = full / "hailo_attempt_receipts"
    immutable = receipts / "attempt_wrong_source.json"
    payload = {
        "schema": "onnx-splitpoint/hailo-build-attempt-receipt",
        "schema_version": 2,
        "attempt_id": "wrong_source",
        "terminal": True,
        "semantic_status": "timeout",
        "hw_arch": "hailo8",
        "source_onnx_sha256": "c" * 64,
        "ended_at_epoch_s": 2.0,
        "compiler_phase": "compile_prep",
        "last_active_stage": "compile_prep",
        "error_class": "hard_timeout",
        "error": "compiler heartbeat continued beyond policy window",
        "end_nodes": verifier.YOLO11_RAW_END_NODES,
    }
    _write_json(immutable, payload)
    _write_json(receipts / "terminal_attempt.json", {
        **payload,
        "immutable_receipt": str(immutable),
        "immutable_receipt_sha256": _sha(immutable),
    })
    _seal_artifact_index(run)
    with pytest.raises(verifier.GateError, match="hailo8_attempt_source_mismatch"):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=1)


def test_deepx_full_success_is_bound_to_current_dxnn_bytes(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    dxnn = (
        run / "models/yolo11l/benchmark_set/legacy_suite/deepx/deepx_m1/"
        "full/model.dxnn"
    )
    dxnn.write_bytes(dxnn.read_bytes() + b"tampered")
    _seal_artifact_index(run)
    with pytest.raises(
        verifier.GateError, match="deepx_full_contract_artifact_mismatch"
    ):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)


def test_post_seal_artifact_mutation_is_rejected_by_terminal_closure(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    dxnn = (
        run / "models/yolo11l/benchmark_set/legacy_suite/deepx/deepx_m1/"
        "full/model.dxnn"
    )
    dxnn.write_bytes(dxnn.read_bytes() + b"post-seal-tamper")
    with pytest.raises(
        verifier.GateError,
        match="artifact_index_(size|sha)_mismatch:.*full/model.dxnn",
    ):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)


def test_deepx_b067_success_is_bound_to_current_dxnn_bytes(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    dxnn = (
        run / "models/yolo11l/benchmark_set/legacy_suite/b067/deepx/"
        "deepx_m1/part1/model.dxnn"
    )
    dxnn.write_bytes(dxnn.read_bytes() + b"tampered")
    _seal_artifact_index(run)
    with pytest.raises(
        verifier.GateError, match="deepx_b067_status_artifact_mismatch"
    ):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)


def test_deepx_b067_success_is_bound_to_current_part1_bytes(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    part1 = (
        run / "models/yolo11l/benchmark_set/legacy_suite/b067/"
        "yolo11l_part1_b67.onnx"
    )
    part1.write_bytes(part1.read_bytes() + b"tampered")
    _seal_artifact_index(run)
    with pytest.raises(verifier.GateError, match="hailo8_source_artifact_mismatch"):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)


def test_deepx_b067_status_must_declare_current_part1_sha(tmp_path: Path) -> None:
    verifier = _load_verifier()
    run = _fixture(tmp_path)
    status_path = (
        run / "models/yolo11l/benchmark_set/legacy_suite/b067/deepx/"
        "deepx_m1/part1/deepx_part1_artifact_status.json"
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["source_onnx_sha256"] = "d" * 64
    _write_json(status_path, status)
    _seal_artifact_index(run)
    with pytest.raises(verifier.GateError, match="deepx_b067_source_mismatch"):
        verifier.verify(gate_output=run, expected_profile=PROFILE, workflow_rc=0)


def test_deepx_cache_receipt_rejects_same_size_current_byte_tamper_in_fast_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onnx_splitpoint_tool.deepx.artifacts import (
        cache_dxnn_artifact,
        deepx_cached_artifact_compatible,
        deepx_cached_artifact_identity_compatible,
        sha256_file,
    )

    source = tmp_path / "source.dxnn"
    source.write_bytes(b"a" * (1024 * 1024))
    contract = {"schema": "v2796-deepx-current-byte-test", "case_id": "b067"}
    cached = cache_dxnn_artifact(
        dxnn_path=source,
        manifest={"cache_contract": contract},
        cache_root=tmp_path / "cache",
        cache_key="part1-b067",
    )
    monkeypatch.setenv("ONNX_SPLITPOINT_INTEGRITY_MODE", "fast")
    monkeypatch.setenv("ONNX_SPLITPOINT_HASH_CACHE", str(tmp_path / "hash-cache.json"))
    original_digest = sha256_file(cached)
    original_stat = cached.stat()
    tampered = bytearray(cached.read_bytes())
    tampered[128 * 1024] = ord("b")
    cached.write_bytes(tampered)
    os.utime(
        cached,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert sha256_file(cached) == original_digest

    ok, reason, _ = deepx_cached_artifact_compatible(
        cache_dir=cached.parent,
        expected_contract=contract,
        require_artifact_identity=True,
    )
    assert not ok
    assert reason == "artifact_identity_mismatch"
    ok, reason, _ = deepx_cached_artifact_identity_compatible(
        cache_dir=cached.parent,
    )
    assert not ok
    assert reason == "artifact_identity_mismatch"


def test_deepx_part1_producer_requires_receipt_identity_and_strict_hashes() -> None:
    source = (
        ROOT / "onnx_splitpoint_tool/gui/benchmark_workflow.py"
    ).read_text(encoding="utf-8")
    start = source.index("def _materialize_manual_deepx_part1_artifacts(")
    implementation = source[start:]
    assert "require_artifact_identity=True" in implementation
    assert "source_onnx_sha256 = strict_artifact_sha256(p1)" in implementation
    assert "dxnn_sha256 = strict_artifact_sha256(dst)" in implementation
    assert "output_contract_sha256 = strict_artifact_sha256(contract_path)" in implementation
