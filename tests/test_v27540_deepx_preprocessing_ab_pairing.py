from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool.deepx.config import (
    CLASSIFICATION_PREPROCESSING_CURRENT,
    CLASSIFICATION_PREPROCESSING_IMAGENET,
    deepx_classification_preprocessing_contract,
)
from onnx_splitpoint_tool.deepx.preprocessing_ab import (
    ArmEvidence,
    compare_preprocessing_arms,
    compare_run_directories,
    load_arm_evidence,
)


def _arm(tmp_path: Path, mode: str, hits: list[bool]) -> ArmEvidence:
    suffix = "a" if mode == CLASSIFICATION_PREPROCESSING_CURRENT else "b"
    records = tuple(
        {
            "image_id": f"image-{index}",
            "label_id": index,
            "candidate": {
                "top1_hit": bool(hit),
                "top5_hit": bool(hit or index % 2 == 0),
            },
        }
        for index, hit in enumerate(hits)
    )
    return ArmEvidence(
        run_dir=tmp_path / suffix,
        mode=mode,
        profile_name=f"profile-{suffix}",
        cache_dir=f"/cache/{suffix}",
        cache_key=f"deepx_m1_full_{suffix}_v2_key",
        cache_contract_sha256=suffix * 64,
        task="classification",
        target="deepx_m1",
        source_onnx_sha256="1" * 64,
        build_onnx_sha256=("2" if suffix == "a" else "3") * 64,
        dxcom_config_sha256="d" * 64,
        build_options_sha256="e" * 64,
        dxnn_sha256=("4" if suffix == "a" else "5") * 64,
        calibration_identity_sha256="6" * 64,
        calibration_contract_sha256="f" * 64,
        calibration_items_identity_sha256="sha256:" + "7" * 64,
        compiler_identity_sha256="8" * 64,
        compiler_contract_sha256="0" * 64,
        validation_manifest_sha256="9" * 64,
        validation_image_ids_sha256="a" * 64,
        validation_ground_truth_sha256="b" * 64,
        prepared_input_evidence_sha256="1" * 64,
        preprocessing_contract_sha256="2" * 64,
        policy_sha256="c" * 64,
        request_path=tmp_path / suffix / "request.json",
        candidate_path=tmp_path / suffix / "candidate.json",
        records=records,
        deepx_result={"status": "completed", "decision": "pass" if suffix == "b" else "fail"},
        trt_result={"status": "completed", "decision": "pass"},
        workflow_status="ok",
    )


def test_pairing_reports_direct_paired_improvement_and_unlocks_standard_plus(
    tmp_path: Path,
) -> None:
    arm_a = _arm(
        tmp_path, CLASSIFICATION_PREPROCESSING_CURRENT,
        [False] * 20 + [True] * 20,
    )
    arm_b = _arm(
        tmp_path, CLASSIFICATION_PREPROCESSING_IMAGENET, [True] * 40,
    )
    result = compare_preprocessing_arms(
        arm_a, arm_b, bootstrap_repetitions=200, bootstrap_seed=7,
    )
    assert result["status"] == "verified"
    assert result["paired"]["top1"]["arm_a_hits"] == 20
    assert result["paired"]["top1"]["arm_b_hits"] == 40
    assert result["paired"]["top1"]["corrected_by_b"] == 20
    assert result["paired"]["top1"]["regressed_by_b"] == 0
    assert result["standard_plus_ready"] is True
    assert result["next_step"] == "standard_plus"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"compiler_identity_sha256": "f" * 64}, "cohort identity differs"),
        ({"cache_dir": "/cache/a"}, "cache namespaces"),
        ({"build_onnx_sha256": "2" * 64}, "build ONNX"),
        ({"dxnn_sha256": "4" * 64}, "DXNN"),
        ({"build_options_sha256": "1" * 64}, "cohort identity differs"),
    ],
)
def test_pairing_rejects_confounded_arms(
    tmp_path: Path, mutation: dict[str, str], match: str,
) -> None:
    arm_a = _arm(tmp_path, CLASSIFICATION_PREPROCESSING_CURRENT, [False, True])
    arm_b = replace(
        _arm(tmp_path, CLASSIFICATION_PREPROCESSING_IMAGENET, [True, True]),
        **mutation,
    )
    with pytest.raises(ValueError, match=match):
        compare_preprocessing_arms(arm_a, arm_b, bootstrap_repetitions=20)


def test_pairing_rejects_reorder_and_keeps_standard_plus_blocked_on_quality_fail(
    tmp_path: Path,
) -> None:
    arm_a = _arm(tmp_path, CLASSIFICATION_PREPROCESSING_CURRENT, [False, True])
    arm_b = _arm(tmp_path, CLASSIFICATION_PREPROCESSING_IMAGENET, [True, True])
    reordered = replace(arm_b, records=tuple(reversed(arm_b.records)))
    with pytest.raises(ValueError, match="sample order"):
        compare_preprocessing_arms(arm_a, reordered, bootstrap_repetitions=20)

    failed = replace(arm_b, deepx_result={"status": "completed", "decision": "fail"})
    result = compare_preprocessing_arms(arm_a, failed, bootstrap_repetitions=20)
    assert result["standard_plus_ready"] is False
    assert result["next_step"] == "keep_standard_plus_blocked_and_isolate_next_deepx_variable"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_synthetic_run(root: Path, *, mode: str, hits: list[bool]) -> None:
    root.mkdir(parents=True)
    cache_dir = root.parent / "cache" / mode
    contract_body = {
        "schema": "onnx-splitpoint/deepx-full-cache-contract",
        "schema_version": 2,
        "target": "deepx_m1",
        "variant": "full",
        "task": "classification",
        "classification_preprocessing": mode,
        "source_onnx_sha256": "3" * 64,
        "build_onnx_sha256": (
            "3" if mode == CLASSIFICATION_PREPROCESSING_CURRENT else "5"
        ) * 64,
        "dxcom_config_sha256": "d" * 64,
        "preprocessing_contract": {"mode": mode},
        "calibration_manifest_contract": {
            "status": "resolved",
            "items_identity_sha256": "sha256:" + "7" * 64,
        },
        "compiler_identity": {
            "status": "resolved", "evidence": {"version": "test"},
        },
        "build_options": {
            "calibration_method": "ema", "calibration_count": 2, "opt_level": 0,
        },
    }
    for key in ("calibration_manifest_contract", "compiler_identity"):
        subsection = contract_body[key]
        subsection["identity_sha256"] = hashlib.sha256(json.dumps(
            subsection, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
    contract_sha = hashlib.sha256(json.dumps(
        contract_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    cache_contract = {**contract_body, "contract_sha256": contract_sha}
    key_payload = {
        "schema": "onnx-splitpoint/deepx-cache-key",
        "schema_version": 2,
        "target": "deepx_m1",
        "variant": f"full_{mode}",
        "onnx_sha256": cache_contract["build_onnx_sha256"],
        "config_sha256": cache_contract["dxcom_config_sha256"],
        "cache_contract": cache_contract,
    }
    key_digest = hashlib.sha256(json.dumps(
        key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    cache_key = f"deepx_m1_full_{mode}_v2_{key_digest[:32]}"
    dxnn = cache_dir / cache_key / "model.dxnn"
    dxnn.parent.mkdir(parents=True)
    dxnn.write_bytes(("dxnn-" + mode).encode("utf-8"))
    dxnn_sha = hashlib.sha256(dxnn.read_bytes()).hexdigest()
    if mode == CLASSIFICATION_PREPROCESSING_CURRENT:
        adapter = {
            "schema": "onnx-splitpoint/deepx-build-onnx-adapter",
            "schema_version": 1,
            "kind": "identity",
            "source_onnx_sha256": cache_contract["source_onnx_sha256"],
            "build_onnx_sha256": cache_contract["build_onnx_sha256"],
        }
    else:
        adapter_identity = {
            "schema": "onnx-splitpoint/deepx-build-onnx-adapter",
            "schema_version": 1,
            "source_onnx_sha256": cache_contract["source_onnx_sha256"],
            "input_name": "input",
            "consumer_count": 1,
            "preprocessing": deepx_classification_preprocessing_contract(mode),
        }
        adapter = {
            **adapter_identity,
            "adapter_contract_sha256": hashlib.sha256(json.dumps(
                adapter_identity, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")).hexdigest(),
            "build_onnx_path": "build.onnx",
            "build_onnx_sha256": cache_contract["build_onnx_sha256"],
            "build_onnx_bytes": 1,
        }
    _write_json(dxnn.parent / "build_manifest.json", {
        "schema": "onnx-splitpoint/deepx-full-cache-receipt",
        "schema_version": 2,
        "cache_key": cache_key,
        "cache_contract": cache_contract,
        "build_onnx_adapter": adapter,
        "artifact": {"sha256": dxnn_sha, "bytes": dxnn.stat().st_size},
    })
    _write_json(root / "profile_start_snapshot.json", {"profile_name": root.name})
    (root / "profile.yaml").write_text(yaml.safe_dump({
        "name": root.name,
        "deepx_build": {
            "classification_preprocessing": mode,
            "cache_dir": str(cache_dir),
        },
    }), encoding="utf-8")
    _write_json(
        root / "models/resnet50/benchmark_set/deepx/deepx_artifact_status.json",
        {
            "status": "ok",
            "classification_preprocessing": mode,
            "cache_contract": cache_contract,
            "artifacts": [{
                "variant": "full",
                "cache_key": cache_key,
                "dxnn_path": str(dxnn),
            }],
        },
    )
    records = [
        {
            "image_id": f"image-{index}",
            "label_id": index,
            "candidate": {"top1_hit": hit, "top5_hit": hit},
        }
        for index, hit in enumerate(hits)
    ]
    request_dir = (
        root / "models/resnet50/benchmark_results/quality_inputs/setup/"
        "results/deepx_m1_full/task_quality_inputs"
    )
    candidate_path = request_dir / "full_candidate.json"
    _write_json(candidate_path, {"record_count": len(records), "records": records})
    _write_json(request_dir / "full_request.json", {
        "candidate": {
            "path": candidate_path.name,
            "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            "size_bytes": candidate_path.stat().st_size,
        },
        "expected_image_ids": [row["image_id"] for row in records],
        "prepared_input_evidence_sha256": "1" * 64,
        "preprocessing_contract_sha256": "2" * 64,
        "policy_sha256": "9" * 64,
        "producer_identity": {
            "model": {
                "source_onnx_sha256": "3" * 64,
                "runtime_artifact_sha256": dxnn_sha,
            },
            "dataset": {
                "manifest_sha256": "a" * 64,
                "image_ids_sha256": "b" * 64,
                "ground_truth_sha256": "c" * 64,
            },
        },
    })
    _write_json(root / "quality_management/central_quality_summary.json", {
        "technical_status": "completed",
        "results": [
            {
                "source_run_id": "deepx_m1_full",
                "status": "completed",
                "technical_status": "completed",
                "decision": "pass" if mode == CLASSIFICATION_PREPROCESSING_IMAGENET else "fail",
            },
            {
                "source_run_id": "native_full_tensorrt",
                "status": "completed",
                "technical_status": "completed",
                "decision": "pass",
            },
        ],
    })
    _write_json(root / "run_manifest.json", {"status": "ok"})


def test_real_run_directory_loader_seals_receipt_candidate_and_quality_axes(
    tmp_path: Path,
) -> None:
    arm_a = tmp_path / "arm-a"
    arm_b = tmp_path / "arm-b"
    _write_synthetic_run(
        arm_a, mode=CLASSIFICATION_PREPROCESSING_CURRENT,
        hits=[False] * 20 + [True] * 20,
    )
    _write_synthetic_run(
        arm_b, mode=CLASSIFICATION_PREPROCESSING_IMAGENET, hits=[True] * 40,
    )
    result = compare_run_directories(
        arm_a_dir=arm_a,
        arm_b_dir=arm_b,
        expected_records=40,
        bootstrap_repetitions=200,
    )
    assert result["status"] == "verified"
    assert result["technical_evidence_complete"] is True
    assert result["standard_plus_ready"] is True


def test_standard_plus_requires_both_setup_local_tensorrt_controls(
    tmp_path: Path,
) -> None:
    arm_a = _arm(
        tmp_path, CLASSIFICATION_PREPROCESSING_CURRENT,
        [False] * 20 + [True] * 20,
    )
    arm_b = _arm(
        tmp_path, CLASSIFICATION_PREPROCESSING_IMAGENET, [True] * 40,
    )
    arm_b = replace(
        arm_b,
        trt_result={"status": "completed", "decision": "fail"},
    )
    result = compare_preprocessing_arms(
        arm_a, arm_b, bootstrap_repetitions=200, bootstrap_seed=7,
    )
    assert result["setup_local_tensorrt_controls_pass"] is False
    assert result["standard_plus_ready"] is False


def test_standard_plus_requires_positive_paired_top1_and_top5_intervals(
    tmp_path: Path,
) -> None:
    hits = [False] * 20 + [True] * 20
    arm_a = _arm(tmp_path, CLASSIFICATION_PREPROCESSING_CURRENT, hits)
    arm_b = _arm(tmp_path, CLASSIFICATION_PREPROCESSING_IMAGENET, hits)
    result = compare_preprocessing_arms(
        arm_a, arm_b, bootstrap_repetitions=200, bootstrap_seed=7,
    )
    assert result["paired_improvement_ci_positive"] is False
    assert result["standard_plus_ready"] is False


def test_run_loader_rejects_cache_contract_self_hash_tamper(tmp_path: Path) -> None:
    root = tmp_path / "arm-a"
    _write_synthetic_run(
        root, mode=CLASSIFICATION_PREPROCESSING_CURRENT, hits=[True, True],
    )
    status_path = (
        root / "models/resnet50/benchmark_set/deepx/deepx_artifact_status.json"
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["cache_contract"]["contract_sha256"] = "0" * 64
    _write_json(status_path, status)
    with pytest.raises(ValueError, match="self-hash"):
        load_arm_evidence(
            root,
            expected_mode=CLASSIFICATION_PREPROCESSING_CURRENT,
            expected_records=2,
        )


def test_run_loader_rejects_artifact_outside_arm_cache(tmp_path: Path) -> None:
    root = tmp_path / "arm-a"
    _write_synthetic_run(
        root, mode=CLASSIFICATION_PREPROCESSING_CURRENT, hits=[True, True],
    )
    status_path = (
        root / "models/resnet50/benchmark_set/deepx/deepx_artifact_status.json"
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    original = Path(status["artifacts"][0]["dxnn_path"])
    outside = tmp_path / "outside" / "model.dxnn"
    outside.parent.mkdir()
    outside.write_bytes(original.read_bytes())
    status["artifacts"][0]["dxnn_path"] = str(outside)
    _write_json(status_path, status)
    with pytest.raises(ValueError, match="outside the isolated"):
        load_arm_evidence(
            root,
            expected_mode=CLASSIFICATION_PREPROCESSING_CURRENT,
            expected_records=2,
        )


def test_run_loader_rejects_build_onnx_adapter_tamper(tmp_path: Path) -> None:
    root = tmp_path / "arm-b"
    _write_synthetic_run(
        root, mode=CLASSIFICATION_PREPROCESSING_IMAGENET, hits=[True, True],
    )
    status = json.loads((
        root / "models/resnet50/benchmark_set/deepx/deepx_artifact_status.json"
    ).read_text(encoding="utf-8"))
    artifact = Path(status["artifacts"][0]["dxnn_path"])
    receipt_path = artifact.parent / "build_manifest.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["build_onnx_adapter"]["preprocessing"]["build_onnx_adapter"][
        "mean"
    ][0] = 0.0
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="adapter contract"):
        load_arm_evidence(
            root,
            expected_mode=CLASSIFICATION_PREPROCESSING_IMAGENET,
            expected_records=2,
        )
