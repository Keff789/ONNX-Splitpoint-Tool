from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.campaign import create_dataset_manifest
from scripts import pin_v27541_deepx_calibration_baseline as baseline_pin
from scripts.pin_v27541_deepx_calibration_baseline import PinError, pin_manifest


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> str:
    return _sha(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8"))


@pytest.fixture(autouse=True)
def _stub_full_b500_loader(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Keep this unit fixture small; v2.75.40 tests cover the full arm loader."""
    def fake_loader(run_dir: str | Path, **_: object) -> SimpleNamespace:
        status_path = (
            Path(run_dir) / "models/resnet50/benchmark_set/deepx/"
            "deepx_artifact_status.json"
        )
        contract = json.loads(status_path.read_text(encoding="utf-8"))["cache_contract"]
        calibration = contract["calibration_manifest_contract"]
        base = SimpleNamespace(
            cache_contract_sha256=contract["contract_sha256"],
            calibration_identity_sha256=calibration["identity_sha256"],
            calibration_items_identity_sha256=calibration["items_identity_sha256"],
        )
        return SimpleNamespace(base=base)

    monkeypatch.setattr(
        baseline_pin.canary_authority, "load_calibration_arm", fake_loader,
    )
    if not request.node.name.startswith("test_frozen_b500_authority_"):
        monkeypatch.setattr(
            baseline_pin,
            "validate_frozen_b500_authority",
            lambda _evidence: {
                "ok": True,
                "authority": "unit_fixture_mechanics_only",
                "observed": {},
            },
        )


def _write_manifest(root: Path, *, count: int = 500) -> Path:
    for index in range(count):
        class_dir = root / f"n{index:08d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        data = f"image-{index}".encode()
        image = class_dir / f"image-{index}.JPEG"
        image.write_bytes(data)
    path = root.parent / "manifest.json"
    return create_dataset_manifest(
        task="classification",
        role="calibration",
        dataset_id="ilsvrc2012-train-calibration",
        split="train",
        root=root,
        output=path,
        hash_mode="content",
        max_items=0,
        selection_strategy="class_stratified",
        selection_seed=20260710,
    )


def _write_run(run: Path, manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    calibration = {
        "status": "resolved",
        "item_count": 500,
        "effective_count": 500,
        "manifest_file_sha256": _sha(manifest.read_bytes()),
        "manifest_payload_sha256": payload["manifest_payload_sha256"],
        "items_identity_sha256": payload["items_identity_sha256"],
    }
    calibration["identity_sha256"] = _canonical(calibration)
    compiler = {"status": "resolved", "evidence": {"version": "test"}}
    compiler["identity_sha256"] = _canonical(compiler)
    contract = {
        "schema": "onnx-splitpoint/deepx-full-cache-contract",
        "schema_version": 2,
        "task": "classification",
        "target": "deepx_m1",
        "classification_preprocessing": "imagenet_mean_std",
        "calibration_manifest_contract": calibration,
        "compiler_identity": compiler,
        "build_options": {
            "calibration_count": 500,
            "calibration_method": "ema",
            "opt_level": 0,
        },
    }
    contract["contract_sha256"] = _canonical(contract)
    status = {
        "status": "ok",
        "classification_preprocessing": "imagenet_mean_std",
        "cache_contract": contract,
    }
    path = run / "models/resnet50/benchmark_set/deepx/deepx_artifact_status.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(status), encoding="utf-8")


def test_pin_creates_then_reuses_identical_manifest(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "data")
    run = tmp_path / "run"
    _write_run(run, manifest)
    output = tmp_path / "pinned/manifest.json"
    first = pin_manifest(baseline_run=run, source=manifest, output=output)
    second = pin_manifest(baseline_run=run, source=manifest, output=output)
    assert first["action"] == "created"
    assert second["action"] == "reused_identical"
    assert output.read_bytes() == manifest.read_bytes()


def test_pin_refuses_source_not_bound_by_run(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "data")
    run = tmp_path / "run"
    _write_run(run, manifest)
    status_path = run / "models/resnet50/benchmark_set/deepx/deepx_artifact_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["cache_contract"]["calibration_manifest_contract"]["items_identity_sha256"] = "f" * 64
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(PinError, match="self-hash mismatch|does not match"):
        pin_manifest(
            baseline_run=run,
            source=manifest,
            output=tmp_path / "pinned/manifest.json",
        )


def test_pin_never_overwrites_different_destination(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "data")
    run = tmp_path / "run"
    _write_run(run, manifest)
    output = tmp_path / "pinned/manifest.json"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"different")
    with pytest.raises(PinError, match="not overwritten"):
        pin_manifest(baseline_run=run, source=manifest, output=output)
    assert output.read_bytes() == b"different"


def test_pin_refuses_tampered_outer_cache_contract(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "data")
    run = tmp_path / "run"
    _write_run(run, manifest)
    status_path = run / "models/resnet50/benchmark_set/deepx/deepx_artifact_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["cache_contract"]["build_options"]["calibration_method"] = "minmax"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(PinError, match="self-hash mismatch"):
        pin_manifest(
            baseline_run=run,
            source=manifest,
            output=tmp_path / "pinned/manifest.json",
        )


def test_pin_refuses_symlink_source_and_destination(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "data")
    run = tmp_path / "run"
    _write_run(run, manifest)
    source_link = tmp_path / "source-link.json"
    source_link.symlink_to(manifest)
    with pytest.raises(PinError, match="not a regular file"):
        pin_manifest(
            baseline_run=run,
            source=source_link,
            output=tmp_path / "pinned/source-case.json",
        )

    destination_target = tmp_path / "destination-target.json"
    destination_target.write_bytes(manifest.read_bytes())
    destination_link = tmp_path / "destination-link.json"
    destination_link.symlink_to(destination_target)
    with pytest.raises(PinError, match="not a regular file"):
        pin_manifest(
            baseline_run=run,
            source=manifest,
            output=destination_link,
        )
    assert destination_target.read_bytes() == manifest.read_bytes()


def _quality_result(
    *, source_run_id: str, top1: int, top5: int, decision: str,
    source_request_sha256: str, top5_decision: str | None = None,
) -> dict[str, object]:
    authority = baseline_pin.canary_authority
    top5_decision = top5_decision or decision
    return {
        "schema": "onnx-splitpoint/management-paired-quality-result",
        "status": "completed",
        "technical_status": "completed",
        "source_run_id": source_run_id,
        "source_setup_id": authority.EXPECTED_SETUP_ID,
        "source_request_sha256": source_request_sha256,
        "case_id": "full",
        "variant": "full",
        "task": "classification",
        "execution_role": "full_quality_only",
        "backend": (
            "tensorrt" if source_run_id == "native_full_tensorrt" else "deepx_m1"
        ),
        "performance_claims_emitted": False,
        "n": 500,
        "algorithm_version": authority.EXPECTED_QUALITY_ALGORITHM,
        "decision": decision,
        "primary": {
            "decision": decision,
            "sample_count": 500,
            "candidate_hits": top1,
            "reference_hits": authority.EXPECTED_REFERENCE_TOP1_HITS,
        },
        "guardrails": {
            "top5_accuracy": {
                "decision": top5_decision,
                "sample_count": 500,
                "candidate_hits": top5,
                "reference_hits": authority.EXPECTED_REFERENCE_TOP5_HITS,
            },
        },
    }


def _record_identity(
    records: tuple[dict[str, object], ...],
) -> tuple[list[str], str, str]:
    return baseline_pin.canary_authority._classification_record_identity(
        records, label="unit B500 endpoint",
    )


def _write_quality_endpoint(
    tmp_path: Path,
    *,
    source_run_id: str,
    records: tuple[dict[str, object], ...],
    decision: str,
    top5_decision: str | None = None,
) -> SimpleNamespace:
    authority = baseline_pin.canary_authority
    image_ids, image_ids_sha256, ground_truth_sha256 = _record_identity(records)
    producer = {
        "setup_id": authority.EXPECTED_SETUP_ID,
        "dataset": {
            "manifest_sha256": authority.EXPECTED_VALIDATION_MANIFEST_SHA256,
            "image_ids_sha256": image_ids_sha256,
            "ground_truth_sha256": ground_truth_sha256,
            "image_count": 500,
        },
    }
    request = {
        "setup_id": authority.EXPECTED_SETUP_ID,
        "expected_image_ids": image_ids,
        "expected_image_ids_sha256": image_ids_sha256,
        "producer_identity": producer,
    }
    endpoint_dir = tmp_path / source_run_id
    endpoint_dir.mkdir(parents=True)
    request_path = endpoint_dir / "full_request.json"
    request_path.write_text(
        json.dumps(request, sort_keys=True), encoding="utf-8",
    )
    candidate_path = endpoint_dir / "full_candidate.json"
    candidate_path.write_text("{}\n", encoding="utf-8")
    top1 = sum(bool(row["candidate"]["top1_hit"]) for row in records)
    top5 = sum(bool(row["candidate"]["top5_hit"]) for row in records)
    result = _quality_result(
        source_run_id=source_run_id,
        top1=top1,
        top5=top5,
        decision=decision,
        top5_decision=top5_decision,
        source_request_sha256=_sha(request_path.read_bytes()),
    )
    return SimpleNamespace(
        source_run_id=source_run_id,
        setup_id=authority.EXPECTED_SETUP_ID,
        request_path=request_path,
        candidate_path=candidate_path,
        request=request,
        producer=producer,
        candidate={},
        records=records,
        result=result,
    )


def _reseal_endpoint_records(
    endpoint: SimpleNamespace,
    records: tuple[dict[str, object], ...],
) -> SimpleNamespace:
    image_ids, image_ids_sha256, ground_truth_sha256 = _record_identity(records)
    producer = json.loads(json.dumps(endpoint.producer))
    producer["dataset"]["image_ids_sha256"] = image_ids_sha256
    producer["dataset"]["ground_truth_sha256"] = ground_truth_sha256
    request = json.loads(json.dumps(endpoint.request))
    request["expected_image_ids"] = image_ids
    request["expected_image_ids_sha256"] = image_ids_sha256
    request["producer_identity"] = producer
    endpoint.request_path.write_text(
        json.dumps(request, sort_keys=True), encoding="utf-8",
    )
    result = json.loads(json.dumps(endpoint.result))
    result["source_request_sha256"] = _sha(endpoint.request_path.read_bytes())
    return SimpleNamespace(
        **{
            **vars(endpoint),
            "request": request,
            "producer": producer,
            "records": records,
            "result": result,
        },
    )


def _arm_with_endpoints(
    arm: SimpleNamespace,
    *,
    deepx_endpoint: SimpleNamespace | None = None,
    trt_endpoint: SimpleNamespace | None = None,
) -> SimpleNamespace:
    deepx = deepx_endpoint or arm.deepx_endpoint
    trt = trt_endpoint or arm.trt_endpoint
    base = SimpleNamespace(**vars(arm.base))
    base.request_path = deepx.request_path
    base.candidate_path = deepx.candidate_path
    base.records = deepx.records
    base.deepx_result = deepx.result
    base.trt_result = trt.result
    return SimpleNamespace(
        **{
            **vars(arm),
            "base": base,
            "deepx_endpoint": deepx,
            "trt_endpoint": trt,
        },
    )


def test_pin_uses_full_frozen_arm_loader_and_explicit_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path / "data")
    run = tmp_path / "run"
    _write_run(run, manifest)
    status_path = run / "models/resnet50/benchmark_set/deepx/deepx_artifact_status.json"
    contract = json.loads(status_path.read_text(encoding="utf-8"))["cache_contract"]
    calls: list[dict[str, object]] = []

    def strict_loader(run_dir: str | Path, **kwargs: object) -> SimpleNamespace:
        calls.append({"run_dir": Path(run_dir), **kwargs})
        calibration = contract["calibration_manifest_contract"]
        return SimpleNamespace(base=SimpleNamespace(
            cache_contract_sha256=contract["contract_sha256"],
            calibration_identity_sha256=calibration["identity_sha256"],
            calibration_items_identity_sha256=calibration["items_identity_sha256"],
        ))

    monkeypatch.setattr(
        baseline_pin.canary_authority, "load_calibration_arm", strict_loader,
    )
    baseline_pin._cache_contract(run, manifest)
    assert calls == [{
        "run_dir": run,
        "label": "B500",
        "expected_calibration_count": 500,
        "calibration_manifest_path": manifest,
    }]


def test_frozen_b500_authority_binds_both_endpoint_records_and_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = baseline_pin.canary_authority
    assert (
        authority.EXPECTED_SETUP_ID,
        authority.EXPECTED_B500_DEEPX_TOP1_HITS,
        authority.EXPECTED_B500_DEEPX_TOP5_HITS,
        authority.EXPECTED_REFERENCE_TOP1_HITS,
        authority.EXPECTED_REFERENCE_TOP5_HITS,
        authority.EXPECTED_VALIDATION_IMAGE_IDS_SHA256,
        authority.EXPECTED_VALIDATION_GROUND_TRUTH_SHA256,
    ) == (
        "orin_nx_deepx_m1_01", 401, 474, 406, 476,
        "71032a98e158ca71711a5567de5d46fbf04f05777baa940f74f3c39fdf0c083f",
        "87775e86ef3bcf3ec1e0d0a79696bf2f58fa215d5320ce80167bb1f417c7a177",
    )
    deepx_records = tuple({
        "image_id": f"image-{index:03d}",
        "label_id": index % 1000,
        "candidate": {
            "top1_hit": index < authority.EXPECTED_B500_DEEPX_TOP1_HITS,
            "top5_hit": index < authority.EXPECTED_B500_DEEPX_TOP5_HITS,
        },
    } for index in range(500))
    trt_records = tuple({
        "image_id": row["image_id"],
        "label_id": row["label_id"],
        "candidate": {
            "top1_hit": index < authority.EXPECTED_REFERENCE_TOP1_HITS,
            "top5_hit": index < authority.EXPECTED_REFERENCE_TOP5_HITS,
        },
    } for index, row in enumerate(deepx_records))
    _, ids_sha256, ground_truth_sha256 = _record_identity(deepx_records)
    monkeypatch.setattr(
        authority, "EXPECTED_VALIDATION_IMAGE_IDS_SHA256", ids_sha256,
    )
    monkeypatch.setattr(
        authority, "EXPECTED_VALIDATION_GROUND_TRUTH_SHA256", ground_truth_sha256,
    )
    monkeypatch.setattr(
        authority,
        "_validate_endpoint_scientific_binding",
        lambda _arm, _endpoint: {
            "policy_sha256": authority.EXPECTED_QUALITY_POLICY_SHA256,
        },
    )
    deepx = _write_quality_endpoint(
        tmp_path,
        source_run_id="deepx_m1_full",
        records=deepx_records,
        decision="inconclusive",
        top5_decision="pass",
    )
    trt = _write_quality_endpoint(
        tmp_path,
        source_run_id="native_full_tensorrt",
        records=trt_records,
        decision="pass",
    )
    base = SimpleNamespace(
        run_dir=tmp_path,
        mode="imagenet_mean_std",
        profile_name=authority.BASELINE_PROFILE_NAME,
        cache_dir=authority.B500_CACHE_NAMESPACE,
        cache_key=authority.EXPECTED_B500_CACHE_KEY,
        cache_contract_sha256=authority.EXPECTED_B500_CACHE_CONTRACT_SHA256,
        task="classification",
        target="deepx_m1",
        source_onnx_sha256=authority.EXPECTED_SOURCE_ONNX_SHA256,
        build_onnx_sha256=authority.EXPECTED_BUILD_ONNX_SHA256,
        dxcom_config_sha256=authority.EXPECTED_B500_DXCOM_CONFIG_SHA256,
        build_options_sha256=authority.EXPECTED_B500_BUILD_OPTIONS_SHA256,
        dxnn_sha256=authority.EXPECTED_B500_DXNN_SHA256,
        calibration_identity_sha256=(
            authority.EXPECTED_B500_CALIBRATION_IDENTITY_SHA256
        ),
        calibration_contract_sha256=(
            authority.EXPECTED_B500_CALIBRATION_CONTRACT_SHA256
        ),
        calibration_items_identity_sha256=(
            authority.EXPECTED_B500_CALIBRATION_ITEMS_SHA256
        ),
        compiler_identity_sha256=authority.EXPECTED_COMPILER_IDENTITY_SHA256,
        compiler_contract_sha256=authority.EXPECTED_COMPILER_CONTRACT_SHA256,
        validation_manifest_sha256=authority.EXPECTED_VALIDATION_MANIFEST_SHA256,
        validation_image_ids_sha256=ids_sha256,
        validation_ground_truth_sha256=ground_truth_sha256,
        prepared_input_evidence_sha256=(
            authority.EXPECTED_PREPARED_INPUT_EVIDENCE_SHA256
        ),
        preprocessing_contract_sha256=(
            authority.EXPECTED_PREPROCESSING_CONTRACT_SHA256
        ),
        policy_sha256=authority.EXPECTED_QUALITY_POLICY_SHA256,
        request_path=deepx.request_path,
        candidate_path=deepx.candidate_path,
        records=deepx.records,
        deepx_result=deepx.result,
        trt_result=trt.result,
        workflow_status="ok",
    )
    arm = SimpleNamespace(
        base=base,
        evaluation_run_id="synthetic-evaluation-run",
        deepx_endpoint=deepx,
        trt_endpoint=trt,
    )
    assert baseline_pin.validate_frozen_b500_authority(arm)["ok"] is True

    frozen_base_fields = (
        "mode", "profile_name", "cache_dir", "cache_key",
        "cache_contract_sha256", "task", "target", "source_onnx_sha256",
        "build_onnx_sha256", "dxcom_config_sha256", "build_options_sha256",
        "dxnn_sha256", "calibration_identity_sha256",
        "calibration_contract_sha256", "calibration_items_identity_sha256",
        "compiler_identity_sha256", "compiler_contract_sha256",
        "validation_manifest_sha256", "validation_image_ids_sha256",
        "validation_ground_truth_sha256", "prepared_input_evidence_sha256",
        "preprocessing_contract_sha256", "policy_sha256",
    )
    for field in frozen_base_fields:
        changed_base = SimpleNamespace(**vars(base))
        setattr(changed_base, field, "jointly-resealed-alternate")
        changed_arm = SimpleNamespace(**{**vars(arm), "base": changed_base})
        with pytest.raises(PinError):
            baseline_pin.validate_frozen_b500_authority(changed_arm)

    changed = json.loads(json.dumps(deepx.records))
    changed[0]["candidate"]["top1_hit"] = False
    changed_deepx = SimpleNamespace(
        **{**vars(deepx), "records": tuple(changed)},
    )
    with pytest.raises(PinError, match="deepx_record_top1_hits"):
        baseline_pin.validate_frozen_b500_authority(
            _arm_with_endpoints(arm, deepx_endpoint=changed_deepx),
        )

    original_deepx_request = deepx.request_path.read_bytes()
    for mutation in ("image_id", "label_id"):
        changed = json.loads(json.dumps(deepx.records))
        changed[0][mutation] = (
            "different-image" if mutation == "image_id" else 999
        )
        resealed_deepx = _reseal_endpoint_records(deepx, tuple(changed))
        with pytest.raises(PinError, match="validation|record"):
            baseline_pin.validate_frozen_b500_authority(
                _arm_with_endpoints(arm, deepx_endpoint=resealed_deepx),
            )
        deepx.request_path.write_bytes(original_deepx_request)

    changed_result = json.loads(json.dumps(deepx.result))
    changed_result["source_request_sha256"] = "f" * 64
    changed_deepx = SimpleNamespace(
        **{**vars(deepx), "result": changed_result},
    )
    with pytest.raises(PinError, match="result/request binding"):
        baseline_pin.validate_frozen_b500_authority(
            _arm_with_endpoints(arm, deepx_endpoint=changed_deepx),
        )

    deepx.request_path.write_text(
        json.dumps({**deepx.request, "status": "resealed"}, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(PinError, match="on-disk bytes"):
        baseline_pin.validate_frozen_b500_authority(arm)
    deepx.request_path.write_bytes(original_deepx_request)

    changed_trt_records = json.loads(json.dumps(trt.records))
    changed_trt_records[0]["candidate"]["top1_hit"] = False
    changed_trt = SimpleNamespace(
        **{**vars(trt), "records": tuple(changed_trt_records)},
    )
    with pytest.raises(PinError, match="trt_record_top1_hits"):
        baseline_pin.validate_frozen_b500_authority(
            _arm_with_endpoints(arm, trt_endpoint=changed_trt),
        )

    original_trt_request = trt.request_path.read_bytes()
    changed_trt_records = json.loads(json.dumps(trt.records))
    changed_trt_records[0]["image_id"] = "different-trt-image"
    resealed_trt = _reseal_endpoint_records(trt, tuple(changed_trt_records))
    with pytest.raises(PinError, match="validation|record"):
        baseline_pin.validate_frozen_b500_authority(
            _arm_with_endpoints(arm, trt_endpoint=resealed_trt),
        )
    trt.request_path.write_bytes(original_trt_request)
