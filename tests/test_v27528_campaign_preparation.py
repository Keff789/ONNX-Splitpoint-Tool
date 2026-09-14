from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool import campaign
from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.workflow.artifacts import (
    sha256_file,
    sha256_json,
)


def _touch(path: Path, payload: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _prepared_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    campaign_dir = tmp_path / "FinalCampaign"
    inputs = campaign_dir / "campaign_inputs"
    for name in (
        "imagenet_train_calibration_manifest.json",
        "imagenet_val_manifest.json",
        "coco2017_train_calibration_manifest.json",
        "coco2017_val_manifest.json",
        "instances_val2017.json",
    ):
        _touch(inputs / name, b"{}\n")

    models = tmp_path / "Models"
    _touch(models / "resnet50.onnx", b"resnet50")
    _touch(models / "yolo26s.onnx", b"yolo26s")
    _touch(models / "yolov7_paper.onnx", b"yolov7-paper")
    collector = _touch(tmp_path / "bin" / "urecs-data-collector", b"collector")
    power = _touch(tmp_path / "bin" / "power_calculations", b"power")
    _touch(campaign_dir / "final_profile_working.yaml", b"legacy: true\n")
    sealed = _touch(campaign_dir / "sealed_final_profile.yaml", b"sealed-old\n")

    result = campaign.prepare_evaluated_matrix_campaign(
        campaign_dir=campaign_dir,
        models_root=models,
        energy_method_attested_by="Kevin Mika",
        accept_validated_energy_method_reuse=True,
        collector_binary=str(collector),
        power_calculations_binary=str(power),
    )
    assert sealed.read_bytes() == b"sealed-old\n"
    return campaign_dir, models, result


def test_prepare_evaluated_matrix_materializes_exact_locked_contracts(
    tmp_path: Path,
) -> None:
    campaign_dir, _models, result = _prepared_fixture(tmp_path)
    assert result["ok"] is True
    assert result["sealed_profile_written"] is False
    assert result["model_ids"] == ["resnet50", "yolo26s", "yolov7_paper"]
    assert (campaign_dir / "final_profile_working.pre_v27528.yaml").read_text(
        encoding="utf-8"
    ) == "legacy: true\n"

    profile = yaml.safe_load(
        (campaign_dir / "final_profile_working.yaml").read_text(encoding="utf-8")
    )
    assert validate_evaluation_profile_payload(profile) == profile
    rows = profile["model_suite"]["primary"]
    assert [row["id"] for row in rows] == [
        "resnet50",
        "yolo26s",
        "yolov7_paper",
    ]
    assert all(Path(row["path"]).is_file() for row in rows)
    assert all(str(row["model_sha256"]).startswith("sha256:") for row in rows)
    assert profile["measurement_campaign"]["system_power"] == {
        **profile["measurement_campaign"]["system_power"],
        "scope": "FS",
        "channel": 0,
        "sample_rate_hz": 2000,
        "measurement_point": "complete_system_input",
        "required_setup_ids": [
            "orin_nx_hailo8_01",
            "orin_nx_hailo10_01",
            "orin_nx_deepx_m1_01",
        ],
        "energy_evidence_mode": "inherited_validated_method",
    }

    pipeline_path = Path(str(result["pipeline_contract_manifest"]))
    pipeline_manifest = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline_check = campaign.verify_pipeline_contract_manifest(
        pipeline_manifest,
        require_locked=True,
        manifest_path=pipeline_path,
    )
    assert pipeline_check["ok"] is True
    assert pipeline_check["placeholder_free_ok"] is True
    assert pipeline_check["evaluated_matrix_semantics"] == {
        "ok": True,
        "preprocessing_ok": True,
        "preprocessing_tasks": ["classification", "detection"],
        "decoder_matrix_ok": True,
        "nms_ok": True,
    }

    generated_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((pipeline_path.parent / "configs").glob("*.yaml"))
    )
    assert "TODO" not in generated_text
    assert "REPLACE_WITH" not in generated_text
    assert "short_side" not in generated_text
    assert "stride_alignment" not in generated_text
    assert "yolov7_paper" in generated_text


def test_inherited_energy_method_needs_no_new_numeric_calibration(
    tmp_path: Path,
) -> None:
    _campaign_dir, _models, result = _prepared_fixture(tmp_path)
    manifest_path = Path(str(result["energy_method_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["evidence_mode"] == "inherited_validated_method"
    assert manifest["calibration"] == {}
    assert manifest["verification"] == {}
    assert manifest["uncertainty"] == {}
    assert manifest["reuse_attestation"]["new_calibration_required"] is False
    check = campaign.verify_energy_calibration_manifest(
        manifest,
        require_final=True,
        expected_channel_bindings=campaign.FINAL_MATRIX_SETUP_BINDINGS,
    )
    assert check["ok"] is True
    assert check["inherited_validated_method_final_ok"] is True
    assert check["channel_binding_ids"] == [
        "orin_nx_deepx_m1_01",
        "orin_nx_hailo10_01",
        "orin_nx_hailo8_01",
    ]


def test_internally_rehashed_wrong_energy_ip_is_rejected(
    tmp_path: Path,
) -> None:
    _campaign_dir, _models, result = _prepared_fixture(tmp_path)
    manifest = json.loads(
        Path(str(result["energy_method_manifest"])).read_text(encoding="utf-8")
    )
    mutated = copy.deepcopy(manifest)
    mutated["channel_bindings"][0]["urecs_address"] = "192.168.0.1"
    mutated["channel_binding_set_sha256"] = sha256_json(
        mutated["channel_bindings"]
    )
    mutated["manifest_payload_sha256"] = sha256_json(
        {
            key: value
            for key, value in mutated.items()
            if key != "manifest_payload_sha256"
        }
    )
    check = campaign.verify_energy_calibration_manifest(
        mutated,
        require_final=True,
        expected_channel_bindings=campaign.FINAL_MATRIX_SETUP_BINDINGS,
    )
    assert check["ok"] is False
    assert check["expected_channel_bindings_ok"] is False


def test_evaluated_matrix_requires_both_hashed_measurement_binaries(
    tmp_path: Path,
) -> None:
    _campaign_dir, _models, result = _prepared_fixture(tmp_path)
    manifest = json.loads(
        Path(str(result["energy_method_manifest"])).read_text(encoding="utf-8")
    )
    mutated = copy.deepcopy(manifest)
    mutated["artifacts"] = [
        row
        for row in mutated["artifacts"]
        if row["id"] != "power_calculations_binary"
    ]
    mutated["artifact_set_sha256"] = sha256_json(mutated["artifacts"])
    mutated["manifest_payload_sha256"] = sha256_json(
        {
            key: value
            for key, value in mutated.items()
            if key != "manifest_payload_sha256"
        }
    )
    check = campaign.verify_energy_calibration_manifest(
        mutated,
        require_final=True,
        expected_channel_bindings=campaign.FINAL_MATRIX_SETUP_BINDINGS,
    )
    assert check["ok"] is False
    assert check["expected_implementation_artifacts_ok"] is False


def test_prepare_requires_explicit_method_reuse_attestation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit acceptance"):
        campaign.prepare_evaluated_matrix_campaign(
            campaign_dir=tmp_path,
            models_root=tmp_path,
            energy_method_attested_by="Kevin Mika",
            accept_validated_energy_method_reuse=False,
        )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("final_ready", 0),
        ("development_ready", 0),
        ("incomplete", 0),
        ("ready", 0),
        ("blocked", 2),
        ("unknown", 2),
    ],
)
def test_preflight_cli_exit_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected: int,
) -> None:
    profile = _touch(tmp_path / "profile.yaml", b"{}\n")
    monkeypatch.setattr(
        campaign,
        "build_campaign_readiness",
        lambda *_args, **_kwargs: {"status": status, "checks": []},
    )
    assert campaign.main(["preflight", "--profile", str(profile)]) == expected


def test_locked_placeholder_contract_is_rejected(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    contracts = []
    for kind, task in (
        ("preprocessing", "classification"),
        ("preprocessing", "detection"),
        ("decoder", "detection"),
        ("nms", "detection"),
    ):
        path = configs / f"{kind}_{task}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                {
                    "locked": True,
                    "value": "TODO_REPLACE_WITH_REAL_CONTRACT",
                }
            ),
            encoding="utf-8",
        )
        contracts.append(
            {
                "id": path.stem,
                "kind": kind,
                "task": task,
                "path": str(path),
            }
        )
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump({"contracts": contracts}), encoding="utf-8")
    manifest_path = campaign.create_pipeline_contract_manifest(
        spec=spec, output=tmp_path / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check = campaign.verify_pipeline_contract_manifest(
        manifest, require_locked=True
    )
    assert check["ok"] is False
    assert check["placeholder_free_ok"] is False


def _rehash_pipeline_manifest(
    manifest: dict[str, object], *, manifest_path: Path
) -> None:
    rows = manifest["contracts"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = manifest_path.parent / path
        row["sha256"] = sha256_file(path)
        structured = yaml.safe_load(path.read_text(encoding="utf-8"))
        row["normalized_sha256"] = sha256_json(structured)
        row["size_bytes"] = path.stat().st_size
    manifest["contract_set_sha256"] = sha256_json(rows)
    manifest["manifest_payload_sha256"] = sha256_json(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_payload_sha256"
        }
    )


def test_profile_scope_keeps_matrix_semantics_mandatory_when_manifest_scope_is_removed(
    tmp_path: Path,
) -> None:
    _campaign_dir, _models, result = _prepared_fixture(tmp_path)
    manifest_path = Path(str(result["pipeline_contract_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nms_row = next(row for row in manifest["contracts"] if row["kind"] == "nms")
    nms_path = Path(nms_row["path"])
    if not nms_path.is_absolute():
        nms_path = manifest_path.parent / nms_path
    nms = yaml.safe_load(nms_path.read_text(encoding="utf-8"))
    nms["confidence_threshold"] = 0.90
    nms_path.write_text(yaml.safe_dump(nms, sort_keys=False), encoding="utf-8")
    manifest["claim_scope"] = ""
    _rehash_pipeline_manifest(manifest, manifest_path=manifest_path)

    check = campaign.verify_pipeline_contract_manifest(
        manifest,
        require_locked=True,
        expected_claim_scope="evaluated_matrix",
    )
    assert check["ok"] is False
    assert check["claim_scope_binding_ok"] is False
    assert check["evaluated_matrix_semantics"]["nms_ok"] is False


def test_rehashed_decoder_semantic_contradiction_is_rejected(
    tmp_path: Path,
) -> None:
    _campaign_dir, _models, result = _prepared_fixture(tmp_path)
    manifest_path = Path(str(result["pipeline_contract_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decoder_row = next(
        row for row in manifest["contracts"] if row["kind"] == "decoder"
    )
    decoder_path = Path(decoder_row["path"])
    if not decoder_path.is_absolute():
        decoder_path = manifest_path.parent / decoder_path
    decoder = yaml.safe_load(decoder_path.read_text(encoding="utf-8"))
    decoder["class_count"] = 79
    decoder["model_contracts"]["yolo26s"]["integrated_output_format"] = (
        "raw_head"
    )
    decoder_path.write_text(
        yaml.safe_dump(decoder, sort_keys=False), encoding="utf-8"
    )
    _rehash_pipeline_manifest(manifest, manifest_path=manifest_path)

    check = campaign.verify_pipeline_contract_manifest(
        manifest,
        require_locked=True,
        expected_claim_scope="evaluated_matrix",
    )
    assert check["ok"] is False
    assert check["evaluated_matrix_semantics"]["decoder_matrix_ok"] is False


def test_prepare_is_idempotent_but_refuses_to_overwrite_manual_review(
    tmp_path: Path,
) -> None:
    campaign_dir, models, first = _prepared_fixture(tmp_path)
    collector = tmp_path / "bin" / "urecs-data-collector"
    power = tmp_path / "bin" / "power_calculations"
    second = campaign.prepare_evaluated_matrix_campaign(
        campaign_dir=campaign_dir,
        models_root=models,
        energy_method_attested_by="Kevin Mika",
        accept_validated_energy_method_reuse=True,
        collector_binary=str(collector),
        power_calculations_binary=str(power),
    )
    assert second["working_profile"] == first["working_profile"]
    assert not (campaign_dir / "campaign_inputs" / ".generated_v27528.rollback").exists()

    working = campaign_dir / "final_profile_working.yaml"
    reviewed = yaml.safe_load(working.read_text(encoding="utf-8"))
    reviewed["purpose"] = "MANUAL_REVIEW_MUST_SURVIVE"
    working.write_text(yaml.safe_dump(reviewed, sort_keys=False), encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite reviewed work"):
        campaign.prepare_evaluated_matrix_campaign(
            campaign_dir=campaign_dir,
            models_root=models,
            energy_method_attested_by="Kevin Mika",
            accept_validated_energy_method_reuse=True,
            collector_binary=str(collector),
            power_calculations_binary=str(power),
        )
    assert "MANUAL_REVIEW_MUST_SURVIVE" in working.read_text(encoding="utf-8")


def test_prepared_pipeline_contract_remains_valid_after_campaign_move(
    tmp_path: Path,
) -> None:
    campaign_dir, _models, _result = _prepared_fixture(tmp_path)
    moved = tmp_path / "MovedFinalCampaign"
    campaign_dir.rename(moved)
    manifest_path = (
        moved
        / "campaign_inputs"
        / "generated_v27528"
        / "pipeline_contract_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["path_resolution"] == "manifest_relative"
    assert all(not Path(row["path"]).is_absolute() for row in manifest["contracts"])
    check = campaign.verify_pipeline_contract_manifest(
        manifest,
        require_locked=True,
        expected_claim_scope="evaluated_matrix",
        manifest_path=manifest_path,
    )
    assert check["ok"] is True


@pytest.mark.parametrize("reference_style", ["relative", "dot", "absolute"])
def test_sealed_generated_profile_prevents_evidence_regeneration(
    tmp_path: Path, reference_style: str,
) -> None:
    campaign_dir, models, result = _prepared_fixture(tmp_path)
    working = Path(str(result["working_profile"]))
    sealed = campaign_dir / "sealed_final_profile.yaml"
    sealed_payload = yaml.safe_load(working.read_text(encoding="utf-8"))
    pipeline_target = (
        campaign_dir
        / "campaign_inputs"
        / "generated_v27528"
        / "pipeline_contract_manifest.json"
    )
    if reference_style == "dot":
        pipeline_reference = (
            "./campaign_inputs/generated_v27528/pipeline_contract_manifest.json"
        )
    elif reference_style == "absolute":
        pipeline_reference = str(pipeline_target.resolve())
    else:
        pipeline_reference = (
            "campaign_inputs/generated_v27528/pipeline_contract_manifest.json"
        )
    sealed_payload["campaign"]["pipeline_contract_manifest"] = pipeline_reference
    # Isolate the pipeline alias under test; either generated reference alone
    # must seal the shared evidence directory.
    sealed_payload["campaign"]["energy_calibration_manifest"] = ""
    sealed.write_text(
        yaml.safe_dump(sealed_payload, sort_keys=False), encoding="utf-8"
    )
    manifest = Path(str(result["energy_method_manifest"]))
    before = sha256_file(manifest)
    with pytest.raises(RuntimeError, match="sealed campaign evidence"):
        campaign.prepare_evaluated_matrix_campaign(
            campaign_dir=campaign_dir,
            models_root=models,
            energy_method_attested_by="Kevin Mika",
            accept_validated_energy_method_reuse=True,
            collector_binary=str(tmp_path / "bin" / "urecs-data-collector"),
            power_calculations_binary=str(tmp_path / "bin" / "power_calculations"),
        )
    assert sha256_file(manifest) == before


def test_structured_contract_must_be_parseable_and_self_locked(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(": definitely [invalid yaml", encoding="utf-8")
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "contracts": [
                    {
                        "id": "invalid",
                        "kind": "preprocessing",
                        "task": "classification",
                        "path": str(invalid),
                        "locked": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty or invalid"):
        campaign.create_pipeline_contract_manifest(
            spec=spec, output=tmp_path / "manifest.json"
        )


def test_matrix_manifest_schema_version_is_exact(tmp_path: Path) -> None:
    _campaign_dir, _models, result = _prepared_fixture(tmp_path)
    manifest_path = Path(str(result["pipeline_contract_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    manifest["manifest_payload_sha256"] = sha256_json(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_payload_sha256"
        }
    )
    check = campaign.verify_pipeline_contract_manifest(
        manifest,
        require_locked=True,
        expected_claim_scope="evaluated_matrix",
        manifest_path=manifest_path,
    )
    assert check["ok"] is False
    assert check["matrix_schema_version_ok"] is False


def test_matrix_energy_rejects_contradictory_channel_and_reuse_policy(
    tmp_path: Path,
) -> None:
    _campaign_dir, _models, result = _prepared_fixture(tmp_path)
    manifest = json.loads(
        Path(str(result["energy_method_manifest"])).read_text(encoding="utf-8")
    )
    manifest["channel_id"] = "channel_99"
    manifest["method"]["implementation_policy"] = "changes_allowed"
    manifest["manifest_payload_sha256"] = sha256_json(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_payload_sha256"
        }
    )
    check = campaign.verify_energy_calibration_manifest(
        manifest,
        require_final=True,
        expected_channel_bindings=campaign.FINAL_MATRIX_SETUP_BINDINGS,
    )
    assert check["ok"] is False
    assert check["expected_method_identity_ok"] is False
