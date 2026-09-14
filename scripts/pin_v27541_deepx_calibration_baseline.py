#!/usr/bin/env python3
"""Pin the proven v2.75.40 B500 calibration manifest before provisioning B1000.

The v2.75.40 profile used the mutable final-dataset manifest pointer.  A later
1000-item provisioning action may legitimately update that pointer.  This
utility first proves that the current 500-item manifest is exactly the one
bound by the completed B500 DeepX cache contract, verifies every dataset file,
and then creates an immutable count/seed-specific manifest copy.  It never
modifies the source manifest, dataset images, cache, or EvaluationRun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.campaign import verify_dataset_manifest
from onnx_splitpoint_tool.deepx import calibration_size_canary as canary_authority
from onnx_splitpoint_tool.deepx.config import CLASSIFICATION_PREPROCESSING_IMAGENET


EXPECTED_COUNT = 500
EXPECTED_SEED = 20260710
DEFAULT_SOURCE = (
    Path.home()
    / ".onnx_splitpoint_tool"
    / "final_datasets"
    / "manifests"
    / "imagenet_train_calibration_manifest.json"
)
DEFAULT_OUTPUT = (
    Path.home()
    / ".onnx_splitpoint_tool"
    / "final_datasets"
    / "v27541_imagenet_n500_s20260710"
    / "manifests"
    / "imagenet_train_calibration_manifest.json"
)


class PinError(RuntimeError):
    """The mutable source could not be proven as the completed B500 input."""


def _bare_sha(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return raw.split(":", 1)[1] if raw.startswith("sha256:") else raw


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, default=str,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _require_sha256(value: Any, *, label: str) -> str:
    digest = _bare_sha(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise PinError(f"{label} is not a SHA-256 identity")
    return digest


def _verified_subcontract_identity(
    value: Mapping[str, Any], *, label: str,
) -> str:
    declared = _require_sha256(value.get("identity_sha256"), label=label)
    unhashed = dict(value)
    unhashed.pop("identity_sha256", None)
    if _canonical_sha256(unhashed) != declared:
        raise PinError(f"{label} self-hash mismatch")
    return declared


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PinError(f"{label} is missing or not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PinError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PinError(f"{label} must contain one JSON object: {path}")
    return dict(payload)


def _absolute_without_following_leaf(path: str | Path) -> Path:
    """Return an absolute path while preserving a leaf symlink for checks."""
    return Path(os.path.abspath(Path(path).expanduser()))


def _seed_values(payload: Mapping[str, Any]) -> list[int]:
    values: list[int] = []
    for block_name in ("selection", "provisioning_selection"):
        block = payload.get(block_name)
        if not isinstance(block, Mapping) or "seed" not in block:
            continue
        try:
            values.append(int(block.get("seed")))
        except (TypeError, ValueError):
            values.append(-1)
    return values


def validate_frozen_b500_authority(evidence: Any) -> dict[str, Any]:
    """Bind a strictly loaded arm to the delivered v2.75.40 B500 result.

    The caller must supply ``CalibrationArmEvidence`` produced by the frozen
    canary's ``load_calibration_arm``.  Unlike the legacy ``ArmEvidence``, that
    object contains both real Full-quality endpoints.  Reuse the canary's
    scientific validator, then independently freeze the historical hit counts
    and the on-disk result-to-request bindings here.
    """

    base = getattr(evidence, "base", None)
    deepx_endpoint = getattr(evidence, "deepx_endpoint", None)
    trt_endpoint = getattr(evidence, "trt_endpoint", None)
    if base is None or deepx_endpoint is None or trt_endpoint is None:
        raise PinError(
            "B500 authority requires frozen load_calibration_arm evidence with "
            "both DeepX and TensorRT endpoints"
        )

    try:
        deepx_policy = canary_authority._validate_endpoint_scientific_binding(
            evidence, deepx_endpoint,
        )
        trt_policy = canary_authority._validate_endpoint_scientific_binding(
            evidence, trt_endpoint,
        )
    except Exception as exc:
        raise PinError(f"B500 Full-quality endpoint binding is invalid: {exc}") from exc
    if deepx_policy != trt_policy:
        raise PinError("B500 DeepX/TensorRT quality policy contracts differ")

    def endpoint_projection(endpoint: Any, *, label: str) -> dict[str, Any]:
        request_path = Path(str(getattr(endpoint, "request_path", "") or ""))
        request = getattr(endpoint, "request", None)
        producer = getattr(endpoint, "producer", None)
        result = getattr(endpoint, "result", None)
        records = tuple(getattr(endpoint, "records", ()) or ())
        if (
            not isinstance(request, Mapping)
            or not isinstance(producer, Mapping)
            or not isinstance(result, Mapping)
        ):
            raise PinError(f"{label} endpoint envelope is incomplete")
        current_request = _read_object(request_path, label=f"{label} quality request")
        if current_request != dict(request):
            raise PinError(f"{label} loaded request differs from its on-disk bytes")
        request_producer = request.get("producer_identity")
        if not isinstance(request_producer, Mapping) or dict(
            request_producer
        ) != dict(producer):
            raise PinError(f"{label} Request/Producer identity differs")
        request_sha256 = _sha256_bytes(request_path.read_bytes())
        if _require_sha256(
            result.get("source_request_sha256"),
            label=f"{label} result source request",
        ) != request_sha256:
            raise PinError(f"{label} central result/request binding mismatch")
        try:
            image_ids, image_ids_sha256, ground_truth_sha256 = (
                canary_authority._classification_record_identity(
                    records, label=label,
                )
            )
        except Exception as exc:
            raise PinError(f"{label} validation record identity is invalid: {exc}") from exc
        dataset = producer.get("dataset")
        if not isinstance(dataset, Mapping):
            raise PinError(f"{label} producer dataset identity is missing")
        if list(request.get("expected_image_ids") or []) != image_ids:
            raise PinError(f"{label} request/record validation order differs")
        declared = {
            "request_image_ids_sha256": _bare_sha(
                request.get("expected_image_ids_sha256")
            ),
            "dataset_manifest_sha256": _bare_sha(dataset.get("manifest_sha256")),
            "dataset_image_ids_sha256": _bare_sha(dataset.get("image_ids_sha256")),
            "dataset_ground_truth_sha256": _bare_sha(
                dataset.get("ground_truth_sha256")
            ),
        }
        required = {
            "request_image_ids_sha256": _bare_sha(
                getattr(base, "validation_image_ids_sha256", "")
            ),
            "dataset_manifest_sha256": _bare_sha(
                getattr(base, "validation_manifest_sha256", "")
            ),
            "dataset_image_ids_sha256": image_ids_sha256,
            "dataset_ground_truth_sha256": ground_truth_sha256,
        }
        if declared != required:
            mismatches = sorted(
                key for key in required if declared.get(key) != required[key]
            )
            raise PinError(
                f"{label} Request/Producer/record validation authority differs: "
                + ",".join(mismatches)
            )
        if (
            image_ids_sha256
            != _bare_sha(getattr(base, "validation_image_ids_sha256", ""))
            or ground_truth_sha256
            != _bare_sha(getattr(base, "validation_ground_truth_sha256", ""))
        ):
            raise PinError(f"{label} records violate frozen validation ID/GT authority")
        try:
            image_count = int(dataset.get("image_count"))
        except (TypeError, ValueError):
            image_count = -1
        if image_count != EXPECTED_COUNT or len(records) != EXPECTED_COUNT:
            raise PinError(f"{label} validation record count differs from authority")

        top1_hits = 0
        top5_hits = 0
        for row in records:
            candidate = row.get("candidate") if isinstance(row, Mapping) else None
            if not isinstance(candidate, Mapping) or not isinstance(
                candidate.get("top1_hit"), bool
            ) or not isinstance(candidate.get("top5_hit"), bool):
                raise PinError(f"{label} classification hits are incomplete")
            top1_hits += int(candidate["top1_hit"])
            top5_hits += int(candidate["top5_hit"])
        return {
            "request_sha256": request_sha256,
            "image_ids": tuple(image_ids),
            "image_ids_sha256": image_ids_sha256,
            "ground_truth_sha256": ground_truth_sha256,
            "top1_hits": top1_hits,
            "top5_hits": top5_hits,
        }

    deepx_records = endpoint_projection(deepx_endpoint, label="B500 DeepX")
    trt_records = endpoint_projection(trt_endpoint, label="B500 TensorRT")
    if (
        deepx_records["image_ids"] != trt_records["image_ids"]
        or deepx_records["image_ids_sha256"] != trt_records["image_ids_sha256"]
        or deepx_records["ground_truth_sha256"]
        != trt_records["ground_truth_sha256"]
    ):
        raise PinError("B500 DeepX/TensorRT validation cohort or ground truth differs")

    base_request_path = Path(str(getattr(base, "request_path", "") or ""))
    base_candidate_path = Path(str(getattr(base, "candidate_path", "") or ""))
    if (
        base_request_path.absolute() != Path(deepx_endpoint.request_path).absolute()
        or base_candidate_path.absolute() != Path(deepx_endpoint.candidate_path).absolute()
        or tuple(getattr(base, "records", ()) or ()) != tuple(deepx_endpoint.records)
        or dict(getattr(base, "deepx_result", {}) or {}) != dict(deepx_endpoint.result)
        or dict(getattr(base, "trt_result", {}) or {}) != dict(trt_endpoint.result)
    ):
        raise PinError("B500 base evidence and strict Full-quality endpoints differ")

    request = deepx_endpoint.request
    producer = deepx_endpoint.producer

    def result_projection(value: Any, *, label: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise PinError(f"{label} central quality result is missing")
        primary = value.get("primary")
        guardrails = value.get("guardrails")
        top5 = guardrails.get("top5_accuracy") if isinstance(
            guardrails, Mapping
        ) else None
        if not isinstance(primary, Mapping) or not isinstance(top5, Mapping):
            raise PinError(f"{label} central quality components are missing")
        setup_values = [
            str(value.get(field) or "").strip()
            for field in ("source_setup_id", "setup_id")
            if str(value.get(field) or "").strip()
        ]
        if not setup_values or any(
            setup != canary_authority.EXPECTED_SETUP_ID for setup in setup_values
        ):
            setup = "|".join(setup_values)
        else:
            setup = setup_values[0]
        return {
            "setup_id": setup,
            "source_run_id": str(value.get("source_run_id") or ""),
            "algorithm": str(value.get("algorithm_version") or ""),
            "decision": str(value.get("decision") or ""),
            "n": value.get("n"),
            "source_request_sha256": _bare_sha(
                value.get("source_request_sha256")
            ),
            "primary_decision": str(primary.get("decision") or ""),
            "primary_sample_count": primary.get("sample_count"),
            "top5_decision": str(top5.get("decision") or ""),
            "top5_sample_count": top5.get("sample_count"),
            "top1_candidate_hits": primary.get("candidate_hits"),
            "top1_reference_hits": primary.get("reference_hits"),
            "top5_candidate_hits": top5.get("candidate_hits"),
            "top5_reference_hits": top5.get("reference_hits"),
        }

    deepx_quality = result_projection(
        deepx_endpoint.result, label="B500 DeepX",
    )
    trt_quality = result_projection(
        trt_endpoint.result, label="B500 TensorRT",
    )
    observed = {
        "mode": str(getattr(base, "mode", "") or ""),
        "profile_name": str(getattr(base, "profile_name", "") or ""),
        "cache_dir": str(getattr(base, "cache_dir", "") or ""),
        "cache_key": str(getattr(base, "cache_key", "") or ""),
        "cache_contract_sha256": _bare_sha(
            getattr(base, "cache_contract_sha256", "")
        ),
        "task": str(getattr(base, "task", "") or ""),
        "target": str(getattr(base, "target", "") or ""),
        "source_onnx_sha256": _bare_sha(
            getattr(base, "source_onnx_sha256", "")
        ),
        "build_onnx_sha256": _bare_sha(
            getattr(base, "build_onnx_sha256", "")
        ),
        "dxcom_config_sha256": _bare_sha(
            getattr(base, "dxcom_config_sha256", "")
        ),
        "build_options_sha256": _bare_sha(
            getattr(base, "build_options_sha256", "")
        ),
        "dxnn_sha256": _bare_sha(getattr(base, "dxnn_sha256", "")),
        "calibration_identity_sha256": _bare_sha(
            getattr(base, "calibration_identity_sha256", "")
        ),
        "calibration_contract_sha256": _bare_sha(
            getattr(base, "calibration_contract_sha256", "")
        ),
        "calibration_items_identity_sha256": _bare_sha(
            getattr(base, "calibration_items_identity_sha256", "")
        ),
        "compiler_identity_sha256": _bare_sha(
            getattr(base, "compiler_identity_sha256", "")
        ),
        "compiler_contract_sha256": _bare_sha(
            getattr(base, "compiler_contract_sha256", "")
        ),
        "validation_manifest_sha256": _bare_sha(
            getattr(base, "validation_manifest_sha256", "")
        ),
        "validation_image_ids_sha256": _bare_sha(
            getattr(base, "validation_image_ids_sha256", "")
        ),
        "validation_ground_truth_sha256": _bare_sha(
            getattr(base, "validation_ground_truth_sha256", "")
        ),
        "prepared_input_evidence_sha256": _bare_sha(
            getattr(base, "prepared_input_evidence_sha256", "")
        ),
        "preprocessing_contract_sha256": _bare_sha(
            getattr(base, "preprocessing_contract_sha256", "")
        ),
        "policy_sha256": _bare_sha(getattr(base, "policy_sha256", "")),
        "request_setup_id": str(request.get("setup_id") or ""),
        "producer_setup_id": str(producer.get("setup_id") or ""),
        "trt_request_setup_id": str(trt_endpoint.request.get("setup_id") or ""),
        "trt_producer_setup_id": str(trt_endpoint.producer.get("setup_id") or ""),
        "deepx_record_top1_hits": deepx_records["top1_hits"],
        "deepx_record_top5_hits": deepx_records["top5_hits"],
        "trt_record_top1_hits": trt_records["top1_hits"],
        "trt_record_top5_hits": trt_records["top5_hits"],
        "deepx_request_sha256": deepx_records["request_sha256"],
        "trt_request_sha256": trt_records["request_sha256"],
        "deepx_result_setup_id": deepx_quality["setup_id"],
        "deepx_result_source_run_id": deepx_quality["source_run_id"],
        "deepx_result_algorithm": deepx_quality["algorithm"],
        "deepx_result_decision": deepx_quality["decision"],
        "deepx_result_n": deepx_quality["n"],
        "deepx_result_source_request_sha256": deepx_quality[
            "source_request_sha256"
        ],
        "deepx_result_primary_decision": deepx_quality["primary_decision"],
        "deepx_result_primary_sample_count": deepx_quality[
            "primary_sample_count"
        ],
        "deepx_result_top5_decision": deepx_quality["top5_decision"],
        "deepx_result_top5_sample_count": deepx_quality["top5_sample_count"],
        "deepx_result_top1_candidate_hits": deepx_quality["top1_candidate_hits"],
        "deepx_result_top1_reference_hits": deepx_quality["top1_reference_hits"],
        "deepx_result_top5_candidate_hits": deepx_quality["top5_candidate_hits"],
        "deepx_result_top5_reference_hits": deepx_quality["top5_reference_hits"],
        "trt_result_setup_id": trt_quality["setup_id"],
        "trt_result_source_run_id": trt_quality["source_run_id"],
        "trt_result_algorithm": trt_quality["algorithm"],
        "trt_result_decision": trt_quality["decision"],
        "trt_result_n": trt_quality["n"],
        "trt_result_source_request_sha256": trt_quality[
            "source_request_sha256"
        ],
        "trt_result_primary_decision": trt_quality["primary_decision"],
        "trt_result_primary_sample_count": trt_quality[
            "primary_sample_count"
        ],
        "trt_result_top5_decision": trt_quality["top5_decision"],
        "trt_result_top5_sample_count": trt_quality["top5_sample_count"],
        "trt_result_top1_candidate_hits": trt_quality["top1_candidate_hits"],
        "trt_result_top1_reference_hits": trt_quality["top1_reference_hits"],
        "trt_result_top5_candidate_hits": trt_quality["top5_candidate_hits"],
        "trt_result_top5_reference_hits": trt_quality["top5_reference_hits"],
    }
    expected = {
        "mode": CLASSIFICATION_PREPROCESSING_IMAGENET,
        "profile_name": canary_authority.BASELINE_PROFILE_NAME,
        "cache_dir": canary_authority.B500_CACHE_NAMESPACE,
        "cache_key": canary_authority.EXPECTED_B500_CACHE_KEY,
        "cache_contract_sha256": (
            canary_authority.EXPECTED_B500_CACHE_CONTRACT_SHA256
        ),
        "task": "classification",
        "target": "deepx_m1",
        "source_onnx_sha256": canary_authority.EXPECTED_SOURCE_ONNX_SHA256,
        "build_onnx_sha256": canary_authority.EXPECTED_BUILD_ONNX_SHA256,
        "dxcom_config_sha256": canary_authority.EXPECTED_B500_DXCOM_CONFIG_SHA256,
        "build_options_sha256": (
            canary_authority.EXPECTED_B500_BUILD_OPTIONS_SHA256
        ),
        "dxnn_sha256": canary_authority.EXPECTED_B500_DXNN_SHA256,
        "calibration_identity_sha256": (
            canary_authority.EXPECTED_B500_CALIBRATION_IDENTITY_SHA256
        ),
        "calibration_contract_sha256": (
            canary_authority.EXPECTED_B500_CALIBRATION_CONTRACT_SHA256
        ),
        "calibration_items_identity_sha256": (
            canary_authority.EXPECTED_B500_CALIBRATION_ITEMS_SHA256
        ),
        "compiler_identity_sha256": (
            canary_authority.EXPECTED_COMPILER_IDENTITY_SHA256
        ),
        "compiler_contract_sha256": (
            canary_authority.EXPECTED_COMPILER_CONTRACT_SHA256
        ),
        "validation_manifest_sha256": (
            canary_authority.EXPECTED_VALIDATION_MANIFEST_SHA256
        ),
        "validation_image_ids_sha256": (
            canary_authority.EXPECTED_VALIDATION_IMAGE_IDS_SHA256
        ),
        "validation_ground_truth_sha256": (
            canary_authority.EXPECTED_VALIDATION_GROUND_TRUTH_SHA256
        ),
        "prepared_input_evidence_sha256": (
            canary_authority.EXPECTED_PREPARED_INPUT_EVIDENCE_SHA256
        ),
        "preprocessing_contract_sha256": (
            canary_authority.EXPECTED_PREPROCESSING_CONTRACT_SHA256
        ),
        "policy_sha256": canary_authority.EXPECTED_QUALITY_POLICY_SHA256,
        "request_setup_id": canary_authority.EXPECTED_SETUP_ID,
        "producer_setup_id": canary_authority.EXPECTED_SETUP_ID,
        "trt_request_setup_id": canary_authority.EXPECTED_SETUP_ID,
        "trt_producer_setup_id": canary_authority.EXPECTED_SETUP_ID,
        "deepx_record_top1_hits": (
            canary_authority.EXPECTED_B500_DEEPX_TOP1_HITS
        ),
        "deepx_record_top5_hits": (
            canary_authority.EXPECTED_B500_DEEPX_TOP5_HITS
        ),
        "trt_record_top1_hits": (
            canary_authority.EXPECTED_REFERENCE_TOP1_HITS
        ),
        "trt_record_top5_hits": (
            canary_authority.EXPECTED_REFERENCE_TOP5_HITS
        ),
        "deepx_result_setup_id": canary_authority.EXPECTED_SETUP_ID,
        "deepx_result_source_run_id": "deepx_m1_full",
        "deepx_result_algorithm": canary_authority.EXPECTED_QUALITY_ALGORITHM,
        "deepx_result_decision": "inconclusive",
        "deepx_result_n": EXPECTED_COUNT,
        "deepx_result_primary_decision": "inconclusive",
        "deepx_result_primary_sample_count": EXPECTED_COUNT,
        "deepx_result_top5_decision": "pass",
        "deepx_result_top5_sample_count": EXPECTED_COUNT,
        "deepx_result_top1_candidate_hits": (
            canary_authority.EXPECTED_B500_DEEPX_TOP1_HITS
        ),
        "deepx_result_top1_reference_hits": (
            canary_authority.EXPECTED_REFERENCE_TOP1_HITS
        ),
        "deepx_result_top5_candidate_hits": (
            canary_authority.EXPECTED_B500_DEEPX_TOP5_HITS
        ),
        "deepx_result_top5_reference_hits": (
            canary_authority.EXPECTED_REFERENCE_TOP5_HITS
        ),
        "trt_result_setup_id": canary_authority.EXPECTED_SETUP_ID,
        "trt_result_source_run_id": "native_full_tensorrt",
        "trt_result_algorithm": canary_authority.EXPECTED_QUALITY_ALGORITHM,
        "trt_result_decision": "pass",
        "trt_result_n": EXPECTED_COUNT,
        "trt_result_primary_decision": "pass",
        "trt_result_primary_sample_count": EXPECTED_COUNT,
        "trt_result_top5_decision": "pass",
        "trt_result_top5_sample_count": EXPECTED_COUNT,
        "trt_result_top1_candidate_hits": (
            canary_authority.EXPECTED_REFERENCE_TOP1_HITS
        ),
        "trt_result_top1_reference_hits": (
            canary_authority.EXPECTED_REFERENCE_TOP1_HITS
        ),
        "trt_result_top5_candidate_hits": (
            canary_authority.EXPECTED_REFERENCE_TOP5_HITS
        ),
        "trt_result_top5_reference_hits": (
            canary_authority.EXPECTED_REFERENCE_TOP5_HITS
        ),
    }
    mismatches = sorted(
        key for key, expected_value in expected.items()
        if observed.get(key) != expected_value
    )
    if mismatches:
        raise PinError(
            "B500 EvaluationRun differs from the frozen v2.75.40 Arm-B "
            "authority: " + ",".join(mismatches)
        )
    return {
        "ok": True,
        "authority": "delivered_v2.75.40_arm_b",
        "observed": observed,
    }


def _cache_contract(
    run_dir: Path,
    calibration_manifest_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    status_path = (
        run_dir
        / "models"
        / "resnet50"
        / "benchmark_set"
        / "deepx"
        / "deepx_artifact_status.json"
    )
    status = _read_object(status_path, label="B500 DeepX artifact status")
    if str(status.get("status") or "") != "ok":
        raise PinError("B500 DeepX artifact status is not ok")
    contract = status.get("cache_contract")
    if not isinstance(contract, Mapping):
        raise PinError("B500 DeepX cache contract is missing")
    if (
        str(contract.get("schema") or "")
        != "onnx-splitpoint/deepx-full-cache-contract"
        or int(contract.get("schema_version") or 0) != 2
        or str(contract.get("task") or "") != "classification"
        or str(contract.get("target") or "") != "deepx_m1"
        or str(contract.get("classification_preprocessing") or "")
        != "imagenet_mean_std"
    ):
        raise PinError("B500 DeepX exact-v2 cache contract is not the expected arm")
    declared_contract_sha = _require_sha256(
        contract.get("contract_sha256"), label="B500 DeepX cache contract",
    )
    unhashed_contract = dict(contract)
    unhashed_contract.pop("contract_sha256", None)
    if _canonical_sha256(unhashed_contract) != declared_contract_sha:
        raise PinError("B500 DeepX cache contract self-hash mismatch")
    calibration = contract.get("calibration_manifest_contract")
    build = contract.get("build_options")
    if not isinstance(calibration, Mapping) or not isinstance(build, Mapping):
        raise PinError("B500 calibration/build subcontracts are missing")
    calibration_identity = _verified_subcontract_identity(
        calibration, label="B500 calibration contract",
    )
    if (
        str(calibration.get("status") or "") != "resolved"
        or int(calibration.get("item_count") or -1) != EXPECTED_COUNT
        or int(calibration.get("effective_count") or -1) != EXPECTED_COUNT
        or int(build.get("calibration_count") or -1) != EXPECTED_COUNT
        or str(build.get("calibration_method") or "") != "ema"
        or int(
            build.get("opt_level")
            if build.get("opt_level") is not None else -1
        ) != 0
    ):
        raise PinError("B500 calibration count, EMA, or opt-level contract differs")

    # Reuse the frozen final-canary loader as the high-assurance authority.
    # It validates the explicit immutable B500 manifest plus both real DeepX
    # and TensorRT request/candidate/result endpoints; in particular it binds
    # each central result to the SHA-256 of its own source request.
    try:
        arm = canary_authority.load_calibration_arm(
            run_dir,
            label="B500",
            expected_calibration_count=EXPECTED_COUNT,
            calibration_manifest_path=calibration_manifest_path,
        )
    except Exception as exc:
        raise PinError(f"B500 EvaluationRun evidence is invalid: {exc}") from exc
    evidence = arm.base
    if (
        evidence.cache_contract_sha256 != declared_contract_sha
        or evidence.calibration_identity_sha256 != calibration_identity
        or _bare_sha(evidence.calibration_items_identity_sha256)
        != _bare_sha(calibration.get("items_identity_sha256"))
    ):
        raise PinError("B500 validated evidence and cache contract differ")
    frozen_authority = validate_frozen_b500_authority(arm)
    return status_path, dict(calibration), frozen_authority


def pin_manifest(
    *, baseline_run: str | Path, source: str | Path = DEFAULT_SOURCE,
    output: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    run_dir = Path(baseline_run).expanduser().resolve()
    source_path = _absolute_without_following_leaf(source)
    output_path = _absolute_without_following_leaf(output)
    if not run_dir.is_dir():
        raise PinError(f"B500 EvaluationRun is missing: {run_dir}")
    if output_path == source_path:
        raise PinError("pin destination must not be the mutable source pointer")
    status_path, calibration, frozen_authority = _cache_contract(
        run_dir, source_path,
    )
    payload = _read_object(source_path, label="current B500 calibration manifest")
    try:
        item_count = int(payload.get("item_count") or -1)
    except (TypeError, ValueError):
        item_count = -1
    seeds = _seed_values(payload)
    if (
        str(payload.get("schema") or "") != "onnx-splitpoint/dataset-manifest"
        or str(payload.get("task") or "") != "classification"
        or str(payload.get("role") or "") != "calibration"
        or item_count != EXPECTED_COUNT
        or len(list(payload.get("items") or [])) != EXPECTED_COUNT
        or not seeds
        or any(seed != EXPECTED_SEED for seed in seeds)
    ):
        raise PinError("source is not the expected 500-item, seed-20260710 manifest")
    verification = verify_dataset_manifest(
        payload, verify_files=True, verification_mode="full"
    )
    if verification.get("ok") is not True:
        raise PinError(
            "source manifest or its 500 dataset files failed full verification: "
            + json.dumps(verification, sort_keys=True)
        )
    source_bytes = source_path.read_bytes()
    source_sha = _sha256_bytes(source_bytes)
    required = {
        "manifest_file_sha256": source_sha,
        "manifest_payload_sha256": _bare_sha(payload.get("manifest_payload_sha256")),
        "items_identity_sha256": _bare_sha(payload.get("items_identity_sha256")),
    }
    observed = {
        "manifest_file_sha256": _bare_sha(calibration.get("manifest_file_sha256")),
        "manifest_payload_sha256": _bare_sha(calibration.get("manifest_payload_sha256")),
        "items_identity_sha256": _bare_sha(calibration.get("items_identity_sha256")),
    }
    if not all(required[key] and required[key] == observed[key] for key in required):
        raise PinError(
            "current manifest does not match the completed B500 cache contract: "
            + json.dumps({"manifest": required, "run_contract": observed}, sort_keys=True)
        )

    action = "created"
    if output_path.exists():
        if output_path.is_symlink() or not output_path.is_file():
            raise PinError(f"pin destination exists but is not a regular file: {output_path}")
        if output_path.read_bytes() != source_bytes:
            raise PinError(
                "pin destination already exists with different bytes; it was not overwritten: "
                f"{output_path}"
            )
        action = "reused_identical"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + f".tmp-{os.getpid()}")
        try:
            with temporary.open("xb") as handle:
                handle.write(source_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Atomic create-without-overwrite.  os.replace() is unsafe here:
                # another process could create the destination after our
                # exists() check and would then be silently overwritten.
                os.link(temporary, output_path)
                directory_fd = os.open(output_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except FileExistsError:
                if output_path.is_symlink() or not output_path.is_file():
                    raise PinError(
                        "pin destination appeared but is not a regular file: "
                        f"{output_path}"
                    )
                if output_path.read_bytes() != source_bytes:
                    raise PinError(
                        "pin destination appeared with different bytes; it was not "
                        f"overwritten: {output_path}"
                    )
                action = "reused_identical"
        finally:
            if temporary.exists():
                temporary.unlink()
    if output_path.read_bytes() != source_bytes:
        raise PinError("pinned manifest byte verification failed")
    return {
        "schema": "onnx-splitpoint/v27541-deepx-b500-manifest-pin",
        "schema_version": 1,
        "status": "ok",
        "action": action,
        "baseline_run": str(run_dir),
        "deepx_artifact_status": str(status_path),
        "source": str(source_path),
        "pinned_manifest": str(output_path),
        "manifest_file_sha256": source_sha,
        "manifest_payload_sha256": required["manifest_payload_sha256"],
        "items_identity_sha256": required["items_identity_sha256"],
        "item_count": EXPECTED_COUNT,
        "selection_seed": EXPECTED_SEED,
        "frozen_b500_authority": frozen_authority,
        "dataset_images_modified": False,
        "evaluation_run_modified": False,
        "cache_modified": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pin the exact proven v2.75.40 B500 calibration manifest"
    )
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = pin_manifest(
            baseline_run=args.baseline_run, source=args.source, output=args.out
        )
    except Exception as exc:
        print(
            json.dumps({
                "schema": "onnx-splitpoint/v27541-deepx-b500-manifest-pin",
                "schema_version": 1,
                "status": "blocked",
                "error": f"{type(exc).__name__}: {exc}",
                "no_overwrite_performed": True,
            }, indent=2, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
