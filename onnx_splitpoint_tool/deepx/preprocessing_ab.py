"""Fail-closed comparison of two DeepX Full preprocessing canary runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import (
    CLASSIFICATION_PREPROCESSING_CURRENT,
    CLASSIFICATION_PREPROCESSING_IMAGENET,
    deepx_classification_preprocessing_contract,
)


PAIR_SCHEMA = "onnx-splitpoint/deepx-classification-preprocessing-ab"
PAIR_SCHEMA_VERSION = 1


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    digest = _bare_sha(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} is not a SHA-256 identity")
    return digest


def _verified_subcontract_identity(
    value: Mapping[str, Any], *, label: str,
) -> str:
    declared = _require_sha256(value.get("identity_sha256"), label=label)
    unhashed = dict(value)
    unhashed.pop("identity_sha256", None)
    if _canonical_sha256(unhashed) != declared:
        raise ValueError(f"{label} self-hash mismatch")
    return declared


def _single(paths: Sequence[Path], *, label: str) -> Path:
    existing = sorted({path.resolve() for path in paths if path.is_file()})
    if len(existing) != 1:
        raise ValueError(f"expected exactly one {label}, found {len(existing)}")
    return existing[0]


def _bare_sha(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


@dataclass(frozen=True)
class ArmEvidence:
    run_dir: Path
    mode: str
    profile_name: str
    cache_dir: str
    cache_key: str
    cache_contract_sha256: str
    task: str
    target: str
    source_onnx_sha256: str
    build_onnx_sha256: str
    dxcom_config_sha256: str
    build_options_sha256: str
    dxnn_sha256: str
    calibration_identity_sha256: str
    calibration_contract_sha256: str
    calibration_items_identity_sha256: str
    compiler_identity_sha256: str
    compiler_contract_sha256: str
    validation_manifest_sha256: str
    validation_image_ids_sha256: str
    validation_ground_truth_sha256: str
    prepared_input_evidence_sha256: str
    preprocessing_contract_sha256: str
    policy_sha256: str
    request_path: Path
    candidate_path: Path
    records: tuple[dict[str, Any], ...]
    deepx_result: dict[str, Any]
    trt_result: dict[str, Any]
    workflow_status: str


def _result_for(summary: Mapping[str, Any], source_run_id: str) -> dict[str, Any]:
    matches = [
        dict(row) for row in list(summary.get("results") or [])
        if isinstance(row, Mapping) and str(row.get("source_run_id") or "") == source_run_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"quality summary requires exactly one {source_run_id} result; found {len(matches)}"
        )
    result = matches[0]
    if str(result.get("status") or "") != "completed" or str(
        result.get("technical_status") or "completed"
    ) != "completed":
        raise ValueError(f"quality result is not technically completed: {source_run_id}")
    return result


def load_arm_evidence(
    run_dir: str | Path,
    *,
    expected_mode: str,
    expected_records: int = 500,
    model_id: str = "resnet50",
) -> ArmEvidence:
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"EvaluationRun not found: {root}")
    profile = _read_object(root / "profile_start_snapshot.json") if (
        root / "profile_start_snapshot.json"
    ).is_file() else _read_object(root / "profile_resolution.json")
    # The fully materialized YAML/JSON is the authority for deepx_build.  The
    # start snapshot is retained above as a mandatory provenance artifact.
    profile_yaml = root / "profile.yaml"
    if not profile_yaml.is_file():
        raise FileNotFoundError(profile_yaml)
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to verify A/B EvaluationRuns") from exc
    resolved_profile = yaml.safe_load(profile_yaml.read_text(encoding="utf-8"))
    if not isinstance(resolved_profile, Mapping):
        raise ValueError("resolved profile.yaml is not an object")
    deepx_build = resolved_profile.get("deepx_build")
    if not isinstance(deepx_build, Mapping):
        raise ValueError("resolved profile lacks deepx_build")
    mode = str(deepx_build.get("classification_preprocessing") or "").strip()
    if mode != expected_mode:
        raise ValueError(f"DeepX preprocessing arm mismatch: expected {expected_mode}, got {mode}")
    cache_dir = str(deepx_build.get("cache_dir") or "").strip()
    if not cache_dir:
        raise ValueError("resolved profile lacks an isolated DeepX cache_dir")

    if not model_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in model_id):
        raise ValueError("invalid model_id")
    model_root = root / "models" / model_id
    status_path = model_root / "benchmark_set" / "deepx" / "deepx_artifact_status.json"
    artifact_status = _read_object(status_path)
    if str(artifact_status.get("status") or "") != "ok":
        raise ValueError("DeepX artifact status is not ok")
    if str(artifact_status.get("classification_preprocessing") or "") != mode:
        raise ValueError("DeepX artifact/profile preprocessing modes differ")
    cache_contract = artifact_status.get("cache_contract")
    if not isinstance(cache_contract, Mapping):
        raise ValueError("DeepX full v2 cache contract missing")
    if str(cache_contract.get("schema") or "") != "onnx-splitpoint/deepx-full-cache-contract":
        raise ValueError("DeepX cache contract schema invalid")
    if int(cache_contract.get("schema_version") or 0) != 2:
        raise ValueError("DeepX cache contract is not v2")
    if str(cache_contract.get("task") or "") != "classification":
        raise ValueError("DeepX A/B cache contract task is not classification")
    if str(cache_contract.get("target") or "") != "deepx_m1":
        raise ValueError("DeepX A/B cache contract target is not deepx_m1")
    if str(cache_contract.get("classification_preprocessing") or "") != mode:
        raise ValueError("DeepX cache contract preprocessing mode mismatch")
    declared_contract_sha = _require_sha256(
        cache_contract.get("contract_sha256"), label="DeepX cache contract",
    )
    unhashed_contract = dict(cache_contract)
    unhashed_contract.pop("contract_sha256", None)
    if _canonical_sha256(unhashed_contract) != declared_contract_sha:
        raise ValueError("DeepX cache contract self-hash mismatch")
    calibration = cache_contract.get("calibration_manifest_contract")
    compiler = cache_contract.get("compiler_identity")
    if not isinstance(calibration, Mapping) or calibration.get("status") != "resolved":
        raise ValueError("DeepX calibration manifest contract is unresolved")
    if not isinstance(compiler, Mapping) or compiler.get("status") != "resolved":
        raise ValueError("DeepX compiler identity is unresolved")
    calibration_identity = _verified_subcontract_identity(
        calibration, label="DeepX calibration contract",
    )
    compiler_identity = _verified_subcontract_identity(
        compiler, label="DeepX compiler contract",
    )
    build_options = cache_contract.get("build_options")
    if not isinstance(build_options, Mapping):
        raise ValueError("DeepX cache contract build options are missing")
    artifacts = [
        dict(row) for row in list(artifact_status.get("artifacts") or [])
        if isinstance(row, Mapping) and str(row.get("variant") or "") == "full"
    ]
    if len(artifacts) != 1:
        raise ValueError(f"DeepX artifact status requires one Full artifact, found {len(artifacts)}")
    cache_key = str(artifacts[0].get("cache_key") or "")
    if "_v2_" not in cache_key:
        raise ValueError("DeepX Full artifact did not use the v2 cache namespace")
    build_onnx_sha = _require_sha256(
        cache_contract.get("build_onnx_sha256"), label="DeepX build ONNX",
    )
    config_sha = _require_sha256(
        cache_contract.get("dxcom_config_sha256"), label="DeepX DX-COM config",
    )
    key_payload = {
        "schema": "onnx-splitpoint/deepx-cache-key",
        "schema_version": 2,
        "target": "deepx_m1",
        "variant": f"full_{mode}",
        "onnx_sha256": build_onnx_sha,
        "config_sha256": config_sha,
        "cache_contract": dict(cache_contract),
    }
    expected_cache_key = (
        f"deepx_m1_full_{mode}_v2_{_canonical_sha256(key_payload)[:32]}"
    )
    if cache_key != expected_cache_key:
        raise ValueError("DeepX v2 cache key does not match its exact contract")
    artifact_path = Path(str(artifacts[0].get("dxnn_path") or "")).expanduser()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"DeepX cache artifact missing: {artifact_path}")
    expected_artifact_path = (
        Path(cache_dir).expanduser().resolve() / cache_key / "model.dxnn"
    )
    if artifact_path.resolve() != expected_artifact_path:
        raise ValueError("DeepX artifact is outside the isolated arm cache namespace")
    artifact_sha256 = _sha256_file(artifact_path)
    receipt = _read_object(artifact_path.parent / "build_manifest.json")
    if (
        str(receipt.get("schema") or "")
        != "onnx-splitpoint/deepx-full-cache-receipt"
        or int(receipt.get("schema_version") or 0) != 2
        or str(receipt.get("cache_key") or "") != cache_key
    ):
        raise ValueError("DeepX v2 cache receipt envelope is invalid")
    receipt_contract = receipt.get("cache_contract")
    receipt_artifact = receipt.get("artifact")
    if not isinstance(receipt_contract, Mapping) or dict(receipt_contract) != dict(cache_contract):
        raise ValueError("DeepX cache receipt contract differs from run contract")
    if not isinstance(receipt_artifact, Mapping):
        raise ValueError("DeepX cache receipt artifact identity missing")
    if (
        _bare_sha(receipt_artifact.get("sha256")) != artifact_sha256
        or int(receipt_artifact.get("bytes") or -1) != artifact_path.stat().st_size
    ):
        raise ValueError("DeepX cache receipt artifact identity mismatch")
    adapter = receipt.get("build_onnx_adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError("DeepX build ONNX adapter receipt is missing")
    if (
        str(adapter.get("schema") or "")
        != "onnx-splitpoint/deepx-build-onnx-adapter"
        or int(adapter.get("schema_version") or 0) != 1
        or _require_sha256(
            adapter.get("source_onnx_sha256"), label="DeepX adapter source ONNX",
        ) != _require_sha256(
            cache_contract.get("source_onnx_sha256"), label="DeepX source ONNX",
        )
        or _require_sha256(
            adapter.get("build_onnx_sha256"), label="DeepX adapter build ONNX",
        ) != build_onnx_sha
    ):
        raise ValueError("DeepX build ONNX adapter identity mismatch")
    if mode == CLASSIFICATION_PREPROCESSING_CURRENT:
        if (
            str(adapter.get("kind") or "") != "identity"
            or build_onnx_sha
            != _require_sha256(
                cache_contract.get("source_onnx_sha256"),
                label="DeepX source ONNX",
            )
        ):
            raise ValueError("DeepX scale-only arm did not preserve the source ONNX")
    else:
        adapter_identity = {
            key: adapter.get(key)
            for key in (
                "schema", "schema_version", "source_onnx_sha256",
                "input_name", "consumer_count", "preprocessing",
            )
        }
        if (
            not str(adapter_identity.get("input_name") or "")
            or int(adapter_identity.get("consumer_count") or 0) <= 0
            or dict(adapter_identity.get("preprocessing") or {})
            != deepx_classification_preprocessing_contract(
                CLASSIFICATION_PREPROCESSING_IMAGENET
            )
            or _require_sha256(
                adapter.get("adapter_contract_sha256"),
                label="DeepX ImageNet adapter contract",
            ) != _canonical_sha256(adapter_identity)
            or int(adapter.get("build_onnx_bytes") or 0) <= 0
        ):
            raise ValueError("DeepX ImageNet build ONNX adapter contract is invalid")

    request_path = _single(
        list(model_root.glob(
            "benchmark_results/quality_inputs/*/results/deepx_m1_full/"
            "task_quality_inputs/full_request.json"
        )),
        label="DeepX Full quality request",
    )
    request = _read_object(request_path)
    descriptor = request.get("candidate")
    if not isinstance(descriptor, Mapping):
        raise ValueError("DeepX request lacks candidate descriptor")
    candidate_path = (request_path.parent / str(descriptor.get("path") or "")).resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    if int(descriptor.get("size_bytes") or -1) != candidate_path.stat().st_size:
        raise ValueError("DeepX candidate byte count mismatch")
    if _bare_sha(descriptor.get("sha256")) != _sha256_file(candidate_path):
        raise ValueError("DeepX candidate SHA-256 mismatch")
    candidate = _read_object(candidate_path)
    records = [dict(row) for row in list(candidate.get("records") or []) if isinstance(row, Mapping)]
    if int(candidate.get("record_count") or -1) != len(records):
        raise ValueError("DeepX candidate record_count mismatch")
    if len(records) != int(expected_records):
        raise ValueError(
            f"DeepX candidate expected {expected_records} records, found {len(records)}"
        )
    image_ids = [str(row.get("image_id") or "") for row in records]
    if any(not value for value in image_ids) or len(image_ids) != len(set(image_ids)):
        raise ValueError("DeepX candidate image IDs are empty or duplicated")
    labels: list[int] = []
    for row in records:
        try:
            labels.append(int(row.get("label_id")))
        except Exception as exc:
            raise ValueError("DeepX candidate label_id is not an integer") from exc
        result = row.get("candidate")
        if not isinstance(result, Mapping):
            raise ValueError("DeepX candidate classification record missing")
        if not isinstance(result.get("top1_hit"), bool) or not isinstance(
            result.get("top5_hit"), bool
        ):
            raise ValueError("DeepX candidate hit decisions are not booleans")

    producer = request.get("producer_identity")
    if not isinstance(producer, Mapping):
        raise ValueError("DeepX request lacks producer identity")
    dataset = producer.get("dataset")
    model = producer.get("model")
    if not isinstance(dataset, Mapping) or not isinstance(model, Mapping):
        raise ValueError("DeepX producer model/dataset identity missing")
    if _bare_sha(model.get("runtime_artifact_sha256")) != artifact_sha256:
        raise ValueError("DeepX producer and cache artifact SHA-256 differ")
    expected_ids = list(request.get("expected_image_ids") or [])
    if expected_ids != image_ids:
        raise ValueError("DeepX request and candidate image order differ")

    quality_summary = _read_object(root / "quality_management" / "central_quality_summary.json")
    deepx_result = _result_for(quality_summary, "deepx_m1_full")
    trt_result = _result_for(quality_summary, "native_full_tensorrt")
    status_artifact = (
        root / "run_status.json"
        if (root / "run_status.json").is_file()
        else root / "run_manifest.json"
    )
    run_status = _read_object(status_artifact) if status_artifact.is_file() else {}
    workflow_status = str(run_status.get("status") or quality_summary.get("technical_status") or "")
    if workflow_status not in {"ok", "completed"}:
        raise ValueError(f"A/B arm is not technically complete: {workflow_status}")

    return ArmEvidence(
        run_dir=root,
        mode=mode,
        profile_name=str(resolved_profile.get("name") or profile.get("profile_name") or ""),
        cache_dir=cache_dir,
        cache_key=cache_key,
        cache_contract_sha256=declared_contract_sha,
        task=str(cache_contract.get("task") or ""),
        target=str(cache_contract.get("target") or ""),
        source_onnx_sha256=_require_sha256(
            cache_contract.get("source_onnx_sha256"), label="DeepX source ONNX",
        ),
        build_onnx_sha256=build_onnx_sha,
        dxcom_config_sha256=config_sha,
        build_options_sha256=_canonical_sha256(build_options),
        dxnn_sha256=artifact_sha256,
        calibration_identity_sha256=calibration_identity,
        calibration_contract_sha256=_canonical_sha256(calibration),
        calibration_items_identity_sha256=_require_sha256(
            calibration.get("items_identity_sha256"),
            label="DeepX calibration item cohort",
        ),
        compiler_identity_sha256=compiler_identity,
        compiler_contract_sha256=_canonical_sha256(compiler),
        validation_manifest_sha256=_require_sha256(
            dataset.get("manifest_sha256"), label="Validation manifest",
        ),
        validation_image_ids_sha256=_require_sha256(
            dataset.get("image_ids_sha256"), label="Validation image IDs",
        ),
        validation_ground_truth_sha256=_require_sha256(
            dataset.get("ground_truth_sha256"), label="Validation ground truth",
        ),
        prepared_input_evidence_sha256=_require_sha256(
            request.get("prepared_input_evidence_sha256"),
            label="DeepX prepared input evidence",
        ),
        preprocessing_contract_sha256=_require_sha256(
            request.get("preprocessing_contract_sha256"),
            label="DeepX preprocessing contract",
        ),
        policy_sha256=_require_sha256(
            request.get("policy_sha256"), label="Task-quality policy",
        ),
        request_path=request_path,
        candidate_path=candidate_path,
        records=tuple(records),
        deepx_result=deepx_result,
        trt_result=trt_result,
        workflow_status=workflow_status,
    )


def _paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    if a.shape != b.shape or a.ndim != 1 or not a.size:
        raise ValueError("paired bootstrap inputs must be non-empty equal vectors")
    delta = b.astype(np.float64) - a.astype(np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, delta.size, size=(int(repetitions), delta.size))
    distribution = delta[indices].mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    return {
        "sample_count": int(delta.size),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "confidence": float(confidence),
        "point_delta_b_minus_a": float(delta.mean()),
        "ci_low": float(np.quantile(distribution, tail)),
        "ci_high": float(np.quantile(distribution, 1.0 - tail)),
    }


def compare_preprocessing_arms(
    arm_a: ArmEvidence,
    arm_b: ArmEvidence,
    *,
    bootstrap_repetitions: int = 500,
    bootstrap_seed: int = 20260710,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if arm_a.mode != CLASSIFICATION_PREPROCESSING_CURRENT:
        raise ValueError("arm A must be current_scale_only")
    if arm_b.mode != CLASSIFICATION_PREPROCESSING_IMAGENET:
        raise ValueError("arm B must be imagenet_mean_std")

    exact_fields = (
        "task",
        "target",
        "source_onnx_sha256",
        "dxcom_config_sha256",
        "build_options_sha256",
        "calibration_identity_sha256",
        "calibration_contract_sha256",
        "calibration_items_identity_sha256",
        "compiler_identity_sha256",
        "compiler_contract_sha256",
        "validation_manifest_sha256",
        "validation_image_ids_sha256",
        "validation_ground_truth_sha256",
        "prepared_input_evidence_sha256",
        "preprocessing_contract_sha256",
        "policy_sha256",
    )
    mismatches = [
        field for field in exact_fields
        if not getattr(arm_a, field) or getattr(arm_a, field) != getattr(arm_b, field)
    ]
    if mismatches:
        raise ValueError(f"A/B cohort identity differs: {','.join(mismatches)}")
    if arm_a.cache_dir == arm_b.cache_dir or arm_a.cache_key == arm_b.cache_key:
        raise ValueError("A/B cache namespaces are not isolated")
    if arm_a.build_onnx_sha256 == arm_b.build_onnx_sha256:
        raise ValueError("A/B build ONNX identities unexpectedly match")
    if arm_a.dxnn_sha256 == arm_b.dxnn_sha256:
        raise ValueError("A/B DXNN identities unexpectedly match")

    ids_a = [str(row.get("image_id") or "") for row in arm_a.records]
    ids_b = [str(row.get("image_id") or "") for row in arm_b.records]
    labels_a = [int(row.get("label_id")) for row in arm_a.records]
    labels_b = [int(row.get("label_id")) for row in arm_b.records]
    if ids_a != ids_b or labels_a != labels_b:
        raise ValueError("A/B candidate sample order or ground truth differs")

    def hits(arm: ArmEvidence, key: str) -> np.ndarray:
        return np.asarray(
            [int(bool(dict(row.get("candidate") or {}).get(key))) for row in arm.records],
            dtype=np.int8,
        )

    a_top1, b_top1 = hits(arm_a, "top1_hit"), hits(arm_b, "top1_hit")
    a_top5, b_top5 = hits(arm_a, "top5_hit"), hits(arm_b, "top5_hit")
    top1 = _paired_bootstrap(
        a_top1, b_top1,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    top5 = _paired_bootstrap(
        a_top5, b_top5,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed + 1,
        confidence=confidence,
    )
    top1.update({
        "arm_a_hits": int(a_top1.sum()),
        "arm_b_hits": int(b_top1.sum()),
        "corrected_by_b": int(np.sum((a_top1 == 0) & (b_top1 == 1))),
        "regressed_by_b": int(np.sum((a_top1 == 1) & (b_top1 == 0))),
    })
    top5.update({
        "arm_a_hits": int(a_top5.sum()),
        "arm_b_hits": int(b_top5.sum()),
        "corrected_by_b": int(np.sum((a_top5 == 0) & (b_top5 == 1))),
        "regressed_by_b": int(np.sum((a_top5 == 1) & (b_top5 == 0))),
    })
    b_decision = str(arm_b.deepx_result.get("decision") or "")
    technically_complete = all(
        str(result.get("status") or "") == "completed"
        for result in (
            arm_a.deepx_result, arm_a.trt_result,
            arm_b.deepx_result, arm_b.trt_result,
        )
    )
    controls_pass = all(
        str(result.get("decision") or "") == "pass"
        for result in (arm_a.trt_result, arm_b.trt_result)
    )
    paired_improvement_ci_positive = bool(
        float(top1["ci_low"]) > 0.0 and float(top5["ci_low"]) > 0.0
    )
    standard_plus_ready = bool(
        technically_complete
        and controls_pass
        and b_decision == "pass"
        and paired_improvement_ci_positive
    )
    return {
        "schema": PAIR_SCHEMA,
        "schema_version": PAIR_SCHEMA_VERSION,
        "status": "verified",
        "cohort": {
            field: getattr(arm_a, field) for field in exact_fields
        },
        "arm_a": {
            "run_dir": str(arm_a.run_dir),
            "mode": arm_a.mode,
            "cache_dir": arm_a.cache_dir,
            "cache_key": arm_a.cache_key,
            "cache_contract_sha256": arm_a.cache_contract_sha256,
            "build_onnx_sha256": arm_a.build_onnx_sha256,
            "dxnn_sha256": arm_a.dxnn_sha256,
            "quality_decision": str(arm_a.deepx_result.get("decision") or ""),
        },
        "arm_b": {
            "run_dir": str(arm_b.run_dir),
            "mode": arm_b.mode,
            "cache_dir": arm_b.cache_dir,
            "cache_key": arm_b.cache_key,
            "cache_contract_sha256": arm_b.cache_contract_sha256,
            "build_onnx_sha256": arm_b.build_onnx_sha256,
            "dxnn_sha256": arm_b.dxnn_sha256,
            "quality_decision": b_decision,
        },
        "paired": {"top1": top1, "top5": top5},
        "technical_evidence_complete": technically_complete,
        "setup_local_tensorrt_controls_pass": controls_pass,
        "paired_improvement_ci_positive": paired_improvement_ci_positive,
        "standard_plus_ready": standard_plus_ready,
        "next_step": (
            "standard_plus"
            if standard_plus_ready
            else "keep_standard_plus_blocked_and_isolate_next_deepx_variable"
        ),
    }


def compare_run_directories(
    *,
    arm_a_dir: str | Path,
    arm_b_dir: str | Path,
    expected_records: int = 500,
    bootstrap_repetitions: int = 500,
    bootstrap_seed: int = 20260710,
    model_id: str = "resnet50",
) -> dict[str, Any]:
    arm_a = load_arm_evidence(
        arm_a_dir,
        expected_mode=CLASSIFICATION_PREPROCESSING_CURRENT,
        expected_records=expected_records, model_id=model_id,
    )
    arm_b = load_arm_evidence(
        arm_b_dir,
        expected_mode=CLASSIFICATION_PREPROCESSING_IMAGENET,
        expected_records=expected_records, model_id=model_id,
    )
    return compare_preprocessing_arms(
        arm_a,
        arm_b,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
    )
