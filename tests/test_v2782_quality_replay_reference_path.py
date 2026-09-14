from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.quality_replay import (
    OfflineQualityReplayError,
    _management_reference_member,
    _preflight_rows,
)


MODEL_ID = "yolo11l"


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _request_row(
    run_dir: Path, management_reference: dict[str, object]
) -> dict[str, object]:
    request_path = run_dir / "models" / MODEL_ID / "quality" / "full_request.json"
    _write_json(request_path.parent / "full_candidate.json", {"records": []})
    request_sha256 = _write_json(
        request_path,
        {"candidate": {"path": "full_candidate.json"}},
    )
    return {
        "model_id": MODEL_ID,
        "source_request": request_path.relative_to(run_dir).as_posix(),
        "source_request_sha256": request_sha256,
        "management_cpu_reference": management_reference,
        "reference_predictions_sha256": "1" * 64,
        "candidate_predictions_sha256": "2" * 64,
        "annotations_sha256": "3" * 64,
    }


@pytest.mark.parametrize("path_form", ["omitted", "absolute", "stale_absolute"])
def test_replay_keeps_historical_fixed_reference_compatibility(
    tmp_path: Path, path_form: str,
) -> None:
    run_dir = (tmp_path / "copied_run").resolve()
    reference_path = (
        run_dir
        / "quality_management"
        / "references"
        / MODEL_ID
        / "canonical_cpu_reference.json"
    )
    reference_sha256 = _write_json(reference_path, {"records": ["legacy"]})
    descriptor: dict[str, object] = {"reference_sha256": reference_sha256}
    if path_form == "absolute":
        descriptor["reference_path"] = str(reference_path)
    elif path_form == "stale_absolute":
        descriptor["reference_path"] = str(
            Path("/retired/host/original_run")
            / reference_path.relative_to(run_dir)
        )

    prepared = _preflight_rows(run_dir, [_request_row(run_dir, descriptor)])

    assert len(prepared) == 1
    assert prepared[0][2] == reference_path.resolve()


def test_replay_rebases_and_consumes_exact_immutable_source_contract_reference(
    tmp_path: Path,
) -> None:
    run_dir = (tmp_path / "copied_run").resolve()
    source_contract_sha256 = "a" * 64
    relative_reference = Path(
        "quality_management",
        "references",
        MODEL_ID,
        "by_source_contract",
        source_contract_sha256,
        "canonical_cpu_reference.json",
    )
    reference_path = run_dir / relative_reference
    reference_sha256 = _write_json(reference_path, {"records": ["immutable"]})

    # Mutable or legacy neighbours must not influence the row-bound selection.
    _write_json(
        run_dir
        / "quality_management/references"
        / MODEL_ID
        / "latest/canonical_cpu_reference.json",
        {"records": ["latest"]},
    )
    _write_json(
        run_dir
        / "quality_management/references"
        / MODEL_ID
        / "canonical_cpu_reference.json",
        {"records": ["legacy"]},
    )
    descriptor: dict[str, object] = {
        "reference_path": str(Path("/retired/original_run") / relative_reference),
        "reference_sha256": f"sha256:{reference_sha256}",
        "reference_size_bytes": reference_path.stat().st_size,
        "source_contract_sha256": f"sha256:{source_contract_sha256}",
        "reference_storage": "immutable_source_contract",
        "reference_immutable": True,
    }

    prepared = _preflight_rows(run_dir, [_request_row(run_dir, descriptor)])

    assert len(prepared) == 1
    assert prepared[0][2] == reference_path.resolve()
    assert any(
        snapshot["role"] == "management CPU reference"
        and snapshot["sha256"] == reference_sha256
        for snapshot in prepared[0][3]
    )


@pytest.mark.parametrize(
    ("descriptor", "message"),
    [
        (
            {
                "reference_path": (
                    "/old/run/quality_management/references/yolo11l/"
                    "latest/canonical_cpu_reference.json"
                ),
                "source_contract_sha256": "a" * 64,
            },
            "mutable/unbound latest reference",
        ),
        (
            {
                "reference_path": "/old/run/outside/reference.json",
                "source_contract_sha256": "a" * 64,
            },
            "not a unique management reference member",
        ),
        (
            {
                "reference_path": (
                    "/old/run/quality_management/references/yolo11l/"
                    f"by_source_contract/{'a' * 64}/canonical_cpu_reference.json"
                ),
                "source_contract_sha256": "b" * 64,
            },
            "source-contract path is not bound",
        ),
        (
            {
                "reference_path": (
                    "/old/run/quality_management/references/yolo11l/"
                    f"by_source_contract/{'a' * 64}/canonical_cpu_reference.json"
                ),
                "source_contract_sha256": "a" * 64,
                "reference_immutable": True,
            },
            "reference_storage does not seal",
        ),
        (
            {
                "reference_path": (
                    "/old/run/quality_management/references/yolo11l/"
                    f"by_source_contract/{'a' * 64}/canonical_cpu_reference.json"
                ),
                "source_contract_sha256": "a" * 64,
                "reference_storage": "immutable_source_contract",
            },
            "reference_immutable does not seal",
        ),
        (
            {
                "reference_path": (
                    "/old/run/quality_management/references/yolo11l/"
                    f"by_source_contract/{'A' * 64}/canonical_cpu_reference.json"
                ),
                "source_contract_sha256": "a" * 64,
                "reference_storage": "immutable_source_contract",
                "reference_immutable": True,
                "reference_size_bytes": 1,
            },
            "source-contract path is not bound",
        ),
        (
            {
                "reference_path": (
                    "/old/run/quality_management/references/yolo11l/"
                    f"by_source_contract/{'a' * 64}/canonical_cpu_reference.json"
                ),
                "source_contract_sha256": "a" * 64,
                "reference_storage": "immutable_source_contract",
                "reference_immutable": True,
            },
            "has no reference_size_bytes seal",
        ),
        (
            {
                "reference_path": (
                    "/old/run/quality_management/references/yolo11l/"
                    f"by_source_contract/{'a' * 64}/canonical_cpu_reference.json"
                ),
                "source_contract_sha256": "a" * 64,
                "reference_storage": "mutable_latest",
                "reference_immutable": True,
            },
            "reference_storage does not seal",
        ),
        (
            {
                "reference_path": (
                    "/old/run/quality_management/references/yolo11l/"
                    f"by_source_contract/{'a' * 64}/canonical_cpu_reference.json"
                ),
                "source_contract_sha256": "a" * 64,
                "reference_storage": "immutable_source_contract",
                "reference_immutable": False,
            },
            "reference_immutable does not seal",
        ),
    ],
)
def test_replay_rejects_unbound_or_mutable_management_reference_paths(
    tmp_path: Path,
    descriptor: dict[str, object],
    message: str,
) -> None:
    run_dir = (tmp_path / "run").resolve()
    run_dir.mkdir(parents=True)

    with pytest.raises(OfflineQualityReplayError, match=message):
        _management_reference_member(
            descriptor,
            run_dir,
            MODEL_ID,
            label="row.reference_path",
        )


def test_replay_rejects_reference_hash_mismatch_at_rebased_member(
    tmp_path: Path,
) -> None:
    run_dir = (tmp_path / "run").resolve()
    source_contract_sha256 = "a" * 64
    reference_path = (
        run_dir
        / "quality_management/references"
        / MODEL_ID
        / "by_source_contract"
        / source_contract_sha256
        / "canonical_cpu_reference.json"
    )
    _write_json(reference_path, {"records": ["immutable"]})
    descriptor: dict[str, object] = {
        "reference_path": str(reference_path),
        "reference_sha256": "f" * 64,
        "reference_size_bytes": reference_path.stat().st_size,
        "source_contract_sha256": source_contract_sha256,
        "reference_storage": "immutable_source_contract",
        "reference_immutable": True,
    }

    with pytest.raises(
        OfflineQualityReplayError,
        match="management reference SHA-256 mismatch",
    ):
        _preflight_rows(run_dir, [_request_row(run_dir, descriptor)])


def test_replay_rejects_reference_size_mismatch_at_rebased_member(
    tmp_path: Path,
) -> None:
    run_dir = (tmp_path / "run").resolve()
    source_contract_sha256 = "a" * 64
    reference_path = (
        run_dir
        / "quality_management/references"
        / MODEL_ID
        / "by_source_contract"
        / source_contract_sha256
        / "canonical_cpu_reference.json"
    )
    reference_sha256 = _write_json(reference_path, {"records": ["immutable"]})
    descriptor: dict[str, object] = {
        "reference_path": str(reference_path),
        "reference_sha256": reference_sha256,
        "reference_size_bytes": reference_path.stat().st_size + 1,
        "source_contract_sha256": source_contract_sha256,
        "reference_storage": "immutable_source_contract",
        "reference_immutable": True,
    }

    with pytest.raises(
        OfflineQualityReplayError,
        match="management reference size mismatch",
    ):
        _preflight_rows(run_dir, [_request_row(run_dir, descriptor)])


def test_replay_rejects_symbolic_link_in_immutable_reference_path(
    tmp_path: Path,
) -> None:
    run_dir = (tmp_path / "run").resolve()
    source_contract_sha256 = "a" * 64
    target = run_dir / "real_reference.json"
    reference_sha256 = _write_json(target, {"records": ["immutable"]})
    reference_path = (
        run_dir
        / "quality_management/references"
        / MODEL_ID
        / "by_source_contract"
        / source_contract_sha256
        / "canonical_cpu_reference.json"
    )
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.symlink_to(target)
    descriptor: dict[str, object] = {
        "reference_path": str(reference_path),
        "reference_sha256": reference_sha256,
        "reference_size_bytes": target.stat().st_size,
        "source_contract_sha256": source_contract_sha256,
        "reference_storage": "immutable_source_contract",
        "reference_immutable": True,
    }

    with pytest.raises(OfflineQualityReplayError, match="symbolic link"):
        _management_reference_member(
            descriptor,
            run_dir,
            MODEL_ID,
            label="row.reference_path",
        )
