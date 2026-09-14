from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from onnx_splitpoint_tool.validation import official_coco


def test_normalize_sha256_accepts_only_supported_representations() -> None:
    digest = "a1" * 32
    assert official_coco.normalize_sha256(digest) == digest
    assert official_coco.normalize_sha256(f"sha256:{digest}") == digest
    assert official_coco.normalize_sha256(digest.upper()) == digest


@pytest.mark.parametrize(
    "value",
    [
        "",
        "a" * 63,
        "a" * 65,
        "sha256:" + "a" * 63,
        "SHA256:" + "a" * 64,
        "md5:" + "a" * 64,
        "sha256:sha256:" + "a" * 64,
        " " + "a" * 64,
        "a" * 64 + " ",
        None,
    ],
)
def test_normalize_sha256_rejects_malformed_values(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_sha256"):
        official_coco.normalize_sha256(value)


@pytest.mark.parametrize("declared_prefix", ["", "sha256:"])
def test_official_coco_accepts_bare_or_qualified_declared_hash_against_qualified_actual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declared_prefix: str,
) -> None:
    annotations = tmp_path / "instances_val2017.json"
    annotations.write_bytes(b"{}\n")
    digest = hashlib.sha256(annotations.read_bytes()).hexdigest()

    # v2.75.47 stripped the declared prefix but compared it directly with this
    # qualified value, causing the completed 500-image probe to fail here.
    assert official_coco.sha256_file(annotations) == f"sha256:{digest}"
    monkeypatch.setattr(
        official_coco,
        "pycocotools_status",
        lambda: {"available": False, "error": "not installed in unit test"},
    )

    payload = official_coco.evaluate_coco_bbox(
        annotations=annotations,
        predictions=[],
        output_dir=tmp_path / f"result-{declared_prefix or 'bare'}",
        annotations_provenance={
            "source_sha256": f"{declared_prefix}{digest}",
            "exact_use_snapshot_sha256": f"{declared_prefix}{digest}",
        },
    )

    assert payload["status"] == "unavailable"
    assert payload["reason"] == "pycocotools_missing"


@pytest.mark.parametrize(
    "declared",
    [
        "sha256:sha256:" + "a" * 64,
        "sha1:" + "a" * 64,
        "a" * 63,
    ],
)
def test_official_coco_fails_closed_on_malformed_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declared: str,
) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_bytes(b"{}\n")
    monkeypatch.setattr(
        official_coco,
        "pycocotools_status",
        lambda: {"available": False, "error": "not installed in unit test"},
    )

    with pytest.raises(
        ValueError,
        match="annotations_provenance_source_sha256_invalid_sha256",
    ):
        official_coco.evaluate_coco_bbox(
            annotations=annotations,
            predictions=[],
            output_dir=tmp_path / "result",
            annotations_provenance={"source_sha256": declared},
        )


def test_official_coco_still_rejects_true_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_bytes(b"{}\n")
    monkeypatch.setattr(
        official_coco,
        "pycocotools_status",
        lambda: {"available": False, "error": "not installed in unit test"},
    )

    with pytest.raises(
        ValueError,
        match="annotations_provenance_sha256_mismatch:source_sha256",
    ):
        official_coco.evaluate_coco_bbox(
            annotations=annotations,
            predictions=[],
            output_dir=tmp_path / "result",
            annotations_provenance={"source_sha256": "0" * 64},
        )
