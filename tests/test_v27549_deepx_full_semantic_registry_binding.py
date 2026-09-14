from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from onnx_splitpoint_tool.native_detection_postprocess import (
    verify_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_MODEL_ID,
    YOLOV7_PAPER_ONNX_SHA256,
)
from scripts import native_full_semantic_dump as semantic_dump


def _archived_deepx_full_contract() -> dict[str, object]:
    """Relevant projection of the v2.75.48 DeepX Full output contract."""
    return {
        "artifact_kind": "dxnn",
        "backend": "deepx_m1",
        "source_onnx_sha256": None,
        "input": {
            "name": "images",
            "shape": [640, 640, 3],
            "dtype": "uint8",
            "layout": "HWC",
            "normalization": "embedded_dxcom_preprocessing",
            "color_space": "RGB",
            "preprocess_mode": "letterbox",
            "letterbox_pad_value": 114,
        },
        "outputs": [
            {"name": "p3"},
            {"name": "p4"},
            {"name": "p5"},
        ],
    }


def test_registered_yolov7_uses_registry_when_transport_hash_is_absent() -> None:
    assert semantic_dump._bound_model_sha256(
        _archived_deepx_full_contract(),
        {},
        model_id=YOLOV7_PAPER_MODEL_ID,
    ) == YOLOV7_PAPER_ONNX_SHA256


def test_correct_explicit_yolov7_hash_remains_an_accepted_assertion() -> None:
    assert semantic_dump._bound_model_sha256(
        {"source_onnx_sha256": "sha256:" + YOLOV7_PAPER_ONNX_SHA256.upper()},
        {"model_sha256": YOLOV7_PAPER_ONNX_SHA256},
        model_id=YOLOV7_PAPER_MODEL_ID,
    ) == YOLOV7_PAPER_ONNX_SHA256


@pytest.mark.parametrize(
    ("sources", "reason"),
    [
        (({"source_onnx_sha256": "sha256:not-a-digest"},),
         "raw_head_model_sha256_invalid"),
        (({"source_onnx_sha256": "1" * 64},
          {"model_sha256": "2" * 64}),
         "raw_head_model_sha256_conflicting"),
        (({"source_onnx_sha256": "0" * 64},),
         "raw_head_model_sha256_registry_mismatch"),
    ],
)
def test_explicit_invalid_or_conflicting_yolov7_hashes_fail_closed(
    sources: tuple[dict[str, str], ...], reason: str,
) -> None:
    with pytest.raises(RuntimeError, match=reason):
        semantic_dump._bound_model_sha256(
            *sources, model_id=YOLOV7_PAPER_MODEL_ID,
        )


def test_unregistered_model_still_requires_one_explicit_hash() -> None:
    with pytest.raises(
        RuntimeError, match="raw_head_model_sha256_missing_or_conflicting",
    ):
        semantic_dump._bound_model_sha256({}, model_id="future_yolo")


def test_deepx_raw_head_dump_builds_frozen_contract_from_registry_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "semantic.png"
    Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8), mode="RGB").save(
        image
    )
    dxnn = tmp_path / "model.dxnn"
    dxnn.write_bytes(b"registered-yolov7-deepx-full")
    contract = _archived_deepx_full_contract()
    heads = [
        np.full((1, 3, size, size, 85), -20.0, dtype=np.float32)
        for size in (80, 40, 20)
    ]

    class FakeInferenceEngine:
        def __init__(self, path: str) -> None:
            assert path == str(dxnn)

        def run(self, inputs):
            assert len(inputs) == 1
            assert np.asarray(inputs[0]).shape == (640, 640, 3)
            return heads

    monkeypatch.setitem(
        sys.modules,
        "dx_engine",
        SimpleNamespace(InferenceEngine=FakeInferenceEngine),
    )
    monkeypatch.setattr(
        semantic_dump,
        "_find_deepx_contract",
        lambda _benchmark_set: (dxnn, contract),
    )
    monkeypatch.setattr(
        semantic_dump,
        "load_authoritative_output_contract",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        semantic_dump,
        "_contract",
        lambda *_args, **_kwargs: {"contract_family": "raw_head"},
    )
    captured: dict[str, object] = {}

    def capture_manifests(**kwargs):
        frozen = verify_frozen_postprocess_contract(
            kwargs["frozen_postprocess_contract"], outputs=kwargs["outputs"],
        )
        captured["frozen"] = frozen
        output = tmp_path / "native_full_outputs_manifest.json"
        input_manifest = tmp_path / "native_full_input_manifest.json"
        output.write_text("{}", encoding="utf-8")
        input_manifest.write_text("{}", encoding="utf-8")
        return output, input_manifest

    monkeypatch.setattr(semantic_dump, "_write_manifests", capture_manifests)

    result = semantic_dump._run_deepx(
        tmp_path,
        image,
        tmp_path / "semantic",
        YOLOV7_PAPER_MODEL_ID,
        "detection",
        "orin_nx_deepx_m1_01",
        "deepx",
    )

    assert result["ok"] is True
    assert result["input_candidate"] == "contract"
    assert captured["frozen"]["model_sha256"] == YOLOV7_PAPER_ONNX_SHA256
    assert captured["frozen"]["anchor_table_id"] == (
        "yolov7_paper_standard_anchors_640_v1"
    )


def test_packaged_remote_semantic_dump_is_byte_identical() -> None:
    assert Path("scripts/native_full_semantic_dump.py").read_bytes() == Path(
        "onnx_splitpoint_tool/resources/remote_scripts/"
        "native_full_semantic_dump.py"
    ).read_bytes()
