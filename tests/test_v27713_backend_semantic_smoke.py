from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool import backend_semantic_smoke as smoke
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    prepare_rgb_uint8_image,
)


def _spec(*cases: dict[str, object]) -> dict[str, object]:
    return {"schema": smoke.SPEC_SCHEMA, "cases": list(cases)}


def test_prepared_input_identity_exact_and_mismatch(tmp_path: Path) -> None:
    source = np.arange(19 * 31 * 3, dtype=np.uint16).reshape(19, 31, 3)
    source = np.asarray(source % 256, dtype=np.uint8)
    source_path = tmp_path / "source.npy"
    np.save(source_path, source)
    contract = canonical_image_preprocessing_contract(
        "detection", [64, 64]
    )
    prepared, _geometry = prepare_rgb_uint8_image(source, contract)
    for producer in ("cpu", "tensorrt", "hailo"):
        np.save(tmp_path / f"{producer}.npy", prepared)

    case = {
        "id": "prepared",
        "kind": "prepared_input_identity",
        "task": "detection",
        "target_hw": [64, 64],
        "source_image": "source.npy",
        "observations": [
            {"producer": value, "path": f"{value}.npy"}
            for value in ("cpu", "tensorrt", "hailo")
        ],
    }
    result = smoke.run_backend_semantic_smokes(
        _spec(case), base_dir=tmp_path
    )

    assert result["status"] == "PASS"
    assert result["hardware_invoked"] is False
    assert result["cases"][0]["details"]["observations"][2][
        "exact_match"
    ] is True

    changed = prepared.copy()
    changed[0, 0, 0] ^= np.uint8(1)
    np.save(tmp_path / "hailo.npy", changed)
    failed = smoke.run_backend_semantic_smokes(
        _spec(case), base_dir=tmp_path
    )
    assert failed["status"] == "FAIL"
    assert failed["cases"][0]["reason"] == (
        "prepared_input_identity_mismatch"
    )


def test_prepared_input_requires_two_distinct_producers(
    tmp_path: Path,
) -> None:
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    np.save(tmp_path / "source.npy", source)
    contract = canonical_image_preprocessing_contract(
        "classification", [4, 4]
    )
    prepared, _geometry = prepare_rgb_uint8_image(source, contract)
    np.save(tmp_path / "cpu.npy", prepared)

    payload = smoke.run_backend_semantic_smokes(
        _spec({
            "id": "single-producer",
            "kind": "prepared_input_identity",
            "task": "classification",
            "target_hw": [4, 4],
            "source_image": "source.npy",
            "observations": [
                {"producer": "cpu", "path": "cpu.npy"},
            ],
        }),
        base_dir=tmp_path,
    )

    assert payload["status"] == "FAIL"
    assert payload["cases"][0]["reason"] == (
        "prepared_input_at_least_two_producers_required"
    )

    duplicate = smoke.run_backend_semantic_smokes(
        _spec({
            "id": "duplicate-producer",
            "kind": "prepared_input_identity",
            "task": "classification",
            "target_hw": [4, 4],
            "source_image": "source.npy",
            "observations": [
                {"producer": "cpu", "path": "cpu.npy"},
                {"producer": "cpu", "path": "cpu.npy"},
            ],
        }),
        base_dir=tmp_path,
    )
    assert duplicate["status"] == "FAIL"
    assert duplicate["cases"][0]["reason"] == (
        "prepared_input_producer_duplicate:cpu"
    )


def test_optional_evidence_is_structured_skip(tmp_path: Path) -> None:
    payload = smoke.run_backend_semantic_smokes(
        _spec({
            "id": "optional",
            "kind": "prepared_input_identity",
            "task": "classification",
            "target_hw": [224, 224],
            "source_image": "not-present.jpg",
            "observations": [
                {"producer": "cpu", "path": "not-present-cpu.npy"},
                {"producer": "hailo", "path": "not-present-hailo.npy"},
            ],
        }),
        base_dir=tmp_path,
    )
    assert payload["status"] == "SKIP"
    assert payload["counts"] == {"PASS": 0, "FAIL": 0, "SKIP": 1}
    assert payload["cases"][0]["reason"] == "source_image_missing"


def test_prepared_input_mismatch_wins_over_later_skip(
    tmp_path: Path,
) -> None:
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    np.save(tmp_path / "source.npy", source)
    contract = canonical_image_preprocessing_contract(
        "classification", [4, 4]
    )
    prepared, _geometry = prepare_rgb_uint8_image(source, contract)
    changed = prepared.copy()
    changed[0, 0, 0] ^= np.uint8(1)
    np.save(tmp_path / "changed.npy", changed)

    payload = smoke.run_backend_semantic_smokes(
        _spec({
            "id": "mismatch-before-missing",
            "kind": "prepared_input_identity",
            "task": "classification",
            "target_hw": [4, 4],
            "source_image": "source.npy",
            "observations": [
                {"producer": "cpu", "path": "changed.npy"},
                {"producer": "hailo", "path": "missing.npy"},
            ],
        }),
        base_dir=tmp_path,
    )

    assert payload["status"] == "FAIL"
    assert payload["cases"][0]["reason"] == (
        "prepared_input_identity_mismatch"
    )
    assert [
        row["status"]
        for row in payload["cases"][0]["details"]["observations"]
    ] == ["FAIL", "SKIP"]


class _FakeSession:
    def __init__(
        self,
        path: str,
        *,
        offset: float = 0.0,
        fixed_input_name: str = "images",
        fixed_input_type: str = "tensor(float)",
        fixed_input_shape=(1, 3, 4, 4),
        fixed_output_type: str = "tensor(float)",
        fixed_output_dtype=np.float32,
        **_kwargs,
    ) -> None:
        self.fixed = "fixed" in Path(path).name
        self.offset = float(offset)
        self.output_dtype = np.dtype(
            fixed_output_dtype if self.fixed else np.float32
        )
        self._input = SimpleNamespace(
            name=(fixed_input_name if self.fixed else "images"),
            shape=(list(fixed_input_shape) if self.fixed else [1, 3, 4, 4]),
            type=(fixed_input_type if self.fixed else "tensor(float)"),
        )
        self._output = SimpleNamespace(
            name="logits",
            shape=[1, 3],
            type=(fixed_output_type if self.fixed else "tensor(float)"),
        )

    def get_inputs(self):
        return [self._input]

    def get_outputs(self):
        return [self._output]

    def run(self, names, feed):
        assert names == ["logits"]
        total = float(np.asarray(feed["images"]).sum())
        value = np.asarray(
            [[total, total / 2.0, -total]], dtype=self.output_dtype
        )
        if self.fixed:
            value = np.asarray(value + self.offset, dtype=self.output_dtype)
        return [value]


def _fake_runtime(offset: float = 0.0, **session_kwargs):
    class Runtime:
        class SessionOptions:
            pass

        @staticmethod
        def InferenceSession(path, **kwargs):
            return _FakeSession(
                path,
                offset=offset,
                **session_kwargs,
                **kwargs,
            )

    return Runtime


def test_onnx_pair_cpu_parity_with_optional_runtime_adapter(
    tmp_path: Path,
) -> None:
    (tmp_path / "original.onnx").write_bytes(b"original")
    (tmp_path / "fixed.onnx").write_bytes(b"fixed")
    case = {
        "id": "onnx-pair",
        "kind": "onnx_cpu_pair",
        "original_model": "original.onnx",
        "fixed_model": "fixed.onnx",
        "seed": 7,
    }

    passed = smoke.audit_onnx_cpu_pair(
        case, base_dir=tmp_path, runtime_module=_fake_runtime()
    )
    assert passed["status"] == "PASS"
    assert passed["details"]["input_source"] == (
        "deterministic_seeded_static_shape"
    )
    assert passed["details"]["outputs"][0]["argmax_match"] is True

    failed = smoke.audit_onnx_cpu_pair(
        case,
        base_dir=tmp_path,
        runtime_module=_fake_runtime(offset=0.25),
    )
    assert failed["status"] == "FAIL"
    assert failed["reason"] == "onnx_cpu_pair_numerical_mismatch"
    assert failed["details"]["outputs"][0]["max_abs_error"] == pytest.approx(
        0.25
    )


@pytest.mark.parametrize("same_path", [True, False])
def test_onnx_pair_requires_independent_model_artifacts(
    tmp_path: Path,
    same_path: bool,
) -> None:
    original = tmp_path / "original.onnx"
    fixed = original if same_path else tmp_path / "fixed.onnx"
    original.write_bytes(b"identical-model")
    if not same_path:
        fixed.write_bytes(b"identical-model")

    with pytest.raises(
        smoke.SemanticSmokeFailure,
        match="onnx_pair_models_not_independent",
    ):
        smoke.audit_onnx_cpu_pair(
            {
                "id": "independent-models",
                "original_model": original.name,
                "fixed_model": fixed.name,
            },
            base_dir=tmp_path,
            runtime_module=_fake_runtime(),
        )


def test_onnx_pair_nonfinite_outputs_are_hashable_structured_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "original.onnx").write_bytes(b"original")
    (tmp_path / "fixed.onnx").write_bytes(b"fixed")
    monkeypatch.setattr(
        smoke.importlib,
        "import_module",
        lambda name: _fake_runtime(offset=float("nan")),
    )

    payload = smoke.run_backend_semantic_smokes(
        _spec({
            "id": "nonfinite-output",
            "kind": "onnx_cpu_pair",
            "original_model": "original.onnx",
            "fixed_model": "fixed.onnx",
        }),
        base_dir=tmp_path,
    )

    assert payload["status"] == "FAIL"
    row = payload["cases"][0]["details"]["outputs"][0]
    assert row["finite"] is False
    assert row["max_abs_error"] is None
    assert len(payload["result_sha256"]) == 64
    __import__("json").dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("session_kwargs", "reason"),
    [
        (
            {
                "fixed_output_type": "tensor(float16)",
                "fixed_output_dtype": np.float16,
            },
            "onnx_pair_output_dtype_contract_mismatch:logits:logits",
        ),
        (
            {"fixed_output_dtype": np.float16},
            "onnx_pair_output_dtype_mismatch:logits:logits",
        ),
    ],
)
def test_onnx_pair_requires_declared_and_actual_output_dtype_identity(
    tmp_path: Path,
    session_kwargs: dict[str, object],
    reason: str,
) -> None:
    (tmp_path / "original.onnx").write_bytes(b"original")
    (tmp_path / "fixed.onnx").write_bytes(b"fixed")

    with pytest.raises(smoke.SemanticSmokeFailure, match=reason):
        smoke.audit_onnx_cpu_pair(
            {
                "id": "output-dtype",
                "original_model": "original.onnx",
                "fixed_model": "fixed.onnx",
            },
            base_dir=tmp_path,
            runtime_module=_fake_runtime(**session_kwargs),
        )


def test_missing_onnxruntime_is_skip_not_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "original.onnx").write_bytes(b"original")
    (tmp_path / "fixed.onnx").write_bytes(b"fixed")

    def unavailable(name: str):
        assert name == "onnxruntime"
        raise ImportError("optional runtime absent")

    monkeypatch.setattr(smoke.importlib, "import_module", unavailable)
    result = smoke.run_backend_semantic_smokes(
        _spec({
            "id": "onnx-pair",
            "kind": "onnx_cpu_pair",
            "original_model": "original.onnx",
            "fixed_model": "fixed.onnx",
        }),
        base_dir=tmp_path,
    )
    assert result["status"] == "SKIP"
    assert result["cases"][0]["reason"] == "onnxruntime_unavailable"


@pytest.mark.parametrize(
    ("session_kwargs", "reason"),
    [
        ({"fixed_input_name": "pixels"}, "onnx_pair_input_names_mismatch"),
        (
            {"fixed_input_type": "tensor(float16)"},
            "onnx_pair_input_dtype_mismatch:images",
        ),
        (
            {"fixed_input_shape": (1, 3, 8, 8)},
            "onnx_pair_input_shape_contract_mismatch:images",
        ),
    ],
)
def test_onnx_pair_requires_exact_input_contract(
    tmp_path: Path,
    session_kwargs: dict[str, object],
    reason: str,
) -> None:
    (tmp_path / "original.onnx").write_bytes(b"original")
    (tmp_path / "fixed.onnx").write_bytes(b"fixed")
    with pytest.raises(smoke.SemanticSmokeFailure, match=reason):
        smoke.audit_onnx_cpu_pair(
            {
                "id": "input-contract",
                "original_model": "original.onnx",
                "fixed_model": "fixed.onnx",
            },
            base_dir=tmp_path,
            runtime_module=_fake_runtime(**session_kwargs),
        )


class _TwoOutputSession(_FakeSession):
    def __init__(self, path: str, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self._outputs = [
            SimpleNamespace(name="kept", shape=[1], type="tensor(float)"),
            SimpleNamespace(name="different", shape=[1], type="tensor(float)"),
        ]

    def get_outputs(self):
        return self._outputs

    def run(self, names, feed):
        assert "images" in feed
        values = {
            "kept": np.asarray([1.0], dtype=np.float32),
            "different": np.asarray(
                [99.0 if self.fixed else 2.0], dtype=np.float32
            ),
        }
        return [values[name] for name in names]


class _TwoOutputRuntime:
    class SessionOptions:
        pass

    @staticmethod
    def InferenceSession(path, **kwargs):
        return _TwoOutputSession(path, **kwargs)


def test_onnx_output_pairs_must_be_complete_bijection(
    tmp_path: Path,
) -> None:
    (tmp_path / "original.onnx").write_bytes(b"original")
    (tmp_path / "fixed.onnx").write_bytes(b"fixed")
    case = {
        "id": "output-coverage",
        "original_model": "original.onnx",
        "fixed_model": "fixed.onnx",
        "output_pairs": [{"original": "kept", "fixed": "kept"}],
    }

    with pytest.raises(
        smoke.SemanticSmokeFailure,
        match="onnx_output_pairs_not_complete_bijection",
    ):
        smoke.audit_onnx_cpu_pair(
            case, base_dir=tmp_path, runtime_module=_TwoOutputRuntime
        )

    case["output_pairs"].append({
        "original": "different",
        "fixed": "different",
    })
    result = smoke.audit_onnx_cpu_pair(
        case, base_dir=tmp_path, runtime_module=_TwoOutputRuntime
    )
    assert result["status"] == "FAIL"
    assert [row["allclose"] for row in result["details"]["outputs"]] == [
        True,
        False,
    ]


_EXPECTED_DETECTIONS = {
    # YOLO11 DFL16: uniform zero logits have E[bin]=7.5.  At stride 8,
    # grid cell (10,10) is centred at (84,84), hence 84 +/- 7.5*8.
    "yolo11l": [[24.0, 24.0, 144.0, 144.0, 1.0, 11.0]],
    # YOLO26 direct LTRB: four unit distances at stride 8 around (84,84).
    "yolo26m": [[76.0, 76.0, 92.0, 92.0, 1.0, 7.0]],
}


def _reference_bn6(model_id: str) -> np.ndarray:
    """Build the completed oracle without invoking either decoder under test."""

    return np.asarray([_EXPECTED_DETECTIONS[model_id]], dtype=np.float32)


def _yolo11_outputs() -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for level, side in enumerate((80, 40, 20)):
        outputs[f"reg{level}"] = np.zeros(
            (side, side, 64), dtype=np.float32
        )
        outputs[f"cls{level}"] = np.full(
            (side, side, 80), -20.0, dtype=np.float32
        )
    outputs["cls0"][10, 10, 11] = 20.0
    return outputs


def _yolo26_outputs_in_vendor_order() -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    # Deliberately exercise the observed 20 -> 80 -> 40 vendor order.
    for ordinal, side in enumerate((20, 80, 40)):
        outputs[f"conv{ordinal * 2 + 1}"] = np.ones(
            (side, side, 4), dtype=np.float32
        )
        outputs[f"conv{ordinal * 2 + 2}"] = np.full(
            (side, side, 80), -20.0, dtype=np.float32
        )
    outputs["conv4"][10, 10, 7] = 20.0
    return outputs


@pytest.mark.parametrize(
    ("model_id", "factory", "expected_orders"),
    [
        ("yolo11l", _yolo11_outputs, 1),
        ("yolo26m", _yolo26_outputs_in_vendor_order, 4),
    ],
)
def test_detection_frozen_host_tail_and_reference_parity(
    tmp_path: Path,
    model_id: str,
    factory,
    expected_orders: int,
) -> None:
    raw = factory()
    reference = _reference_bn6(model_id)
    raw_path = tmp_path / f"{model_id}_raw.npz"
    reference_path = tmp_path / f"{model_id}_reference.npz"
    np.savez(raw_path, **raw)
    np.savez(reference_path, output0=reference)

    payload = smoke.run_backend_semantic_smokes(
        _spec({
            "id": model_id,
            "kind": "detection_host_tail",
            "model_id": model_id,
            "raw_outputs": raw_path.name,
            "reference_outputs": reference_path.name,
            "input_hw": [640, 640],
            "original_wh": [640, 640],
        }),
        base_dir=tmp_path,
    )
    assert payload["status"] == "PASS", payload
    case = payload["cases"][0]
    assert case["details"]["raw_detection_count"] == 1
    assert case["details"]["reference_format"] == "bn6_detections"
    assert case["details"]["parity"]["match"] is True
    assert case["details"]["parity"]["max_score_error"] == 0.0
    assert case["details"]["parity"]["max_coordinate_error"] == 0.0
    assert len(case["details"]["order_checks"]) == expected_orders
    assert all(
        row["match"] for row in case["details"]["order_checks"]
    )
    if model_id == "yolo26m":
        assert [
            row["grid_hw"]
            for row in case["details"]["yolo26_regcls_geometry"]
        ] == [[80, 80], [40, 40], [20, 20]]
    else:
        assert case["details"]["decoder_id"] == (
            "yolo11_regcls_dfl16_classaware_nms_v1"
        )
        assert case["details"]["decoder_format"] == "ultralytics_regcls"
        assert [
            row["grid_hw"]
            for row in case["details"]["yolo11_regcls_geometry"]
        ] == [[80, 80], [40, 40], [20, 20]]


def _detection_case(**updates: object) -> dict[str, object]:
    case: dict[str, object] = {
        "id": "detection",
        "kind": "detection_host_tail",
        "model_id": "yolo11l",
        "raw_outputs": "raw.npz",
        "reference_outputs": "reference.npz",
        "input_hw": [640, 640],
        "original_wh": [640, 640],
    }
    case.update(updates)
    return case


def test_detection_requires_independent_completed_bn6_reference(
    tmp_path: Path,
) -> None:
    raw = _yolo11_outputs()
    np.savez(tmp_path / "raw.npz", **raw)

    same = smoke.run_backend_semantic_smokes(
        _spec(_detection_case(reference_outputs="raw.npz")),
        base_dir=tmp_path,
    )
    assert same["status"] == "FAIL"
    assert same["cases"][0]["reason"] == (
        "detection_reference_not_independent"
    )

    distinct_raw = _yolo11_outputs()
    distinct_raw["cls1"][0, 0, 0] = -19.0
    np.savez(tmp_path / "reference.npz", **distinct_raw)
    not_completed = smoke.run_backend_semantic_smokes(
        _spec(_detection_case()), base_dir=tmp_path
    )
    assert not_completed["status"] == "FAIL"
    assert not_completed["cases"][0]["reason"] == (
        "completed_bn6_detection_reference_required"
    )


def test_detection_rejects_completed_bn6_as_yolo11_raw_head(
    tmp_path: Path,
) -> None:
    np.savez(
        tmp_path / "raw.npz",
        output0=_reference_bn6("yolo11l"),
    )
    different_reference = _reference_bn6("yolo11l").copy()
    different_reference[0, 0, 4] = np.float32(0.99)
    np.savez(tmp_path / "reference.npz", output0=different_reference)

    payload = smoke.run_backend_semantic_smokes(
        _spec(_detection_case()), base_dir=tmp_path
    )

    assert payload["status"] == "FAIL"
    assert payload["cases"][0]["reason"] == (
        "yolo11_exact_six_regcls_tensors_required"
    )


def test_detection_rejects_nonfinite_raw_dump(tmp_path: Path) -> None:
    raw = _yolo11_outputs()
    raw["cls0"][0, 0, 0] = np.nan
    np.savez(tmp_path / "raw.npz", **raw)
    np.savez(tmp_path / "reference.npz", output0=_reference_bn6("yolo11l"))

    payload = smoke.run_backend_semantic_smokes(
        _spec(_detection_case()), base_dir=tmp_path
    )
    assert payload["status"] == "FAIL"
    assert payload["cases"][0]["reason"] == "raw_output_dump_nonfinite"


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"minimum_detections": 0}, "minimum_detections_invalid"),
        ({"score_atol": float("inf")}, "detection_smoke_tolerance_invalid"),
        (
            {"coordinate_atol": float("nan")},
            "detection_smoke_tolerance_invalid",
        ),
    ],
)
def test_detection_rejects_noop_or_nonfinite_comparison_policy(
    tmp_path: Path,
    updates: dict[str, object],
    reason: str,
) -> None:
    np.savez(tmp_path / "raw.npz", **_yolo11_outputs())
    np.savez(tmp_path / "reference.npz", output0=_reference_bn6("yolo11l"))

    payload = smoke.run_backend_semantic_smokes(
        _spec(_detection_case(**updates)), base_dir=tmp_path
    )
    assert payload["status"] == "FAIL"
    assert payload["cases"][0]["reason"] == reason


def test_cli_example_and_structured_all_skip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert smoke.main(["--example"]) == 0
    assert smoke.SPEC_SCHEMA in capsys.readouterr().out

    spec_path = tmp_path / "smokes.json"
    spec_path.write_text(
        __import__("json").dumps(_spec({
            "id": "missing",
            "kind": "detection_host_tail",
            "model_id": "yolo11l",
            "raw_outputs": "missing-raw.npz",
            "reference_outputs": "missing-ref.npz",
        })),
        encoding="utf-8",
    )
    assert smoke.main(["--spec", str(spec_path), "--compact"]) == 0
    emitted = __import__("json").loads(capsys.readouterr().out)
    assert emitted["status"] == "SKIP"
    assert emitted["hardware_invoked"] is False

    assert smoke.main([
        "--spec",
        str(spec_path),
        "--compact",
        "--require-pass",
    ]) == 1
    required = __import__("json").loads(capsys.readouterr().out)
    assert required["status"] == "SKIP"
