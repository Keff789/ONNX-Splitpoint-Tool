from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator():
    return _load_script(
        "v270i_native_full_validation_p0",
        Path("scripts/native_producer_validate_visualize.py"),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _full_validation_fixture(
    tmp_path: Path, *, setup_id: str = "h8",
) -> dict[str, object]:
    benchmark_set = tmp_path / "eval" / "resnet50" / "benchmark_set"
    dump_dir = (
        benchmark_set
        / "native_full_outputs"
        / "model=resnet50"
        / "backend=native_full_hailo8"
        / f"setup={setup_id}"
        / "comparison=hailo8"
    )
    dump_dir.mkdir(parents=True)

    image = benchmark_set / "images" / "sample.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"full-validation-image")
    image_sha = _sha256(image)

    input_hwc = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    input_dump = dump_dir / "input_rgb_uint8.bin"
    input_dump.write_bytes(input_hwc.tobytes())

    runtime_tensor = (
        input_hwc.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    )
    runtime_input = dump_dir / "runtime_input.bin"
    runtime_input.write_bytes(runtime_tensor.tobytes())

    logits = np.asarray([[0.1, 0.2, 9.0, 0.4, 0.3]], dtype=np.float32)
    output = dump_dir / "output_00_logits.bin"
    output.write_bytes(logits.tobytes())

    input_manifest = dump_dir / "native_full_input_manifest.json"
    input_payload = {
        "schema": "onnx-splitpoint/native-full-input-dump",
        "schema_version": 2,
        "backend": "native_full_hailo8",
        "model": "resnet50",
        "setup_id": setup_id,
        "comparison_backend": "hailo8",
        "case": "full",
        "image": str(image),
        "image_sha256": image_sha,
        "input_image": str(image),
        "input_image_sha256": image_sha,
        "input_dump": input_dump.name,
        "input_dump_sha256": _sha256(input_dump),
        "input_dump_bytes": input_dump.stat().st_size,
        "input_shape_hwc": [2, 2, 3],
        "runtime_input_name": "input",
        "runtime_input_shape": [1, 3, 2, 2],
        "runtime_input_dtype": "float32",
        "runtime_input_file": runtime_input.name,
        "runtime_input_sha256": _sha256(runtime_input),
        "runtime_input_bytes": runtime_input.stat().st_size,
        "preprocess": {
            "layout": "NCHW",
            "ort_model_scale": "norm",
        },
    }
    _write_json(input_manifest, input_payload)

    output_manifest = dump_dir / "native_full_outputs_manifest.json"
    output_payload = {
        "schema": "onnx-splitpoint/runner-output-dump",
        "schema_version": 4,
        "producer": "native_full_hailo8",
        "backend": "native_full_hailo8",
        "model": "resnet50",
        "setup_id": setup_id,
        "comparison_backend": "hailo8",
        "case": "full",
        "execution_mode": "native_full_baseline",
        "task": "classification",
        "stage": "classification_logits",
        "contract_family": "classification_logits",
        "input_image": str(image),
        "input_image_sha256": image_sha,
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": _sha256(input_manifest),
        # Deliberately no boundary_manifest/native_boundary_manifest aliases.
        "outputs": [
            {
                "name": "logits",
                "file": output.name,
                "dtype": "float32",
                "shape": [1, 5],
                "bytes": output.stat().st_size,
            }
        ],
    }
    _write_json(output_manifest, output_payload)

    model = benchmark_set / "models" / "resnet50.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"test-onnx-placeholder")
    return {
        "benchmark_set": benchmark_set,
        "dump_dir": dump_dir,
        "input_manifest": input_manifest,
        "input_payload": input_payload,
        "output_manifest": output_manifest,
        "output_payload": output_payload,
        "runtime_tensor": runtime_tensor,
        "logits": logits,
        "model": model,
    }


def test_native_full_self_reference_uses_full_input_manifest_without_boundary(
    validator, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fixture = _full_validation_fixture(tmp_path)
    output_manifest = fixture["output_manifest"]
    input_manifest = fixture["input_manifest"]
    logits = fixture["logits"]

    found, kind, reason = validator._find_self_reference_input_manifest(
        output_manifest,
        roots=[tmp_path],
    )
    assert found == input_manifest
    assert kind == "native_full_input_manifest"
    assert reason == ""

    class FakeSession:
        def __init__(self, model_path: str, providers: list[str]):
            assert Path(model_path) == fixture["model"]
            assert providers == ["CPUExecutionProvider"]

        @staticmethod
        def get_inputs():
            return [
                SimpleNamespace(
                    name="input",
                    shape=[1, 3, 2, 2],
                    type="tensor(float)",
                )
            ]

        @staticmethod
        def get_outputs():
            return [SimpleNamespace(name="logits")]

        @staticmethod
        def run(_outputs, feeds):
            assert list(feeds) == ["input"]
            assert feeds["input"].shape == (1, 3, 2, 2)
            return [np.asarray(logits)]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )
    monkeypatch.setattr(
        validator,
        "load_dump",
        lambda _manifest: ({"logits": np.asarray(logits)}, {}),
    )

    result = validator._full_onnx_self_reference_classification(
        output_manifest,
        fixture["benchmark_set"],
        roots=[tmp_path],
    )

    assert result["available"] is True
    assert result["semantic_available"] is True
    assert result["semantic_ok"] is True
    assert result["self_reference_input_manifest_kind"] == (
        "native_full_input_manifest"
    )
    assert Path(result["boundary_manifest"]) == input_manifest


def test_collected_hailo_full_chain_rebases_smoke_remote_manifest_binding(
    validator, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Replay the exact remote path shape collected by the 7-Aug Smoke."""
    setup_id = "orin_nx_hailo8_01"
    fixture = _full_validation_fixture(tmp_path, setup_id=setup_id)
    output_manifest = Path(fixture["output_manifest"])
    input_manifest = Path(fixture["input_manifest"])
    logits = np.asarray(fixture["logits"])
    remote_input_manifest = (
        "/home/nx/native_fifo_evalsets/"
        "resnet_yolo26s_yolo7_20260807_103543/resnet50/benchmark_set/"
        "native_full_outputs/model=resnet50/backend=native_full_hailo8/"
        "setup=orin_nx_hailo8_01/comparison=hailo8/"
        "native_full_input_manifest.json"
    )
    output_payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    output_payload["input_manifest"] = remote_input_manifest
    output_payload.pop("input_manifest_sha256")
    _write_json(output_manifest, output_payload)

    command_contract = {
        "schema": "onnx-splitpoint/native-full-command-contract",
        "schema_version": 1,
        "backend": "native_full_hailo8",
        "model": "resnet50",
        "case": "full",
        "setup_id": setup_id,
        "comparison_backend": "hailo8",
        "complete": True,
        "artifacts": {
            "input_manifest": {
                "path": remote_input_manifest,
                "sha256": _sha256(input_manifest),
            },
        },
    }
    command_sha = validator._canonical_json_sha256(command_contract)
    command_contract["contract_sha256"] = command_sha
    evidence = {
        "full_command_contract": command_contract,
        "full_command_contract_sha256": command_sha,
    }

    class FakeSession:
        def __init__(self, model_path: str, providers: list[str]):
            assert Path(model_path) == fixture["model"]
            assert providers == ["CPUExecutionProvider"]

        @staticmethod
        def get_inputs():
            return [SimpleNamespace(
                name="input", shape=[1, 3, 2, 2], type="tensor(float)",
            )]

        @staticmethod
        def get_outputs():
            return [SimpleNamespace(name="logits")]

        @staticmethod
        def run(_outputs, feeds):
            assert feeds["input"].shape == (1, 3, 2, 2)
            return [logits]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )
    monkeypatch.setattr(
        validator,
        "load_dump",
        lambda _manifest: ({"logits": logits}, {}),
    )

    result = validator._full_onnx_self_reference_classification(
        output_manifest,
        Path(fixture["benchmark_set"]),
        roots=[tmp_path],
        endpoint_evidence=evidence,
    )

    assert result["available"] is True
    assert result["semantic_ok"] is True
    assert Path(result["boundary_manifest"]) == input_manifest

    evidence["full_command_contract_sha256"] = "0" * 64
    found, kind, reason = validator._find_self_reference_input_manifest(
        output_manifest,
        roots=[tmp_path],
        endpoint_evidence=evidence,
    )
    assert found is None
    assert kind == "native_full_input_manifest"
    assert reason == "native_full_input_manifest_missing_or_invalid"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model", "other_model"),
        ("backend", "native_full_hailo10h"),
        ("setup_id", "other_setup"),
        ("comparison_backend", "hailo10h"),
        ("input_image_sha256", "0" * 64),
        ("runtime_input_name", ""),
        ("runtime_input_shape", [1, 3, 2, 3]),
        ("runtime_input_dtype", "float16"),
        ("runtime_input_sha256", "1" * 64),
        ("runtime_input_bytes", 47),
        ("input_dump_sha256", "2" * 64),
        ("input_dump_bytes", 11),
    ],
)
def test_native_full_input_manifest_identity_and_tensor_tampering_fails_closed(
    validator,
    tmp_path: Path,
    field: str,
    bad_value: object,
) -> None:
    fixture = _full_validation_fixture(tmp_path)
    input_manifest = fixture["input_manifest"]
    payload = json.loads(input_manifest.read_text(encoding="utf-8"))
    payload[field] = bad_value
    _write_json(input_manifest, payload)
    # Keep the outer manifest binding valid so each case exercises the named
    # identity/tensor field rather than failing only on a stale envelope hash.
    output_manifest = fixture["output_manifest"]
    output_payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    output_payload["input_manifest_sha256"] = _sha256(input_manifest)
    _write_json(output_manifest, output_payload)

    found, kind, reason = validator._find_self_reference_input_manifest(
        output_manifest,
        roots=[tmp_path],
    )

    assert found is None
    assert kind == "native_full_input_manifest"
    assert reason == "native_full_input_manifest_missing_or_invalid"


def test_native_full_input_manifest_envelope_hash_tampering_fails_closed(
    validator, tmp_path: Path,
) -> None:
    fixture = _full_validation_fixture(tmp_path)
    output_manifest = fixture["output_manifest"]
    output_payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    output_payload["input_manifest_sha256"] = "f" * 64
    _write_json(output_manifest, output_payload)

    found, kind, reason = validator._find_self_reference_input_manifest(
        output_manifest,
        roots=[tmp_path],
    )

    assert found is None
    assert kind == "native_full_input_manifest"
    assert reason == "native_full_input_manifest_missing_or_invalid"


def test_split_validation_never_borrows_native_full_input_manifest(
    validator, tmp_path: Path,
) -> None:
    full = _full_validation_fixture(tmp_path)
    split_manifest = (
        full["benchmark_set"]
        / "native_pipeline"
        / "b052"
        / "hailo_to_trt"
        / "uint8_dequant_fp16"
        / "native_outputs"
        / "native_outputs_manifest.json"
    )
    _write_json(
        split_manifest,
        {
            "schema": "onnx-splitpoint/runner-output-dump",
            "schema_version": 4,
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": "b052",
            "execution_mode": "native_split",
            # Even an explicit pointer to Full evidence cannot satisfy Split.
            "input_manifest": str(full["input_manifest"]),
            "outputs": [],
        },
    )

    found, kind, reason = validator._find_self_reference_input_manifest(
        split_manifest,
        roots=[tmp_path],
    )

    assert found is None
    assert kind == "split_boundary_manifest"
    assert reason == "boundary_manifest_missing"
    assert validator._find_boundary_manifest_for_output(
        split_manifest,
        roots=[tmp_path],
    ) is None


def test_validator_keeps_runtime_and_buildability_when_semantics_fail(
    validator, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fixture = _full_validation_fixture(tmp_path)
    summary = tmp_path / "native_producer_summary.json"
    report = tmp_path / "native_full_report.json"
    _write_json(report, {"ok": True})
    _write_json(
        summary,
        {
            "rows": [
                {
                    "ok": True,
                    "buildable": True,
                    "runtime_ok": True,
                    "runtime_executable": True,
                    "backend": "native_full_hailo8",
                    "model": "resnet50",
                    "case": "full",
                    "setup_id": "h8",
                    "comparison_backend": "hailo8",
                    "execution_mode": "native_full_baseline",
                    "output_dump_manifest": str(fixture["output_manifest"]),
                    "native_report": str(report),
                }
            ]
        },
    )
    out_dir = tmp_path / "validation"

    monkeypatch.setattr(
        validator,
        "_eval_root_from_summary",
        lambda _summary: fixture["benchmark_set"],
    )
    monkeypatch.setattr(validator, "_native_split_authority", lambda _root: {})
    monkeypatch.setattr(validator, "_find_report", lambda _row, _roots: report)
    monkeypatch.setattr(
        validator,
        "_find_dump",
        lambda _report, _roots, _row: (
            fixture["output_manifest"],
            "full_row_explicit_manifest",
        ),
    )
    monkeypatch.setattr(
        validator,
        "_find_reference_report",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        validator,
        "_validate_hailo_trt_interface_contract",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        validator,
        "_native_full_e2e_contract_gate",
        lambda *_args, **_kwargs: {
            "e2e_claim_eligible": True,
            "e2e_scope": "full_task_pipeline",
            "contract_consistent": True,
        },
    )
    monkeypatch.setattr(
        validator,
        "_validate_tensor_dump",
        lambda *_args, **_kwargs: {"ok": True, "output_count": 1},
    )

    def semantic_failure(_manifest, row_dir, _topk, _reference):
        _write_json(
            row_dir / "classification_validation.json",
            {
                "ok": False,
                "semantic_available": True,
                "semantic_ok": False,
                "top1_match": False,
                "top5_overlap": 0,
            },
        )
        (row_dir / "classification_validation.md").write_text(
            "# semantic mismatch\n",
            encoding="utf-8",
        )
        return {
            "ok": False,
            "semantic_available": True,
            "semantic_ok": False,
            "top1_match": False,
            "top5_overlap": 0,
        }

    monkeypatch.setattr(
        validator,
        "_validate_classification",
        semantic_failure,
    )
    monkeypatch.setattr(
        validator,
        "_full_onnx_self_reference_classification",
        lambda *_args, **_kwargs: {
            "available": False,
            "reason": "deliberate_semantic_test_failure",
        },
    )
    monkeypatch.setattr(
        validator,
        "_bind_central_quality_evidence",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        validator,
        "_enforce_historical_split_diagnostic_only",
        lambda _row: None,
    )
    monkeypatch.setattr(
        validator,
        "apply_native_split_quality_authority",
        None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_producer_validate_visualize.py",
            "--summary",
            str(summary),
            "--out-dir",
            str(out_dir),
        ],
    )

    assert validator.main() == 0
    payload = json.loads(
        (out_dir / "native_producer_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["row_count"] == 1
    assert payload["buildable_count"] == 1
    assert payload["runtime_executable_count"] == 1
    assert payload["semantic_fail_count"] == 1
    row = payload["rows"][0]
    assert row["buildable"] is True
    assert row["runtime_executable"] is True
    assert row["semantic_available"] is True
    assert row["semantic_ok"] is False
    assert row["gate_status"] != "not_buildable"
    assert row["accuracy_gate_reason"] != "not_buildable"
