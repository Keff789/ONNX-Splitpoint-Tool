from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from PIL import Image

from onnx_splitpoint_tool.benchmark.remote_run import (
    _remote_result_collect_script,
)
from onnx_splitpoint_tool.runners.native_full_input import (
    prepare_and_seal_deepx_native_full_input,
)
from onnx_splitpoint_tool.workflow.execution_binding import (
    _materialize_deepx_prepared_input,
)
from onnx_splitpoint_tool.workflow.native_transfer import (
    build_native_transfer_inventory,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v27510_{path.stem}_{id(path)}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deepx_contract() -> dict[str, object]:
    return {
        "input": {
            "name": "images",
            "shape": [1, 4, 4, 3],
            "dtype": "uint8",
            "layout": "NHWC",
            "color_space": "RGB",
            "normalization": "embedded_dxcom_preprocessing",
            "preprocess_mode": "letterbox",
            "letterbox_pad_value": 114,
        }
    }


def _prepare_remote_deepx_result(tmp_path: Path) -> tuple[Path, dict]:
    suite = tmp_path / "remote_suite"
    image = suite / "resources/validation.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (7, 3), color=(11, 22, 33)).save(image)
    prepared = suite / "results/deepx_m1_full/prepared_input"
    sealed = prepare_and_seal_deepx_native_full_input(
        image_path=image,
        input_contract=_deepx_contract(),
        task="detection",
        out_dir=prepared,
        model="yolo26s",
        setup_id="orin_nx_deepx_m1_01",
        comparison_backend="deepx",
    )
    payload = sealed["payload"]
    runtime = prepared / "runtime_input.bin"
    row = {
        "run_id": "deepx_m1_full",
        "backend": "deepx_m1",
        "provider": "deepx_m1",
        "variant": "full",
        "task": "detection",
        "deepx_prepared_feed_benchmark": {
            "status": "ok",
            "task": "detection",
            "prepared_input_binding_verified": True,
            "prepared_input_manifest_sha256": sealed["manifest_sha256"],
            "prepared_input_sha256": hashlib.sha256(
                runtime.read_bytes()
            ).hexdigest(),
            "prepared_input_file_sha256": hashlib.sha256(
                runtime.read_bytes()
            ).hexdigest(),
            "prepared_input_bytes": runtime.stat().st_size,
            "prepared_input_source_image_sha256": payload[
                "input_image_sha256"
            ],
            "input_contract": {"model_id": "yolo26s"},
        },
    }
    (suite / "benchmark_results_deepx_m1_full_auto.json").write_text(
        json.dumps([row]), encoding="utf-8",
    )
    # These files must never cross the narrow portable-input handoff.
    (prepared / "rogue.bin").write_bytes(b"not-a-contract-role")
    (prepared.parent / "model.dxnn").write_bytes(b"compiled-model")
    return suite, sealed


def test_deepx_prepared_input_is_collected_validated_and_native_transferable(
    tmp_path: Path,
) -> None:
    remote_suite, sealed = _prepare_remote_deepx_result(tmp_path)
    local_run = tmp_path / "downloaded_run"
    downloaded = local_run / "results"
    script = _remote_result_collect_script(
        remote_results_dir=str(downloaded),
        remote_suite_dir=str(remote_suite),
    )
    subprocess.run(["bash", "-lc", script], check=True)

    source = downloaded / "deepx_m1_full/prepared_input"
    expected_roles = {
        "native_full_input_manifest.json",
        "runtime_input.bin",
        "input_rgb_uint8.bin",
    }
    assert {path.name for path in source.iterdir()} == expected_roles
    for role in expected_roles:
        assert (source / role).read_bytes() == (
            Path(sealed["manifest_path"]).parent / role
        ).read_bytes()
    assert not (downloaded / "deepx_m1_full/model.dxnn").exists()

    authoritative_suite = tmp_path / "model/benchmark_set/legacy_suite"
    result_dir = tmp_path / "evaluation_results"
    receipt = _materialize_deepx_prepared_input(
        remote_local_run_dir=local_run,
        suite_dir=authoritative_suite,
        result_dir=result_dir,
        model_id="yolo26s",
        target_id="orin_nx_deepx_m1_01",
    )
    assert receipt["status"] == "verified_exact"
    assert Path(receipt["receipt_path"]).is_file()
    destination = (
        authoritative_suite / "results/deepx_m1_full/prepared_input"
    )
    assert {path.name for path in destination.iterdir()} == expected_roles
    inventory = build_native_transfer_inventory(authoritative_suite)
    for role in expected_roles:
        assert (
            f"results/deepx_m1_full/prepared_input/{role}"
            in inventory["relative_paths"]
        )


def test_normal_deepx_prepared_input_still_requires_one_performance_row(
    tmp_path: Path,
) -> None:
    remote_suite, _sealed = _prepare_remote_deepx_result(tmp_path)
    local_run = tmp_path / "downloaded_run"
    downloaded = local_run / "results"
    script = _remote_result_collect_script(
        remote_results_dir=str(downloaded),
        remote_suite_dir=str(remote_suite),
    )
    subprocess.run(["bash", "-lc", script], check=True)
    for path in downloaded.glob("benchmark_results_*.json"):
        path.unlink()

    with pytest.raises(
        RuntimeError,
        match="deepx_prepared_input_canonical_row_count:0",
    ):
        _materialize_deepx_prepared_input(
            remote_local_run_dir=local_run,
            suite_dir=tmp_path / "authoritative_suite",
            result_dir=tmp_path / "evaluation_results",
            model_id="yolo26s",
            target_id="orin_nx_deepx_m1_01",
        )


def test_deepx_contract_without_model_id_is_rejected_with_specific_reason(
    tmp_path: Path,
) -> None:
    remote_suite, _sealed = _prepare_remote_deepx_result(tmp_path)
    result_path = remote_suite / "benchmark_results_deepx_m1_full_auto.json"
    rows = json.loads(result_path.read_text(encoding="utf-8"))
    rows[0]["deepx_prepared_feed_benchmark"]["input_contract"].pop(
        "model_id"
    )
    result_path.write_text(json.dumps(rows), encoding="utf-8")

    local_run = tmp_path / "downloaded_run"
    subprocess.run(
        ["bash", "-lc", _remote_result_collect_script(
            remote_results_dir=str(local_run / "results"),
            remote_suite_dir=str(remote_suite),
        )],
        check=True,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "deepx_prepared_input_result_binding_invalid:"
            "input_contract_model_id_mismatch"
        ),
    ):
        _materialize_deepx_prepared_input(
            remote_local_run_dir=local_run,
            suite_dir=tmp_path / "authoritative_suite",
            result_dir=tmp_path / "evaluation_results",
            model_id="yolo26s",
            target_id="orin_nx_deepx_m1_01",
        )


def test_deepx_contract_conflicting_model_id_is_rejected_with_specific_reason(
    tmp_path: Path,
) -> None:
    remote_suite, _sealed = _prepare_remote_deepx_result(tmp_path)
    result_path = remote_suite / "benchmark_results_deepx_m1_full_auto.json"
    rows = json.loads(result_path.read_text(encoding="utf-8"))
    rows[0]["deepx_prepared_feed_benchmark"]["input_contract"][
        "model_id"
    ] = "different_model"
    result_path.write_text(json.dumps(rows), encoding="utf-8")

    local_run = tmp_path / "downloaded_run"
    subprocess.run(
        ["bash", "-lc", _remote_result_collect_script(
            remote_results_dir=str(local_run / "results"),
            remote_suite_dir=str(remote_suite),
        )],
        check=True,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "deepx_prepared_input_result_binding_invalid:"
            "input_contract_model_id_mismatch"
        ),
    ):
        _materialize_deepx_prepared_input(
            remote_local_run_dir=local_run,
            suite_dir=tmp_path / "authoritative_suite",
            result_dir=tmp_path / "evaluation_results",
            model_id="yolo26s",
            target_id="orin_nx_deepx_m1_01",
        )


@pytest.mark.parametrize("mutation", ["missing", "tampered", "extra"])
def test_deepx_prepared_input_materialization_fails_before_native_on_tamper(
    tmp_path: Path, mutation: str,
) -> None:
    remote_suite, _sealed = _prepare_remote_deepx_result(tmp_path)
    local_run = tmp_path / "downloaded_run"
    downloaded = local_run / "results"
    subprocess.run(
        ["bash", "-lc", _remote_result_collect_script(
            remote_results_dir=str(downloaded),
            remote_suite_dir=str(remote_suite),
        )],
        check=True,
    )
    source = downloaded / "deepx_m1_full/prepared_input"
    if mutation == "missing":
        (source / "runtime_input.bin").unlink()
    elif mutation == "tampered":
        path = source / "runtime_input.bin"
        data = bytearray(path.read_bytes())
        data[0] ^= 0xFF
        path.write_bytes(data)
    else:
        (source / "unexpected.bin").write_bytes(b"unexpected")

    authoritative_suite = tmp_path / "model/benchmark_set/legacy_suite"
    with pytest.raises(RuntimeError, match="deepx_prepared_input"):
        _materialize_deepx_prepared_input(
            remote_local_run_dir=local_run,
            suite_dir=authoritative_suite,
            result_dir=tmp_path / "evaluation_results",
            model_id="yolo26s",
            target_id="orin_nx_deepx_m1_01",
        )
    assert not (
        authoritative_suite / "results/deepx_m1_full/prepared_input"
    ).exists()


def test_completed_v2_projection_preserves_persisted_result_identity() -> None:
    visualizer = _load_script("native_producer_validate_visualize.py")
    source = {
        "completed_task_result_artifact_saved": True,
        "completed_task_result_artifact": {"detections": []},
        "completed_task_result_artifact_path": "/portable/completed.json",
        "completed_task_result_artifact_sha256": "a" * 64,
        "completed_task_result_artifact_file_sha256": "b" * 64,
        "completed_task_result_artifact_verification_status": "verified",
    }
    target: dict = {}
    visualizer._copy_completed_v2_projection(target, source)
    assert target == source


def test_dequant_sweep_requires_exact_hash_and_remains_diagnostic() -> None:
    sweep = _load_script("native_boundary_dequant_sweep.py")
    assert sweep._same_input_identity(
        "/a/image.jpg", "a" * 64, "/b/image.jpg", "a" * 64,
    ) is True
    assert sweep._same_input_identity(
        "/a/image.jpg", "a" * 64, "/b/image.jpg", "b" * 64,
    ) is False
    assert sweep._same_input_identity(
        "/a/image.jpg", "a" * 64, "/b/other.jpg", "a" * 64,
    ) is False
    source = (ROOT / "scripts/native_boundary_dequant_sweep.py").read_text(
        encoding="utf-8",
    )
    assert "'diagnostic_only': True" in source
    assert "'claim_eligible': False" in source
    assert "'production_contract_eligible': False" in source
    assert "'selection_export_permitted': False" in source


def test_wrong_image_detection_reference_is_unavailable_not_zero_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    visualizer = _load_script("native_producer_validate_visualize.py")
    native_image = tmp_path / "000000005600.jpg"
    external_image = tmp_path / "000000308165.jpg"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(native_image)
    Image.new("RGB", (8, 8), color=(9, 8, 7)).save(external_image)
    manifest = tmp_path / "native_manifest.json"
    manifest.write_text(json.dumps({
        "model": "yolo26s",
        "input_image": str(native_image),
        "input_image_sha256": hashlib.sha256(
            native_image.read_bytes()
        ).hexdigest(),
    }), encoding="utf-8")
    reference = tmp_path / "validation_report.json"
    reference.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "visual"
    out_dir.mkdir()

    monkeypatch.setattr(
        visualizer, "load_dump",
        lambda _path: ({"detections": np.zeros((1, 1, 6), dtype=np.float32)}, {}),
    )
    monkeypatch.setattr(
        visualizer, "_reference_detections",
        lambda _path: ([{
            "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
            "score": 0.9, "class_id": 1,
        }], external_image.name),
    )

    def find_image(_root, image_name, **_kwargs):
        return native_image if Path(str(image_name)).name == native_image.name else external_image

    monkeypatch.setattr(visualizer, "_find_image", find_image)
    monkeypatch.setattr(
        visualizer, "_choose_detection_candidate",
        lambda *_args, **_kwargs: ([{
            "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
            "score": 0.9, "class_id": 1,
        }], "decoded_nms_bn6", {"reason": "ok", "candidates": []}),
    )
    monkeypatch.setattr(
        visualizer, "_decode_sweep",
        lambda *_args, **_kwargs: {"sweep": []},
    )
    monkeypatch.setattr(
        visualizer, "_full_onnx_self_reference_detection",
        lambda *_args, **_kwargs: {
            "available": False, "reason": "fixture_unavailable",
        },
    )
    monkeypatch.setattr(visualizer, "_save_png_gray", lambda *_a, **_k: None)
    monkeypatch.setattr(visualizer, "_draw_boxes", lambda *_a, **_k: None)

    report = visualizer._visualize_detection(
        manifest, out_dir, 0, reference, tmp_path,
    )
    assert report["semantic_available"] is False
    assert report["semantic_ok"] == "unavailable"
    assert report["semantic_reference_source"] == "none"
    assert report["external_reference_input_mismatch"] is True
    assert report["external_reference_detection_count"] == 1
    assert report["match"]["ref_count"] == 0
    assert report["numerical_similarity_reason"] == (
        "reference_input_identity_mismatch"
    )
