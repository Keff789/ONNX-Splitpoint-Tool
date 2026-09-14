from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.release_identity import BUILD_ID as CURRENT_BUILD_ID
from onnx_splitpoint_tool.v2796_smoke import BUILD_ID as V2796_BUILD_ID


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"
VENDORED = ROOT / "onnx_splitpoint_tool/resources/remote_scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"
RESOURCE = ROOT / "onnx_splitpoint_tool/resources/native_concurrent_three_stage_yolov7"


def _load_helper():
    name = "v2796_three_stage_closure_helper"
    spec = importlib.util.spec_from_file_location(name, HELPER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _invocation(module, tmp_path: Path, *, count: int = 32, claim: bool = True):
    files = {}
    for name in ("corpus_manifest.json", "dataset_manifest.json", "reference.json", "binding.json"):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        files[name] = path
    out_root = tmp_path / "out"
    out_root.mkdir()
    return module.ThreeStageInvocation(
        benchmark_set=tmp_path,
        case_id="b066",
        setup_id="orin_nx_hailo8_01",
        corpus_manifest=files["corpus_manifest.json"],
        dataset_manifest=files["dataset_manifest.json"],
        reference_report=files["reference.json"],
        quality_binding=files["binding.json"],
        out_root=out_root,
        expected_item_count=count,
        model_id="yolov7_paper",
        precision="uint8_dequant_fp16",
        execution_scope=module.CLAIM_SCOPE if claim else module.DIAGNOSTIC_SCOPE,
        claim_eligible=claim,
        corpus_manifest_sha256=_sha(files["corpus_manifest.json"]),
        dataset_manifest_sha256=_sha(files["dataset_manifest.json"]),
        reference_report_sha256=_sha(files["reference.json"]),
        quality_binding_sha256=_sha(files["binding.json"]),
        out_root_binding_sha256=module._out_root_binding_sha256(
            out_root,
            execution_scope=module.CLAIM_SCOPE if claim else module.DIAGNOSTIC_SCOPE,
            model_id="yolov7_paper",
            case_id="b066",
            setup_id="orin_nx_hailo8_01",
        ),
    )


def test_three_stage_invocation_is_frozen_and_hash_bound(tmp_path: Path):
    module = _load_helper()
    # The helper is current product code even though this inherited regression
    # file keeps its v2796 name.
    assert V2796_BUILD_ID == "v2.79.6-remaining-changes-yolo11-admission-closure"
    assert module.BUILD_ID == CURRENT_BUILD_ID
    assert CURRENT_BUILD_ID == "v2.79.13-platform-power-calibration-operational-repair"
    assert V2796_BUILD_ID != CURRENT_BUILD_ID
    assert (
        "from onnx_splitpoint_tool.release_identity import BUILD_ID"
        in HELPER.read_text(encoding="utf-8")
    )
    invocation = _invocation(module, tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        invocation.expected_item_count = 1
    receipt = module._invocation_receipt(invocation)
    assert len(receipt["invocation_sha256"]) == 64
    assert receipt["out_root_binding_sha256"] == invocation.out_root_binding_sha256


def test_three_stage_adapter_passes_corpus_manifest_file_not_directory(tmp_path: Path):
    module = _load_helper()
    invocation = _invocation(module, tmp_path)
    command = module.build_three_stage_command(
        invocation,
        script=RESOURCE / "three_stage_canary.py",
        tool_root=ROOT,
        expected_artifacts=RESOURCE / "expected_artifacts.json",
        repetitions=3,
        frames=1000,
        warmup=100,
        p1_queue_depth=3,
        post_queue_depth=4,
    )
    assert command[command.index("--corpus") + 1].endswith("corpus_manifest.json")
    assert Path(command[command.index("--corpus") + 1]).is_file()
    assert command[command.index("--reference-report") + 1] == str(invocation.reference_report)
    assert command[command.index("--out-root") + 1] == str(invocation.out_root)
    assert command[command.index("--expected-corpus-count") + 1] == "32"


def test_three_stage_adapter_binds_exact_reference_report(tmp_path: Path):
    module = _load_helper()
    invocation = _invocation(module, tmp_path)
    command = module.build_three_stage_command(
        invocation,
        script=RESOURCE / "three_stage_canary.py",
        tool_root=ROOT,
        expected_artifacts=RESOURCE / "expected_artifacts.json",
        repetitions=1,
        frames=1,
        warmup=0,
        p1_queue_depth=1,
        post_queue_depth=1,
    )
    assert command[command.index("--reference-report") + 1] == str(invocation.reference_report)
    assert invocation.reference_report_sha256 == _sha(invocation.reference_report)


def test_three_stage_adapter_binds_out_root(tmp_path: Path):
    module = _load_helper()
    invocation = _invocation(module, tmp_path)
    command = module.build_three_stage_command(
        invocation,
        script=RESOURCE / "three_stage_canary.py",
        tool_root=ROOT,
        expected_artifacts=RESOURCE / "expected_artifacts.json",
        repetitions=1,
        frames=1,
        warmup=0,
        p1_queue_depth=1,
        post_queue_depth=1,
    )
    assert command[command.index("--out-root") + 1] == str(invocation.out_root)
    assert invocation.out_root_binding_sha256


def test_three_stage_adapter_passes_dataset_manifest_to_stager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_helper()
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}\n", encoding="utf-8")
    corpus_dir = tmp_path / "corpus"
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        corpus_dir.mkdir(parents=True)
        (corpus_dir / "corpus_manifest.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="PASS\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    manifest = module._stage_claim_corpus(
        resource=RESOURCE,
        dataset_manifest=dataset,
        corpus_dir=corpus_dir,
        images_root="",
        annotations="",
        work_dir=tmp_path,
    )
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--dataset-manifest") + 1] == str(dataset)
    assert command[command.index("--count") + 1] == "32"
    assert manifest == corpus_dir / "corpus_manifest.json"


def test_yolov7_product_fixture_materializes_exactly_32_ordered_items(tmp_path: Path):
    module = _load_helper()
    dataset_sha = "a" * 64
    dataset_items = []
    corpus_items = []
    reference_rows = []
    corpus_root = tmp_path / "corpus"
    images = corpus_root / "images"
    images.mkdir(parents=True)
    for index in range(32):
        image_id = 1000 + index
        relative = f"{image_id:012d}.jpg"
        staged = images / f"{index:02d}_{relative}"
        staged.write_bytes(f"image-{index}".encode())
        digest = _sha(staged)
        dataset_items.append(
            {
                "image_id": image_id,
                "relative_path": relative,
                "sha256": digest,
                "size_bytes": staged.stat().st_size,
            }
        )
        corpus_items.append(
            {
                "index": index,
                "image_id": image_id,
                "relative_path": relative,
                "staged_file": f"images/{staged.name}",
                "staged_sha256": digest,
            }
        )
        reference_rows.append({"image_id": image_id})
    manifest = corpus_root / "corpus_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "onnx-splitpoint/yolov7-fast-decode-parity-corpus",
                "requested_count": 32,
                "selected_count": 32,
                "dataset_manifest_sha256": dataset_sha,
                "items": corpus_items,
            }
        ),
        encoding="utf-8",
    )
    payload, digest = module._validate_corpus_manifest(
        manifest,
        dataset={"items": dataset_items},
        dataset_manifest_sha256=dataset_sha,
        reference={"rows": reference_rows},
        expected_item_count=32,
    )
    assert payload["selected_count"] == 32
    assert len(digest) == 64
    with pytest.raises(RuntimeError, match="item_count_mismatch"):
        module._validate_corpus_manifest(
            manifest,
            dataset={"items": dataset_items},
            dataset_manifest_sha256=dataset_sha,
            reference={"rows": reference_rows},
            expected_item_count=1,
        )


def test_three_stage_adapter_rejects_one_item_fallback_in_product_mode(tmp_path: Path):
    module = _load_helper()
    corpus_root = tmp_path / "claim_corpus"
    images = corpus_root / "images"
    images.mkdir(parents=True)
    staged = images / "00_000000000632.jpg"
    staged.write_bytes(b"sentinel")
    staged_sha = _sha(staged)
    dataset_sha = "b" * 64
    corpus_manifest = corpus_root / "corpus_manifest.json"
    corpus_manifest.write_text(
        json.dumps(
            {
                "schema": "onnx-splitpoint/yolov7-product-smoke-corpus",
                "requested_count": 1,
                "selected_count": 1,
                "dataset_manifest_sha256": dataset_sha,
                "items": [
                    {
                        "index": 0,
                        "image_id": 632,
                        "relative_path": "000000000632.jpg",
                        "staged_file": "images/00_000000000632.jpg",
                        "staged_sha256": staged_sha,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="item_count_mismatch"):
        module._validate_corpus_manifest(
            corpus_manifest,
            dataset={
                "items": [
                    {
                        "image_id": 632,
                        "relative_path": "000000000632.jpg",
                        "sha256": staged_sha,
                    }
                ]
            },
            dataset_manifest_sha256=dataset_sha,
            reference={"rows": [{"image_id": 632}]},
            expected_item_count=32,
        )

    invocation = _invocation(module, tmp_path, count=1, claim=False)
    report_path = tmp_path / "report.json"
    report_path.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        model_id="yolov7_paper",
        case="b066",
        setup_id="orin_nx_hailo8_01",
        eval_run_id="eval",
        source_run_id="hailo8_to_trt",
        precision="uint8_dequant_fp16",
        native_split_quality_binding=str(invocation.quality_binding),
    )
    result = module._project_result(
        report={
            "status": "PASS_THREE_STAGE_TARGET_MET",
            "aggregate": {
                "raw_fps_median": 97.0,
                "completed_fps_median": 96.9,
            },
            "postflight_quality_oracle": {"all_exact": True},
        },
        args=args,
        binding={},
        report_path=report_path,
        resource=RESOURCE,
        invocation=invocation,
    )
    assert result["ok"] is True
    assert result["execution_scope"] == module.DIAGNOSTIC_SCOPE
    assert result["claim_eligible"] is False
    assert result["expected_item_count"] == 1


def test_three_stage_adapter_collects_failure_report_after_rc2(tmp_path: Path):
    module = _load_helper()
    invocation = _invocation(module, tmp_path)
    report_dir = invocation.out_root / "run"
    report_dir.mkdir()
    report_path = report_dir / "three_stage_canary_report.json"
    original_reason = "BindingMismatch: exact quality binding changed"
    report_path.write_text(
        json.dumps(
            {
                "status": "FAIL_CANARY_INFRASTRUCTURE",
                "failure_class": "BindingMismatch",
                "failure_reason": original_reason,
            }
        ),
        encoding="utf-8",
    )
    assert module._find_report_only_below(invocation.out_root) == report_path
    args = SimpleNamespace(
        model_id="yolov7_paper",
        case="b066",
        setup_id="orin_nx_hailo8_01",
        eval_run_id="eval",
        source_run_id="hailo8_to_trt",
        precision="uint8_dequant_fp16",
        native_split_quality_binding=str(invocation.quality_binding),
    )
    result = module._project_result(
        report=json.loads(report_path.read_text()),
        args=args,
        binding={},
        report_path=report_path,
        resource=RESOURCE,
        invocation=invocation,
        child_returncode=2,
    )
    assert result["ok"] is False
    assert result["child_returncode"] == 2
    assert result["failure_class"] == "BindingMismatch"
    assert result["failure_reason"] == original_reason
    assert result["out_root"] == str(invocation.out_root)


def test_three_stage_adapter_preserves_original_failure_class():
    module = _load_helper()
    failure_class, failure_reason = module._child_failure_details(
        {
            "status": "FAIL_CANARY_INFRASTRUCTURE",
            "failure_class": "ExactBindingMismatch",
            "failure_reason": "quality binding SHA-256 changed",
        }
    )
    assert failure_class == "ExactBindingMismatch"
    assert failure_reason == "quality binding SHA-256 changed"


def test_no_generic_argparse_guessing_or_silent_corpus_fallback():
    text = HELPER.read_text(encoding="utf-8")
    for forbidden in (
        "_capture_parser",
        "_semantic_value",
        "_argv_from_parser",
        "_fallback_one_image_corpus",
        "allow_synthetic_corpus",
    ):
        assert forbidden not in text
    assert 'choices=(DIAGNOSTIC_SCOPE, CLAIM_SCOPE)' in text
    assert 'claim_gate_explicit_three_stage_out_root_required' in text


def test_bundled_claim_reference_is_exactly_hash_bound_and_has_32_rows():
    module = _load_helper()
    reference = RESOURCE / "reference/multi_image_fast_decode_canary_report.json"
    dataset = RESOURCE / "reference/dataset_detection_validation.json"
    assert _sha(reference) == module.REFERENCE_REPORT_SHA256
    assert _sha(dataset) == module.DATASET_MANIFEST_SHA256
    payload = json.loads(reference.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 32
    assert payload["corpus"]["selected_count"] == 32


def test_vendored_three_stage_scripts_are_byte_identical():
    assert HELPER.read_bytes() == VENDORED.read_bytes()
