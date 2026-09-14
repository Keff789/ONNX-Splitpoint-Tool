from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from onnx_splitpoint_tool.native_performance_reporting import (
    collect_native_performance_matrix,
)
from onnx_splitpoint_tool.runners.native_full_input import (
    prepare_and_seal_deepx_native_full_input,
)
from onnx_splitpoint_tool.validation.accuracy_gates import (
    apply_accuracy_gate_to_row,
)
from scripts import native_full_semantic_dump as semantic_dump
from scripts import native_producer_final_report as producer_collector
from scripts import native_producer_validate_visualize as validator


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests/fixtures/v27518/deepx_full_sealed_chain_resnet50.json"
)


def test_real_deepx_sealed_v3_writer_to_report_chain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Reproduce the archived DeepX chain with a reduced real-smoke fixture."""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    archived = fixture["archived_evidence"]
    reduced = fixture["reduced_fixture"]
    assert archived["prepared_feed_contract_version"] == (
        "deepx-sealed-runtime-input-v3"
    )
    assert archived["performance_input_contract_mode"] == "explicit"
    assert archived["archived_output_input_contract_mode"] == (
        "diagnostic_autodetect"
    )

    image = tmp_path / "source.png"
    Image.fromarray(
        np.asarray(reduced["rgb_pixels"], dtype=np.uint8), mode="RGB",
    ).save(image)
    contract = {
        "input": {
            "name": archived["prepared_input_name"],
            "shape": reduced["input_shape"],
            "dtype": reduced["input_dtype"],
            "layout": reduced["input_layout"],
            "normalization": reduced["input_normalization"],
            "color_space": "RGB",
            "preprocess_mode": reduced["input_preprocess_mode"],
            "letterbox_pad_value": 0,
        },
        "outputs": [{"name": "logits"}],
    }
    sealed = prepare_and_seal_deepx_native_full_input(
        image_path=image,
        input_contract=contract,
        task=archived["task"],
        out_dir=tmp_path / "sealed",
        model=archived["model"],
        setup_id=archived["setup_id"],
        comparison_backend=archived["comparison_backend"],
    )
    dxnn = tmp_path / "model.dxnn"
    dxnn.write_bytes(b"derived-deepx-fixture")
    logits = np.asarray(
        reduced["classification_logits"], dtype=np.float32,
    )

    class FakeInferenceEngine:
        def __init__(self, path: str) -> None:
            assert path == str(dxnn)

        def run(self, inputs):
            assert len(inputs) == 1
            assert list(np.asarray(inputs[0]).shape) == reduced["input_shape"]
            return [logits]

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
    endpoint_hash = "a" * 64

    def classification_contract(*_args, **_kwargs):
        return {
            "task": "classification",
            "output_format": "classification_logits",
            "contract_family": "classification_logits",
            "stage": "classification_logits",
            "contract_source": (
                "authoritative_suite_contract_plus_runtime_tensor:v4"
            ),
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": endpoint_hash,
            "output_endpoint_attestation": {
                "attested": True,
                "status": "passed",
                "stage": "classification_logits",
                "endpoint_contract_hash": endpoint_hash,
            },
            "e2e_scope": "full_task_pipeline",
            "claim_eligible_e2e": True,
        }

    monkeypatch.setattr(semantic_dump, "_contract", classification_contract)
    semantic_result = semantic_dump._run_deepx(
        tmp_path,
        image,
        tmp_path / "semantic",
        archived["model"],
        archived["task"],
        archived["setup_id"],
        archived["comparison_backend"],
        prepared_input_manifest=Path(sealed["manifest_path"]),
    )
    manifest = Path(semantic_result["output_manifest"])
    serialized = json.loads(manifest.read_text(encoding="utf-8"))
    assert semantic_result["input_candidate"] == (
        archived["semantic_input_candidate"]
    )
    assert serialized["provenance"]["runtime"]["input_contract_mode"] == (
        "explicit"
    )
    assert serialized["input_contract_mode"] == "explicit"

    # The collector preserves the independent performance-row assertion, and
    # the final validator requires both copies to agree before admitting E2E.
    serialized_performance_row = {
        "backend": "native_full_deepx",
        "model": archived["model"],
        "case": "full",
        "task": archived["task"],
        "setup_id": archived["setup_id"],
        "comparison_backend": archived["comparison_backend"],
        "execution_mode": "native_full_baseline",
        "precision": "uint8_cast_fp16",
        "performance_input_contract_mode": archived[
            "performance_input_contract_mode"
        ],
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "contract_family": "classification_logits",
        "stage": "classification_logits",
        "buildable": True,
        "runtime_executable": True,
        "status": "ok",
        "ok": True,
        "fps_makespan": archived["fps_makespan"],
        "outer_makespan_verified": True,
        "native_output_manifest": str(manifest),
    }
    producer_root = tmp_path / "producer"
    analysis = producer_root / "analysis_tables"
    analysis.mkdir(parents=True)
    (analysis / "native_full_baseline_eval.json").write_text(
        json.dumps({"rows": [serialized_performance_row]}),
        encoding="utf-8",
    )
    collected = producer_collector._rows_from_native_full(producer_root)
    assert len(collected) == 1
    row = collected[0]
    assert row["performance_input_contract_mode"] == "explicit"
    mode, projection_status = (
        validator._project_performance_input_contract_mode(
            row, {}, serialized,
        )
    )
    row["performance_input_contract_mode"] = mode
    row["performance_input_contract_mode_projection_status"] = (
        projection_status
    )
    row.update(
        validator._native_full_e2e_contract_gate(
            manifest, row, archived["task"],
        )
    )
    apply_accuracy_gate_to_row(row)
    assert projection_status == "projected_consistent"
    assert row["e2e_scope"] == "full_task_pipeline"
    assert row["e2e_claim_eligible"] is True
    assert row["structural_contract_status"] == "pass"
    assert row["structural_contract_pass"] is True

    reports = tmp_path / "report_chain/reports"
    reports.mkdir(parents=True)
    payload = {"schema": "test", "schema_version": 1, "rows": [row]}
    (reports / "native_stage_concise_summary.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    (reports / "native_producer_summary.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    matrix = collect_native_performance_matrix(tmp_path / "report_chain")
    observation = matrix["observations"][0]
    assert observation["e2e_scope"] == "full_task_pipeline"
    assert observation["structural_contract_status"] == "pass"
    assert observation["structural_contract_pass"] is True


def test_deepx_unverified_probe_sources_remain_diagnostic() -> None:
    assert semantic_dump._deepx_input_contract_mode("contract") == "explicit"
    assert semantic_dump._deepx_input_contract_mode(
        "shared_pre_timing_sealed_manifest"
    ) == "explicit"
    for source in ("", "hwc_uint8", "nhwc_uint8", "unknown_future_source"):
        assert semantic_dump._deepx_input_contract_mode(source) == (
            "diagnostic_autodetect"
        )
