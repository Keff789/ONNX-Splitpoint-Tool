from __future__ import annotations

import importlib.util
from collections import Counter
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_performance_reporting import (
    collect_native_performance_matrix,
)
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy
from onnx_splitpoint_tool.workflow.checkpoints import (
    native_coordinator_input_hash,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"test_v276_{path.stem}", path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quality_policy(name: str = "frozen_fixture") -> dict[str, object]:
    return {
        "schema": "onnx-splitpoint/task-quality-policy",
        "schema_version": 3,
        "name": name,
        "profile_id": name,
    }


def _minimal_run(tmp_path: Path) -> Path:
    run = tmp_path / "evalrun"
    (run / "reports").mkdir(parents=True)
    (run / "native_producers" / "variant").mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"run_id": run.name}), encoding="utf-8",
    )
    (run / "profile.yaml").write_text(
        json.dumps({"quality_gate": _quality_policy()}), encoding="utf-8",
    )
    (run / "quality_management").mkdir()
    (run / "quality_management" / "central_quality_summary.json").write_text(
        json.dumps({"results": []}), encoding="utf-8",
    )
    (run / "reports" / "native_expected_matrix.json").write_text(
        json.dumps({
            "expected_row_count": 24,
            "present_expected_row_count": 24,
            "successful_expected_row_count": 24,
            "row_presence_complete": True,
            "execution_success_complete": True,
        }),
        encoding="utf-8",
    )
    (run / "reports" / "native_producer_stage.json").write_text(
        json.dumps({"status": "failed_after_measurement"}), encoding="utf-8",
    )
    return run


def test_native_policy_is_inherited_and_drift_is_rejected(
    tmp_path: Path,
) -> None:
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    run = _minimal_run(tmp_path)
    cfg = {
        "validation": {"enabled": True},
        "_workflow_context": {"execution_preset": {"id": "standard"}},
    }

    effective, source = coordinator._bind_evalrun_quality_gate_policy(
        run, cfg,
    )
    assert source == "inherited_from_evalrun_profile"
    assert effective["quality_gate_policy"] == _quality_policy()
    assert "quality_gate_policy" not in cfg

    drifted = dict(cfg)
    drifted["quality_gate_policy"] = _quality_policy("different")
    with pytest.raises(ValueError, match="drifts"):
        coordinator._bind_evalrun_quality_gate_policy(run, drifted)


def test_native_coordinator_checkpoint_hash_separates_quality_policies(
    tmp_path: Path,
) -> None:
    run = _minimal_run(tmp_path)
    config = tmp_path / "native.json"
    config.write_text("{}", encoding="utf-8")

    first = native_coordinator_input_hash(
        run, config, quality_gate_policy_sha256="a" * 64,
    )
    second = native_coordinator_input_hash(
        run, config, quality_gate_policy_sha256="b" * 64,
    )
    assert first != second


def test_offline_native_replay_rejects_23_of_24_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _load_script("replay_native_supplement_reports.py")
    run = _minimal_run(tmp_path)
    destination = tmp_path / "projection"

    native_rows = [{"ok": True, "case": f"b{index:03d}"}
                   for index in range(24)]
    (run / "reports" / "native_producer_summary.json").write_text(
        json.dumps({"rows": native_rows}), encoding="utf-8",
    )
    validation_rows = [
        {
            "central_quality_binding_status": "exact_identity_match",
            "central_quality_evidence_verified": True,
            "precision_quality_binding_verified": True,
        }
        for _ in range(23)
    ]

    def fake_run(command: list[str], *, cwd: Path) -> dict[str, object]:
        del cwd
        script = Path(command[2]).name
        out_dir = Path(command[command.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        if script == "native_producer_validate_visualize.py":
            (out_dir / "native_producer_validation_summary.json").write_text(
                json.dumps({"rows": validation_rows}), encoding="utf-8",
            )
        return {"ok": True}

    monkeypatch.setattr(replay, "_run", fake_run)
    with pytest.raises(
        RuntimeError, match="did not retain every performance row",
    ):
        replay.main([
            "--run-dir", str(run),
            "--output-dir", str(destination),
        ])
    assert not destination.exists()


def test_offline_native_replay_retains_quality_fails_as_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _load_script("replay_native_supplement_reports.py")
    run = _minimal_run(tmp_path)
    destination = tmp_path / "projection"
    native_rows = [
        {
            "ok": True,
            "status": "ok",
            "runtime_executable": True,
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": f"b{index:03d}",
            "precision": "float32_layout_fp16",
            "setup_id": "orin_nx_hailo8_01",
            "comparison_backend": "hailo8",
            "fps_makespan": 100.0 + index,
            "fps_median": 99.0 + index,
            "repetition_count_requested": 3,
            "repetition_count_attempted": 3,
            "repetition_count_valid": 3,
            "repetition_status": "complete",
            "repetition_aggregation": "median",
            "repetition_claim_identity_status": "exact",
        }
        for index in range(24)
    ]
    (run / "reports" / "native_producer_summary.json").write_text(
        json.dumps({"rows": native_rows}), encoding="utf-8",
    )
    policy_sha = AccuracyGatePolicy.from_mapping(_quality_policy()).sha256()
    decisions = ["pass"] * 9 + ["fail"] * 11 + ["inconclusive"] * 4
    validation_rows = [
        {
            "case": f"b{index:03d}",
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "precision": "float32_layout_fp16",
            "setup_id": "orin_nx_hailo8_01",
            "comparison_backend": "hailo8",
            "central_quality_binding_status": "exact_identity_match",
            "central_quality_evidence_verified": True,
            "precision_quality_binding_verified": True,
            "accuracy_gate_policy_sha256": policy_sha,
            "task_quality_policy_sha256": policy_sha,
            "task_quality_status": decision,
            "eligible_for_ranking": False,
            "status": "claim_ok",
            "runtime_executable": True,
            "tensor_ok": True,
        }
        for index, decision in enumerate(decisions)
    ]

    invoked_scripts: list[str] = []
    drift_final_report = [False]

    def fake_run(command: list[str], *, cwd: Path) -> dict[str, object]:
        del cwd
        script = Path(command[2]).name
        invoked_scripts.append(script)
        out_dir = Path(command[command.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        if script == "native_producer_final_report.py":
            quality_summary = Path(
                command[command.index("--quality-summary") + 1]
            )
            assert quality_summary.is_file()
            combined_rows = [
                {
                    **native_row,
                    "task_quality_status": validation_row[
                        "task_quality_status"
                    ],
                    "task_quality_pass": (
                        validation_row["task_quality_status"] == "pass"
                    ),
                    "quality_central_evidence_verified": True,
                    "precision_quality_binding_verified": True,
                    "task_quality_observation_valid": True,
                    "quality_claim_result_verified": False,
                    "quality_evidence_verified": True,
                    "quality_accuracy_gate_pass": (
                        validation_row["task_quality_status"] == "pass"
                    ),
                    "quality_eligible_for_ranking": False,
                    "quality_gate_status": "screening_only",
                }
                for native_row, validation_row in zip(
                    native_rows, validation_rows, strict=True,
                )
            ]
            if drift_final_report[0]:
                combined_rows[0]["fps_makespan"] += 1.0
            (out_dir / "native_producer_combined_summary.json").write_text(
                json.dumps({
                    "rows": combined_rows,
                    "quality_evidence": {
                        "status": "loaded",
                        "schema_valid": True,
                        "quality_row_count": 24,
                        "unique_identity_count": 24,
                        "duplicate_identity_count": 0,
                        "verified_performance_row_count": 24,
                        "missing_performance_row_count": 0,
                        "failed_performance_row_count": 0,
                        "ambiguous_performance_row_count": 0,
                    },
                }),
                encoding="utf-8",
            )
            for extension in ("csv", "md"):
                (out_dir / f"native_producer_combined_summary.{extension}").write_text(
                    "quality-bound\n", encoding="utf-8",
                )
            return {"ok": True, "rows": 24}

        assert script == "native_producer_validate_visualize.py"
        staged_rows = []
        for index, row in enumerate(validation_rows):
            row_dir = out_dir / f"row_{index:02d}"
            row_dir.mkdir()
            tensor_validation = row_dir / "tensor_dump_validation.json"
            task_validation = row_dir / "detection_visual_validation.json"
            visual_artifact = row_dir / "detection_visual_validation.md"
            box_overlay = row_dir / "detection_boxes_overlay.png"
            tensor_validation.write_text("{}", encoding="utf-8")
            task_validation.write_text(
                json.dumps({"box_overlay": str(box_overlay)}),
                encoding="utf-8",
            )
            visual_artifact.write_text(
                f"overlay: {box_overlay}\n", encoding="utf-8",
            )
            box_overlay.write_bytes(b"fake-png")
            staged_rows.append({
                **row,
                "tensor_validation": str(tensor_validation),
                "task_validation": str(task_validation),
                "visual_artifact": str(visual_artifact),
                "box_overlay": str(box_overlay),
            })
        (out_dir / "native_producer_validation_summary.json").write_text(
            json.dumps({"rows": staged_rows}), encoding="utf-8",
        )
        (out_dir / "native_producer_validation_summary.csv").write_text(
            f"task_validation\n{staged_rows[0]['task_validation']}\n",
            encoding="utf-8",
        )
        (out_dir / "native_producer_validation_summary.md").write_text(
            f"artifact: {staged_rows[0]['visual_artifact']}\n",
            encoding="utf-8",
        )
        return {
            "ok": True,
            "summary": str(
                out_dir / "native_producer_validation_summary.json"
            ),
        }

    monkeypatch.setattr(replay, "_run", fake_run)
    assert replay.main([
        "--run-dir", str(run),
        "--output-dir", str(destination),
    ]) == 0
    status = json.loads(
        (destination / "reports" / "native_replay_status.json").read_text(
            encoding="utf-8",
        )
    )
    assert status["central_quality_decisions"] == {
        "pass": 9, "fail": 11, "inconclusive": 4,
    }
    assert status["technical_error_count"] == 0
    assert status["claim_eligible_count"] == 0
    assert status["scientific_pass"] is False
    assert invoked_scripts == [
        "native_producer_validate_visualize.py",
        "native_producer_final_report.py",
    ]
    combined = json.loads(
        (
            destination / "reports"
            / "native_producer_combined_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert combined["quality_evidence"] == {
        "status": "loaded",
        "schema_valid": True,
        "quality_row_count": 24,
        "unique_identity_count": 24,
        "duplicate_identity_count": 0,
        "verified_performance_row_count": 24,
        "missing_performance_row_count": 0,
        "failed_performance_row_count": 0,
        "ambiguous_performance_row_count": 0,
    }
    assert len(combined["rows"]) == 24
    assert all(
        row["quality_central_evidence_verified"] is True
        for row in combined["rows"]
    )
    matrix = collect_native_performance_matrix(destination)
    observations = matrix["observations"]
    assert matrix["observation_count"] == 24
    assert Counter(
        row["task_quality_status"] for row in observations
    ) == {"pass": 9, "fail": 11, "inconclusive": 4}
    assert sum(
        row["quality_central_evidence_verified"] is True
        for row in observations
    ) == 24
    assert sum(
        row["precision_quality_binding_verified"] is True
        for row in observations
    ) == 24
    assert sum(
        row["task_quality_pass"] is True for row in observations
    ) == 9
    assert sum(
        row["quality_claim_result_verified"] is True
        for row in observations
    ) == 0
    published_validation = json.loads(
        (
            destination / "reports" / "native_validation"
            / "native_producer_validation_summary.json"
        ).read_text(encoding="utf-8")
    )
    published_row = published_validation["rows"][0]
    for field in (
        "tensor_validation", "task_validation", "visual_artifact",
        "box_overlay",
    ):
        assert str(published_row[field]).startswith(str(destination))
        assert ".partial-" not in str(published_row[field])
        assert Path(str(published_row[field])).is_file()
    task_payload = json.loads(
        Path(str(published_row["task_validation"])).read_text(
            encoding="utf-8",
        )
    )
    assert task_payload["box_overlay"] == published_row["box_overlay"]
    assert status["validation"]["summary"].startswith(str(destination))
    for path in destination.rglob("*"):
        if (
            path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in {".json", ".csv", ".md"}
        ):
            assert ".partial-" not in path.read_text(encoding="utf-8")

    drift_final_report[0] = True
    drifted_destination = tmp_path / "projection_drifted"
    with pytest.raises(
        RuntimeError, match="changed preserved performance",
    ):
        replay.main([
            "--run-dir", str(run),
            "--output-dir", str(drifted_destination),
        ])
    assert not drifted_destination.exists()


@pytest.mark.parametrize(
    ("backend", "pipeline_dir", "precision", "filename"),
    [
        (
            "deepx_to_trt",
            "deepx_to_trt",
            "float32_layout_fp16",
            "deepx_native_fifo_e2e_results.json",
        ),
        (
            "hailo10h_to_trt",
            "hailo10h_to_trt",
            "float32_layout_fp16",
            "hailo10_native_fifo_e2e_results.json",
        ),
        (
            "hailo8_to_trt",
            "hailo_to_trt",
            "uint8_dequant_fp16",
            "native_fifo_results.json",
        ),
    ],
)
def test_native_report_rebase_is_model_scoped_for_duplicate_b002(
    tmp_path: Path,
    backend: str,
    pipeline_dir: str,
    precision: str,
    filename: str,
) -> None:
    validator = _load_script("native_producer_validate_visualize.py")
    copied_root = tmp_path / "copied_native_producers"
    relative = (
        Path("benchmark_set") / "native_pipeline" / "b002" /
        pipeline_dir / precision / filename
    )
    expected = copied_root / "yolo26s" / relative
    competing = copied_root / "yolov7_paper" / relative
    expected.parent.mkdir(parents=True)
    competing.parent.mkdir(parents=True)

    portable_binding = {
        "schema": "onnx-splitpoint/native-split-quality-binding",
        "schema_version": 1,
        "model_id": "yolo26s",
        "case_id": "b002",
        "backend": backend,
        "precision": precision,
        "binding_sha256": "a" * 64,
    }
    competing_binding = {
        **portable_binding,
        "model_id": "yolov7_paper",
        "binding_sha256": "b" * 64,
    }
    expected.write_text(
        json.dumps({
            "model": "yolo26s",
            "native_split_quality_binding": portable_binding,
        }, sort_keys=True),
        encoding="utf-8",
    )
    competing.write_text(
        json.dumps({
            "model": "yolov7_paper",
            "native_split_quality_binding": competing_binding,
        }, sort_keys=True),
        encoding="utf-8",
    )
    expected_bytes = expected.read_bytes()
    competing_bytes = competing.read_bytes()

    stale_report = (
        Path("/stale/native_host") / "yolo26s" / relative
    )
    row = {
        "model": "yolo26s",
        "case": "b002",
        "backend": backend,
        "precision": precision,
        "report": str(stale_report),
        "native_split_quality_binding": portable_binding,
    }
    selected = validator._find_report(row, [copied_root])

    assert selected is not None
    assert selected.resolve() == expected.resolve()
    merged: dict[str, object] = {}
    validator._merge_native_split_quality_evidence(
        merged,
        row,
        json.loads(selected.read_text(encoding="utf-8")),
    )
    assert merged["native_split_quality_binding"] == portable_binding
    assert "native_split_quality_provenance_conflict" not in merged
    # Report lookup/replay must consume the portable authority as-is; it must
    # never rewrite or reseal either archived producer binding.
    assert expected.read_bytes() == expected_bytes
    assert competing.read_bytes() == competing_bytes
