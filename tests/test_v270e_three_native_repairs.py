from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import onnx_splitpoint_tool.native_detection_postprocess as postprocess
from onnx_splitpoint_tool.native_command_contract import (
    verify_split_energy_artifact_stats,
)
from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenPostprocessError,
    build_frozen_postprocess_contract,
    canonical_json_sha256,
    frozen_postprocess_invariant_identity,
    verify_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_ONNX_SHA256,
)
from onnx_splitpoint_tool.workflow.runner import (
    _native_selection_contract_runs_v270e,
    _native_split_case_support_v270e,
)


def _write_split_manifest(
    benchmark_set: Path,
    case: str,
    part2_inputs: list[str],
) -> None:
    case_dir = benchmark_set / case
    case_dir.mkdir(parents=True)
    (case_dir / "split_manifest.json").write_text(
        (
            '{"part2_external_inputs":['
            + ",".join(f'"{name}"' for name in part2_inputs)
            + "]}\n"
        ),
        encoding="utf-8",
    )


def test_native_plan_uses_all_selected_single_input_cases(
    tmp_path: Path,
) -> None:
    selected = {
        "resnet50": ["b039", "b052", "b082"],
        "yolo26s": ["b036", "b038", "b142"],
        "yolov7_paper": ["b044", "b116", "b216"],
    }
    benchmark_sets = {
        model: tmp_path / model / "benchmark_set"
        for model in selected
    }
    multi_input = {"b142", "b116", "b216"}
    for model, cases in selected.items():
        for case in cases:
            _write_split_manifest(
                benchmark_sets[model],
                case,
                [f"{case}_main", f"{case}_skip_a", f"{case}_skip_b"]
                if case in multi_input
                else [f"{case}_main"],
            )

    support = _native_split_case_support_v270e(
        benchmark_sets, selected,
    )
    supported = {
        model: [
            row["case"]
            for row in support
            if row["model"] == model and row["native_supported"]
        ]
        for model in selected
    }
    supported = {model: cases for model, cases in supported.items() if cases}
    excluded = {
        (row["model"], row["case"], row["part2_input_count"], row["reason"])
        for row in support
        if not row["native_supported"]
    }

    assert supported == {
        "resnet50": ["b039", "b052", "b082"],
        "yolo26s": ["b036", "b038"],
        "yolov7_paper": ["b044"],
    }
    assert excluded == {
        ("yolo26s", "b142", 3, "part2_input_count_not_one"),
        ("yolov7_paper", "b116", 3, "part2_input_count_not_one"),
        ("yolov7_paper", "b216", 3, "part2_input_count_not_one"),
    }

    planned_rows = []
    for backend in ("hailo8", "hailo10h", "deepx"):
        runs = _native_selection_contract_runs_v270e(backend, supported)
        for run in runs:
            for model, cases in run["case_map"].items():
                planned_rows.extend(
                    (backend, model, case, run["precision"])
                    for case in cases
                )
    assert len(planned_rows) == 18
    assert {
        backend
        for backend, model, case, _precision in planned_rows
        if model == "yolov7_paper" and case == "b044"
    } == {"hailo8", "hailo10h", "deepx"}
    assert not any(case in multi_input for _, _, case, _ in planned_rows)


def test_native_plan_fails_closed_when_split_interface_is_missing(
    tmp_path: Path,
) -> None:
    benchmark_set = tmp_path / "resnet50" / "benchmark_set"
    (benchmark_set / "b001").mkdir(parents=True)
    rows = _native_split_case_support_v270e(
        {"resnet50": benchmark_set},
        {"resnet50": ["b001"]},
    )
    assert len(rows) == 1
    assert rows[0]["native_supported"] is False
    assert rows[0]["part2_input_count"] is None
    assert rows[0]["reason"].startswith(
        "split_manifest_part2_inputs_unavailable:"
    )


def test_native_plan_console_audits_three_by_three_selection(
    tmp_path: Path,
) -> None:
    selected = {
        "resnet50": ["b039", "b052", "b082"],
        "yolo26s": ["b036", "b038", "b142"],
        "yolov7_paper": ["b044", "b116", "b216"],
    }
    arguments: list[str] = []
    for model, cases in selected.items():
        benchmark_set = tmp_path / "inputs" / model / "benchmark_set"
        benchmark_set.mkdir(parents=True)
        (benchmark_set / "benchmark_set.json").write_text(
            '{"cases":[]}\n', encoding="utf-8",
        )
        for case in cases:
            _write_split_manifest(
                benchmark_set,
                case,
                [f"{case}_main", f"{case}_skip_a", f"{case}_skip_b"]
                if case in {"b142", "b116", "b216"}
                else [f"{case}_main"],
            )
        arguments.extend([
            "--benchmark-set", f"{model}={benchmark_set}",
        ])
    result_root = tmp_path / "console_result"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/native_console_smoke.py"
            ),
            "native-plan",
            *arguments,
            "--result-root", str(result_root),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(
        (result_root / "console_smoke_summary.json").read_text(
            encoding="utf-8",
        )
    )
    assert summary["ok"] is True
    assert summary["planned_split_row_count"] == 18
    assert summary["capability_exclusion_count"] == 9
    assert summary["supported_case_map"]["yolov7_paper"] == ["b044"]


def _yolov7_outputs() -> dict[str, np.ndarray]:
    return {
        "output": np.full((1, 3, 80, 80, 85), -20.0, dtype=np.float32),
        "clone_1": np.full((1, 3, 40, 40, 85), -20.0, dtype=np.float32),
        "clone_2": np.full((1, 3, 20, 20, 85), -20.0, dtype=np.float32),
    }


def test_vendored_and_main_frozen_contracts_share_logical_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=_yolov7_outputs(),
        input_hw=[640, 640],
        original_wh=[1280, 720],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    monkeypatch.setattr(postprocess, "__package__", "splitpoint_runners")
    vendored = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=_yolov7_outputs(),
        input_hw=[640, 640],
        original_wh=[1280, 720],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    assert vendored["implementation_artifacts"] == (
        main["implementation_artifacts"]
    )
    assert vendored["invariant_contract_sha256"] == (
        main["invariant_contract_sha256"]
    )
    assert vendored["contract_sha256"] == main["contract_sha256"]
    assert {
        row["relative_path"]
        for row in vendored["implementation_artifacts"].values()
    } == {
        "onnx_splitpoint_tool/native_detection_postprocess.py",
        "onnx_splitpoint_tool/runners/harness/yolo.py",
        "onnx_splitpoint_tool/runners/harness/base.py",
    }


def test_frozen_contract_still_rejects_changed_implementation_sha() -> None:
    contract = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=_yolov7_outputs(),
        input_hw=[640, 640],
        original_wh=[1280, 720],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    tampered = copy.deepcopy(contract)
    tampered.pop("contract_sha256")
    tampered["implementation_artifacts"]["yolo_harness"]["sha256"] = "0" * 64
    tampered["invariant_identity"] = frozen_postprocess_invariant_identity(
        tampered
    )
    tampered["invariant_contract_sha256"] = canonical_json_sha256(
        tampered["invariant_identity"]
    )
    tampered["contract_sha256"] = canonical_json_sha256(tampered)
    with pytest.raises(
        FrozenPostprocessError,
        match="yolov7_sigmoid_arithmetic_implementation_mismatch",
    ):
        verify_frozen_postprocess_contract(tampered)


def _verified_stat_row(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def test_split_energy_stat_guard_accepts_epoch_mtime_and_detects_change(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "source_part2.onnx"
    artifact.write_bytes(b"immutable-onnx")
    os.utime(artifact, ns=(0, 0))
    binding = {"verified_files": [_verified_stat_row(artifact)]}
    assert binding["verified_files"][0]["mtime_ns"] == 0
    assert verify_split_energy_artifact_stats(binding) == (
        True,
        "preflight_file_stats_unchanged",
    )

    artifact.write_bytes(b"changed-onnx")
    assert verify_split_energy_artifact_stats(binding) == (
        False,
        "split_energy_verified_file_changed_after_preflight",
    )


@pytest.mark.parametrize("field", ["size_bytes", "mtime_ns", "device", "inode"])
@pytest.mark.parametrize("invalid", [None, True, "not-an-integer"])
def test_split_energy_stat_guard_rejects_invalid_required_fields(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"x")
    row = _verified_stat_row(artifact)
    row[field] = invalid
    assert verify_split_energy_artifact_stats(
        {"verified_files": [row]},
    ) == (False, "split_energy_verified_file_stat_invalid")
