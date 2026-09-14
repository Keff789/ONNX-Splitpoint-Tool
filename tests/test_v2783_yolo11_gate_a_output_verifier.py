from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from v2783_gate_a_fixture_builder import (
    write_gate_a_anchor_fixture,
    write_gate_a_budget_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/yolo11l_v2783_hailo8_first_b5_gate_a.yaml"
SCRIPT = ROOT / "scripts/verify_v2783_yolo11_gate_a_output.py"


def _module():
    name = f"v2783_gate_a_output_verifier_{id(SCRIPT)}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _verify(gate_output: Path, *, workflow_rc: int = 1):
    return _module().verify_gate_a_output(
        gate_output=gate_output,
        workflow_rc=workflow_rc,
        expected_profile_path=PROFILE,
    )


def _tree_observation(root: Path) -> dict[str, tuple[int, int, str]]:
    observed = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        stat_result = path.stat()
        observed[path.relative_to(root).as_posix()] = (
            stat_result.st_size,
            stat_result.st_mtime_ns,
            __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        )
    return observed


def test_production_shaped_stop_after_rc1_is_strictly_admitted_read_only(
    tmp_path: Path,
) -> None:
    gate = tmp_path / "gate"
    gate.mkdir()
    run = write_gate_a_anchor_fixture(gate, PROFILE)
    receipt = json.loads(
        (
            run
            / "models/yolo11l/benchmark_set/hailo_feasibility_receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["state"]["targets"] == ["hailo10", "hailo8"]
    before = _tree_observation(gate)

    verdict = _verify(gate, workflow_rc=1)

    assert verdict["status"] == "PASS", verdict["errors"]
    assert verdict["ok"] is True
    assert verdict["outcome"] == "ANCHOR_FOUND"
    assert verdict["anchor_boundary"] == 67
    assert verdict["expected_stop_after_partial"] is True
    assert len(verdict["full_source_onnx_sha256"]) == 64
    assert set(verdict["artifacts"]) == {"hailo8", "hailo10"}
    assert all(
        len(row["cache_key"]) == 64
        for row in verdict["artifacts"].values()
    )
    assert _tree_observation(gate) == before


def test_yaml_presentation_order_cannot_replace_production_run_plan_order(
    tmp_path: Path,
) -> None:
    gate = tmp_path / "gate"
    gate.mkdir()
    run = write_gate_a_anchor_fixture(gate, PROFILE)
    receipt_path = (
        run / "models/yolo11l/benchmark_set/hailo_feasibility_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["state"]["targets"] = ["hailo8", "hailo10"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    verdict = _verify(gate)

    assert verdict["ok"] is False
    assert verdict["errors"] == ["receipt_state_target_order_mismatch"]


def test_only_zero_or_expected_stop_after_rc1_can_be_admitted(tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    gate.mkdir()
    write_gate_a_anchor_fixture(gate, PROFILE)

    assert _verify(gate, workflow_rc=0)["ok"] is True
    rejected = _verify(gate, workflow_rc=2)

    assert rejected["ok"] is False
    assert rejected["errors"] == ["unexpected_workflow_rc:2"]


def test_partial_status_is_required_even_with_real_anchor_and_rc1(tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    gate.mkdir()
    write_gate_a_anchor_fixture(gate, PROFILE, workflow_status="ok")

    verdict = _verify(gate, workflow_rc=1)

    assert verdict["ok"] is False
    assert verdict["errors"] == ["run_manifest_not_partial"]


def test_manifest_cannot_mask_tampered_stage_checkpoint(tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    gate.mkdir()
    run = write_gate_a_anchor_fixture(gate, PROFILE)
    stage = (
        run
        / "models/yolo11l/stages/build_backend_artifacts/stage_result.json"
    )
    payload = json.loads(stage.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    stage.write_text(json.dumps(payload), encoding="utf-8")

    verdict = _verify(gate)

    assert verdict["ok"] is False
    assert verdict["errors"] == [
        "stage_not_allowed_status:build_backend_artifacts:failed"
    ]


def test_receipt_and_manifest_cannot_mask_tampered_hef(tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    gate.mkdir()
    run = write_gate_a_anchor_fixture(gate, PROFILE)
    hef = (
        run
        / "models/yolo11l/benchmark_set/legacy_suite/b067/"
        "hailo/hailo10/part1/compiled.hef"
    )
    hef.write_bytes(b"tampered-hef")

    verdict = _verify(gate)

    assert verdict["ok"] is False
    assert verdict["errors"] == ["anchor_material_revalidation_failed"]


def test_receipt_only_fixture_is_rejected_before_rc1_admission(tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    run = gate / "fake_run"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text("{}", encoding="utf-8")

    verdict = _verify(gate)

    assert verdict["ok"] is False
    assert verdict["errors"] == [
        "profile_source_unsafe_or_missing:BuildEvidenceError"
    ]


def test_expected_profile_symlink_is_rejected_without_fd_growth(tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    gate.mkdir()
    write_gate_a_anchor_fixture(gate, PROFILE)
    linked = tmp_path / "profile.yaml"
    linked.symlink_to(PROFILE)
    verifier = _module()
    fd_dir = Path("/proc/self/fd")
    before = len(list(fd_dir.iterdir())) if fd_dir.is_dir() else None

    for _ in range(5):
        verdict = verifier.verify_gate_a_output(
            gate_output=gate,
            workflow_rc=1,
            expected_profile_path=linked,
        )
        assert verdict["ok"] is False
        assert verdict["errors"] == [
            "expected_profile_unsafe_or_missing:BuildEvidenceError"
        ]

    if before is not None:
        assert len(list(fd_dir.iterdir())) == before


def test_shell_cli_reports_production_stop_after_pass(tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    gate.mkdir()
    write_gate_a_anchor_fixture(gate, PROFILE)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--gate-output",
            str(gate),
            "--workflow-rc",
            "1",
            "--expected-profile",
            str(PROFILE),
            "--format",
            "shell",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "GATE_A_VERIFICATION=PASS" in completed.stdout
    assert "GATE_A_OUTCOME=ANCHOR_FOUND" in completed.stdout
    assert "ANCHOR_BOUNDARY=b67" in completed.stdout
    assert "FULL_SOURCE_ONNX_SHA256=" in completed.stdout
    assert "EXPECTED_STOP_AFTER_PARTIAL=PASS" in completed.stdout
    assert "WORKFLOW_RC=1" in completed.stdout
    assert "B5_BLOCKED=NO" in completed.stdout


def test_production_shaped_budget_exhaustion_is_blocked_read_only(
    tmp_path: Path,
) -> None:
    gate = tmp_path / "gate"
    gate.mkdir()
    run = write_gate_a_budget_fixture(gate, PROFILE)
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert "build_backend_artifacts" not in manifest["models"]["yolo11l"]["stages"]
    assert (
        manifest["models"]["yolo11l"]["stages"]["generate_benchmark_set"][
            "status"
        ]
        == "failed"
    )
    before = _tree_observation(gate)

    verdict = _verify(gate, workflow_rc=1)

    assert verdict["status"] == "BLOCKED", verdict["errors"]
    assert verdict["ok"] is False
    assert verdict["valid_terminal"] is True
    assert verdict["outcome"] == "CANARY_BUDGET_EXHAUSTED"
    assert verdict["errors"] == ["canary_budget_exhausted"]
    assert len(verdict["full_source_onnx_sha256"]) == 64
    assert _tree_observation(gate) == before


def test_budget_exhaustion_cli_has_distinct_terminal_rc3(tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    gate.mkdir()
    write_gate_a_budget_fixture(gate, PROFILE)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--gate-output",
            str(gate),
            "--workflow-rc",
            "1",
            "--expected-profile",
            str(PROFILE),
            "--format",
            "shell",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 3, completed.stderr + completed.stdout
    assert "GATE_A_VERIFICATION=BLOCKED" in completed.stdout
    assert "GATE_A_OUTCOME=CANARY_BUDGET_EXHAUSTED" in completed.stdout
    assert "CANARY_BUDGET_EXHAUSTED=YES" in completed.stdout
    assert "ANCHOR_FOUND=NO" in completed.stdout
    assert "B5_BLOCKED=YES" in completed.stdout
