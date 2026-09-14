from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_v27911_yolov7_claim_gate_32.py"
HISTORICAL_SCRIPT = ROOT / "scripts" / "verify_v27910_yolov7_claim_gate_32.py"
HISTORICAL_TEST = ROOT / "tests" / "test_v27910_yolov7_claim_gate_32.py"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load(SCRIPT, "v27911_yolov7_claim_gate")
historical_fixture = _load(HISTORICAL_TEST, "v27911_historical_gate_fixture")


def build_current_claim_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Reuse the exhaustive synthetic evidence builder with v2.79.11 IDs."""

    monkeypatch.setattr(historical_fixture, "gate", gate)
    paths = historical_fixture._fixture(tmp_path, monkeypatch)
    result_path = Path(paths["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["product_execution_context"]["source_snapshot_prefix"] = (
        "ONNX-Splitpoint-Tool_v2.79.11"
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    artifact_path = Path(paths["artifact_index"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    canonical = artifact["artifacts"]["canonical_result"]
    canonical["sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    canonical["size_bytes"] = result_path.stat().st_size
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    # The validation functions execute with the frozen base module as globals.
    # Mirror the fixture-only frozen-reference substitutions into that module.
    monkeypatch.setattr(
        gate._base, "DATASET_MANIFEST_SHA256", gate.DATASET_MANIFEST_SHA256
    )
    monkeypatch.setattr(
        gate._base, "REFERENCE_REPORT_SHA256", gate.REFERENCE_REPORT_SHA256
    )
    return paths


def test_v27911_exact_fixture_passes_current_compatibility_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = build_current_claim_fixture(tmp_path, monkeypatch)
    receipt = gate.verify_claim_gate(
        result_json=paths["result"],
        evidence_root=paths["evidence"],
        source_root=paths["source"],
        expected_source_manifest_sha256=paths["source_manifest_sha"],
    )
    assert receipt["status"] == "PASS"
    assert receipt["valid_claim"] is True
    assert receipt["passed_item_count"] == 32
    assert receipt["verification_identity"] == {
        "package_version": "2.79.11",
        "build_id": "v2.79.11-platform-power-energy-evidence-closure",
        "source_manifest_sha256": paths["source_manifest_sha"],
        "verifier": "verify_v27911_yolov7_claim_gate_32.py",
        "frozen_base_verifier": {
            "path": "verify_v27910_yolov7_claim_gate_32.py",
            "sha256": gate.BASE_VERIFIER_SHA256,
        },
    }


def test_v27911_current_verifier_still_fails_closed_on_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = build_current_claim_fixture(tmp_path, monkeypatch)
    payload = json.loads(paths["result"].read_text(encoding="utf-8"))
    payload["claim_eligible"] = False
    paths["result"].write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(gate.ClaimGateError, match="result_contract_claim_eligible"):
        gate.verify_claim_gate(
            result_json=paths["result"],
            evidence_root=paths["evidence"],
            source_root=paths["source"],
            expected_source_manifest_sha256=paths["source_manifest_sha"],
        )


def test_v27911_frozen_historical_verifier_is_byte_pinned(tmp_path: Path) -> None:
    assert hashlib.sha256(HISTORICAL_SCRIPT.read_bytes()).hexdigest() == (
        gate.BASE_VERIFIER_SHA256
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    current = scripts / SCRIPT.name
    historical = scripts / HISTORICAL_SCRIPT.name
    shutil.copyfile(SCRIPT, current)
    historical.write_bytes(HISTORICAL_SCRIPT.read_bytes() + b"\n# drift\n")
    completed = subprocess.run(
        [sys.executable, str(current), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode != 0
    assert "v27910_claim_gate_base_sha256_mismatch" in completed.stderr


def test_v27911_snapshot_member_pins_remain_frozen() -> None:
    assert gate.CARRY_FORWARD_ORIGIN["current_snapshot_member_sha256"] == {
        "onnx_splitpoint_tool/__init__.py": (
            "bc0fe68cdf8c1993bd1c84d4194d1dd3b239dc49df53afee751083bce73fd6d8"
        ),
        "onnx_splitpoint_tool/release_identity.py": (
            "294376afb46f5fa010c2ac9bf15d0e8ba8cb760f9105bbf73d1ed991863b32c8"
        ),
        "onnx_splitpoint_tool/native_three_stage.py": (
            "6ab076454a231f5a7939c05489a885dba92785e2711aa0595125a79aca25311f"
        ),
        "scripts/native_hailo_trt_fifo_from_benchmarkset.py": (
            "da285602e2fb6373b6ee7a3c5eafd7ececc5d8728cc560ad78bcdd38ae674800"
        ),
        "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py": (
            "ba187ca8bd36cc58c24775477fc18877b324f30bda776bcf83d9f08794de6ecf"
        ),
    }


def test_v27911_carry_forward_computational_surface_remains_exact() -> None:
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    rows = {str(row["path"]): dict(row) for row in manifest["files"]}
    attestation = gate._validate_carry_forward_computational_source(ROOT, rows)
    assert attestation["file_count"] == 21
    assert attestation["inventory_sha256"] == (
        gate.CARRY_FORWARD_COMPUTATIONAL_TREE_SHA256
    )
    assert attestation["byte_identical_to_measurement_release"] is True


def test_v27911_verifier_is_python38_compatible_and_executable() -> None:
    assert SCRIPT.stat().st_mode & 0o111
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'print("V27911_YOLOV7_CLAIM_GATE=PASS")' in text
    assert ".removeprefix(" not in text
    assert ".removesuffix(" not in text
    assert ".is_relative_to(" not in text
    subprocess.run([sys.executable, "-m", "py_compile", str(SCRIPT)], check=True)
