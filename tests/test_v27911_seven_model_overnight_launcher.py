from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_v27911_seven_model_long_overnight.sh"
PROFILE = ROOT / "profiles/complete_set_7models_v27911_b500_audit20.yaml"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_v27911_launcher_identity_profile_and_current_alias_separation() -> None:
    assert os.access(LAUNCHER, os.X_OK)
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = _text(LAUNCHER)
    profile_sha = hashlib.sha256(PROFILE.read_bytes()).hexdigest()
    assert 'expected = "2.79.11"' in text
    assert "V27911_RELEASE_IDENTITY=PASS" in text
    assert 'ACCEPTANCE="$TOOL/scripts/run_v27911_small_acceptance.sh"' in text
    assert 'STATUS_HELPER="$TOOL/scripts/launcher_status_v27911.py"' in text
    assert 'PROFILE="$TOOL/profiles/complete_set_7models_v27911_b500_audit20.yaml"' in text
    assert f"PROFILE_EXPECTED_SHA256={profile_sha}" in text
    assert 'profile.get("name") == "complete_set_7models_v27911_b500_audit20"' in text
    generic = _text(ROOT / "scripts/run_v279_seven_model_long_overnight.sh")
    assert 'CURRENT_LAUNCHER="$SCRIPT_DIR/run_v27913_seven_model_long_overnight.sh"' in generic


def test_v27911_launcher_retains_exact_historical_gate_receipts() -> None:
    text = _text(LAUNCHER)
    assert 'GATE_VERIFIER="$TOOL/scripts/verify_v27910_yolo11_r8b_gate.py"' in text
    assert 'GATE_PROFILE="$TOOL/profiles/yolo11l_v27910_r8b_full_b067_gate.yaml"' in text
    assert 'YOLOV7_CLAIM_VERIFIER="$TOOL/scripts/verify_v27911_yolov7_claim_gate_32.py"' in text
    assert 'gate_verdict.get("version") == "2.79.10"' in text
    assert '"build_id": "v2.79.10-urecs-platform-power-release-closure"' in text
    assert 'V27910_YOLO11_FULL_GATE=PASS' in text
    assert 'V27911_YOLOV7_CLAIM_GATE_32=PASS' in text


def test_v27911_launcher_preflight_is_non_dispatching_and_interlocked() -> None:
    text = _text(LAUNCHER)
    assert 'PLATFORM_INTERLOCK="$PLATFORM_LOCK_PARENT/workflow_platform_interlock.lock"' in text
    assert 'exec 7<>"$PLATFORM_INTERLOCK"' in text
    assert "flock -n -s 7" in text
    branch_start = text.index("if (( PREFLIGHT_ONLY )); then")
    branch_end = text.index("\nfi\n", branch_start) + len("\nfi\n")
    branch = text[branch_start:branch_end]
    assert "V27911_SEVEN_MODEL_LONG_PREFLIGHT=PASS" in branch
    assert "LONG_RUN_STARTED=NO" in branch
    assert "nohup" not in branch
    assert "exit 0" in branch


def test_v27911_launcher_help_and_source_guard() -> None:
    help_result = subprocess.run(
        ["bash", str(LAUNCHER), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "run_v27911_seven_model_long_overnight.sh" in help_result.stdout
    sourced = subprocess.run(
        [
            "bash", "-c",
            'source "$1"; printf "SOURCE_CONTINUED=YES\\n"',
            "launcher-source-test", str(LAUNCHER),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert sourced.returncode == 0
    assert sourced.stdout == "SOURCE_CONTINUED=YES\n"


def _load_current_fixture_module() -> object:
    path = ROOT / "tests" / "test_v27911_yolov7_claim_gate_32.py"
    spec = importlib.util.spec_from_file_location(
        "v27911_launcher_claim_fixture", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refresh_fixture_source_manifest(paths: dict[str, object]) -> None:
    source = Path(paths["source"])
    manifest_path = source / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for row in manifest["files"]:
        relative = str(row["path"])
        path = source / relative
        rows.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest["files"] = rows
    manifest["file_count"] = len(rows)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    source_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    paths["source_manifest_sha"] = source_manifest_sha
    capture_path = Path(paths["capture"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["source_manifest_sha256"] = source_manifest_sha
    capture_path.write_text(json.dumps(capture, indent=2) + "\n", encoding="utf-8")


def _prepare_launcher_claim_fixture(paths: dict[str, object]) -> None:
    source = Path(paths["source"])
    for name in (
        "verify_v27910_yolov7_claim_gate_32.py",
        "verify_v27911_yolov7_claim_gate_32.py",
    ):
        shutil.copyfile(ROOT / "scripts" / name, source / "scripts" / name)
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
    _refresh_fixture_source_manifest(paths)


def _write_fake_tool_python(
    path: Path, *, dataset_sha: str, reference_sha: str, recovery: Path
) -> None:
    real_python = sys.executable
    source = textwrap.dedent(
        f"""\
        #!{real_python}
        import importlib.util
        import json
        import os
        from pathlib import Path
        import sys

        DATASET_SHA = {dataset_sha!r}
        REFERENCE_SHA = {reference_sha!r}
        RECOVERY = {str(recovery)!r}

        args = list(sys.argv[1:])
        while args and args[0] == "-B":
            args.pop(0)
        first = args[0] if args else ""
        if first.endswith("verify_v27911_yolov7_claim_gate_32.py"):
            spec = importlib.util.spec_from_file_location("fixture_current_gate", first)
            if spec is None or spec.loader is None:
                raise SystemExit(90)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.DATASET_MANIFEST_SHA256 = DATASET_SHA
            module.REFERENCE_REPORT_SHA256 = REFERENCE_SHA
            module._base.DATASET_MANIFEST_SHA256 = DATASET_SHA
            module._base.REFERENCE_REPORT_SHA256 = REFERENCE_SHA
            raise SystemExit(module.main(args[1:]))
        if first.endswith("verify_v27910_yolo11_r8b_gate.py"):
            print("{{}}")
            raise SystemExit(0)
        if first == "-":
            program = sys.stdin.read()
            rest = args[1:]
            if "unsafe directory" in program and rest:
                Path(rest[0]).mkdir(parents=True, exist_ok=True)
                print(str(Path(rest[0])))
            elif "retained R8B verdict" in program and len(rest) >= 2:
                print("0\\t" + RECOVERY)
            elif "print(json.dumps(receipt" in program:
                print("{{}}")
            else:
                print("FIXTURE_PYTHON_CHECK=PASS")
            raise SystemExit(0)
        print("{{}}")
        """
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_v27911_launcher_preflight_executes_current_claim_verifier_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_module = _load_current_fixture_module()
    claim_root = tmp_path / "claim"
    claim_root.mkdir()
    paths = fixture_module.build_current_claim_fixture(claim_root, monkeypatch)
    _prepare_launcher_claim_fixture(paths)
    source = Path(paths["source"])
    evidence = Path(paths["evidence"])

    source_receipt = evidence / "yolov7_claim_gate_32_verification.json"
    source_receipt.write_text(
        json.dumps(
            {
                "schema": "onnx-splitpoint/yolov7-claim-gate-32/v1",
                "status": "PASS",
                "ok": True,
                "valid_claim": True,
                "model_id": "yolov7_paper",
                "case_id": "b066",
                "claim_gate": "claim_gate_32",
                "required_item_count": 32,
                "passed_item_count": 32,
                "claim_eligible": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    profile_target = source / "profiles" / PROFILE.name
    profile_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROFILE, profile_target)
    shutil.copyfile(
        ROOT / "profiles" / "yolo11l_v27910_r8b_full_b067_gate.yaml",
        source / "profiles" / "yolo11l_v27910_r8b_full_b067_gate.yaml",
    )
    for relative in (
        "scripts/verify_v27910_yolo11_r8b_gate.py",
        "scripts/prepare_v27910_yolo11_r8b_recovery.py",
        "scripts/launcher_status_v27911.py",
        "scripts/run_fresh_standard_workflow.sh",
    ):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        target.chmod(0o755)
    acceptance = source / "scripts" / "run_v27911_small_acceptance.sh"
    acceptance.write_text(
        "#!/usr/bin/env bash\nprintf 'V27911_TEST_ACCEPTANCE=PASS\\n'\n",
        encoding="utf-8",
    )
    acceptance.chmod(0o755)

    r8b = tmp_path / "r8b"
    r8b.mkdir()
    (r8b / "yolo11_r8b_verification.json").write_text("{}\n", encoding="utf-8")
    recovery = r8b / "recovery.json"
    recovery.write_text("{}\n", encoding="utf-8")

    fake_python = source / ".venv" / "bin" / "python"
    _write_fake_tool_python(
        fake_python,
        dataset_sha=fixture_module.gate.DATASET_MANIFEST_SHA256,
        reference_sha=fixture_module.gate.REFERENCE_REPORT_SHA256,
        recovery=recovery,
    )
    hardware = tmp_path / "hardware_setups.yaml"
    hardware.write_text("hardware_setups:\n  - id: fixture\n", encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    out_root = tmp_path / "out"
    home = tmp_path / "home"
    home.mkdir()

    env = dict(os.environ)
    env.update(
        {
            "HOME": str(home),
            "TOOL": str(source),
            "TOOL_PYTHON": str(fake_python),
        }
    )
    completed = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--yolo11-r8b-output",
            str(r8b),
            "--yolov7-claim-output",
            str(evidence),
            "--out-root",
            str(out_root),
            "--models-root",
            str(models),
            "--hardware-setups",
            str(hardware),
            "--preflight-only",
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "V27911_YOLOV7_CLAIM_GATE=PASS" in completed.stdout
    assert "V27911_YOLOV7_CLAIM_GATE_32=PASS" in completed.stdout
    assert "V27911_SEVEN_MODEL_LONG_PREFLIGHT=PASS" in completed.stdout
    assert "LONG_RUN_STARTED=NO" in completed.stdout
    controls = sorted(out_root.glob("v27911_7model_b500_audit20_launch_*"))
    assert len(controls) == 1
    receipt = json.loads(
        (controls[0] / "yolov7_claim_gate_32_verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["verification_identity"]["verifier"] == (
        "verify_v27911_yolov7_claim_gate_32.py"
    )
    assert receipt["valid_claim"] is True
