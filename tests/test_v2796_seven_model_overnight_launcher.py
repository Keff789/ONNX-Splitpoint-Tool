from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_v2796_seven_model_long_overnight.sh"
GENERIC_LAUNCHER = ROOT / "scripts/run_v279_seven_model_long_overnight.sh"
STATUS_HELPER = ROOT / "scripts/launcher_status_v2796.py"
HISTORICAL_STATUS_HELPER = ROOT / "scripts/launcher_status_v2795.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_v2796_launcher_is_source_safe_helpful_and_executable() -> None:
    assert os.access(LAUNCHER, os.X_OK)
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    help_result = _run("--help")
    assert help_result.returncode == 0
    assert "run_v2796_seven_model_long_overnight.sh" in help_result.stdout
    assert "--yolo11-r8b-output PATH" in help_result.stdout
    assert "--yolov7-claim-output PATH" in help_result.stdout
    assert "(required)" in help_result.stdout
    assert "--preflight-only" in help_result.stdout

    sourced = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "SOURCE_CONTINUED=YES\\n"',
            "launcher-source-test",
            str(LAUNCHER),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert sourced.returncode == 0
    assert sourced.stdout == "SOURCE_CONTINUED=YES\n"
    assert "was sourced" in sourced.stderr


def test_r8b_output_option_fails_closed_before_host_preflight() -> None:
    missing = _run("--preflight-only")
    assert missing.returncode == 64
    assert "--yolo11-r8b-output is required" in missing.stderr

    missing_yolov7 = _run("--yolo11-r8b-output", "/tmp")
    assert missing_yolov7.returncode == 64
    assert "--yolov7-claim-output is required" in missing_yolov7.stderr

    duplicate = _run(
        "--yolo11-r8b-output",
        "/tmp",
        "--yolo11-r8b-output",
        "/tmp",
    )
    assert duplicate.returncode == 64
    assert "only once" in duplicate.stderr

    duplicate_yolov7 = _run(
        "--yolo11-r8b-output", "/tmp",
        "--yolov7-claim-output", "/tmp",
        "--yolov7-claim-output", "/tmp",
    )
    assert duplicate_yolov7.returncode == 64
    assert "--yolov7-claim-output may be supplied only once" in duplicate_yolov7.stderr

    relative = _run(
        "--yolo11-r8b-output", "relative/gate",
        "--yolov7-claim-output", "/tmp",
    )
    assert relative.returncode == 64
    assert "YOLO11_R8B_OUTPUT muss absolut" in relative.stderr

    noncanonical = _run(
        "--yolo11-r8b-output", "/tmp/../tmp",
        "--yolov7-claim-output", "/tmp",
    )
    assert noncanonical.returncode == 64
    assert "YOLO11_R8B_OUTPUT ist nicht kanonisch" in noncanonical.stderr

    relative_yolov7 = _run(
        "--yolo11-r8b-output", "/tmp",
        "--yolov7-claim-output", "relative/evidence",
    )
    assert relative_yolov7.returncode == 64
    assert "YOLOV7_CLAIM_OUTPUT muss absolut" in relative_yolov7.stderr

    unsupported = _run("--unsupported-v2796-option")
    assert unsupported.returncode == 64
    assert "unsupported option" in unsupported.stderr


def test_v2796_launcher_pins_release_profile_identity_and_acceptance() -> None:
    text = _read(LAUNCHER)
    assert 'expected = "2.79.6"' in text
    assert "V2796_RELEASE_IDENTITY=PASS" in text
    assert 'ACCEPTANCE="$TOOL/scripts/run_v2796_small_acceptance.sh"' in text
    assert 'STATUS_HELPER="$TOOL/scripts/launcher_status_v2796.py"' in text
    assert 'PROFILE="$TOOL/profiles/complete_set_7models_v2796_b500_audit20.yaml"' in text
    assert (
        "PROFILE_EXPECTED_SHA256="
        "9804fc7f99bd79c17b71d152ad5a20cb4a42d7f1488eefbf414ee961f9ab7930"
    ) in text
    assert 'profile.get("name") == "complete_set_7models_v2796_b500_audit20"' in text
    assert 'hailo_build.get("cold_build_timeout_s") == 0' in text
    assert '[0, "off", "none", "unlimited", "disabled"]' in text
    assert 'hailo_build.get("immutable_attempt_receipts") is True' in text
    assert '"last_attempt_even_on_failure"' in text
    assert 'FROZEN_HARDWARE="$FROZEN_PARENT/v2784_7model_hardware_setups.yaml"' in text
    assert ".removeprefix(" not in text
    assert 'declared = declared[7:] if declared.startswith("sha256:") else declared' in text
    assert "> \"$ADMISSION_TMP\" <<'PY'\nfrom __future__ import annotations" in text


def test_r8b_receipt_is_exact_and_precedes_every_launch_path() -> None:
    text = _read(LAUNCHER)
    assert 'GATE_VERIFIER="$TOOL/scripts/verify_v2796_yolo11_r8b_gate.py"' in text
    assert 'GATE_PROFILE="$TOOL/profiles/yolo11l_v2796_r8b_full_b067_gate.yaml"' in text
    assert 'R8B_SOURCE_RECEIPT="$YOLO11_R8B_OUTPUT/yolo11_r8b_verification.json"' in text
    assert '"schema": "onnx-splitpoint/yolo11-r8b-terminal-gate/v1"' in text
    assert text.count('"version": "2.79.6"') >= 3
    assert text.count(
        '"build_id": "v2.79.6-remaining-changes-yolo11-admission-closure"'
    ) >= 3
    assert text.count(
        '"workflow_version": "v2.79.6-remaining-changes-yolo11-admission-closure"'
    ) >= 2
    assert '"required_path_count": 6' in text
    assert '"quality_decision_used_for_terminal": False' in text

    expected_paths = {
        "hailo8_full",
        "hailo10h_full",
        "deepx_full",
        "hailo8_b067_composed",
        "hailo10h_b067_composed",
        "deepx_b067_composed",
    }
    for path_id in expected_paths:
        assert f'"{path_id}"' in text
    assert 'row.get("terminal_status") not in {"success", "blocked"}' in text
    assert 'gate_verdict.get("version") == "2.79.6"' in text
    assert 'gate.get("version") != "2.79.6"' in text

    verify_call = text.index('"$TOOL_PYTHON" -B "$GATE_VERIFIER" \\\n')
    gate_pass = text.index('print("V2796_YOLO11_FULL_GATE=PASS")', verify_call)
    admission = text.index('ADMISSION="$CONTROL/long_run_admission.json"', gate_pass)
    preflight = text.index("if (( PREFLIGHT_ONLY )); then", admission)
    detached = text.index("nohup setsid --fork --wait bash -c", preflight)
    assert verify_call < gate_pass < admission < preflight < detached


def test_yolov7_claim_gate_is_strict_receipt_bound_and_pre_dispatch() -> None:
    text = _read(LAUNCHER)
    assert 'YOLOV7_CLAIM_VERIFIER="$TOOL/scripts/verify_v2796_yolov7_claim_gate_32.py"' in text
    assert 'YOLOV7_CLAIM_RESULT="$YOLOV7_CLAIM_OUTPUT/native_three_stage_result.json"' in text
    assert (
        'YOLOV7_CLAIM_SOURCE_RECEIPT="$YOLOV7_CLAIM_OUTPUT/'
        'yolov7_claim_gate_32_verification.json"'
    ) in text
    for marker in (
        '"schema": "onnx-splitpoint/yolov7-claim-gate-32/v1"',
        '"valid_claim": True',
        '"model_id": "yolov7_paper"',
        '"case_id": "b066"',
        '"claim_gate": "claim_gate_32"',
        '"required_item_count": 32',
        '"passed_item_count": 32',
        '"claim_eligible": True',
        '--expected-source-manifest-sha256 "$SOURCE_MANIFEST_SHA256"',
    ):
        assert marker in text
    verifier = text.index('"$TOOL_PYTHON" -B "$YOLOV7_CLAIM_VERIFIER"')
    pass_marker = text.index('print("V2796_YOLOV7_CLAIM_GATE_32=PASS")', verifier)
    admission = text.index('ADMISSION="$CONTROL/long_run_admission.json"', pass_marker)
    detached = text.index("nohup setsid --fork --wait bash -c", admission)
    assert verifier < pass_marker < admission < detached
    assert 'printf "%s  %s\\n" "$yolov7_claim_sha" "$yolov7_claim_receipt" | sha256sum -c -' in text


def test_preflight_is_receipt_bound_and_cannot_dispatch() -> None:
    text = _read(LAUNCHER)
    branch_start = text.index("if (( PREFLIGHT_ONLY )); then")
    branch_end = text.index("\nfi\n", branch_start) + len("\nfi\n")
    branch = text[branch_start:branch_end]
    assert "V2796_SEVEN_MODEL_LONG_PREFLIGHT=PASS" in branch
    assert "V2796_YOLO11_FULL_GATE=PASS" in branch
    assert "V2796_YOLOV7_CLAIM_GATE_32=PASS" in branch
    assert "LONG_RUN_STARTED=NO" in branch
    assert "YOLO11_R8B_WORKFLOW_RERUN=NO" in branch
    assert "YOLOV7_CLAIM_GATE_WORKFLOW_RERUN=NO" in branch
    assert "YOLO11_R8B_VERDICT" in branch
    assert "YOLOV7_CLAIM_VERDICT" in branch
    assert "exit 0" in branch
    assert "nohup" not in branch


def test_v2796_status_helper_is_frozen_and_current_alias_tracks_v27913() -> None:
    generic = _read(GENERIC_LAUNCHER)
    assert (
        'CURRENT_LAUNCHER="$SCRIPT_DIR/run_v27913_seven_model_long_overnight.sh"'
        in generic
    )
    assert 'exec bash "$CURRENT_LAUNCHER" "$@"' in generic
    assert "run_v2796_seven_model_long_overnight.sh" not in generic
    assert (ROOT / "scripts/run_v27913_seven_model_long_overnight.sh").is_file()
    subprocess.run(["bash", "-n", str(GENERIC_LAUNCHER)], check=True)

    expected = _read(HISTORICAL_STATUS_HELPER).replace(
        "for v2.79.5 campaigns", "for v2.79.6 campaigns"
    )
    assert _read(STATUS_HELPER) == expected
    subprocess.run(["python3", "-m", "py_compile", str(STATUS_HELPER)], check=True)
