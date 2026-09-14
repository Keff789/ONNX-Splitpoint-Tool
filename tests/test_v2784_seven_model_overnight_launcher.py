from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_v2784_seven_model_long_overnight.sh"


def _launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_is_shell_valid_with_help_and_source_guard() -> None:
    assert LAUNCHER.is_file()
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    help_result = subprocess.run(
        ["bash", str(LAUNCHER), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--gate-output PATH" in help_result.stdout
    assert "--preflight-only" in help_result.stdout

    source_result = subprocess.run(
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
    assert source_result.returncode == 0
    assert source_result.stdout == "SOURCE_CONTINUED=YES\n"
    assert "was sourced" in source_result.stderr

    invalid_result = subprocess.run(
        ["bash", str(LAUNCHER), "--unsupported-v2784-option"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid_result.returncode == 64
    assert "unsupported option" in invalid_result.stderr


def test_launcher_reuses_gate_a_read_only_and_never_reruns_it() -> None:
    text = _launcher_text()

    assert "verify_v2783_yolo11_gate_a_output.py" in text
    assert '--gate-output "$GATE_OUTPUT"' in text
    assert '--workflow-rc "$WORKFLOW_RC"' in text
    assert '--expected-profile "$GATE_PROFILE"' in text
    assert '"outcome": "ANCHOR_FOUND"' in text
    assert '"anchor_boundary": 67' in text
    assert '"expected_stop_after_partial": True' in text
    assert '"workflow_rc": 1' in text
    assert '"full_source_onnx_sha256"' in text
    assert 'gate_full_sha == models["yolo11l"]["sha256"]' in text
    assert "_load_valid_hailo_receipt" in text
    assert 'for target in ("hailo8", "hailo10")' in text
    assert "GATE_A_RERUN=NO" in text
    assert "YOLO11_ANCHOR_REUSED=b067" in text

    # The only admitted Gate A is the retained output.  The online producer is
    # deliberately absent from this launcher.
    assert "run_v2783_yolo11_hailo8_first_gate" not in text
    assert "hailo8_first_common_anchor" not in text
    assert "--resume" not in text
    assert "rm -" not in text


def test_launcher_fail_closed_admission_freezes_the_exact_campaign() -> None:
    text = _launcher_text()

    assert "V2784_RELEASE_IDENTITY=PASS" in text
    assert "run_v278_small_acceptance.sh" in text
    assert "complete_set_7models_v2784_b500_audit20.yaml" in text
    assert 'PROFILE="$CONTROL/profile.yaml"' in text
    assert 'cmp --silent -- "$PROFILE_SOURCE" "$PROFILE"' in text
    assert "load_evaluation_profile" in text
    assert "build_effective_execution_plan" in text

    for model_id in (
        "resnet50",
        "yolo26s",
        "yolov7_paper",
        "mobilenet_v3_large",
        "regnet_x_1_6gf",
        "yolo26m",
        "yolo11l",
    ):
        assert f'"{model_id}"' in text

    assert 'selection.get("audit_size") == 20' in text
    assert 'selection.get("minimum_valid_audit_candidates") == 10' in text
    assert 'selection.get("audit_seed") == 20260710' in text
    assert 'selection.get("forced_cases") == {"yolo11l": ["b067"]}' in text
    assert '"execution_union_order_policy": "forced_deployment_then_audit"' in text
    assert 'limits.get("classification") == 500' in text
    assert 'limits.get("detection") == 500' in text
    assert 'scheduler.get("max_workers") == 3' in text
    assert 'scheduler.get("cpu_tokens") == 8' in text
    assert 'scheduler.get("ram_mb") == 0' in text
    assert 'scheduler.get("ram_reserve_mb") == 2048' in text
    assert 'workflow.get("parallel_remote_setups") is True' in text
    assert 'workflow.get("max_parallel_setups") == 3' in text
    assert 'dict(profile.get("native_producers") or {}).get("enabled") is False' in text
    assert 'energy.get("enabled") is False' in text

    for setup_id in (
        "orin_nx_hailo8_01",
        "orin_nx_hailo10_01",
        "orin_nx_deepx_m1_01",
    ):
        assert f'"{setup_id}"' in text
    assert "normalize_hardware_targets" in text
    assert "hardware_host:" in text
    assert "hardware_user:" in text
    assert "build_environment:" in text


def test_launcher_detaches_only_the_project_native_fresh_workflow() -> None:
    text = _launcher_text()

    assert "run_fresh_standard_workflow.sh" in text
    assert "nohup setsid --fork --wait bash -c" in text
    assert '--profile "$profile"' in text
    assert '--out "$run_root"' in text
    assert "--output-root" not in text
    assert "--only-model" not in text
    assert "flock -n" in text
    assert "STATE=STARTING" in text
    assert "write_status RUNNING" in text
    assert "write_status COMPLETED 0" in text
    assert "write_status FAILED" in text
    assert "LONG_RUN_PID_FILE" in text
    assert "LONG_RUN_STATUS" in text
    assert "LONG_RUN_LOG" in text
    assert "DETACHED=YES" in text
    assert "NATIVE_RUN=NO" in text
    assert "ENERGY_RUN=NO" in text
