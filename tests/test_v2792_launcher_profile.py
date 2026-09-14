from __future__ import annotations

from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_v2792_seven_model_long_overnight.sh"
PROFILE = ROOT / "profiles/complete_set_7models_v2792_b500_audit20.yaml"
STATUS = ROOT / "scripts/launcher_status_v2792.py"


def test_v2792_launcher_has_consistent_release_profile_and_registry() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    help_result = subprocess.run(
        ["bash", str(LAUNCHER), "--help"],
        text=True, capture_output=True, check=False,
    )
    assert help_result.returncode == 0
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'expected = "2.79.2"' in text
    assert "complete_set_7models_v2792_b500_audit20.yaml" in text
    assert "v2792_7model_hardware_setups.yaml" in text
    assert "launcher_status_v2792.py" in text
    assert "--preflight-only" in text
    assert "V279_TOOL_PYTHON_VENV=PASS" in text
    assert "hailo10_to_tensorrt" in text
    assert '"hailo10_to_trt"' not in text
    assert "--worker-pid" in text and "--monitor-pid" in text
    assert "--workflow-rc" in text


def test_v2792_profile_seals_global_scope_before_compiler() -> None:
    payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    required = payload["campaign"]["required_run_scope"]
    assert required == {
        "seal_before_first_compiler_dispatch": True,
        "authoritative_source": "global_effective_execution_plan",
        "model_local_plan_may_shrink_scope": False,
        "build_failure_may_shrink_scope": False,
        "terminal_outcome_required": True,
    }
    hailo = payload["hailo_build"]
    assert hailo["immutable_attempt_receipts"] is True
    assert hailo["terminal_attempt_selection"] == "last_attempt_even_on_failure"
    assert hailo["timeout_s"] == 9000
    assert set(hailo["hard_timeout_disable_tokens"]) == {0, "off", "none"}
    assert payload["ranking_validation"]["methods"] == ["cut_bytes_only"]
    assert payload["native_producers"]["enabled"] is False
    assert payload["energy"]["enabled"] is False


def test_v2792_status_helper_records_periodic_fields() -> None:
    text = STATUS.read_text(encoding="utf-8")
    for marker in (
        "progress_completed", "progress_total", "log_age_s",
        "worker_pid", "monitor_pid", "workflow_rc", "atomic_write_launcher_status",
    ):
        assert marker in text
