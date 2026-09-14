from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_v2792_seven_model_long_overnight.sh"

def test_v279_launcher_syntax_help_and_known_fixes() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    result = subprocess.run(["bash", str(LAUNCHER), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "V279_TOOL_PYTHON_VENV=PASS" in text
    assert 'hailo10_to_tensorrt' in text
    assert '"hailo10_to_trt"' not in text
    assert "complete_set_7models_v2792_b500_audit20.yaml" in text
    assert 'expected = "2.79.2"' in text

def test_v279_launcher_keeps_gate_a_and_campaign_frozen() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "verify_v2783_yolo11_gate_a_output.py" in text
    assert "GATE_A_RERUN=NO" in text
    assert "YOLO11_ANCHOR_REUSED=b067" in text
    assert 'selection.get("audit_size") == 20' in text
    assert 'selection.get("audit_seed") == 20260710' in text
    assert 'dict(profile.get("native_producers") or {}).get("enabled") is False' in text
    assert 'energy.get("enabled") is False' in text
