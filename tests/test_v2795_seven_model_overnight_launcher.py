from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_v2795_seven_model_long_overnight.sh"
V2792_LAUNCHER = ROOT / "scripts" / "run_v2792_seven_model_long_overnight.sh"
V2792_STATUS_HELPER = ROOT / "scripts" / "launcher_status_v2792.py"
V2795_STATUS_HELPER = ROOT / "scripts" / "launcher_status_v2795.py"
FROZEN_PROFILE = (
    ROOT / "profiles" / "complete_set_7models_v2792_b500_audit20.yaml"
)
FROZEN_PROFILE_SHA256 = (
    "f609ea640d268bca1a72fd4eb66e58a3dee494624577f3272bfd8bc08b5d6467"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_v2795_launcher_is_the_v2792_campaign_with_release_closure_only() -> None:
    """Guard the scientific campaign while permitting release-local names."""

    expected = _read(V2792_LAUNCHER)
    replacements = (
        (
            "run_v2792_seven_model_long_overnight.sh",
            "run_v2795_seven_model_long_overnight.sh",
        ),
        (
            "Frozen v2.79.2 seven-model profile.",
            "Frozen v2.79.2 scientific seven-model profile.",
        ),
        ("run_v2792_small_acceptance.sh", "run_v2795_small_acceptance.sh"),
        ("launcher_status_v2792.py", "launcher_status_v2795.py"),
        ('expected = "2.79.2"', 'expected = "2.79.5"'),
        ("v2.79.2 required:", "v2.79.5 required:"),
        ("V2792_RELEASE_IDENTITY=PASS", "V2795_RELEASE_IDENTITY=PASS"),
        ("v2792_7model_hardware_setups.yaml", "v2795_7model_hardware_setups.yaml"),
        ("v2792_7model_b500_audit20.lock", "v2795_7model_b500_audit20.lock"),
        (
            "Vorhandener v2.79.2-Hardware-Freeze",
            "Vorhandener v2.79.5-Hardware-Freeze",
        ),
        (
            "v2792_7model_b500_audit20_launch_",
            "v2795_7model_b500_audit20_launch_",
        ),
        (
            "onnx-splitpoint/v2792-seven-model-long-run-admission/v1",
            "onnx-splitpoint/v2795-seven-model-long-run-admission/v1",
        ),
        (
            "V2792_SEVEN_MODEL_LONG_ADMISSION=PASS",
            "V2795_SEVEN_MODEL_LONG_ADMISSION=PASS",
        ),
        (
            "Ein anderer v2.79.2-Sieben-Modell-Langlauf",
            "Ein anderer v2.79.5-Sieben-Modell-Langlauf",
        ),
        (
            "V2792_SEVEN_MODEL_LONG_PREFLIGHT=PASS",
            "V2795_SEVEN_MODEL_LONG_PREFLIGHT=PASS",
        ),
        (
            "V2792_SEVEN_MODEL_LONG_RUN=STARTED",
            "V2795_SEVEN_MODEL_LONG_RUN=STARTED",
        ),
        ("global v2.79.2 campaign lock", "global v2.79.5 campaign lock"),
        ("v2792-seven-model-worker", "v2795-seven-model-worker"),
    )
    for old, new in replacements:
        assert old in expected
        expected = expected.replace(old, new)

    # v2.79.5 additionally holds the historical v2.79.2 campaign lock for the
    # complete worker lifetime.  This is operational admission hardening only;
    # it prevents old and current launchers from sharing the same hardware.
    expected = expected.replace(
        'GLOBAL_LOCK="$LOCK_PARENT/v2795_7model_b500_audit20.lock"\n',
        'GLOBAL_LOCK="$LOCK_PARENT/v2795_7model_b500_audit20.lock"\n'
        'LEGACY_GLOBAL_LOCK="$LOCK_PARENT/v2792_7model_b500_audit20.lock"\n',
    )
    expected = expected.replace(
        '[[ ! -L "$GLOBAL_LOCK" ]] || die 67 "Langlauf-Lock ist ein Symlink"\n',
        '[[ ! -L "$GLOBAL_LOCK" ]] || die 67 "Langlauf-Lock ist ein Symlink"\n'
        '[[ ! -L "$LEGACY_GLOBAL_LOCK" ]] ||\n'
        '  die 67 "Kompatibilitäts-Langlauf-Lock ist ein Symlink"\n',
    )
    expected = expected.replace(
        'flock -u "$probe_fd"\nexec {probe_fd}>&-\n',
        'exec {legacy_probe_fd}>"$LEGACY_GLOBAL_LOCK"\n'
        'if ! flock -n "$legacy_probe_fd"; then\n'
        '  exec {legacy_probe_fd}>&-\n'
        '  flock -u "$probe_fd"\n'
        '  exec {probe_fd}>&-\n'
        '  die 75 "Ein älterer v2.79.2-Sieben-Modell-Langlauf hält bereits den Kompatibilitäts-Lock"\n'
        'fi\n'
        'flock -u "$legacy_probe_fd"\n'
        'exec {legacy_probe_fd}>&-\n'
        'flock -u "$probe_fd"\nexec {probe_fd}>&-\n',
        1,
    )
    expected = expected.replace(
        '  status_helper="${14}"\n',
        '  status_helper="${14}"\n  legacy_lock="${15}"\n',
    )
    expected = expected.replace(
        '    exit 75\n  fi\n\n  printf "%s  %s\\n" "$profile_sha"',
        '    exit 75\n  fi\n'
        '  exec 8>"$legacy_lock"\n'
        '  if ! flock -n 8; then\n'
        '    "$tool/.venv/bin/python" -B "$status_helper" write \\\n'
        '      --status "$status" --state LOCKED --phase lock_admission \\\n'
        '      --log "$log" --run-root "$run_root" --workflow-rc 75 \\\n'
        '      --detail "legacy v2.79.2 campaign compatibility lock unavailable"\n'
        '    exit 75\n'
        '  fi\n\n  printf "%s  %s\\n" "$profile_sha"',
    )
    expected = expected.replace(
        '  "$LOG" "$STATUS_HELPER" \\\n',
        '  "$LOG" "$STATUS_HELPER" "$LEGACY_GLOBAL_LOCK" \\\n',
    )

    assert _read(LAUNCHER) == expected


def test_v2795_status_helper_is_logically_byte_identical_to_v2792() -> None:
    expected = _read(V2792_STATUS_HELPER).replace(
        "for v2.79.2 campaigns", "for v2.79.5 campaigns"
    )
    assert _read(V2795_STATUS_HELPER) == expected
    subprocess.run(
        ["python3", "-m", "py_compile", str(V2795_STATUS_HELPER)], check=True
    )


def test_frozen_v2792_profile_is_byte_identical_and_pinned() -> None:
    profile_bytes = FROZEN_PROFILE.read_bytes()
    assert hashlib.sha256(profile_bytes).hexdigest() == FROZEN_PROFILE_SHA256

    text = _read(LAUNCHER)
    assert "complete_set_7models_v2792_b500_audit20.yaml" in text
    assert 'profile.get("name") == "complete_set_7models_v2792_b500_audit20"' in text
    assert 'PROFILE="$CONTROL/profile.yaml"' in text
    assert 'cmp --silent -- "$PROFILE_SOURCE" "$PROFILE"' in text
    assert 'PROFILE_SHA256="$(sha256sum "$PROFILE"' in text


def test_v2795_launcher_requires_exact_current_identity_and_acceptance() -> None:
    text = _read(LAUNCHER)

    assert 'expected = "2.79.5"' in text
    assert 'distribution = importlib.metadata.version("onnx-splitpoint-tool")' in text
    assert "package != expected or distribution != expected" in text
    assert "V2795_RELEASE_IDENTITY=PASS" in text
    assert 'ACCEPTANCE="$TOOL/scripts/run_v2795_small_acceptance.sh"' in text
    assert 'STATUS_HELPER="$TOOL/scripts/launcher_status_v2795.py"' in text
    assert 'bash "$ACCEPTANCE"' in text
    assert 'LEGACY_GLOBAL_LOCK="$LOCK_PARENT/v2792_7model_b500_audit20.lock"' in text
    assert 'exec 8>"$legacy_lock"' in text
    assert "legacy v2.79.2 campaign compatibility lock unavailable" in text


def test_preflight_only_is_host_independent_and_cannot_start_workflow() -> None:
    text = _read(LAUNCHER)
    branch_start = text.index("if (( PREFLIGHT_ONLY )); then")
    branch_end = text.index("\nfi\n", branch_start) + len("\nfi\n")
    status_start = text.index('"$TOOL_PYTHON" -B "$STATUS_HELPER" write', branch_end)
    detach_start = text.index("nohup setsid --fork --wait bash -c", status_start)
    branch = text[branch_start:branch_end]

    assert "V2795_SEVEN_MODEL_LONG_PREFLIGHT=PASS" in branch
    assert "LONG_RUN_STARTED=NO" in branch
    assert "GATE_A_RERUN=NO" in branch
    assert "exit 0" in branch
    assert branch_end < status_start < detach_start

    # Admission reads the frozen registry and retained receipts locally.  It
    # deliberately performs no network or remote-shell operation.
    for remote_command in ("ssh ", "scp ", "rsync ", "sftp "):
        assert remote_command not in text


def test_historical_v2795_launcher_is_shell_valid_and_source_safe() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "run_v2795_seven_model_long_overnight.sh" in result.stdout
    assert "--preflight-only" in result.stdout

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

    invalid = subprocess.run(
        ["bash", str(LAUNCHER), "--unsupported-v2795-option"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 64
    assert "unsupported option" in invalid.stderr
