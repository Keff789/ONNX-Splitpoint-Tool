from __future__ import annotations

import os
from pathlib import Path
import subprocess
import venv


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_v2798_yolo11_r8b_gate.sh"
BUILD_ID = "v2.79.8-yolo11-gate-profile-schema-closure"


def _source() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_v2798_gate_launcher_is_source_safe_and_has_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    help_result = subprocess.run(
        ["bash", str(LAUNCHER), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "run_v2798_yolo11_r8b_gate.sh" in help_result.stdout

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


def test_standard_venv_python_symlink_passes_semantic_binding(
    tmp_path: Path,
) -> None:
    tool = tmp_path / "tool"
    models = tmp_path / "models"
    output = tmp_path / "output"
    recovery = tmp_path / "v2798_recovery_manifest.json"
    tool.mkdir()
    models.mkdir()
    recovery.write_text("{}\n", encoding="utf-8")
    venv.EnvBuilder(with_pip=False, symlinks=True).create(tool / ".venv")
    python = tool / ".venv/bin/python"
    assert python.is_symlink()

    env = os.environ.copy()
    env.update(
        {
            "TOOL": str(tool),
            "MODELS_ROOT": str(models),
            "OUTPUT_ROOT": str(output),
        }
    )
    result = subprocess.run(
        [
            "bash", str(LAUNCHER),
            "--recovery-manifest", str(recovery),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    # The deliberately incomplete fixture must stop at the first required
    # release file, after the ordinary venv symlink has been accepted.
    assert result.returncode == 66
    assert "Erforderliche reguläre Datei fehlt" in result.stderr
    assert "TOOL_PYTHON muss" not in result.stderr
    assert "Tool-Python gehört nicht zur Tool-venv" not in result.stderr


def test_launcher_pins_v2798_identity_and_all_prehardware_checks() -> None:
    source = _source()
    assert 'expected_version = "2.79.8"' in source
    assert f'expected_build = "{BUILD_ID}"' in source
    assert 'PROFILE="$TOOL/profiles/yolo11l_v2798_r8b_full_b067_gate.yaml"' in source
    assert 'VERIFIER="$TOOL/scripts/verify_v2798_yolo11_r8b_gate.py"' in source
    assert (
        "PROFILE_SHA256="
        "f66935318d34232e8abb896d6ffbc46e7d671ec52e83827878e3798b83ad18be"
        in source
    )
    assert "V2798_R8B_RELEASE_IDENTITY=PASS" in source
    assert "V2798_R8B_SOURCE_MANIFEST=PASS" in source
    assert "V2798_YOLO11_R8B_GATE=PASS" in source
    assert 'importlib.metadata.version("onnx-splitpoint-tool")' in source
    assert "onnx_splitpoint_tool.__release__" in source
    assert "onnx_splitpoint_tool.__build_id__" in source
    assert "WORKFLOW_VERSION" in source
    assert "profile_sha256_mismatch" in source
    assert "model_sha256_mismatch" in source
    assert "hardware_setup_identity_mismatch" in source
    assert '--recovery-manifest "$RECOVERY_MANIFEST"' in source
    assert 'RECOVERY_MANIFEST="$GATE_OUTPUT/v2798_recovery_manifest.json"' in source
    assert "os.O_EXCL" in source
    assert "O_NOFOLLOW" in source

    release = source.index("V2798_R8B_RELEASE_IDENTITY=PASS")
    manifest = source.index("V2798_R8B_SOURCE_MANIFEST=PASS")
    profile = source.index("profile_sha256_mismatch")
    workflow = source.index('"$TOOL_PYTHON" -B "$WORKFLOW"')
    assert release < manifest < profile < workflow


def test_heredocs_are_not_part_of_an_or_list_and_errors_fail_closed() -> None:
    source = _source()
    assert "<<'PY' ||" not in source
    assert source.count("if ! \"$TOOL_PYTHON\"") == 4
    assert source.count("\nPY\nthen\n") == 3
    assert 'die 65 "Installierte v2.79.8-Package-/Workflow-Identität stimmt nicht"' in source
    assert 'die 65 "R8B-Profil-, Modell- oder Hardwareidentität stimmt nicht"' in source


def test_python_path_policy_is_lexical_then_semantic() -> None:
    source = _source()
    assert 'realpath -m -s -- "$value"' in source
    assert "os.path.realpath(sys.prefix) != expected_prefix" in source
    assert "os.path.dirname(requested_executable)" in source
    assert "os.path.realpath(sys.executable)" in source


def test_recovery_manifest_is_mandatory_and_staged_byte_exact(
    tmp_path: Path,
) -> None:
    missing = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 64
    assert "--recovery-manifest ist obligatorisch" in missing.stderr

    source = _source()
    assert "RECOVERY_MANIFEST_SOURCE_SHA256" in source
    assert "RECOVERY_MANIFEST_SOURCE_SIZE" in source
    assert "recovery copy source identity mismatch" in source
    assert "recovery copy target hash mismatch" in source
    assert "Externes Recovery-Manifest wurde während der Übernahme verändert" in source
