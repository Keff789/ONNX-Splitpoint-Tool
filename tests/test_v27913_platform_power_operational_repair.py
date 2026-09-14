from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.energy import method_manifest
from onnx_splitpoint_tool.gui.panels.panel_hardware import (
    _fs_energy_method_badge_state,
    _run_gui_m2_idle_calibration,
    _valid_energy_method_attestor,
)
from onnx_splitpoint_tool import platform_power, source_integrity
from onnx_splitpoint_tool.release_identity import BUILD_ID, VERSION
from scripts import build_source_manifest, refresh_editable_install


PRESERVED_ROOTS = {
    ".venv",
    ".venv-report",
    ".git",
    ".agents",
    ".codex",
    "logs",
    ".install_logs",
    "EvaluationRuns",
    "RemoteBenchmarkRuns",
    "BenchmarkSets",
    "artifact_store",
    "build_evidence",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_tree(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    files = {
        "pyproject.toml": (
            f'[project]\nname = "onnx-splitpoint-tool"\nversion = "{VERSION}"\n'
        ),
        "onnx_splitpoint_tool/release_identity.py": (
            f'VERSION = "{VERSION}"\nBUILD_ID = "{BUILD_ID}"\n'
        ),
        "onnx_splitpoint_tool/runtime.py": "VALUE = 1\n",
        "scripts/run.py": "raise SystemExit(0)\n",
        "profiles/shipped.yaml": "name: shipped\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(
            (root / relative for relative in files),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    ]
    (root / "SOURCE_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema": source_integrity.SOURCE_MANIFEST_SCHEMA,
                "package_version": VERSION,
                "workflow_version": BUILD_ID,
                "file_count": len(rows),
                "files": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "SHA256SUMS.txt").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows),
        encoding="utf-8",
    )
    return root


def test_gui_prepares_exactly_missing_method_before_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "hardware_setups.yaml"
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        method_manifest,
        "verify_configured_energy_method",
        lambda setup_id, *, registry_path: calls.append(("verify", registry_path))
        or {"configured": False, "verified": False, "status": "missing"},
    )

    def prepare(**kwargs: object) -> dict[str, object]:
        calls.append(("prepare", kwargs))
        return {"ok": True, "hardware_action_performed": False}

    monkeypatch.setattr(
        method_manifest,
        "prepare_configured_energy_method",
        prepare,
    )
    monkeypatch.setattr(
        platform_power,
        "calibrate_m2_accelerator_idle_power",
        lambda setup_id, **kwargs: calls.append(("calibrate", kwargs))
        or "calibrated",
    )

    result = _run_gui_m2_idle_calibration(
        "orin_nx_hailo8_01",
        registry_path=registry,
        callback=None,
        prepare_missing_method=True,
        attested_by="Kevin Mika",
    )

    assert result == "calibrated"
    assert [name for name, _payload in calls] == [
        "verify",
        "prepare",
        "calibrate",
    ]
    prepare_call = dict(calls[1][1])
    assert prepare_call == {
        "attested_by": "Kevin Mika",
        "accepted_validated_method_reuse": True,
        "registry_path": registry,
    }
    assert dict(calls[2][1])["registry_path"] == registry


@pytest.mark.parametrize(
    "prepared",
    (
        {"ok": False, "hardware_action_performed": False},
        {"ok": True, "hardware_action_performed": True},
        {"ok": True},
    ),
)
def test_gui_never_calibrates_after_nonexact_preparation_result(
    monkeypatch: pytest.MonkeyPatch,
    prepared: dict[str, object],
) -> None:
    monkeypatch.setattr(
        method_manifest,
        "verify_configured_energy_method",
        lambda *_args, **_kwargs: {
            "configured": False,
            "verified": False,
            "status": "missing",
        },
    )
    monkeypatch.setattr(
        method_manifest,
        "prepare_configured_energy_method",
        lambda **_kwargs: prepared,
    )
    called = False

    def calibrate(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        platform_power,
        "calibrate_m2_accelerator_idle_power",
        calibrate,
    )

    with pytest.raises(
        method_manifest.EnergyMethodManifestError,
        match="hardware-free success result",
    ):
        _run_gui_m2_idle_calibration(
            "orin_nx_hailo8_01",
            registry_path=None,
            callback=None,
            prepare_missing_method=True,
            attested_by="Kevin Mika",
        )
    assert called is False


def test_gui_does_not_overwrite_invalid_or_incomplete_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        method_manifest,
        "verify_configured_energy_method",
        lambda *_args, **_kwargs: {
            "configured": True,
            "verified": False,
            "status": "configured_energy_method_admission_failed",
            "configured_method_admission_errors": ["method_sha256_mismatch"],
        },
    )
    monkeypatch.setattr(
        method_manifest,
        "prepare_configured_energy_method",
        lambda **_kwargs: pytest.fail("invalid method was overwritten"),
    )
    monkeypatch.setattr(
        platform_power,
        "calibrate_m2_accelerator_idle_power",
        lambda *_args, **_kwargs: pytest.fail("invalid method reached hardware path"),
    )

    with pytest.raises(
        method_manifest.EnergyMethodManifestError,
        match="method_sha256_mismatch",
    ):
        _run_gui_m2_idle_calibration(
            "orin_nx_hailo8_01",
            registry_path=None,
            callback=None,
            prepare_missing_method=True,
            attested_by="Kevin Mika",
        )


def test_gui_skips_preparation_if_method_was_prepared_after_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        method_manifest,
        "verify_configured_energy_method",
        lambda *_args, **_kwargs: {
            "configured": True,
            "verified": True,
            "status": "inherited_validated_method_verified",
        },
    )
    monkeypatch.setattr(
        method_manifest,
        "prepare_configured_energy_method",
        lambda **_kwargs: pytest.fail("verified method was prepared again"),
    )
    monkeypatch.setattr(
        platform_power,
        "calibrate_m2_accelerator_idle_power",
        lambda *_args, **_kwargs: "calibrated",
    )

    assert (
        _run_gui_m2_idle_calibration(
            "orin_nx_hailo8_01",
            registry_path=None,
            callback=None,
            prepare_missing_method=True,
            attested_by="Kevin Mika",
        )
        == "calibrated"
    )


def test_gui_verified_normal_path_does_not_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        method_manifest,
        "verify_configured_energy_method",
        lambda *_args, **_kwargs: pytest.fail(
            "normal verified path performed an extra GUI preflight"
        ),
    )
    monkeypatch.setattr(
        method_manifest,
        "prepare_configured_energy_method",
        lambda **_kwargs: pytest.fail("verified method was prepared again"),
    )
    monkeypatch.setattr(
        platform_power,
        "calibrate_m2_accelerator_idle_power",
        lambda *_args, **_kwargs: "calibrated",
    )

    assert (
        _run_gui_m2_idle_calibration(
            "orin_nx_hailo8_01",
            registry_path=None,
            callback=None,
            prepare_missing_method=False,
        )
        == "calibrated"
    )


def test_gui_does_not_prepare_incomplete_missing_reference_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        method_manifest,
        "verify_configured_energy_method",
        lambda *_args, **_kwargs: {
            "configured": False,
            "verified": False,
            "status": "incomplete_configuration",
            "configuration_errors": ["calibration_sha256_missing"],
        },
    )
    monkeypatch.setattr(
        method_manifest,
        "prepare_configured_energy_method",
        lambda **_kwargs: pytest.fail("incomplete method was overwritten"),
    )
    monkeypatch.setattr(
        platform_power,
        "calibrate_m2_accelerator_idle_power",
        lambda *_args, **_kwargs: pytest.fail(
            "incomplete method reached hardware path"
        ),
    )

    with pytest.raises(
        method_manifest.EnergyMethodManifestError,
        match="calibration_sha256_missing",
    ):
        _run_gui_m2_idle_calibration(
            "orin_nx_hailo8_01",
            registry_path=None,
            callback=None,
            prepare_missing_method=True,
            attested_by="Kevin Mika",
        )


@pytest.mark.parametrize(
    "value,expected",
    (
        ("Kevin Mika", True),
        ("kmika", True),
        ("", False),
        (" Kevin", False),
        ("Kevin\nMika", False),
        (None, False),
    ),
)
def test_gui_attestor_is_explicit_and_canonical(
    value: object,
    expected: bool,
) -> None:
    assert _valid_energy_method_attestor(value) is expected


@pytest.mark.parametrize(
    "energy,missing_field",
    (
        ({"calibration_manifest": "/tmp/method.json"}, "calibration_sha256"),
        ({"calibration_sha256": "0" * 64}, "calibration_manifest"),
    ),
)
def test_gui_partial_method_reference_is_invalid_not_missing(
    energy: dict[str, str],
    missing_field: str,
) -> None:
    state = _fs_energy_method_badge_state("setup", energy, verifier=None)
    assert state["status"] == "invalid"
    assert state["level"] == "error"
    assert missing_field in state["detail"]


def test_runtime_and_installed_scope_skip_only_real_preserved_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _release_tree(tmp_path)
    for name in PRESERVED_ROOTS:
        operational = root / name
        operational.mkdir()
        (operational / "large-runtime-output.bin").write_bytes(b"x" * 1024)
        (operational / "current").symlink_to("large-runtime-output.bin")

    runtime_hashes: list[str] = []
    original_runtime_hash = source_integrity._sha256_file

    def runtime_hash(path: Path) -> str:
        runtime_hashes.append(path.relative_to(root).as_posix())
        return original_runtime_hash(path)

    build_hashes: list[str] = []
    original_build_hash = build_source_manifest._sha256

    def build_hash(path: Path) -> str:
        build_hashes.append(path.relative_to(root).as_posix())
        return original_build_hash(path)

    monkeypatch.setattr(source_integrity, "_sha256_file", runtime_hash)
    monkeypatch.setattr(build_source_manifest, "_sha256", build_hash)

    runtime = source_integrity.verify_installed_source_integrity(root)
    installed = build_source_manifest.verify(root, scope="installed")
    release = build_source_manifest.verify(root, scope="release")

    assert runtime["ok"] is True
    assert installed["ok"] is True
    assert release["ok"] is False
    for relative in runtime_hashes + build_hashes:
        assert relative.split("/", 1)[0] not in PRESERVED_ROOTS


@pytest.mark.parametrize("kind", ("symlink", "regular_file"))
def test_preserved_root_name_is_rejected_unless_it_is_a_real_directory(
    tmp_path: Path,
    kind: str,
) -> None:
    root = _release_tree(tmp_path)
    candidate = root / "logs"
    if kind == "symlink":
        candidate.symlink_to(root / "scripts", target_is_directory=True)
    else:
        candidate.write_text("not a directory\n", encoding="utf-8")

    runtime = source_integrity.verify_installed_source_integrity(root)
    installed = build_source_manifest.verify(root, scope="installed")

    assert runtime["ok"] is False
    assert installed["ok"] is False
    assert "logs" in runtime["symlinks"] or any(
        row["path"] == "logs" for row in runtime["unexpected_extras"]
    )
    assert "logs" in installed["symlinks"] or any(
        row["path"] == "logs" for row in installed["unexpected_extras"]
    )


def test_nested_preserved_name_remains_release_owned_and_strict(
    tmp_path: Path,
) -> None:
    root = _release_tree(tmp_path)
    nested = root / "scripts" / "logs"
    nested.mkdir()
    (nested / "unexpected.py").write_text("VALUE = 2\n", encoding="utf-8")

    runtime = source_integrity.verify_installed_source_integrity(root)
    installed = build_source_manifest.verify(root, scope="installed")

    assert runtime["ok"] is False
    assert installed["ok"] is False
    assert any(
        row["path"] == "scripts/logs/unexpected.py"
        for row in runtime["unexpected_extras"]
    )
    assert any(
        row["path"] == "scripts/logs/unexpected.py"
        for row in installed["unexpected_extras"]
    )


def test_runtime_failure_diagnostics_are_bounded_but_keep_exact_count(
    tmp_path: Path,
) -> None:
    root = _release_tree(tmp_path)
    unknown = root / "unknown"
    unknown.mkdir()
    for index in range(105):
        (unknown / f"extra-{index:03d}.txt").write_text("x", encoding="utf-8")

    result = source_integrity.verify_installed_source_integrity(root)

    assert result["ok"] is False
    assert result["unexpected_extra_count"] == 105
    assert len(result["unexpected_extras"]) == 100
    assert result["unexpected_extras_truncated"] is True


def test_preserved_root_policies_and_legacy_launcher_are_exact(
    tmp_path: Path,
) -> None:
    assert source_integrity._PRESERVED_INSTALL_ROOTS == PRESERVED_ROOTS
    assert build_source_manifest.PRESERVED_INSTALL_ROOTS == PRESERVED_ROOTS

    entry = "onnx_splitpoint_tool.platform_power_cli:main"
    launcher = tmp_path / "onnx-splitpoint-platform-power"
    launcher.write_bytes(
        refresh_editable_install._legacy_v27911_launcher_content(entry)
    )
    assert refresh_editable_install._launcher_targets_entry_point(
        launcher, entry
    )

    launcher.write_bytes(launcher.read_bytes() + b"\n")
    assert not refresh_editable_install._launcher_targets_entry_point(
        launcher, entry
    )
    launcher.write_bytes(
        refresh_editable_install._legacy_v27911_launcher_content(entry)
    )
    assert not refresh_editable_install._launcher_targets_entry_point(
        launcher,
        "onnx_splitpoint_tool.platform_power_cli:_parser",
    )
