from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FULL_WRAPPER = ROOT / "scripts" / "run_v27521_full_only_check.sh"
REMOTE_PREFLIGHT = ROOT / "scripts" / "preflight_v27521_native_remotes.py"


def _load_preflight():
    spec = importlib.util.spec_from_file_location(
        "v27521_native_remote_preflight_test",
        REMOTE_PREFLIGHT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v27521_local_dispatch_preflight_precedes_remote_staging_and_ssh() -> None:
    wrapper = FULL_WRAPPER.read_text(encoding="utf-8")
    profile_probe = wrapper.index(
        "validate_setup_local_tensorrt_quality_dispatch(payload)"
    )
    remote_probe = wrapper.index("scripts/preflight_v27521_native_remotes.py")
    snapshot = wrapper.index('RUN_SNAPSHOT="$(mktemp')
    marker = wrapper.index('RUN_MARKER="$(mktemp')
    workflow = wrapper.index(
        "-m onnx_splitpoint_tool.workflow.run_evaluation"
    )
    assert profile_probe < remote_probe < snapshot < marker < workflow

    preflight = REMOTE_PREFLIGHT.read_text(encoding="utf-8")
    local_probe = preflight.index(
        "validate_setup_local_tensorrt_quality_dispatch("
    )
    first_sync = preflight.index("sync_script_fn =")
    first_setup_loop = preflight.index(
        "for setup in resolve_required_remote_setups(profile_payload):"
    )
    assert local_probe < first_sync < first_setup_loop


def test_v27521_invalid_local_dispatch_never_reaches_sync_or_process() -> None:
    preflight = _load_preflight()
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        del args, kwargs
        calls.append("called")
        raise AssertionError("remote side effect must not be reached")

    invalid_profile = {
        "run_profiles": [{
            "id": "hailo8",
            "type": "same_backend_reference",
            "full": "hailo8",
            "stage1": "hailo8",
            "stage2": "hailo8",
        }],
    }
    with pytest.raises(
        RuntimeError,
        match="v27521_setup_local_tensorrt_dispatch_preflight_failed",
    ):
        preflight.run_remote_contract_preflight(
            invalid_profile,
            sync_script=forbidden,
            sync_asset=forbidden,
            verify_module=forbidden,
            process_runner=forbidden,
        )
    assert calls == []


def test_v27521_wrapper_helpers_preserve_rc_and_secure_debug_targets(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    victim = tmp_path / "victim.zip"
    victim.write_bytes(b"unchanged")
    (runs_root / "run_v27521_debug_pack.zip").symlink_to(victim)
    (runs_root / "regular_v27521_debug_pack.zip").write_bytes(b"spoof")

    completed = subprocess.run(
        [
            "bash",
            "-c",
            r'''
set -Eeuo pipefail
source "$1"
set +e
v27521_discovery_failure_rc 130 1
fatal_rc=$?
v27521_discovery_failure_rc 1 7
normal_rc=$?
v27521_terminal_wrapper_rc 1 failed 0 0
failed_terminal_rc=$?
v27521_terminal_wrapper_rc 1 partial 0 0
partial_terminal_rc=$?
set -e
target_one="$(v27521_secure_debug_pack_target "$2" run)"
target_two="$(v27521_secure_debug_pack_target "$2" run)"
printf '%s\n%s\n%s\n%s\n%s\n%s\n' \
  "$fatal_rc" "$normal_rc" "$failed_terminal_rc" "$partial_terminal_rc" \
  "$target_one" "$target_two"
''',
            "v27521-wrapper-helper-test",
            str(FULL_WRAPPER),
            str(runs_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    fatal, normal, failed, partial, raw_one, raw_two = (
        completed.stdout.splitlines()
    )
    assert (fatal, normal, failed, partial) == ("130", "7", "1", "0")
    targets = (Path(raw_one), Path(raw_two))
    assert targets[0] != targets[1]
    for target in targets:
        assert target.parent.parent == runs_root
        assert target.parent.is_dir()
        assert not target.parent.is_symlink()
        assert not target.exists()
    assert victim.read_bytes() == b"unchanged"
    assert (runs_root / "regular_v27521_debug_pack.zip").read_bytes() == b"spoof"


def test_v27521_small_acceptance_names_current_release_scripts() -> None:
    source = (
        ROOT / "scripts" / "run_v27521_small_acceptance.sh"
    ).read_text(encoding="utf-8")
    for name in (
        "run_v27521_cache_canary.sh",
        "run_v27521_small_acceptance.sh",
        "run_v27521_full_only_check.sh",
        "test_v27521_setup_local_trt_dispatch.py",
        "test_v27521_release_wrappers.py",
    ):
        assert name in source
    assert "v2.75.21-setup-local-tensorrt-quality-dispatch-repair" in source
    assert 'tool.__version__ == "2.75.21"' in source
    local = (ROOT / "scripts" / "run_local_acceptance.sh").read_text(
        encoding="utf-8",
    )
    for target in (
        "tests/test_v27521_setup_local_trt_dispatch.py",
        "tests/test_v27521_reporting_and_full_only_authority.py",
        "tests/test_v27521_release_wrappers.py",
    ):
        assert local.count(target) == 2
