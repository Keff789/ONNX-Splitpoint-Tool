from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from onnx_splitpoint_tool.energy.collector import run_fast_firmware_measurement
from onnx_splitpoint_tool.energy.config import EnergyDefaults, EnergySetup


NONCE_TOKEN = "__ONNX_SPLITPOINT_PREFLIGHT_NONCE__"
ATTESTATION_TOKEN = "__ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__"


def _write_executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture()
def preflight_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    order_path = tmp_path / "order.log"
    monkeypatch.setenv("PREFLIGHT_TEST_ORDER", str(order_path))
    helper = _write_executable(
        tmp_path / "preflight_workload.py",
        r'''
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["preflight", "preflight_fail", "preflight_stale", "workload"])
parser.add_argument("--nonce", required=True)
parser.add_argument("--attestation", default="")
args = parser.parse_args()
order = pathlib.Path(os.environ["PREFLIGHT_TEST_ORDER"])
with order.open("a", encoding="utf-8") as handle:
    handle.write(args.mode + "\n")
if args.mode == "preflight_fail":
    print("intentional preflight failure", file=sys.stderr)
    raise SystemExit(7)
if args.mode == "workload":
    print("__SPLITPOINT_WORK_UNITS__=10")
    print("__SPLITPOINT_WORK_UNITS_SOURCE__=completed_frames")
    print("__SPLITPOINT_WORK_UNITS_EXACT__=1")
    raise SystemExit(0)
now = time.time_ns()
created = now if args.mode == "preflight" else now - 120_000_000_000
payload = {
    "schema": "onnx-splitpoint/energy-preflight-attestation",
    "schema_version": 1,
    "ok": True,
    "nonce": args.nonce,
    "created_at_unix_ns": created,
    "expires_at_unix_ns": created + 30_000_000_000,
    "artifact_verification_status": "pass",
    "command_contract_sha256": "a" * 64,
    "runtime_attestation_path": args.attestation,
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
payload["attestation_sha256"] = hashlib.sha256(canonical).hexdigest()
print("__SPLITPOINT_PREFLIGHT_ATTESTATION__=" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
''',
    )
    collector = _write_executable(
        tmp_path / "fake_collector.py",
        r'''
import os
import pathlib
import subprocess
import sys

if "--help" in sys.argv:
    print("fake collector")
    raise SystemExit(0)
order = pathlib.Path(os.environ["PREFLIGHT_TEST_ORDER"])
with order.open("a", encoding="utf-8") as handle:
    handle.write("collector_start\n")
storage = pathlib.Path(next(value[3:] for value in sys.argv if value.startswith("-s=")))
command = pathlib.Path(next(value[3:] for value in sys.argv if value.startswith("-c=")))
storage.mkdir(parents=True, exist_ok=True)
(storage / "samples.parquet").write_bytes(b"PAR1" + b"x" * 4096 + b"PAR1")
raise SystemExit(subprocess.run([str(command)], check=False).returncode)
''',
    )
    defaults = EnergyDefaults(
        collector_binary=str(collector),
        power_calculations_binary="definitely-not-used",
        sample_rate=2000,
        pre_duration_s=0,
        post_duration_s=0,
        run_count=1,
        compare_legacy_window=False,
        postprocess_with_power_calculations=False,
    )
    setup = EnergySetup(setup_id="fake", enabled=True, urecs_address="127.0.0.1")
    return {
        "helper": helper,
        "order": order_path,
        "defaults": defaults,
        "setup": setup,
    }


def _commands(rig: dict[str, object], preflight_mode: str) -> tuple[str, str]:
    helper = Path(rig["helper"])
    py = shlex.quote(sys.executable)
    script = shlex.quote(str(helper))
    preflight = (
        f"{py} {script} {preflight_mode} --nonce {NONCE_TOKEN} "
        f"--attestation {ATTESTATION_TOKEN}"
    )
    workload = (
        f"{py} {script} workload --nonce {NONCE_TOKEN} "
        f"--attestation {ATTESTATION_TOKEN}"
    )
    return preflight, workload


def _run(
    tmp_path: Path,
    rig: dict[str, object],
    *,
    mode: str,
    remote_path: str = "",
    expected_contract_sha256: str = "a" * 64,
) -> tuple[dict, Path]:
    preflight, workload = _commands(rig, mode)
    out = tmp_path / f"measurement_{mode}"
    result = run_fast_firmware_measurement(
        workload,
        out,
        setup=rig["setup"],  # type: ignore[arg-type]
        defaults=rig["defaults"],  # type: ignore[arg-type]
        duration_s=1.0,
        run_count=1,
        postprocess=False,
        compare_legacy_window=False,
        preflight_command=preflight,
        preflight_timeout_s=10.0,
        preflight_attestation_max_age_s=60.0,
        preflight_runtime_attestation_path=remote_path,
        preflight_expected_command_contract_sha256=expected_contract_sha256,
    )
    return result, out


def test_preflight_finishes_before_collector_and_archives_evidence(
    tmp_path: Path, preflight_rig: dict[str, object]
) -> None:
    remote_path = f"/tmp/remote energy/attestation_{NONCE_TOKEN}.json"
    result, out = _run(
        tmp_path, preflight_rig, mode="preflight", remote_path=remote_path
    )

    order = Path(preflight_rig["order"]).read_text(encoding="utf-8").splitlines()
    assert order == ["preflight", "collector_start", "workload"]
    run = result["runs"][0]
    assert run["preflight_status"] == "verified"
    evidence_path = out / "run_000" / "preflight" / "preflight_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["ok"] is True
    assert evidence["rc"] == 0
    assert evidence["collector_started"] is True
    assert evidence["ended_at_unix_ns"] <= evidence["collector_started_at_unix_ns"]
    assert evidence["rendered_command_sha256"]
    assert evidence["command_file_sha256"]
    assert evidence["stdout_sha256"]
    assert evidence["stderr_sha256"]
    assert evidence["attestation_file_sha256"]
    assert evidence["validation"]["status"] == "verified"
    assert evidence["runtime_attestation_path"].startswith("/tmp/remote energy/")
    assert NONCE_TOKEN not in evidence["runtime_attestation_path"]
    rendered_preflight = Path(evidence["command_path"]).read_text(encoding="utf-8")
    rendered_workload = (out / "run_000" / "workload_command.sh").read_text(encoding="utf-8")
    assert evidence["runtime_attestation_path"] in rendered_preflight
    assert evidence["runtime_attestation_path"] in rendered_workload


def test_preflight_failure_prevents_collector_and_workload(
    tmp_path: Path, preflight_rig: dict[str, object]
) -> None:
    result, out = _run(tmp_path, preflight_rig, mode="preflight_fail")

    order = Path(preflight_rig["order"]).read_text(encoding="utf-8").splitlines()
    assert order == ["preflight_fail"]
    assert result["ok"] is False
    run = result["runs"][0]
    assert run["status"] == "preflight_failed"
    assert run["collector_started"] is False
    assert run["workload_started"] is False
    assert run["collector_rc"] is None
    assert run["final_energy_gate_reasons"] == ["preflight_failed"]
    evidence = json.loads(
        (out / "run_000" / "preflight" / "preflight_evidence.json").read_text()
    )
    assert evidence["rc"] == 7
    assert "preflight_command_rc_nonzero" in evidence["validation"]["reasons"]
    assert not (out / "run_000" / "collector_stdout.log").exists()


def test_stale_sealed_attestation_fails_closed_before_sampling(
    tmp_path: Path, preflight_rig: dict[str, object]
) -> None:
    result, out = _run(tmp_path, preflight_rig, mode="preflight_stale")

    order = Path(preflight_rig["order"]).read_text(encoding="utf-8").splitlines()
    assert order == ["preflight_stale"]
    assert result["ok"] is False
    evidence = json.loads(
        (out / "run_000" / "preflight" / "preflight_evidence.json").read_text()
    )
    reasons = evidence["validation"]["reasons"]
    assert "preflight_attestation_not_fresh" in reasons
    assert "preflight_attestation_expired" in reasons
    assert evidence["collector_started"] is False


def test_wrong_successful_command_contract_attestation_fails_closed(
    tmp_path: Path, preflight_rig: dict[str, object]
) -> None:
    result, out = _run(
        tmp_path,
        preflight_rig,
        mode="preflight",
        expected_contract_sha256="b" * 64,
    )

    order = Path(preflight_rig["order"]).read_text(encoding="utf-8").splitlines()
    assert order == ["preflight"]
    assert result["ok"] is False
    evidence = json.loads(
        (out / "run_000" / "preflight" / "preflight_evidence.json").read_text()
    )
    assert "preflight_command_contract_sha256_mismatch" in evidence["validation"][
        "reasons"
    ]
    assert evidence["collector_started"] is False


def test_preflight_is_opt_in_and_legacy_command_needs_no_nonce(
    tmp_path: Path, preflight_rig: dict[str, object]
) -> None:
    result = run_fast_firmware_measurement(
        "printf 'legacy workload\\n'",
        tmp_path / "legacy",
        setup=preflight_rig["setup"],  # type: ignore[arg-type]
        defaults=preflight_rig["defaults"],  # type: ignore[arg-type]
        duration_s=1.0,
        run_count=1,
        postprocess=False,
        compare_legacy_window=False,
    )
    order = Path(preflight_rig["order"]).read_text(encoding="utf-8").splitlines()
    assert order == ["collector_start"]
    assert result["preflight_requested"] is False
    assert result["runs"][0]["preflight_status"] == "not_requested"


def test_energy_cli_exposes_preflight_contract_and_mirror_is_identical() -> None:
    root = Path(__file__).resolve().parents[1]
    cli = root / "scripts" / "energy_measurement_cli.py"
    mirror = (
        root
        / "onnx_splitpoint_tool"
        / "resources"
        / "remote_scripts"
        / "energy_measurement_cli.py"
    )
    assert cli.read_bytes() == mirror.read_bytes()
    help_run = subprocess.run(
        [sys.executable, str(cli), "measure", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_run.returncode == 0, help_run.stderr
    assert "--preflight-command" in help_run.stdout
    assert "--preflight-command-file" in help_run.stdout
    assert "--preflight-runtime-attestation-path" in help_run.stdout
    assert "--preflight-expected-command-contract-sha256" in help_run.stdout
    assert "--exact-run-count" in help_run.stdout
