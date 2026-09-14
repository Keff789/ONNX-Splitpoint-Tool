from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import pytest
import yaml

from onnx_splitpoint_tool.workflow.start_snapshot import (
    build_profile_start_snapshot,
    public_start_snapshot_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ID = "resnet50_v2772_hailo_parallel_build_canary"


def _verifier():
    candidates = (
        ROOT / "scripts/verify_v2772_hailo_parallel_build_canary.py",
        ROOT.parent.parent
        / "v2772_hailo_parallel_canary/scripts"
        / "verify_v2772_hailo_parallel_build_canary.py",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    assert path is not None, "v2.77.2 canary verifier is missing"
    name = f"v2772_hailo_parallel_canary_verifier_{id(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _event(
    target: str,
    *,
    label: str,
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    return {
        "cpu_tokens": 4,
        "elapsed_s": ended_at - started_at,
        "ended_at": ended_at,
        "family": target,
        "metadata": {"label": label, "target": target},
        "name": f"{label}:{target}",
        "ram_mb": 6144,
        "started_at": started_at,
        "status": "ok",
    }


def _profile() -> dict[str, Any]:
    return {
        "name": PROFILE_ID,
        "workflow": {
            "execution_mode": "generate_benchmarksets",
            "skip_runtime_benchmarks": True,
            "stop_after": "build_backend_artifacts",
            "parallel_remote_setups": False,
            "max_parallel_setups": 1,
            "max_parallel_uploads": 0,
            "powercalc_workers": 0,
        },
        "benchmark_execution": {"warmup": 0, "runs": 1, "timeout_s": 0},
        "remote_execution": {"enabled": False},
        "model_preparation": {"mode": "current"},
        "hardware_smoke": {"mode": "disabled"},
        "hailo_build": {
            "backend": "auto",
            "force_build": True,
            "cache_enabled": False,
            "build_full": False,
            "build_part1": True,
            "build_part2": False,
        },
        "build_scheduler": {
            "max_workers": 2,
            "cpu_tokens": 8,
            "ram_mb": 12288,
        },
        "artifact_store": {"enabled": False},
        "native_producers": {"enabled": False},
        "energy": {
            "enabled": False,
            "requested_native_energy": False,
        },
        "ranking_validation": {"enabled": False},
        "validation": {"split_fidelity_reference_mode": "cpu_full"},
    }


def _write_part1_artifact(run_dir: Path, target: str) -> None:
    physical_arch = "hailo10h" if target == "hailo10" else "hailo8"
    artifact_dir = (
        run_dir
        / "models/resnet50/benchmark_set/legacy_suite/b052/hailo"
        / target
        / "part1"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    hef = artifact_dir / "compiled.hef"
    hef.write_bytes(f"synthetic-{target}-hef".encode("ascii"))
    cache_payload = {
        "schema": "onnx-splitpoint/hailo-hef-cache-key-v3",
        "model_sha256": "1" * 64,
        "hw_arch": physical_arch,
        "net_name": "resnet50_part1_b52",
    }
    receipt = {
        "schema": "onnx-splitpoint/hailo-hef-build-receipt/v2",
        "source_onnx_sha256": "2" * 64,
        "compiler_onnx_sha256": "1" * 64,
        "hef_sha256": hashlib.sha256(hef.read_bytes()).hexdigest(),
        "hef_size_bytes": hef.stat().st_size,
        "hw_arch": physical_arch,
        "net_name": "resnet50_part1_b52",
        "cache_payload": cache_payload,
        "cache_key": _cache_key(cache_payload),
    }
    result = {
        "ok": True,
        "hw_arch": physical_arch,
        "backend": "venv",
        "error": None,
        "skipped": False,
        "timed_out": False,
        # The live v2.77.1 format carries the receipt in both of these
        # containers and records the cache decision under details.
        "calib_info": {"build_receipt": receipt},
        "details": {"cache_hit": False, "build_receipt": receipt},
    }
    _write_json(artifact_dir / "hailo_hef_build_receipt.json", receipt)
    _write_json(artifact_dir / "hailo_hef_build_result.json", result)


def _build_run(
    tmp_path: Path,
    *,
    extra_full_scheduler: bool = False,
    extra_full_artifacts: bool = False,
) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    profile = _profile()
    (run_dir / "profile.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "profile_source.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "profile_id": PROFILE_ID,
            "tool_version": "2.77.2",
            "workflow_version": (
                "v2.77.2-hailo-full-scope-and-parallel-canary-closure"
            ),
            "status": "partial",
            "models": {
                "resnet50": {
                    "model_id": "resnet50",
                    "stages": {
                        "build_backend_artifacts": {
                            "stage": "build_backend_artifacts",
                            "status": "ok",
                        }
                    },
                }
            },
            "root_stages": {
                "resolve_profile": {"status": "ok"},
                "campaign_preflight": {"status": "ok"},
            },
        },
    )
    full_snapshot = build_profile_start_snapshot(
        profile_request="synthetic-canary-profile.yaml",
        source_profile=profile,
        resolved_profile=profile,
        profile_id=PROFILE_ID,
        profile_path="/synthetic/resnet50_v2772_canary.yaml",
        profile_source="file",
        runtime_bindings={},
        schema_version=2,
    )
    _write_json(
        run_dir / "profile_start_snapshot.json",
        public_start_snapshot_metadata(full_snapshot),
    )
    workflow_lines = [
        "[benchmarkset:resnet50] [build-scheduler] Hailo pair "
        "requested=True effective=True targets=hailo10,hailo8 "
        "backend_requested=auto backend_effective=venv "
        "reason=resources_available",
        "[benchmarkset:resnet50] b52: Hailo HEF generation requested "
        "(backend=auto, targets=['hailo10', 'hailo8'], full=False, "
        "part1=True, part2=False)",
    ]
    events = [
        _event("hailo8", label="hailo-targets:b52", started_at=100.0, ended_at=130.0),
        _event(
            "hailo10",
            label="hailo-targets:b52",
            started_at=100.25,
            ended_at=142.0,
        ),
    ]
    if extra_full_scheduler:
        workflow_lines.append(
            "[benchmarkset:resnet50] [build-scheduler] start "
            "hailo-full-targets:hailo8"
        )
        events.extend(
            [
                _event(
                    "hailo8",
                    label="hailo-full-targets",
                    started_at=142.1,
                    ended_at=162.1,
                ),
                _event(
                    "hailo10",
                    label="hailo-full-targets",
                    started_at=142.2,
                    ended_at=172.2,
                ),
            ]
        )
    (run_dir / "evaluation_workflow.log").write_text(
        "\n".join(workflow_lines) + "\n",
        encoding="utf-8",
    )
    scheduler_log = tmp_path / "build_scheduler.jsonl"
    scheduler_log.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    for target in ("hailo8", "hailo10"):
        _write_part1_artifact(run_dir, target)
        if extra_full_artifacts:
            full = (
                run_dir
                / "models/resnet50/benchmark_set/legacy_suite/b052/hailo"
                / target
                / "full/compiled.hef"
            )
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_bytes(f"unexpected-full-{target}".encode("ascii"))
    return run_dir, scheduler_log


def _verify(run_dir: Path, scheduler_log: Path) -> dict[str, Any]:
    verifier = _verifier()
    verdict, _artifact_files = verifier.verify_run(
        run_dir=run_dir,
        scheduler_log=scheduler_log,
        workflow_exit_code=1,
        receipt_validator=lambda _hef: {},
    )
    return verdict


def test_exact_part1_pair_fixture_passes_without_hardware(tmp_path: Path) -> None:
    run_dir, scheduler_log = _build_run(tmp_path)

    verdict = _verify(run_dir, scheduler_log)

    assert verdict["status"] == "PASS", verdict["errors"]
    assert verdict["technical_canary_pass"] is True
    assert verdict["process_overlap_s"] == pytest.approx(29.75)
    assert verdict["details"]["part1_pair_diagnostic"] == {
        "observed": True,
        "event_count": 2,
        "targets": ["hailo8", "hailo10"],
        "overlap_s": 29.75,
        "start_skew_s": 0.25,
        "overlap_gate_pass": True,
    }
    assert verdict["details"]["scheduler_contract"] == {
        "total_event_count": 2,
        "unexpected_event_count": 0,
        "pass": True,
    }
    assert set(verdict["artifacts"]) == {"hailo8", "hailo10"}
    assert verdict["warnings"] == [
        "workflow_exit_code=1; the intentional stop-after is verified independently"
    ]


@pytest.mark.parametrize(
    ("extra_full_scheduler", "extra_full_artifacts", "expected_error"),
    [
        (True, False, "scheduler_event_count:4!=2"),
        (False, True, "compiled_hef_total:4!=2"),
    ],
)
def test_extra_full_work_remains_fail_but_part1_overlap_is_diagnostic(
    tmp_path: Path,
    extra_full_scheduler: bool,
    extra_full_artifacts: bool,
    expected_error: str,
) -> None:
    run_dir, scheduler_log = _build_run(
        tmp_path,
        extra_full_scheduler=extra_full_scheduler,
        extra_full_artifacts=extra_full_artifacts,
    )

    verdict = _verify(run_dir, scheduler_log)

    assert verdict["status"] == "FAIL"
    assert verdict["technical_canary_pass"] is False
    assert expected_error in verdict["errors"]
    diagnostic = verdict["details"]["part1_pair_diagnostic"]
    assert diagnostic["observed"] is True
    assert diagnostic["event_count"] == 2
    assert diagnostic["overlap_gate_pass"] is True
    assert diagnostic["overlap_s"] == pytest.approx(29.75)
    assert verdict["process_overlap_s"] == pytest.approx(29.75)


def test_evidence_zip_is_self_contained_and_matches_external_verdict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir, scheduler_log = _build_run(tmp_path)
    verifier = _verifier()
    monkeypatch.setattr(verifier, "_default_receipt_validator", lambda _hef: {})
    verdict_path = tmp_path / "canary_verdict.json"
    evidence_zip = tmp_path / "canary_evidence.zip"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_v2772_hailo_parallel_build_canary.py",
            "--run-dir",
            str(run_dir),
            "--scheduler-log",
            str(scheduler_log),
            "--workflow-exit-code",
            "1",
            "--out-json",
            str(verdict_path),
            "--evidence-zip",
            str(evidence_zip),
        ],
    )

    assert verifier.main() == 0
    external_verdict = verdict_path.read_bytes()
    with zipfile.ZipFile(evidence_zip, "r") as archive:
        assert archive.testzip() is None
        assert archive.read("canary_verdict.json") == external_verdict
        names = set(archive.namelist())
        assert any(name.endswith("hailo/hailo8/part1/compiled.hef") for name in names)
        assert any(name.endswith("hailo/hailo10/part1/compiled.hef") for name in names)
        manifest = json.loads(archive.read("evidence_manifest.json"))
        manifest_sha = archive.read("evidence_manifest.sha256").decode("ascii")
        assert manifest_sha.split()[0] == hashlib.sha256(
            archive.read("evidence_manifest.json")
        ).hexdigest()
        for row in manifest["entries"]:
            payload = archive.read(row["path"])
            assert len(payload) == row["size_bytes"]
            assert hashlib.sha256(payload).hexdigest() == row["sha256"]
