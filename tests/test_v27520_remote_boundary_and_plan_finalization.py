from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.management_reference import (
    finalize_management_cpu_reference_plan_aliases,
)
from onnx_splitpoint_tool.workflow.artifacts import sha256_json
from onnx_splitpoint_tool.workflow.execution_binding import (
    execute_benchmark_suite_if_requested,
    finalize_suite_for_runtime,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_preflight_script():
    path = ROOT / "scripts/preflight_v27520_native_remotes.py"
    spec = importlib.util.spec_from_file_location("v27520_remote_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _central_profile() -> dict:
    return {
        "run_profiles": [{
            "id": "hailo8",
            "full": "hailo8",
            "stage1": "hailo8",
            "stage2": "hailo8",
        }],
        "quality_gate": {
            "statistics": {"execution_location": "central_management"},
            "classification": {"top1_max_drop": 0.01},
        },
    }


def _base_runs() -> list[dict]:
    return [
        {
            "id": "ort_cpu",
            "backend": "ort_cpu",
            "variant": "full",
            "stage1": "cpu",
            "stage2": "cpu",
        },
        {
            "id": "hailo8",
            "type": "matrix",
            "backend": "hailo8_to_tensorrt",
            "variant": "full",
        },
    ]


def _native_split_selection(
    *, applicable: bool, split_backends: list[str], source: str,
) -> dict:
    return {
        "schema": "onnx-splitpoint/native-split-quality-selection",
        "schema_version": 1,
        "applicable": applicable,
        "split_backends": split_backends,
        "split_selection_source": source,
    }


def test_runtime_normalization_precedes_one_canonical_plan_seal(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    formal_dir = run_root / "models" / "resnet18" / "benchmark_set"
    suite_dir = formal_dir / "legacy_suite"
    runs = _base_runs()
    executable_plan = {"runs": runs, "planned_runs": runs, "targets": ["hailo8"]}
    stale_formal_runs = [dict(row) for row in runs]
    stale_formal_runs[1]["formal_only_stale_marker"] = True
    formal_plan = {
        "runs": stale_formal_runs,
        "planned_runs": stale_formal_runs,
        "targets": ["hailo8"],
    }
    executable_contract = {
        "schema": "onnx-splitpoint/benchmark-set",
        "cases": [],
        "planned_runs": runs,
        "plan": {"runs": runs, "planned_runs": runs},
    }
    formal_contract = {
        "schema": "onnx-splitpoint/formal-benchmark-set",
        "planned_runs": stale_formal_runs,
        "plan": {
            "runs": stale_formal_runs,
            "planned_runs": stale_formal_runs,
        },
    }
    _write(suite_dir / "benchmark_plan.json", executable_plan)
    _write(formal_dir / "benchmark_plan.json", formal_plan)
    _write(suite_dir / "benchmark_set.json", executable_contract)
    _write(formal_dir / "benchmark_set.json", formal_contract)

    result = finalize_suite_for_runtime(
        suite_dir=suite_dir,
        run_root=run_root,
        model_id="resnet18",
        suite_payload=executable_contract,
        benchmark_plan=formal_plan,
        profile_payload=_central_profile(),
        model_task="classification",
        quality_gate_policy=_central_profile()["quality_gate"],
    )
    assert result["status"] == "verified"
    assert result["authoritative_alias"] == "executable"

    executable = _read(suite_dir / "benchmark_plan.json")
    formal = _read(formal_dir / "benchmark_plan.json")
    disabled_selection = _native_split_selection(
        applicable=False,
        split_backends=[],
        source="native_disabled",
    )
    assert executable["native_split_quality_selection"] == disabled_selection
    assert formal["native_split_quality_selection"] == disabled_selection
    assert executable["runs"] == executable["planned_runs"]
    assert executable["runs"] == formal["runs"] == formal["planned_runs"]
    assert all("formal_only_stale_marker" not in row for row in executable["runs"])
    for row in executable["runs"]:
        assert row["task"] == "classification"
        assert row["benchmark_task"] == "classification"
        assert row["task_quality_gate"] == _central_profile()["quality_gate"]
        assert len(row["quality_gate_policy_sha256"]) == 64
        assert isinstance(row["stage1"], dict)
        assert isinstance(row["stage2"], dict)

    invariant = executable["management_cpu_reference_invariant"]
    assert invariant["run_plan_sha256"] == sha256_json(executable["runs"])
    assert formal["management_cpu_reference_invariant"] == invariant
    for contract_path in (
        suite_dir / "benchmark_set.json",
        formal_dir / "benchmark_set.json",
    ):
        contract = _read(contract_path)
        assert contract["planned_runs"] == executable["runs"]
        assert contract["plan"]["runs"] == executable["runs"]
        assert contract["plan"]["planned_runs"] == executable["runs"]
        assert contract["native_split_quality_selection"] == disabled_selection
        assert (
            contract["plan"]["native_split_quality_selection"]
            == disabled_selection
        )
        assert contract["management_cpu_reference_invariant"] == invariant

    sealed_bytes = {
        path: path.read_bytes()
        for path in (
            suite_dir / "benchmark_plan.json",
            formal_dir / "benchmark_plan.json",
            suite_dir / "benchmark_set.json",
            formal_dir / "benchmark_set.json",
        )
    }
    second = finalize_suite_for_runtime(
        suite_dir=suite_dir,
        run_root=run_root,
        model_id="resnet18",
        suite_payload=_read(suite_dir / "benchmark_set.json"),
        benchmark_plan=formal,
        profile_payload=_central_profile(),
        model_task="classification",
        quality_gate_policy=_central_profile()["quality_gate"],
    )
    assert second["runtime_preflight"]["repairs"] == []
    assert {path: path.read_bytes() for path in sealed_bytes} == sealed_bytes


def test_runtime_plan_freezes_selected_native_split_quality_backend(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    formal_dir = run_root / "models" / "resnet18" / "benchmark_set"
    suite_dir = formal_dir / "legacy_suite"
    runs = _base_runs()
    plan = {"runs": runs, "planned_runs": runs, "targets": ["hailo8"]}
    contract = {
        "schema": "onnx-splitpoint/benchmark-set",
        "cases": [],
        "planned_runs": runs,
        "plan": {"runs": runs, "planned_runs": runs},
    }
    for path, payload in (
        (suite_dir / "benchmark_plan.json", plan),
        (formal_dir / "benchmark_plan.json", plan),
        (suite_dir / "benchmark_set.json", contract),
        (formal_dir / "benchmark_set.json", contract),
    ):
        _write(path, payload)

    profile = _central_profile()
    profile["native_producers"] = {"enabled": True}
    profile["run_profiles"] = [{
        "id": "hailo8_to_tensorrt",
        "enabled": True,
        "stage1": "hailo8",
        "stage2": "tensorrt",
    }]
    result = finalize_suite_for_runtime(
        suite_dir=suite_dir,
        run_root=run_root,
        model_id="resnet18",
        suite_payload=contract,
        benchmark_plan=plan,
        profile_payload=profile,
        model_task="classification",
        quality_gate_policy=profile["quality_gate"],
    )

    selected = _native_split_selection(
        applicable=True,
        split_backends=["hailo8"],
        source="evaluation_profile.run_profiles",
    )
    assert result["benchmark_plan"]["native_split_quality_selection"] == selected
    for path in (
        suite_dir / "benchmark_plan.json",
        formal_dir / "benchmark_plan.json",
        suite_dir / "benchmark_set.json",
        formal_dir / "benchmark_set.json",
    ):
        payload = _read(path)
        assert payload["native_split_quality_selection"] == selected
        if isinstance(payload.get("plan"), dict):
            assert payload["plan"]["native_split_quality_selection"] == selected


def test_executable_authority_protects_normalized_rows_from_stale_formal_alias(
    tmp_path: Path,
) -> None:
    executable_path = tmp_path / "suite" / "benchmark_plan.json"
    formal_path = tmp_path / "formal" / "benchmark_plan.json"
    final_rows = _base_runs()
    final_rows[1]["task"] = "detection"
    stale_rows = _base_runs()
    stale_rows[1]["task"] = "classification"
    _write(executable_path, {"runs": final_rows, "planned_runs": final_rows})
    _write(formal_path, {"runs": stale_rows, "planned_runs": stale_rows})

    invariant = finalize_management_cpu_reference_plan_aliases(
        executable_plan_path=executable_path,
        formal_plan_path=formal_path,
        profile=_central_profile(),
        cache_verify_enabled=False,
        automatic=True,
        require_existing=True,
        authoritative_alias="executable",
    )
    executable = _read(executable_path)
    formal = _read(formal_path)
    assert executable["runs"] == formal["runs"]
    assert executable["runs"][1]["task"] == "detection"
    assert invariant["run_plan_sha256"] == sha256_json(executable["runs"])


def test_executable_authority_never_resurrects_stale_formal_rows(
    tmp_path: Path,
) -> None:
    executable_path = tmp_path / "suite" / "benchmark_plan.json"
    formal_path = tmp_path / "formal" / "benchmark_plan.json"
    _write(executable_path, {"runs": [], "planned_runs": []})
    _write(formal_path, {
        "runs": _base_runs(),
        "planned_runs": _base_runs(),
    })
    with pytest.raises(
        ValueError, match="management_cpu_reference_recipe_missing",
    ):
        finalize_management_cpu_reference_plan_aliases(
            executable_plan_path=executable_path,
            formal_plan_path=formal_path,
            profile=_central_profile(),
            cache_verify_enabled=False,
            automatic=True,
            require_existing=True,
            authoritative_alias="executable",
        )
    assert _read(executable_path)["runs"] == []


def test_cache_runtime_sync_removes_stale_management_invariant(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    formal_dir = run_root / "models" / "resnet18" / "benchmark_set"
    suite_dir = formal_dir / "legacy_suite"
    runtime_rows = [{
        "id": "hailo8",
        "type": "matrix",
        "backend": "hailo8_to_tensorrt",
        "variant": "full",
    }]
    stale_rows = [dict(runtime_rows[0], stale=True)]
    stale_invariant = {
        "status": "verified",
        "run_plan_sha256": "sha256:" + "0" * 64,
    }
    executable_plan = {
        "runs": runtime_rows,
        "planned_runs": runtime_rows,
        "management_cpu_reference_invariant": stale_invariant,
    }
    formal_plan = {
        "runs": stale_rows,
        "planned_runs": stale_rows,
        "management_cpu_reference_invariant": stale_invariant,
    }
    executable_contract = {
        "planned_runs": runtime_rows,
        "plan": {
            "runs": runtime_rows,
            "planned_runs": runtime_rows,
            "management_cpu_reference_invariant": stale_invariant,
        },
        "management_cpu_reference_invariant": stale_invariant,
        "cases": [],
    }
    formal_contract = {
        "planned_runs": stale_rows,
        "plan": {
            "runs": stale_rows,
            "planned_runs": stale_rows,
            "management_cpu_reference_invariant": stale_invariant,
        },
        "management_cpu_reference_invariant": stale_invariant,
    }
    for path, payload in (
        (suite_dir / "benchmark_plan.json", executable_plan),
        (formal_dir / "benchmark_plan.json", formal_plan),
        (suite_dir / "benchmark_set.json", executable_contract),
        (formal_dir / "benchmark_set.json", formal_contract),
    ):
        _write(path, payload)
    profile = _central_profile()
    profile["execution_guard"] = {"mode": "cache_verify_only"}
    result = finalize_suite_for_runtime(
        suite_dir=suite_dir,
        run_root=run_root,
        model_id="resnet18",
        suite_payload=executable_contract,
        benchmark_plan=formal_plan,
        profile_payload=profile,
        model_task="classification",
        quality_gate_policy=profile["quality_gate"],
    )
    assert result["cpu_reference_invariant"]["status"] == (
        "not_applicable_cache_verify_only"
    )
    expected_rows = _read(suite_dir / "benchmark_plan.json")["runs"]
    for path in (
        suite_dir / "benchmark_plan.json",
        formal_dir / "benchmark_plan.json",
        suite_dir / "benchmark_set.json",
        formal_dir / "benchmark_set.json",
    ):
        payload = _read(path)
        assert "management_cpu_reference_invariant" not in payload
        assert payload.get("runs", expected_rows) == expected_rows
        assert payload["planned_runs"] == expected_rows
        if isinstance(payload.get("plan"), dict):
            assert (
                "management_cpu_reference_invariant"
                not in payload["plan"]
            )


def test_direct_runtime_inserts_cpu_before_normalization_and_hashing(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    formal_dir = run_root / "models" / "resnet18" / "benchmark_set"
    suite_dir = formal_dir / "legacy_suite"
    rows = [{
        "id": "hailo8",
        "backend": "hailo8",
        "type": "matrix",
        "variant": "full",
    }]
    plan = {"runs": rows, "planned_runs": rows, "targets": ["hailo8"]}
    contract = {
        "planned_runs": rows,
        "plan": {"runs": rows, "planned_runs": rows},
        "cases": [],
    }
    for path, payload in (
        (suite_dir / "benchmark_plan.json", plan),
        (formal_dir / "benchmark_plan.json", plan),
        (suite_dir / "benchmark_set.json", contract),
        (formal_dir / "benchmark_set.json", contract),
    ):
        _write(path, payload)
    result = finalize_suite_for_runtime(
        suite_dir=suite_dir,
        run_root=run_root,
        model_id="resnet18",
        suite_payload=contract,
        benchmark_plan=plan,
        profile_payload=_central_profile(),
        model_task="classification",
        quality_gate_policy=_central_profile()["quality_gate"],
    )
    finalized = result["benchmark_plan"]
    cpu = [row for row in finalized["runs"] if row["id"] == "ort_cpu"]
    assert len(cpu) == 1
    assert cpu[0]["task"] == "classification"
    assert cpu[0]["benchmark_task"] == "classification"
    assert cpu[0]["task_quality_gate"] == _central_profile()["quality_gate"]
    assert len(cpu[0]["quality_gate_policy_sha256"]) == 64
    assert (
        finalized["management_cpu_reference_invariant"]["run_plan_sha256"]
        == sha256_json(finalized["runs"])
    )


def test_public_execute_consumes_prefinalized_plan_without_a_second_writer(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    formal_dir = run_root / "models" / "resnet18" / "benchmark_set"
    suite_dir = formal_dir / "legacy_suite"
    plan = {"runs": _base_runs(), "targets": ["hailo8"]}
    contract = {
        "schema": "onnx-splitpoint/benchmark-set",
        "materialized": True,
        "legacy_suite_dir": str(suite_dir),
        "cases": [],
        "planned_runs": _base_runs(),
        "plan": {"runs": _base_runs(), "planned_runs": _base_runs()},
    }
    for path, payload in (
        (suite_dir / "benchmark_plan.json", plan),
        (formal_dir / "benchmark_plan.json", plan),
        (suite_dir / "benchmark_set.json", contract),
        (formal_dir / "benchmark_set.json", contract),
    ):
        _write(path, payload)
    finalization = finalize_suite_for_runtime(
        suite_dir=suite_dir,
        run_root=run_root,
        model_id="resnet18",
        suite_payload=contract,
        benchmark_plan=plan,
        profile_payload=_central_profile(),
        model_task="classification",
        quality_gate_policy=_central_profile()["quality_gate"],
        require_existing_cpu_reference=True,
    )
    sealed = (suite_dir / "benchmark_plan.json").read_bytes()
    result = execute_benchmark_suite_if_requested(
        run_dir=run_root,
        model_id="resnet18",
        options=SimpleNamespace(
            execution_mode="contracts_only",
            skip_benchmarks=True,
            dry_run=False,
            no_remote=True,
            benchmark_execution_backend="auto",
        ),
        benchmark_set_contract=_read(formal_dir / "benchmark_set.json"),
        benchmark_plan=_read(formal_dir / "benchmark_plan.json"),
        profile_payload=_central_profile(),
        model_entry={"task": "classification"},
        runtime_plan_finalization=finalization,
    )
    assert result.status == "skipped"
    assert (suite_dir / "benchmark_plan.json").read_bytes() == sealed


def test_remote_full_runner_preflight_invokes_management_free_helper(
    tmp_path: Path,
) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts/native_full_baseline_eval_runner.py"),
            "--root",
            str(tmp_path),
            "--remote-contract-preflight",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    proof = json.loads(completed.stdout.strip().splitlines()[-1])
    assert proof == {
        "fail_closed_result": [],
        "helper_invoked": True,
        "module": "onnx_splitpoint_tool.hailo_full_contract_promotion",
        "ok": True,
    }
    assert (
        ROOT / "scripts/native_full_baseline_eval_runner.py"
    ).read_bytes() == (
        ROOT
        / "onnx_splitpoint_tool/resources/remote_scripts/"
        "native_full_baseline_eval_runner.py"
    ).read_bytes()


def test_early_preflight_probes_exactly_three_resolved_nodes(
    tmp_path: Path,
) -> None:
    script = _load_preflight_script()
    registry = tmp_path / "hardware_setups.yaml"
    registry.write_text(
        "hardware_setups:\n"
        "  - id: h8\n"
        "    accelerator: hailo8\n"
        "    host: 10.0.0.8\n"
        "    user: nx\n"
        "    remote_venv: /venvs/h8/bin/activate\n"
        "  - id: h10\n"
        "    accelerator: hailo10h\n"
        "    host: 10.0.0.10\n"
        "    user: nx\n"
        "    remote_venv: /venvs/h10/bin/activate\n"
        "  - id: dx\n"
        "    accelerator: deepx_m1\n"
        "    host: 10.0.0.20\n"
        "    user: nx\n"
        "    remote_venv: /venvs/dx/bin/activate\n",
        encoding="utf-8",
    )
    profile = {
        "hardware": {"setups_file": str(registry)},
        "run_profiles": [
            {"id": "hailo8", "full": "hailo8"},
            {"id": "hailo10", "full": "hailo10"},
            {"id": "deepx_m1_full", "full": "deepx_m1"},
        ],
        "native_producers": {
            "remote_tool_dir": "/remote/tool",
            "remotes": {
                "hailo10": {
                    "ssh": "override@10.1.0.10",
                    "env": "source /override/h10/bin/activate",
                    "remote_tool_dir": "/override/tool",
                    "setup_id": "override_h10",
                },
            },
        },
    }
    calls: dict[str, list] = {
        "script": [],
        "asset": [],
        "verify": [],
        "process": [],
    }

    def fake_sync_script(**kwargs):
        calls["script"].append(dict(kwargs))
        return [{"name": "script", "rc": 0}]

    def fake_sync_asset(**kwargs):
        calls["asset"].append(dict(kwargs))
        return [{
            "name": "asset",
            "rc": 0,
            "expected_sha256": "a" * 64,
        }]

    def fake_verify(**kwargs):
        calls["verify"].append(dict(kwargs))
        return {
            "name": "verify",
            "rc": 0,
            "verification": {
                "ok": True,
                "imported_path": kwargs["relative_path"],
                "sha256": kwargs["expected_sha256"],
            },
        }

    def fake_process(command, *, label, timeout_s):
        calls["process"].append((list(command), label, timeout_s))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "ok": True,
                "helper_invoked": True,
                "fail_closed_result": [],
            }) + "\n",
            stderr="",
        )

    result = script.run_remote_contract_preflight(
        profile,
        sync_script=fake_sync_script,
        sync_asset=fake_sync_asset,
        verify_module=fake_verify,
        process_runner=fake_process,
    )
    assert result["ok"] is True
    assert [row["backend"] for row in result["rows"]] == [
        "hailo8", "hailo10h", "deepx",
    ]
    assert result["rows"][1]["ssh"] == "override@10.1.0.10"
    assert result["rows"][1]["remote_env"] == (
        "source /override/h10/bin/activate"
    )
    assert result["rows"][1]["remote_tool_dir"] == "/override/tool"
    assert result["rows"][1]["setup_id"] == "override_h10"
    assert len(calls["script"]) == 9
    closure_count = len(script.native_remote_package_closure())
    assert len(calls["asset"]) == 3 * closure_count
    assert len(calls["verify"]) == 3 * closure_count
    assert len(calls["process"]) == 3
    assert all(
        "--remote-contract-preflight" in call[0][-1]
        for call in calls["process"]
    )
    assert all(
        "native_full_semantic_dump.py --help" in call[0][-1]
        for call in calls["process"]
    )


def test_field_wrapper_orders_remote_probe_before_run_and_always_reports(
) -> None:
    source = (
        ROOT / "scripts/run_v27520_full_only_check.sh"
    ).read_text(encoding="utf-8")
    trap_pos = source.index("trap report_wrapper_exit EXIT")
    remote_probe = source.index(
        "scripts/preflight_v27520_native_remotes.py"
    )
    snapshot = source.index("RUN_SNAPSHOT=\"$(mktemp")
    marker = source.index("RUN_MARKER=\"$(mktemp")
    workflow = source.index(
        "-m onnx_splitpoint_tool.workflow.run_evaluation"
    )
    assert trap_pos < remote_probe < snapshot < marker < workflow
    assert "workflow_rc=$?" in source
    assert "discovery_rc=$?" in source
    assert "postcondition_rc=$?" in source
    assert "native_verify_rc=$?" in source
    assert "Debug-Pack-Pfade:" in source
    assert "RUN_DIR: %s" in source


def test_field_wrapper_shell_helpers_preserve_rc_and_resist_pack_symlinks(
    tmp_path: Path,
) -> None:
    wrapper = ROOT / "scripts/run_v27520_full_only_check.sh"
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    victim = tmp_path / "victim.zip"
    victim.write_bytes(b"unchanged")
    (runs_root / "run_v27520_debug_pack.zip").symlink_to(victim)
    (runs_root / "dangling_v27520_debug_pack.zip").symlink_to(
        tmp_path / "missing.zip"
    )
    (runs_root / "regular_v27520_debug_pack.zip").write_bytes(b"spoof")

    shell = subprocess.run(
        [
            "bash",
            "-c",
            r'''
set -Eeuo pipefail
source "$1"
set +e
v27520_discovery_failure_rc 130 1
fatal_rc=$?
v27520_discovery_failure_rc 1 7
normal_rc=$?
v27520_terminal_wrapper_rc 1 failed 0 0
failed_terminal_rc=$?
v27520_terminal_wrapper_rc 1 partial 0 0
partial_terminal_rc=$?
set -e
target_one="$(v27520_secure_debug_pack_target "$2" run)"
target_two="$(v27520_secure_debug_pack_target "$2" run)"
printf '%s\n%s\n%s\n%s\n%s\n%s\n' \
  "$fatal_rc" "$normal_rc" "$failed_terminal_rc" "$partial_terminal_rc" \
  "$target_one" "$target_two"
''',
            "v27520-wrapper-helper-test",
            str(wrapper),
            str(runs_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert shell.returncode == 0, shell.stderr
    (
        fatal_rc,
        normal_rc,
        failed_terminal_rc,
        partial_terminal_rc,
        raw_one,
        raw_two,
    ) = shell.stdout.splitlines()
    assert fatal_rc == "130"
    assert normal_rc == "7"
    assert failed_terminal_rc == "1"
    assert partial_terminal_rc == "0"
    targets = (Path(raw_one), Path(raw_two))
    assert targets[0] != targets[1]
    for target in targets:
        assert target.parent.parent == runs_root
        assert target.parent.is_dir()
        assert not target.parent.is_symlink()
        assert not target.exists()
    assert victim.read_bytes() == b"unchanged"
    assert (runs_root / "regular_v27520_debug_pack.zip").read_bytes() == b"spoof"
