from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.native_execution_contract import (
    build_native_execution_contract,
    resolve_native_execution_contract,
)
from onnx_splitpoint_tool.remote_runtime_closure import (
    native_remote_package_closure,
)
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope
from onnx_splitpoint_tool.trt_quality_chain import TensorRTQualityChainError
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    WorkflowOptions,
    _verify_remote_module_binding_v263,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v2752_{path.stem}_{id(path)}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_remote_full_dependency_closure_imports_in_isolated_tree(
    tmp_path: Path,
) -> None:
    closure = native_remote_package_closure()
    assert len(closure) == 20
    by_module = {module: (path, tokens) for path, module, tokens in closure}
    assert "onnx_splitpoint_tool.cache_verify_policy" in by_module
    assert "onnx_splitpoint_tool.hailo_attempt_receipts" in by_module
    assert "onnx_splitpoint_tool.hailo_timeout_policy" in by_module
    assert "onnx_splitpoint_tool.runners.native_full_input" in by_module
    assert "onnx_splitpoint_tool.runners.native_split_quality_runtime" in by_module
    ordered_modules = [module for _path, module, _tokens in closure]
    hailo_backend_index = ordered_modules.index(
        "onnx_splitpoint_tool.runners.backends.hailo_backend"
    )
    assert ordered_modules.index(
        "onnx_splitpoint_tool.hailo_attempt_receipts"
    ) < hailo_backend_index
    assert ordered_modules.index(
        "onnx_splitpoint_tool.hailo_timeout_policy"
    ) < hailo_backend_index
    promotion_path, promotion_tokens = by_module[
        "onnx_splitpoint_tool.hailo_full_contract_promotion"
    ]
    assert (
        promotion_path
        == "onnx_splitpoint_tool/hailo_full_contract_promotion.py"
    )
    assert promotion_tokens == (
        "def promote_verified_hailo_full_contracts",
        "def remote_import_preflight",
        "source_onnx_multiscale_raw_head",
    )
    assert "onnx_splitpoint_tool.workflow.benchmark_binding" not in by_module
    assert "onnx_splitpoint_tool.management_reference" not in by_module
    assert "onnx_splitpoint_tool.native_full_quality" not in by_module

    remote = tmp_path / "remote"
    (remote / "scripts").mkdir(parents=True)
    (remote / "onnx_splitpoint_tool/runners").mkdir(parents=True)
    shutil.copy2(
        ROOT / "scripts/native_full_semantic_dump.py",
        remote / "scripts/native_full_semantic_dump.py",
    )
    shutil.copy2(
        ROOT / "scripts/native_full_baseline_eval_runner.py",
        remote / "scripts/native_full_baseline_eval_runner.py",
    )
    shutil.copy2(
        ROOT / "scripts/native_progress.py",
        remote / "scripts/native_progress.py",
    )
    # The remote runtime closure deliberately contains runtime modules, not
    # the management-side public release identity.  Use inert package markers
    # so this test proves the staged runtime closure itself is sufficient and
    # cannot borrow missing modules through the developer checkout.
    for package_init in (
        "onnx_splitpoint_tool/__init__.py",
        "onnx_splitpoint_tool/runners/__init__.py",
        "onnx_splitpoint_tool/runners/backends/__init__.py",
        "onnx_splitpoint_tool/runners/harness/__init__.py",
        "onnx_splitpoint_tool/validation/__init__.py",
        "onnx_splitpoint_tool/workflow/__init__.py",
    ):
        target = remote / package_init
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# isolated remote package marker\n", encoding="utf-8")
    for relative, _module, tokens in closure:
        source = ROOT / relative
        target = remote / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        text = source.read_text(encoding="utf-8")
        assert all(token in text for token in tokens)

    # Replay the v2.75.19 mixed-version condition. These management modules are
    # deliberately unusable; the remote-safe Hailo helper must not import them.
    for stale_name in ("management_reference.py", "native_full_quality.py"):
        (remote / "onnx_splitpoint_tool" / stale_name).write_text(
            "raise RuntimeError('stale_management_module_imported')\n",
            encoding="utf-8",
        )

    isolated_code = (
        "import json,sys;"
        f"sys.path.insert(0,{str(remote)!r});"
        "from onnx_splitpoint_tool.hailo_full_contract_promotion "
        "import remote_import_preflight;"
        "proof=remote_import_preflight();"
        "assert proof['ok'] and proof['helper_invoked'];"
        "assert 'onnx_splitpoint_tool.management_reference' not in sys.modules;"
        "assert 'onnx_splitpoint_tool.native_full_quality' not in sys.modules;"
        "from onnx_splitpoint_tool.runners.backends import hailo_backend;"
        "from onnx_splitpoint_tool.hailo_attempt_receipts "
        "import start_hailo_attempt;"
        "assert hailo_backend.start_hailo_attempt is start_hailo_attempt;"
        "from pathlib import Path;"
        f"receipt_root=Path({str(remote / 'receipt_probe')!r});"
        "successful=start_hailo_attempt(receipt_root,attempt_kind='probe',"
        "endpoint='part1');"
        "successful.finish(value={'ok':True,'status':'success'});"
        "failed=start_hailo_attempt(receipt_root,attempt_kind='probe',"
        "endpoint='part1');"
        "failed.finish(error=RuntimeError('expected_probe_failure'),"
        "returned=False);"
        "terminal=json.loads((receipt_root/'hailo_attempt_receipts'/"
        "'terminal_attempt.json').read_text());"
        "assert terminal['semantic_status']=='failed';"
        "assert terminal['error']=='expected_probe_failure';"
        "print(json.dumps(proof,sort_keys=True))"
    )
    isolated = subprocess.run(
        [sys.executable, "-I", "-B", "-c", isolated_code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert isolated.returncode == 0, isolated.stderr
    assert json.loads(isolated.stdout)["fail_closed_result"] == []

    env = dict(os.environ)
    env["PYTHONPATH"] = str(remote)
    completed = subprocess.run(
        [
            sys.executable, "-B",
            str(remote / "scripts/native_full_semantic_dump.py"),
            "--help",
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr

    # Execute the exact production entry flag from the isolated staged tree.
    # ``-S`` plus an explicit dependency-only site-packages path prevents the
    # editable developer checkout's .pth hook from supplying a missing closure
    # member, while the poisoned management modules replay the .19 condition.
    full_preflight_env = dict(os.environ)
    full_preflight_env["PYTHONPATH"] = os.pathsep.join(
        (str(remote), sysconfig.get_paths()["purelib"])
    )
    full_preflight_command = [
        sys.executable,
        "-S",
        "-B",
        str(remote / "scripts/native_full_baseline_eval_runner.py"),
        "--root",
        ".",
        "--remote-contract-preflight",
    ]
    full_preflight = subprocess.run(
        full_preflight_command,
        cwd=remote,
        env=full_preflight_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert full_preflight.returncode == 0, full_preflight.stderr
    proof = json.loads(full_preflight.stdout.strip().splitlines()[-1])
    assert proof["ok"] is True
    assert proof["helper_invoked"] is True
    assert proof["fail_closed_result"] == []

    promotion_target = remote / promotion_path
    promotion_target.unlink()
    missing_promotion = subprocess.run(
        full_preflight_command,
        cwd=remote,
        env=full_preflight_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert missing_promotion.returncode != 0
    assert "hailo_full_contract_promotion" in missing_promotion.stderr
    shutil.copy2(ROOT / promotion_path, promotion_target)

    (remote / "onnx_splitpoint_tool/runners/native_full_input.py").unlink()
    failed = subprocess.run(
        [
            sys.executable, "-B",
            str(remote / "scripts/native_full_semantic_dump.py"),
            "--help",
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert failed.returncode != 0
    assert "native_full_input" in failed.stderr


@pytest.mark.parametrize(
    ("imported_path", "imported_sha256"),
    (
        ("/remote/old/hailo_full_contract_promotion.py", "a" * 64),
        (
            "/remote/tool/onnx_splitpoint_tool/"
            "hailo_full_contract_promotion.py",
            "b" * 64,
        ),
    ),
)
def test_hailo_promotion_remote_import_drift_fails_closed(
    imported_path: str,
    imported_sha256: str,
) -> None:
    def fake_process_runner(command, *, label, timeout_s):
        del label, timeout_s
        payload = {
            "ok": False,
            "imported_path": imported_path,
            "expected_path": (
                "/remote/tool/onnx_splitpoint_tool/"
                "hailo_full_contract_promotion.py"
            ),
            "sha256": imported_sha256,
            "expected_sha256": "a" * 64,
        }
        return SimpleNamespace(
            args=command,
            returncode=8,
            stdout=json.dumps(payload) + "\n",
            stderr="",
        )

    with pytest.raises(
        RuntimeError,
        match="remote Python module binding verification failed",
    ):
        _verify_remote_module_binding_v263(
            ssh="nx@host",
            remote_tool_dir="/remote/tool",
            remote_env="source /remote/env",
            module_name=(
                "onnx_splitpoint_tool.hailo_full_contract_promotion"
            ),
            relative_path=(
                "onnx_splitpoint_tool/hailo_full_contract_promotion.py"
            ),
            expected_sha256="a" * 64,
            process_runner=fake_process_runner,
        )


def test_hailo_promotion_closure_precedes_full_runner_in_both_coordinators(
) -> None:
    for relative in (
        "onnx_splitpoint_tool/workflow/runner.py",
        "scripts/update_evalset_native_producers.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        closure = source.index(
            "for asset_relative, module_name, asset_tokens in "
            "native_remote_package_closure():"
        )
        sync = source.index("_sync_remote_package_asset_v263(", closure)
        verify = source.index("_verify_remote_module_binding_v263(", sync)
        full_runner = source.index('"native_full_baseline_eval_runner.py"', verify)
        assert closure < sync < verify < full_runner


def test_execution_plan_and_variants_share_one_immutable_standard_contract(
) -> None:
    profile = {
        "execution_preset": {
            "id": "standard",
            "snapshot": {
                "runtime": {
                    "native": {
                        "frames": 100,
                        "warmup": 10,
                        "repetitions": 1,
                        "queue_depth": 2,
                        "inflight": 4,
                    }
                }
            },
        },
        "native_producers": {
            "frames": 1000,
            "warmup": 100,
            "repetitions": 3,
            "queue_depth": 3,
            "inflight": 8,
        },
    }
    plan = build_effective_execution_plan(profile)
    contract = plan["native_execution_contract"]
    assert [contract[field] for field in (
        "frames", "warmup", "repetitions", "queue_depth", "inflight",
    )] == [1000, 100, 3, 3, 8]
    assert plan["native_execution_contract_sha256"] == contract[
        "contract_sha256"
    ]

    variants = _load_script("run_evalrun_native_producer_variants.py")
    base = {
        **profile["native_producers"],
        "_workflow_context": {"execution_preset": {"id": "standard"}},
        "_native_execution_contract": contract,
    }
    merged = variants._merged_variant(
        base, {"id": "yolo", "precision": "uint8_dequant_fp16"},
    )
    assert merged["precision"] == "uint8_dequant_fp16"
    assert merged["_native_execution_contract"] == contract
    with pytest.raises(
        TensorRTQualityChainError,
        match="native_execution_contract_variant_override_forbidden:frames",
    ):
        variants._merged_variant(base, {"id": "bad", "frames": 100})


@pytest.mark.parametrize(
    ("mode", "values"),
    (
        ("smoke", (100, 10, 1, 2, 4)),
        ("standard", (1000, 100, 3, 3, 8)),
        ("final", (1000, 100, 3, 3, 8)),
    ),
)
def test_execution_contract_has_mode_correct_fail_safe_defaults(
    mode: str,
    values: tuple[int, int, int, int, int],
) -> None:
    contract = resolve_native_execution_contract({
        "execution_preset": {"id": mode},
        "native_producers": {},
    })
    assert tuple(contract[field] for field in (
        "frames", "warmup", "repetitions", "queue_depth", "inflight",
    )) == values


def _patch_constructible_energy_rows(
    planner: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def verified_contract(
        raw: Any, *, expected_identity: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        return {
            "contract_sha256": str(
                (raw or {}).get("contract_sha256") or "c" * 64
            ),
            "backend": expected_identity["backend"],
            "model": expected_identity["model"],
            "case": expected_identity["case"],
            "setup_id": expected_identity["setup_id"],
            "comparison_backend": expected_identity[
                "comparison_backend"
            ],
            "runtime_options": {},
            "energy_workload": {},
            "artifacts": {},
        }, "verified_test_contract"

    monkeypatch.setattr(
        planner, "_verify_full_command_contract", verified_contract,
    )
    monkeypatch.setattr(
        planner, "verify_native_energy_command_contract", verified_contract,
    )
    monkeypatch.setattr(
        planner, "verify_native_split_part2_input_contract",
        lambda _contract: ({"inputs": [{}]}, "verified_test_part2"),
    )
    monkeypatch.setattr(
        planner,
        "_split_quality_energy_evidence",
        lambda *_args, **_kwargs: ({
            "native_split_quality_required": False,
            "native_split_energy_binding_valid": True,
            "native_split_energy_binding_status": (
                "runtime_measurement_quality_not_available"
            ),
        }, "runtime_measurement_quality_not_available"),
    )
    monkeypatch.setattr(
        planner, "split_energy_runtime_argv",
        lambda *_args, **_kwargs: ["python", "split-energy.py"],
    )
    monkeypatch.setattr(
        planner, "_full_runtime_argv",
        lambda *_args, **_kwargs: ["python", "full-energy.py"],
    )
    monkeypatch.setattr(
        planner, "_split_preflight_argv",
        lambda *_args, **_kwargs: ["python", "split-preflight.py"],
    )
    monkeypatch.setattr(
        planner, "_full_preflight_argv",
        lambda *_args, **_kwargs: ["python", "full-preflight.py"],
    )
    monkeypatch.setattr(
        planner, "_process_local_runtime_environment",
        lambda _contract: {},
    )


def _runtime_matrix_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for setup, split, full, comparison in (
        ("setup_h8", "hailo8_to_trt", "native_full_hailo8", "hailo8"),
        ("setup_h10", "hailo10h_to_trt", "native_full_hailo10h", "hailo10h"),
        ("setup_dx", "deepx_to_trt", "native_full_deepx", "deepx"),
    ):
        for model in ("resnet50", "yolo26s", "yolov7_paper"):
            task = "classification" if model == "resnet50" else "detection"
            for case in ("b001", "b002", "b003"):
                rows.append({
                    "ok": True,
                    "backend": split,
                    "model": model,
                    "case": case,
                    "precision": "runtime_precision",
                    "setup_id": setup,
                    "comparison_backend": comparison,
                    "part2_input_count": 1,
                    "task": task,
                    "fps_makespan": 10.0,
                    "native_command_contract": {
                        "contract_sha256": "a" * 64,
                    },
                })
            for backend in (full, "native_full_tensorrt"):
                rows.append({
                    "ok": True,
                    "backend": backend,
                    "model": model,
                    "case": "full",
                    "precision": "runtime_precision",
                    "setup_id": setup,
                    "comparison_backend": comparison,
                    "task": task,
                    "fps_makespan": 8.0,
                    "full_command_contract": {
                        "contract_sha256": "b" * 64,
                    },
                })
    assert len(rows) == 45
    return rows


def test_partial_runtime_matrix_starts_energy_but_clamps_every_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _load_script("native_producer_energy_plan.py")
    all_rows = _runtime_matrix_rows()
    successful = all_rows[:7]
    missing = all_rows[7:]
    summary = tmp_path / "native_producer_summary.json"
    validation = tmp_path / "native_producer_validation_summary.json"
    out = tmp_path / "energy-plan"
    summary.write_text(json.dumps({"rows": successful}), encoding="utf-8")
    validation.write_text(json.dumps({"rows": []}), encoding="utf-8")
    (tmp_path / "native_expected_matrix.json").write_text(
        json.dumps({
            "expected_row_count": 45,
            "present_expected_row_count": 7,
            "successful_expected_row_count": 7,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 38,
            "row_presence_complete": False,
            "present_expected_rows": successful,
            "failed_expected_rows": [],
            "missing_expected_rows": missing,
        }),
        encoding="utf-8",
    )
    _patch_constructible_energy_rows(planner, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        str(planner.__file__),
        "--summary", str(summary),
        "--validation-summary", str(validation),
        "--out-dir", str(out),
        "--hailo8-ssh", "hailo8-host",
        "--hailo10-ssh", "hailo10-host",
        "--deepx-ssh", "deepx-host",
        "--duration-s", "1",
        "--screening-energy",
        "--measure-all-runtime-successful",
    ])

    assert planner.main() == 0
    payload = json.loads(
        (out / "native_producer_energy_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["preflight_status"] == "passed"
    assert payload["technical_measurement_contract_valid"] is True
    assert payload["energy_plan_coverage_contract_valid"] is False
    assert payload["preflight"]["measurement_start_allowed"] is True
    assert len(payload["rows"]) == 7
    assert all(
        row["diagnostic_only"] is True
        and row["claim_eligible"] is False
        and row["energy_claim_eligible"] is False
        and row["eligible_for_scientific_claim"] is False
        and row["energy_quality_admission"]["admission_scope"]
        == "native_runtime_observation"
        for row in payload["rows"]
    )

    executor = _load_script(
        "run_native_producer_energy_from_summary.py"
    )
    assert executor._measurement_preflight_allows(payload) is True
    blocked = json.loads(json.dumps(payload))
    blocked["preflight"]["technical_measurement_contract_valid"] = False
    assert executor._measurement_preflight_allows(blocked) is False


def test_terminal_partial_performance_checkpoint_is_energy_eligible(
    tmp_path: Path,
) -> None:
    variants = _load_script("run_evalrun_native_producer_variants.py")
    expected = [
        {
            "execution_mode": "native_split",
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": case,
            "setup_id": "setup_h8",
            "comparison_backend": "hailo8",
        }
        for case in ("b001", "b002")
    ]
    actual = [{**expected[0], "ok": True}]
    matrix = variants._native_performance_expected_matrix(expected, actual)
    for name, payload in (
        ("native_expected_matrix.json", matrix),
        ("native_producer_summary.json", {"rows": actual}),
        ("native_producer_combined_summary.json", {"rows": actual}),
    ):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    completion = variants._native_performance_completion(
        reports=tmp_path,
        expected_rows=expected,
        variant_count=2,
        stage={
            "variant_results": [
                {"id": "a", "rc": 0},
                {"id": "b", "rc": 2},
            ]
        },
        final_report={"rc": 0},
        required_campaign_rows=None,
    )

    assert completion["checkpoint_terminal_valid"] is True
    assert completion["performance_matrix_complete"] is False
    assert completion["scientific_coverage_complete"] is False


def _run_variant_checkpoint_energy_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scenario: str,
) -> tuple[int, list[str], dict[str, Any], dict[str, Any]]:
    """Run the real variant coordinator around a controlled child boundary."""

    coordinator = _load_script(
        "run_evalrun_native_producer_variants.py"
    )
    run_dir = tmp_path / f"EvaluationRun-{scenario}"
    reports = run_dir / "reports"
    (run_dir / "native_producers" / "hailo8").mkdir(
        parents=True,
    )
    for case in ("b001", "b002"):
        (
            run_dir / "models" / "resnet50" / "benchmark_set" / case
        ).mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/evaluation-run-manifest",
            "schema_version": 1,
            "run_id": run_dir.name,
            "workflow_version": "v2.75.2-checkpoint-gate-test",
            "tool_version": "2.75.2",
        }),
        encoding="utf-8",
    )
    (run_dir / "profile.yaml").write_text(
        json.dumps({
            "quality_gate": {
                "schema": "onnx-splitpoint/task-quality-policy",
                "schema_version": 3,
                "name": "fixture_quality_policy",
                "profile_id": "fixture_quality_policy",
            },
        }),
        encoding="utf-8",
    )
    config = {
        "frames": 1000,
        "warmup": 100,
        "repetitions": 3,
        "queue_depth": 3,
        "inflight": 8,
        "backends": ["hailo8"],
        "remotes": {
            "hailo8": {
                "ssh": "nx@test-host",
                "setup_id": "setup_h8",
            }
        },
        "variants": [
            {
                "id": "good",
                "case_map": {"resnet50": ["b001"]},
            },
            {
                "id": "failed",
                "case_map": {"resnet50": ["b002"]},
            },
        ],
        "validation": {"enabled": True},
        "energy": {
            "enabled": True,
            "mode": "measure",
            "duration_s": 1,
            "timeout": 1,
        },
        "_workflow_context": {
            "campaign": {"mode": "development"},
            "execution_preset": {"id": "development"},
        },
    }
    config_path = tmp_path / f"native-{scenario}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    expected = [
        {
            "execution_mode": "native_split",
            "backend_key": "hailo8",
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": case,
            "setup_id": "setup_h8",
            "comparison_backend": "hailo8",
            "variant": variant,
        }
        for case, variant in (("b001", "good"), ("b002", "failed"))
    ]
    successful_row = {
        **expected[0],
        "ok": True,
        "status": "ok",
        "precision": "float32_layout_fp16",
        "fps_makespan": 10.0,
        "part2_input_count": 1,
    }

    monkeypatch.setattr(
        coordinator,
        "_materialize_trt_quality_producer_sets",
        lambda *_args, **_kwargs: (
            {},
            {"status": "test", "errors_by_setup": {}},
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_materialize_native_split_quality_binding_sets",
        lambda *_args, **_kwargs: (
            {},
            {
                "required": False,
                "status": "test",
                "errors_by_variant_setup": {},
            },
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_quality_first_variant_plan",
        lambda _cfg, variants, *_args, **_kwargs: [
            dict(value) for value in variants
        ],
    )
    monkeypatch.setattr(
        coordinator,
        "resolve_native_split_quality_authority",
        lambda *_args, **_kwargs: {
            "valid": True,
            "workflow_version": "v2.75.2-checkpoint-gate-test",
            "tool_version": "2.75.2",
            "errors": [],
        },
    )
    monkeypatch.setattr(
        coordinator,
        "native_split_quality_required_for_row",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        coordinator,
        "_variant_expected_energy_rows",
        lambda *_args, **_kwargs: (
            list(expected),
            {"hailo8": "setup_h8"},
            True,
            False,
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_select_report_python",
        lambda *_args, **_kwargs: (
            sys.executable,
            {"selected": sys.executable, "onnxruntime_ok": True},
        ),
    )

    calls: list[str] = []

    def fake_run(
        cmd: list[str],
        *,
        timeout: int | None = None,
        cwd: Path | None = None,
        label: str = "native-child",
    ) -> dict[str, Any]:
        del timeout, cwd
        calls.append(label)
        if label == "variant:good":
            return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}
        if label == "variant:failed":
            if scenario == "nonterminal":
                return {"stdout_tail": "", "stderr_tail": "interrupted"}
            return {"rc": 2, "stdout_tail": "", "stderr_tail": "failed"}
        if label == "final_report":
            rows = (
                [successful_row, dict(successful_row)]
                if scenario == "ambiguous"
                else [successful_row]
            )
            (reports / "native_producer_combined_summary.json").write_text(
                json.dumps({"rows": rows}),
                encoding="utf-8",
            )
            return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}
        if label == "native_validation":
            validation = (
                reports / "native_validation"
                / "native_producer_validation_summary.json"
            )
            validation.parent.mkdir(parents=True, exist_ok=True)
            validation.write_text(
                json.dumps({
                    "status": "complete",
                    "technical_error_count": 0,
                    "row_count": 1,
                    "rows": [{"ok": True, "status": "technical_pass"}],
                }),
                encoding="utf-8",
            )
            return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}
        if label == "quality_gated_final_report":
            return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}
        if label == "native-energy:measure":
            checkpoint = json.loads(
                (
                    run_dir / "stages" / "run_native_producers"
                    / "native_performance" / "stage_result.json"
                ).read_text(encoding="utf-8")
            )
            assert checkpoint["state"] == "completed"
            assert checkpoint["complete"] is True
            assert checkpoint["details"][
                "checkpoint_terminal_valid"
            ] is True
            assert checkpoint["details"][
                "performance_matrix_complete"
            ] is False
            energy_out = Path(cmd[cmd.index("--out-dir") + 1])
            plan_path = (
                energy_out / "plan" / "native_producer_energy_plan.json"
            )
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(
                json.dumps({
                    "preflight_status": "passed",
                    "preflight": {
                        "status": "passed",
                        "ok": True,
                        "measurement_start_allowed": True,
                        "technical_measurement_contract_valid": True,
                        "energy_plan_coverage_contract_valid": False,
                    },
                    "rows": [successful_row],
                }),
                encoding="utf-8",
            )
            (energy_out / "native_producer_energy_results.json").write_text(
                json.dumps({
                    "ok": True,
                    "complete": True,
                    "status": "ok",
                    "started_measurement_count": 1,
                    "rows": [{"ok": True, "row": successful_row}],
                }),
                encoding="utf-8",
            )
            return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(coordinator, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "run_evalrun_native_producer_variants.py",
        "--eval-run-dir", str(run_dir),
        "--config", str(config_path),
        "--timeout", "1",
    ])

    return_code = coordinator.main()
    stage = json.loads(
        (reports / "native_producer_stage.json").read_text(
            encoding="utf-8"
        )
    )
    performance_checkpoint = json.loads(
        (
            run_dir / "stages" / "run_native_producers"
            / "native_performance" / "stage_result.json"
        ).read_text(encoding="utf-8")
    )
    return return_code, calls, stage, performance_checkpoint


def test_coordinator_terminal_partial_matrix_starts_energy_child_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    return_code, calls, stage, checkpoint = (
        _run_variant_checkpoint_energy_gate(
            tmp_path,
            monkeypatch,
            scenario="terminal_partial",
        )
    )

    assert return_code == 2
    assert calls.count("variant:good") == 1
    assert calls.count("variant:failed") == 1
    assert calls.count("native-energy:measure") == 1
    assert checkpoint["state"] == "completed"
    assert checkpoint["details"]["checkpoint_terminal_valid"] is True
    assert checkpoint["details"]["performance_matrix_complete"] is False
    assert stage["native_energy"]["started_measurement_count"] == 1
    assert stage["performance_claim_eligible"] is False
    assert stage["energy_claim_eligible"] is False
    assert stage["scientific_claim_eligible"] is False


@pytest.mark.parametrize("scenario", ("nonterminal", "ambiguous"))
def test_coordinator_blocks_energy_for_unsealed_performance_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    return_code, calls, stage, checkpoint = (
        _run_variant_checkpoint_energy_gate(
            tmp_path,
            monkeypatch,
            scenario=scenario,
        )
    )

    assert return_code == 2
    assert calls.count("native-energy:measure") == 0
    assert checkpoint["state"] == "failed"
    assert checkpoint["details"]["checkpoint_terminal_valid"] is False
    assert stage["failure_class"] == "native_performance_checkpoint"
    assert stage["upstream_stage"] == "native_performance"


@pytest.mark.parametrize("middle_returncode", [1, 3])
def test_hailo8_known_contract_failure_continues_and_seals_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    middle_returncode: int,
) -> None:
    runner = EvaluationWorkflowRunner(
        WorkflowOptions(profile="", out=str(tmp_path))
    )
    runner.run_id = f"hailo8_checkpoint_rc{middle_returncode}"
    runner.run_dir = tmp_path / runner.run_id
    runner._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(runner.run_id, runner.session_id),
        journal_dir=tmp_path / f"{runner.run_id}_remote_lease_journal",
    )
    runner.profile_payload = {
        "campaign": {"mode": "measurement"},
        "execution_preset": {"id": "standard"},
        "measurement_campaign": {
            "system_power": {"scope": "system", "window": "command"},
        },
    }
    runner.profile_start_snapshot = {}

    models = ["resnet50", "yolo26s", "yolov7_paper"]
    runner.manifest = {"models": {model: {} for model in models}}
    for model in models:
        suite = runner.run_dir / "models" / model / "benchmark_set"
        case_dir = suite / "b001"
        case_dir.mkdir(parents=True)
        (case_dir / "split_manifest.json").write_text(
            json.dumps({"part2_external_inputs": ["boundary_tensor"]}),
            encoding="utf-8",
        )
        (suite / "benchmark_set.json").write_text(
            json.dumps({"cases": [{"id": "b001"}]}),
            encoding="utf-8",
        )

    reports = runner.run_dir / "reports"
    reports.mkdir(parents=True)
    quality = runner.run_dir / "quality_management"
    quality.mkdir(parents=True)
    (quality / "central_quality_summary.json").write_text(
        json.dumps({"results": []}), encoding="utf-8",
    )

    cfg = {
        "enabled": True,
        "models": models,
        "backends": ["hailo8"],
        "precision": "fp16",
        "case_policy": "case_map_only",
        "case_map": {model: ["b001"] for model in models},
        "remotes": {
            "hailo8": {
                "ssh": "nx@hailo8-test",
                "setup_id": "hailo8_setup",
            },
        },
        "copy_benchmarksets": False,
        "build_missing_engines": False,
        "validation": {"enabled": True},
        "full_baselines": {"enabled": False},
        "energy": {"enabled": False},
        "cleanup_remote_native_root": False,
    }
    variants = [
        "hailo8_resnet50_selected_single_part2_input",
        "hailo8_yolo26s_selected_single_part2_input",
        "hailo8_yolov7_paper_selected_single_part2_input",
    ]
    returncodes = {
        variants[0]: 0,
        variants[1]: middle_returncode,
        variants[2]: 0,
    }
    dispatches: list[str] = []
    merge_collections: list[str] = []

    def fake_streaming(
        _command: list[str], *_args: Any, **kwargs: Any,
    ) -> SimpleNamespace:
        label = str(kwargs.get("label") or "")
        split_prefix = "split:hailo8:"
        collect_prefix = "checkpoint-collect:hailo8:"
        if label.startswith(split_prefix):
            variant = label[len(split_prefix):]
            dispatches.append(variant)
            return SimpleNamespace(
                returncode=returncodes[variant],
                stdout=f"runner {variant}",
                stderr=(
                    "injected native child failure"
                    if returncodes[variant] else ""
                ),
            )
        if label.startswith(collect_prefix):
            variant = label[len(collect_prefix):]
            merge_collections.append(variant)
            analysis = (
                runner.run_dir / "native_producers" / "hailo8"
                / "analysis_tables"
            )
            analysis.mkdir(parents=True, exist_ok=True)
            stem = f"native_fifo_eval_runner__{variant}"
            for suffix, content in (
                (".json", json.dumps({"variant": variant})),
                (".csv", f"variant,status\n{variant},terminal\n"),
                (".md", f"# {variant}\n\nterminal\n"),
            ):
                (analysis / f"{stem}{suffix}").write_text(
                    content, encoding="utf-8",
                )
        if label == "final_report":
            rows = []
            for model, variant in zip(models, variants):
                ok = returncodes[variant] == 0
                rows.append({
                    "backend": "hailo8_to_trt",
                    "model": model,
                    "case": "b001",
                    "precision": (
                        "float32_layout_fp16"
                        if model == "resnet50"
                        else "uint8_dequant_fp16"
                    ),
                    "setup_id": "hailo8_setup",
                    "execution_mode": "native_split",
                    "ok": ok,
                    "failure_reason": "" if ok else "native_runner_failed",
                })
            (reports / "native_producer_combined_summary.json").write_text(
                json.dumps({"rows": rows}), encoding="utf-8",
            )
        if label == "validation":
            validation = reports / "native_validation"
            validation.mkdir(parents=True, exist_ok=True)
            (validation / "native_producer_validation_summary.json").write_text(
                json.dumps({
                    "status": "complete",
                    "row_count": 3,
                    "technical_error_count": 0,
                    "rows": [],
                }),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.run_streaming",
        fake_streaming,
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.normalize_hardware_targets",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.benchmark_set_postcondition_v60v",
        lambda path: {"valid": True, "selected_suite_dir": str(path)},
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner._sync_remote_script_v60i",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner._sync_remote_package_asset_v263",
        lambda *_args, **_kwargs: [{
            "name": "sync_fixture",
            "rc": 0,
            "expected_sha256": "a" * 64,
        }],
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner._verify_remote_module_binding_v263",
        lambda *_args, **_kwargs: {
            "name": "verify_fixture",
            "rc": 0,
        },
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner._select_native_report_python",
        lambda *_args, **_kwargs: (
            sys.executable,
            {"selected": sys.executable, "onnxruntime_ok": True},
        ),
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner._native_analysis_diagnostics_v60u",
        lambda _root: {
            "row_count": 3,
            "ok_count": 2,
            "failed_count": 1,
            "failure_reasons": ["native_runner_failed"],
            "evidence_status": "partial",
        },
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.trt_quality_chain.load_split_binding_set_from_central_quality_summary",
        lambda *_args, **_kwargs: {
            "schema": "test/native-split-quality-binding-set",
            "bindings": [],
        },
    )
    monkeypatch.setattr(
        runner,
        "_finish_native_direct_remote_lease",
        lambda *_args, **_kwargs: None,
    )

    with mock.patch.object(runner, "_native_producer_config", return_value=cfg):
        _paths, _details, _message, _status = (
            runner._stage_run_native_producers()
        )

    assert dispatches == variants
    assert merge_collections == variants

    checkpoints = reports / "native_known_contract_checkpoints" / "hailo8"
    payloads = [
        json.loads((checkpoints / f"{variant}.json").read_text(
            encoding="utf-8"
        ))
        for variant in variants
    ]
    assert [payload["runner_returncode"] for payload in payloads] == [
        0, middle_returncode, 0,
    ]
    assert all(payload["terminal"] is True for payload in payloads)
    assert all(
        payload["collection_status"] == "collected"
        and len(payload["collected_artifacts"]) == 3
        for payload in payloads
    )
    assert all(
        int(artifact["size_bytes"]) > 0
        and str(artifact["sha256"]).startswith("sha256:")
        and len(str(artifact["sha256"])) == len("sha256:") + 64
        for payload in payloads
        for artifact in payload["collected_artifacts"]
    )
    assert payloads[1]["failure_reason"] == "native_runner_failed"
    assert payloads[1]["failure_reason"] != "native_transfer_failed"

    stage = json.loads(
        (reports / "native_producer_stage.json").read_text(encoding="utf-8")
    )
    hailo8 = next(
        row for row in stage["backend_results"]
        if row.get("backend") == "hailo8"
    )
    assert hailo8["failure_reason"] == "native_runner_failed"
    assert hailo8["failure_reason"] != "native_transfer_failed"
    assert [
        failure["variant"]
        for failure in hailo8["known_contract_failures"]
    ] == [variants[1]]
