from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.native_split_quality import (
    seal_native_split_quality_binding,
)
from onnx_splitpoint_tool.native_split_quality_authority import (
    CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
)
from onnx_splitpoint_tool.trt_quality_chain import (
    TensorRTQualityChainError,
    split_binding_set_from_central_quality_summary,
)
from tests.test_v269f_native_split_receipt_validation import _fixture_payload


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"test_v269f_variant_{path.stem}", path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evalrun(tmp_path: Path) -> Path:
    run = tmp_path / "eval-native-split-001"
    benchmark_set = run / "models" / "yolo26s" / "benchmark_set"
    case = benchmark_set / "b038"
    case.mkdir(parents=True)
    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps({"benchmark_task": "detection"}), encoding="utf-8",
    )
    (case / "split_manifest.json").write_text("{}", encoding="utf-8")
    full_snapshot = "a" * 64
    selection_snapshot = "b" * 64
    (run / "run_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run.name,
        "workflow_version": CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
        "current_workflow_version": CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
        "tool_version": "2.69.6",
        "current_tool_version": "2.69.6",
        "execution_sessions": [{
            "workflow_version": CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
            "tool_version": "2.69.6",
        }],
        "profile_start_snapshot": {
            "snapshot_sha256": full_snapshot,
            "requested_selection": {"snapshot_sha256": selection_snapshot},
            "resolved_selection": {"snapshot_sha256": selection_snapshot},
        },
    }), encoding="utf-8")
    (run / "profile.yaml").write_text(json.dumps({
        "quality_gate": {
            "schema": "onnx-splitpoint/task-quality-policy",
            "schema_version": 3,
            "name": "fixture_quality_policy",
            "profile_id": "fixture_quality_policy",
        },
    }), encoding="utf-8")
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "native_producer_stage.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/native-producer-stage",
        "schema_version": 3,
        "run_id": run.name,
        "workflow_version": CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
        "tool_version": "2.69.6",
        "profile_start_snapshot_sha256": full_snapshot,
        "profile_selection_snapshot_sha256": selection_snapshot,
        "native_split_quality_first": {"required": True},
    }), encoding="utf-8")
    return run


def _binding_and_summary(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, _ = _fixture_payload(tmp_path / "producer")
    binding = seal_native_split_quality_binding(payload)
    request_sha = "3" * 64
    identity = {
        "identity_valid": True,
        "eval_run_id": binding["eval_run_id"],
        "model_id": "yolo26s",
        "task": "detection",
        "case_id": "b038",
        "source_run_id": "hailo8_to_trt",
        "setup_id": "hailo8_setup",
        "variant": "composed",
        "runtime_precision_identity": "uint8_dequant_fp16",
        "native_split_quality_binding_required": True,
        "native_split_quality_binding": copy.deepcopy(binding),
        "native_split_quality_binding_sha256": binding["binding_sha256"],
        "source_request_sha256": request_sha,
    }
    result = {
        "status": "completed",
        "technical_status": "completed",
        "eval_run_id": binding["eval_run_id"],
        "model_id": "yolo26s",
        "task": "detection",
        "case_id": "b038",
        "source_run_id": "hailo8_to_trt",
        "source_setup_id": "hailo8_setup",
        "variant": "composed",
        "runtime_precision_identity": "uint8_dequant_fp16",
        "native_split_quality_binding_required": True,
        "native_split_quality_binding": copy.deepcopy(binding),
        "native_split_quality_binding_sha256": binding["binding_sha256"],
        "source_request_sha256": request_sha,
        "request_identity": identity,
    }
    summary = {
        "schema": "onnx-splitpoint/central-quality-summary",
        "schema_version": 1,
        "results": [result],
    }
    return binding, summary


def _write_summary(run: Path, summary: dict[str, Any]) -> Path:
    path = run / "quality_management" / "central_quality_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def _variant_cfg(summary_path: Path) -> dict[str, Any]:
    return {
        "backends": ["hailo8"],
        "precision": "uint8_dequant_fp16",
        "remotes": {
            "hailo8": {
                "ssh": "nx@hailo8",
                "setup_id": "hailo8_setup",
            },
        },
        "_workflow_context": {"central_quality_summary": str(summary_path)},
    }


def _binding_set_file(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    _, summary = _binding_and_summary(tmp_path)
    payload = split_binding_set_from_central_quality_summary(
        summary,
        eval_run_id="eval-native-split-001",
        setup_id="hailo8_setup",
        selections=[{
            "model_id": "yolo26s", "case_id": "b038",
            "backend": "hailo8_to_trt", "task": "detection",
            "precision": "uint8_dequant_fp16",
        }],
    )
    path = tmp_path / "native_split_quality_binding_set.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_coordinator_materializes_canonical_variant_setup_set_and_exact_command(
    tmp_path: Path,
) -> None:
    module = _load_script("run_evalrun_native_producer_variants.py")
    run = _evalrun(tmp_path)
    _, summary = _binding_and_summary(tmp_path)
    cfg = _variant_cfg(_write_summary(run, summary))
    cfg, policy_source = module._bind_evalrun_quality_gate_policy(run, cfg)
    assert policy_source == "inherited_from_evalrun_profile"
    variants = [{"id": "yolo", "case_map": {"yolo26s": ["b038"]}}]

    split_paths, split_plan = (
        module._materialize_native_split_quality_binding_sets(
            run, cfg, variants,
        )
    )
    assert set(split_paths) == {"v000_yolo"}
    assert set(split_paths["v000_yolo"]) == {"hailo8_setup"}
    set_path = Path(split_paths["v000_yolo"]["hailo8_setup"])
    binding_set = json.loads(set_path.read_text(encoding="utf-8"))
    assert binding_set["eval_run_id"] == run.name
    assert binding_set["setup_id"] == "hailo8_setup"
    assert set(binding_set["bindings_by_model_case_backend"]) == {
        "yolo26s|b038|hailo8_to_trt",
    }
    assert split_plan["variants"][0]["sha256"] == hashlib.sha256(
        set_path.read_bytes()
    ).hexdigest()

    prepared = module._quality_first_variant_plan(
        cfg, variants, {}, {"owner_by_setup": {}}, split_paths,
    )
    command = module._build_update_cmd(
        run, cfg, prepared[0], refresh_suites=False, timeout_s=60,
    )
    assert command[command.index("--artifact-namespace") + 1] == "v000_yolo"
    assert command[command.index("--hailo8-setup-id") + 1] == "hailo8_setup"
    forwarded = json.loads(
        command[command.index("--native-split-quality-binding-sets") + 1]
    )
    assert forwarded == split_paths["v000_yolo"]
    assert "--native-split-quality-required" in command
    assert "--quality-gate-json" in command
    assert json.loads(
        command[command.index("--quality-gate-json") + 1]
    ) == cfg["quality_gate_policy"]
    assert "--no-build-missing-engines" in command
    assert "--build-missing-engines" not in command
    assert "--native-force-rebuild-engines" not in command


def test_parent_split_set_must_be_byte_semantically_equal_to_central(
    tmp_path: Path,
) -> None:
    module = _load_script("run_evalrun_native_producer_variants.py")
    run = _evalrun(tmp_path)
    _, summary = _binding_and_summary(tmp_path)
    cfg = _variant_cfg(_write_summary(run, summary))
    variants = [{"id": "yolo", "case_map": {"yolo26s": ["b038"]}}]
    paths, _ = module._materialize_native_split_quality_binding_sets(
        run, cfg, variants,
    )
    drifted = json.loads(
        Path(paths["v000_yolo"]["hailo8_setup"]).read_text(encoding="utf-8")
    )
    drifted["eval_run_id"] = "other-eval"
    supplied = tmp_path / "drifted_set.json"
    supplied.write_text(json.dumps(drifted), encoding="utf-8")
    cfg["native_split_quality_binding_sets_by_setup"] = {
        "hailo8_setup": str(supplied),
    }

    with pytest.raises(TensorRTQualityChainError, match="identity/coverage"):
        module._materialize_native_split_quality_binding_sets(
            run, cfg, variants,
        )


def test_updater_rejects_exact_source_alias_before_any_remote_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("update_evalset_native_producers.py")
    run = _evalrun(tmp_path)
    set_path, payload = _binding_set_file(tmp_path)
    binding = copy.deepcopy(
        payload["bindings_by_model_case_backend"][
            "yolo26s|b038|hailo8_to_trt"
        ]
    )
    binding.pop("binding_sha256")
    # The deep validator deliberately normalizes this alias. The variant hand-off
    # is stricter: exact source_run_id must still be hailo8_to_trt.
    binding["source_run_id"] = "hailo8_to_tensorrt"
    receipt = binding["central_quality_selection"]
    receipt.pop("receipt_sha256")
    receipt["source_run_id"] = "hailo8_to_tensorrt"
    producer = copy.deepcopy(binding)
    for field in (
        "producer_binding_sha256", "source_request_sha256",
        "central_result_sha256", "central_quality_selection",
        "central_quality_selection_sha256",
    ):
        producer.pop(field, None)
    binding["producer_binding_sha256"] = canonical_json_sha256(producer)
    receipt["producer_binding_sha256"] = binding["producer_binding_sha256"]
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    binding["central_quality_selection_sha256"] = receipt["receipt_sha256"]
    binding["binding_sha256"] = canonical_json_sha256(binding)
    payload["bindings_by_model_case_backend"][
        "yolo26s|b038|hailo8_to_trt"
    ] = binding
    payload.pop("binding_set_sha256")
    payload["binding_set_sha256"] = canonical_json_sha256(payload)
    set_path.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        module, "_run",
        lambda cmd, **kwargs: calls.append(list(cmd)) or {"rc": 0},
    )

    stage = module._run_native_producers(run, {
        "backends": ["hailo8"],
        "case_policy": "case_map_only",
        "case_map": {"yolo26s": ["b038"]},
        "precision": "uint8_dequant_fp16",
        "remotes": {
            "hailo8": {"ssh": "nx@hailo8", "setup_id": "hailo8_setup"},
        },
        "native_split_quality_required": True,
        "native_split_quality_binding_sets_by_setup": {
            "hailo8_setup": str(set_path),
        },
    }, timeout=10)

    assert stage["status"] == "failed"
    assert "source_run_id" in stage["preflight_errors_by_backend"]["hailo8"]
    assert calls == []


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("eval_run_id", "other-eval"), ("setup_id", "other-setup")],
)
def test_updater_rejects_set_eval_or_setup_before_remote_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    field: str, replacement: str,
) -> None:
    module = _load_script("update_evalset_native_producers.py")
    run = _evalrun(tmp_path)
    set_path, payload = _binding_set_file(tmp_path)
    payload[field] = replacement
    set_path.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        module, "_run",
        lambda cmd, **kwargs: calls.append(list(cmd)) or {"rc": 0},
    )

    stage = module._run_native_producers(run, {
        "backends": ["hailo8"],
        "case_policy": "case_map_only",
        "case_map": {"yolo26s": ["b038"]},
        "precision": "uint8_dequant_fp16",
        "remotes": {
            "hailo8": {"ssh": "nx@hailo8", "setup_id": "hailo8_setup"},
        },
        "native_split_quality_required": True,
        "native_split_quality_binding_sets_by_setup": {
            "hailo8_setup": str(set_path),
        },
    }, timeout=10)

    assert stage["status"] == "failed"
    assert calls == []


def test_split_set_local_and_remote_sha_are_both_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("update_evalset_native_producers.py")
    path = tmp_path / "set.json"
    path.write_text('{"schema":"set"}', encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> dict[str, Any]:
        calls.append(list(cmd))
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(module, "_run", fake_run)
    remote_path, steps = module._stage_remote_native_split_quality_binding_set(
        local_path=path,
        expected_sha256=expected,
        ssh="nx@host",
        remote_root="/remote/eval/variants/v000",
        timeout=30,
    )
    assert remote_path.endswith("/.quality_first/native_split_quality_binding_set.json")
    assert [step["name"] for step in steps] == [
        "mkdir_remote_native_split_quality_set",
        "sync_remote_native_split_quality_set",
        "verify_remote_native_split_quality_set",
    ]
    assert calls[1][:3] == ["rsync", "-a", "--checksum"]
    assert expected in calls[2][-1]

    path.write_text('{"schema":"changed"}', encoding="utf-8")
    calls.clear()
    with pytest.raises(TensorRTQualityChainError, match="changed after local"):
        module._stage_remote_native_split_quality_binding_set(
            local_path=path,
            expected_sha256=expected,
            ssh="nx@host",
            remote_root="/remote/eval/variants/v000",
            timeout=30,
        )
    assert calls == []


def test_two_variant_runs_keep_disjoint_remote_and_local_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("update_evalset_native_producers.py")
    run = _evalrun(tmp_path)
    set_path, _ = _binding_set_file(tmp_path)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append((list(cmd), dict(kwargs)))
        stdout = ""
        if cmd and cmd[0] == "ssh" and "available_kb=" in str(cmd[-1]):
            stdout = "available_kb=1000000\nexisting_target_kb=0\n"
        return {"rc": 0, "stdout_tail": stdout, "stderr_tail": ""}

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_sync_remote_script_v60i", lambda *a, **k: [])
    monkeypatch.setattr(module, "_sync_remote_package_asset_v263", lambda *a, **k: [])
    monkeypatch.setattr(
        module, "_verify_remote_module_binding_v263",
        lambda *a, **k: {"rc": 0},
    )
    monkeypatch.setattr(
        module, "_capture_remote_host_telemetry",
        lambda **kwargs: {"name": f"capture_{kwargs['phase']}", "rc": 0},
    )
    monkeypatch.setattr(
        module, "_summarize_host_telemetry",
        lambda *a, **k: {"name": "summary", "rc": 0},
    )

    def config(namespace: str) -> dict[str, Any]:
        return {
            "backends": ["hailo8"],
            "case_policy": "case_map_only",
            "case_map": {"yolo26s": ["b038"]},
            "precision": "uint8_dequant_fp16",
            "remotes": {
                "hailo8": {
                    "ssh": "nx@hailo8", "setup_id": "hailo8_setup",
                },
            },
            "remote_root": "/remote/native",
            "copy_benchmarksets": True,
            "build_missing_engines": True,
            "native_split_quality_required": True,
            "native_split_quality_binding_sets_by_setup": {
                "hailo8_setup": str(set_path),
            },
            "artifact_namespace": namespace,
            "validation": {"enabled": False},
            "energy": {"enabled": False},
        }

    first = module._run_native_producers(run, config("v000_one"), timeout=10)
    first_root = run / "native_producers" / "variants" / "v000_one" / "hailo8"
    marker = first_root / "preserve.txt"
    marker.write_text("variant-one", encoding="utf-8")
    second = module._run_native_producers(run, config("v001_two"), timeout=10)
    second_root = run / "native_producers" / "variants" / "v001_two" / "hailo8"

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert marker.read_text(encoding="utf-8") == "variant-one"
    assert second_root.is_dir()
    reset_shells = [
        str(cmd[-1]) for cmd, _ in calls
        if cmd and cmd[0] == "ssh" and "rm -rf" in str(cmd[-1])
    ]
    assert any(
        "/eval-native-split-001/variants/v000_one/yolo26s/benchmark_set"
        in shell for shell in reset_shells
    )
    assert any(
        "/eval-native-split-001/variants/v001_two/yolo26s/benchmark_set"
        in shell for shell in reset_shells
    )
    split_shells = [
        str(cmd[-1]) for cmd, kwargs in calls
        if kwargs.get("label") == "split:hailo8"
    ]
    assert len(split_shells) == 2
    assert all("--native-split-quality-binding-set" in shell for shell in split_shells)
    assert all("--setup-id hailo8_setup" in shell for shell in split_shells)
    assert all("--build-missing-engines" not in shell for shell in split_shells)
    assert all("--force-rebuild-engines" not in shell for shell in split_shells)


def test_coordinator_retains_both_variants_through_combined_semantics_and_energy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("run_evalrun_native_producer_variants.py")
    run = _evalrun(tmp_path)
    second_case = run / "models" / "yolo26s" / "benchmark_set" / "b039"
    second_case.mkdir(parents=True)
    (second_case / "split_manifest.json").write_text("{}", encoding="utf-8")
    _, summary = _binding_and_summary(tmp_path)
    summary_path = _write_summary(run, summary)
    cfg = {
        **_variant_cfg(summary_path),
        "variants": [
            {"id": "one", "case_map": {"yolo26s": ["b038"]}},
            {"id": "two", "case_map": {"yolo26s": ["b039"]}},
        ],
        "validation": {"enabled": True},
        # The 2.71 phase-1 Energy preflight requires the theoretical
        # setup-local Split + Vendor Full + TensorRT Full matrix before either
        # variant is streamed. Keep this historical lifecycle test viable by
        # declaring the Full paths it later mocks.
        "full_baselines": {
            "enabled": True,
            "backends_by_producer": {
                "hailo8": ["hailo8", "tensorrt"],
            },
        },
        "energy": {"enabled": True, "mode": "plan"},
    }
    config_path = tmp_path / "variants.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    reports = run / "reports"
    labels: list[str] = []
    trt_producer_set = tmp_path / "trt_quality_producer_set.json"
    trt_producer_set.write_text("{}", encoding="utf-8")
    split_binding_set = tmp_path / "native_split_quality_binding_set.json"
    split_binding_set.write_text("{}", encoding="utf-8")

    def markers() -> list[Path]:
        return sorted(
            (run / "native_producers" / "variants").glob("*/hailo8/marker.json")
        )

    def fake_run(
        cmd: list[str], *, timeout: int | float | None = None,
        cwd: Path | None = None, label: str = "native-child",
    ) -> dict[str, Any]:
        del timeout, cwd
        labels.append(label)
        if label.startswith("variant:"):
            namespace = cmd[cmd.index("--artifact-namespace") + 1]
            marker = (
                run / "native_producers" / "variants" / namespace
                / "hailo8" / "marker.json"
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"namespace": namespace}), encoding="utf-8")
        elif label == "final_report":
            assert len(markers()) == 2
            (reports / "native_producer_combined_summary.json").write_text(
                json.dumps({"rows": [
                    {
                        "execution_mode": "native_split",
                        "backend": "hailo8_to_trt",
                        "model": "yolo26s",
                        "case": "b038",
                        "setup_id": "hailo8_setup",
                        "comparison_backend": "hailo8",
                        "ok": True,
                    },
                    {
                        "execution_mode": "native_split",
                        "backend": "hailo8_to_trt",
                        "model": "yolo26s",
                        "case": "b039",
                        "setup_id": "hailo8_setup",
                        "comparison_backend": "hailo8",
                        "ok": True,
                    },
                    {
                        "execution_mode": "native_full_baseline",
                        "backend": "native_full_hailo8",
                        "model": "yolo26s",
                        "case": "full",
                        "setup_id": "hailo8_setup",
                        "comparison_backend": "hailo8",
                        "ok": True,
                    },
                    {
                        "execution_mode": "native_full_baseline",
                        "backend": "native_full_tensorrt",
                        "model": "yolo26s",
                        "case": "full",
                        "setup_id": "hailo8_setup",
                        "comparison_backend": "hailo8",
                        "ok": True,
                    },
                ]}), encoding="utf-8",
            )
        elif label == "native_validation":
            assert len(markers()) == 2
            validation = reports / "native_validation"
            validation.mkdir(parents=True, exist_ok=True)
            (validation / "native_producer_validation_summary.json").write_text(
                json.dumps({
                    "claim_ok_count": 0,
                    "semantic_ok_count": 4,
                    "technical_error_count": 0,
                    "row_count": 4,
                    "rows": [{"ok": True}] * 4,
                }),
                encoding="utf-8",
            )
        elif label == "native-energy:plan":
            assert len(markers()) == 2
            plan = reports / "native_energy_plan"
            plan.mkdir(parents=True, exist_ok=True)
            (plan / "native_producer_energy_plan.json").write_text(
                json.dumps({"rows": []}), encoding="utf-8",
            )
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "_materialize_trt_quality_producer_sets",
        lambda *a, **k: (
            {"hailo8_setup": str(trt_producer_set)},
            {
                "required_setups": ["hailo8_setup"],
                "owner_by_setup": {"hailo8_setup": 0},
                "owners": [{"setup_id": "hailo8_setup", "variant_index": 0}],
                "models": ["yolo26s"],
                "errors_by_setup": {},
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "_materialize_native_split_quality_binding_sets",
        lambda *a, **k: (
            {
                "v000_one": {"hailo8_setup": str(split_binding_set)},
                "v001_two": {"hailo8_setup": str(split_binding_set)},
            },
            {"variants": [], "errors_by_variant_setup": {}},
        ),
    )
    monkeypatch.setattr(
        module, "_select_report_python",
        lambda *a, **k: (sys.executable, {"onnxruntime_ok": True}),
    )
    monkeypatch.setattr(sys, "argv", [
        "run_evalrun_native_producer_variants.py",
        "--eval-run-dir", str(run),
        "--config", str(config_path),
        "--timeout", "30",
    ])

    assert module.main() == 0
    assert labels.index("variant:one") < labels.index("variant:two")
    assert labels.index("variant:two") < labels.index("native_validation")
    assert labels.index("native_validation") < labels.index("native-energy:plan")
    assert len(markers()) == 2


def test_remote_runtime_closure_and_capabilities_include_split_validator() -> None:
    from onnx_splitpoint_tool.remote_runtime_closure import (
        native_remote_package_closure,
    )

    source = (
        ROOT / "scripts" / "update_evalset_native_producers.py"
    ).read_text(encoding="utf-8")
    closure = {path: (module, tokens) for path, module, tokens in native_remote_package_closure()}
    module, tokens = closure["onnx_splitpoint_tool/native_split_quality.py"]
    assert module == "onnx_splitpoint_tool.native_split_quality"
    assert {
        "validate_native_split_quality_binding",
        "bind_quality_to_native_split",
    } <= set(tokens)
    assert "native_remote_package_closure()" in source
    assert '"--native-split-quality-binding-set"' in source
    assert '"--native-split-quality-binding"' in source
