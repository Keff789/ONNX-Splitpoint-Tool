from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import onnx
from onnx import TensorProto, helper
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_v2776_yolo26_archived_part1.py"
CASE_IDS = ("b066", "b088", "b104", "b199")
SPLIT_LAYER = {66: 6, 88: 8, 104: 8, 199: 16}
BASE_CONV = {66: 14, 88: 24, 104: 24, 199: 50}
OUTPUT_COUNT = {66: 4, 88: 8, 104: 5, 199: 7}
IDENTITY_SLOTS = {66: (1, 2), 88: (2, 3), 104: (2, 3), 199: (2, 3)}


def _load_probe():
    spec = importlib.util.spec_from_file_location("v2776_direct_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case_model(case_id: str) -> tuple[onnx.ModelProto, list[str], dict]:
    boundary = int(case_id[1:])
    split_name = f"/model.{SPLIT_LAYER[boundary]}/Split"
    identity_slots = IDENTITY_SLOTS[boundary]
    identity_names = [
        f"splitpoint_hailo_part1_output_identity_b{boundary}_{slot}"
        for slot in identity_slots
    ]
    split_outputs = [f"{split_name}_output_0", f"{split_name}_output_1"]
    identity_outputs = [f"{value}__hailo_identity" for value in split_outputs]
    nodes = [
        helper.make_node(
            "Split",
            ["input"],
            split_outputs,
            name=split_name,
            axis=1,
        ),
        helper.make_node(
            "Identity",
            [split_outputs[0]],
            [identity_outputs[0]],
            name=identity_names[0],
        ),
        helper.make_node(
            "Identity",
            [split_outputs[1]],
            [identity_outputs[1]],
            name=identity_names[1],
        ),
    ]
    graph_outputs: list[onnx.ValueInfoProto] = []
    archived_end_nodes: list[str] = []
    identity_by_slot = dict(zip(identity_slots, range(2)))
    for slot in range(OUTPUT_COUNT[boundary]):
        if slot in identity_by_slot:
            identity_index = identity_by_slot[slot]
            graph_outputs.append(
                helper.make_tensor_value_info(
                    identity_outputs[identity_index], TensorProto.FLOAT, [1, 2]
                )
            )
            archived_end_nodes.append(identity_names[identity_index])
            continue
        direct_name = f"/fixture/{case_id}/Relu_{slot}"
        direct_output = f"direct_output_{slot}"
        nodes.append(
            helper.make_node(
                "Relu", ["input"], [direct_output], name=direct_name
            )
        )
        graph_outputs.append(
            helper.make_tensor_value_info(
                direct_output, TensorProto.FLOAT, [1, 4]
            )
        )
        archived_end_nodes.append(direct_name)
    graph = helper.make_graph(
        nodes,
        f"fixture_{case_id}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        graph_outputs,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    identity_fix = {
        "applied": True,
        "aliases": {
            identity_outputs[0]: split_outputs[0],
            identity_outputs[1]: split_outputs[1],
        },
        "materialized_original_outputs": split_outputs,
        "reason": (
            "yolo26_part1_feature_splitter_outputs_materialized_through_identity"
        ),
    }
    return model, archived_end_nodes, identity_fix


def _archived_run(tmp_path: Path, *, omit_case: str = "") -> Path:
    run_dir = tmp_path / "source-run"
    root = (
        run_dir
        / "models"
        / "yolo26s"
        / "benchmark_set"
        / "legacy_suite"
        / "_rejected_cases"
    )
    for case_id in CASE_IDS:
        if case_id == omit_case:
            continue
        boundary = int(case_id[1:])
        case_dir = root / case_id
        case_dir.mkdir(parents=True)
        model_name = f"yolo26s_part1_hailo_identity_b{boundary}.onnx"
        model, archived_end_nodes, identity_fix = _case_model(case_id)
        onnx.save(model, case_dir / model_name)
        previous_error = f"'base_conv{BASE_CONV[boundary]}' is not in list"
        manifest = {
            "hailo": {
                "part1_accel_model": model_name,
                "part1_feature_splitter_identity_fix": identity_fix,
                "hefs": {
                    "hailo10": {
                        "part1_error": previous_error,
                        "part1_feature_splitter_identity_fix": identity_fix,
                        "part1_base_conv_resolution": {
                            "attempted": True,
                            "strategy": "explicit_declared_output_producers",
                            "end_node_names": archived_end_nodes,
                        },
                    }
                },
            }
        }
        (case_dir / "split_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return run_dir


def _patch_source_hashes(
    probe,
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = (
        run_dir
        / "models/yolo26s/benchmark_set/legacy_suite/_rejected_cases"
    )
    hashes: dict[str, str] = {}
    for case_id in CASE_IDS:
        case_dir = archive_root / case_id
        model_paths = list(case_dir.glob("*.onnx"))
        if not model_paths:
            continue
        model_path = model_paths[0]
        hashes[case_id] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    monkeypatch.setattr(probe, "EXPECTED_SOURCE_SHA256", hashes)


def _patch_managed_venv(
    probe,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_onnxsim: bool = True,
) -> Path | None:
    managed = tmp_path / "managed"
    python = managed / "bin" / "python"
    python.parent.mkdir(parents=True)
    if os.name == "posix":
        python.symlink_to(Path(sys.executable).resolve())
        assert python.is_symlink()
        assert python.resolve().parent != python.parent
    else:
        python.write_text("", encoding="utf-8")
    onnxsim: Path | None = None
    if with_onnxsim:
        onnxsim = python.parent / "onnxsim"
        onnxsim.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        onnxsim.chmod(0o755)
    monkeypatch.setattr(
        probe,
        "_resolve_managed_venv_python",
        lambda **_kwargs: ("hailo10-test", python, managed / "bin/activate"),
    )
    return onnxsim


def _result(*, ok: bool, error: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        ok=ok,
        elapsed_s=0.01,
        backend="venv",
        error=error,
        har_path="parsed.har" if ok else None,
        fixed_onnx_path="fixed.onnx",
    )


def test_direct_probe_uses_automatic_path_first_for_exact_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe()
    run_dir = _archived_run(tmp_path)
    _patch_source_hashes(probe, run_dir, monkeypatch)
    output_dir = tmp_path / "probe-output"
    onnxsim = _patch_managed_venv(probe, tmp_path, monkeypatch)
    calls: list[tuple[str, list[str] | None]] = []

    def _parse(model_path, **kwargs):
        calls.append((Path(model_path).parent.name, kwargs.get("end_node_names")))
        return _result(ok=True)

    monkeypatch.setattr(probe, "hailo_parse_check_auto", _parse)
    rc = probe.main(
        ["--run-dir", str(run_dir), "--output-dir", str(output_dir)]
    )

    assert rc == 0
    assert calls == [(case_id, None) for case_id in CASE_IDS]
    verdict = json.loads((output_dir / "parse_verdict.json").read_text())
    assert verdict["status"] == "PASS"
    assert verdict["passed_cases"] == 4
    assert verdict["parser_attempt_count"] == 4
    assert verdict["retry_case_count"] == 0
    assert verdict["capability_status"] == "ALL_PARSE_SUPPORTED"
    assert verdict["source_mutation_detected"] is False
    assert onnxsim is not None
    assert verdict["onnxsim_path"] == str(onnxsim.resolve())
    for row in verdict["cases"]:
        boundary = int(row["case_id"][1:])
        assert len(row["archived_end_node_names"]) == OUTPUT_COUNT[boundary]
        assert len(row["end_node_names"]) == OUTPUT_COUNT[boundary] - 1
        assert not any(
            name.startswith("splitpoint_hailo_part1_output_identity_")
            for name in row["end_node_names"]
        )
        assert sum(name.endswith("/Split") for name in row["end_node_names"]) == 1
        assert len(row["endpoint_projection"]) == OUTPUT_COUNT[boundary]
        assert [
            item["graph_output_slot"] for item in row["endpoint_projection"]
        ] == list(range(OUTPUT_COUNT[boundary]))


def test_base_conv_uses_one_attested_dfc_visible_retry_per_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe()
    run_dir = _archived_run(tmp_path)
    _patch_source_hashes(probe, run_dir, monkeypatch)
    output_dir = tmp_path / "probe-output"
    _patch_managed_venv(probe, tmp_path, monkeypatch)
    calls: list[tuple[str, list[str] | None]] = []

    def _parse(model_path, **kwargs):
        case_id = Path(model_path).parent.name
        end_nodes = kwargs.get("end_node_names")
        calls.append((case_id, end_nodes))
        if end_nodes is None:
            boundary = int(case_id[1:])
            return _result(
                ok=False,
                error=f"'base_conv{BASE_CONV[boundary]}' is not in list",
            )
        boundary = int(case_id[1:])
        assert len(end_nodes) == OUTPUT_COUNT[boundary] - 1
        assert sum(name.endswith("/Split") for name in end_nodes) == 1
        assert not any("output_identity" in name for name in end_nodes)
        return _result(ok=True)

    monkeypatch.setattr(probe, "hailo_parse_check_auto", _parse)
    rc = probe.main(
        ["--run-dir", str(run_dir), "--output-dir", str(output_dir)]
    )

    assert rc == 0
    assert len(calls) == 8
    assert [case_id for case_id, _nodes in calls[::2]] == list(CASE_IDS)
    assert all(nodes is None for _case_id, nodes in calls[::2])
    assert all(nodes is not None for _case_id, nodes in calls[1::2])
    verdict = json.loads((output_dir / "parse_verdict.json").read_text())
    assert verdict["status"] == "PASS"
    assert verdict["passed_cases"] == 4
    assert verdict["parser_attempt_count"] == 8
    assert verdict["retry_case_count"] == 4
    assert all(row["retry_attempted"] for row in verdict["cases"])


def test_missing_optional_onnxsim_can_still_close_exact_base_conv_terminals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe()
    run_dir = _archived_run(tmp_path)
    _patch_source_hashes(probe, run_dir, monkeypatch)
    output_dir = tmp_path / "probe-output"
    _patch_managed_venv(
        probe, tmp_path, monkeypatch, with_onnxsim=False
    )
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))
    calls: list[str] = []

    def _base_conv_failure(model_path, **_kwargs):
        case_id = Path(model_path).parent.name
        calls.append(case_id)
        boundary = int(case_id[1:])
        return _result(
            ok=False,
            error=f"'base_conv{BASE_CONV[boundary]}' is not in list",
        )

    monkeypatch.setattr(probe, "hailo_parse_check_auto", _base_conv_failure)
    rc = probe.main(
        ["--run-dir", str(run_dir), "--output-dir", str(output_dir)]
    )

    assert rc == 0
    assert calls == [case_id for case_id in CASE_IDS for _ in range(2)]
    verdict = json.loads((output_dir / "parse_verdict.json").read_text())
    assert verdict["status"] == "PASS"
    assert verdict["passed_cases"] == 0
    assert verdict["base_conv_terminal_cases"] == 4
    assert verdict["capability_status"] == (
        "ALL_BASE_CONV_UNSUPPORTED_IN_PROVISIONED_ENVIRONMENT"
    )
    assert verdict["onnxsim_origin"] == "missing"
    assert verdict["warnings"]


def test_unknown_retry_failure_remains_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe()
    run_dir = _archived_run(tmp_path)
    _patch_source_hashes(probe, run_dir, monkeypatch)
    output_dir = tmp_path / "probe-output"
    _patch_managed_venv(probe, tmp_path, monkeypatch)

    def _parse(model_path, **kwargs):
        if kwargs.get("end_node_names") is None:
            boundary = int(Path(model_path).parent.name[1:])
            return _result(
                ok=False,
                error=f"'base_conv{BASE_CONV[boundary]}' is not in list",
            )
        return _result(ok=False, error="unexpected parser endpoint failure")

    monkeypatch.setattr(probe, "hailo_parse_check_auto", _parse)
    rc = probe.main(
        ["--run-dir", str(run_dir), "--output-dir", str(output_dir)]
    )

    assert rc == 1
    verdict = json.loads((output_dir / "parse_verdict.json").read_text())
    assert verdict["status"] == "FAIL"
    assert verdict["capability_status"] == "UNEXPECTED_PARSE_FAILURE"
    assert verdict["evidence_complete"] is False
    assert verdict["parser_attempt_count"] == 8
    assert verdict["retry_case_count"] == 4


@pytest.mark.parametrize(
    "error_template",
    [
        "'base_conv99' is not in list",
        "'base_conv{expected}' is not in list\nunexpected extra detail",
    ],
)
def test_nonexact_base_conv_failure_never_triggers_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_template: str,
) -> None:
    probe = _load_probe()
    run_dir = _archived_run(tmp_path)
    _patch_source_hashes(probe, run_dir, monkeypatch)
    output_dir = tmp_path / "probe-output"
    _patch_managed_venv(probe, tmp_path, monkeypatch)
    calls: list[str] = []

    def _parse(model_path, **kwargs):
        assert kwargs.get("end_node_names") is None
        case_id = Path(model_path).parent.name
        calls.append(case_id)
        boundary = int(case_id[1:])
        return _result(
            ok=False,
            error=error_template.format(expected=BASE_CONV[boundary]),
        )

    monkeypatch.setattr(probe, "hailo_parse_check_auto", _parse)
    rc = probe.main(
        ["--run-dir", str(run_dir), "--output-dir", str(output_dir)]
    )

    assert rc == 1
    assert calls == list(CASE_IDS)
    verdict = json.loads((output_dir / "parse_verdict.json").read_text())
    assert verdict["status"] == "FAIL"
    assert verdict["capability_status"] == "UNEXPECTED_PARSE_FAILURE"
    assert verdict["parser_attempt_count"] == 4
    assert verdict["retry_case_count"] == 0
    assert all(not row["retry_attempted"] for row in verdict["cases"])
    assert all(
        row["failure_kind"] == "unexpected_base_conv_failure"
        for row in verdict["cases"]
    )


def test_identity_alias_drift_stops_before_any_parser_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe()
    run_dir = _archived_run(tmp_path)
    _patch_source_hashes(probe, run_dir, monkeypatch)
    case_manifest = (
        run_dir
        / "models/yolo26s/benchmark_set/legacy_suite/_rejected_cases"
        / "b104/split_manifest.json"
    )
    payload = json.loads(case_manifest.read_text())
    identity_fix = payload["hailo"]["part1_feature_splitter_identity_fix"]
    aliases = identity_fix["aliases"]
    first_key = next(iter(aliases))
    aliases[first_key] = "tampered_tensor"
    payload["hailo"]["hefs"]["hailo10"][
        "part1_feature_splitter_identity_fix"
    ] = identity_fix
    case_manifest.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "probe-output"
    _patch_managed_venv(probe, tmp_path, monkeypatch)
    calls: list[object] = []
    monkeypatch.setattr(
        probe,
        "hailo_parse_check_auto",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    rc = probe.main(
        ["--run-dir", str(run_dir), "--output-dir", str(output_dir)]
    )

    assert rc == 2
    assert calls == []
    verdict = json.loads((output_dir / "parse_verdict.json").read_text())
    assert verdict["status"] == "PREFLIGHT_FAIL"
    assert verdict["parser_started"] is False


def test_source_mutation_is_never_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe()
    run_dir = _archived_run(tmp_path)
    _patch_source_hashes(probe, run_dir, monkeypatch)
    output_dir = tmp_path / "probe-output"
    _patch_managed_venv(probe, tmp_path, monkeypatch)
    mutated = False

    def _parse(model_path, **_kwargs):
        nonlocal mutated
        if not mutated:
            with Path(model_path).open("ab") as handle:
                handle.write(b"mutation")
            mutated = True
        return _result(ok=True)

    monkeypatch.setattr(probe, "hailo_parse_check_auto", _parse)
    rc = probe.main(
        ["--run-dir", str(run_dir), "--output-dir", str(output_dir)]
    )

    assert rc == 2
    verdict = json.loads((output_dir / "parse_verdict.json").read_text())
    assert verdict["status"] == "FAIL"
    assert verdict["source_mutation_detected"] is True
    assert verdict["evidence_complete"] is False


def test_source_mutation_after_automatic_failure_blocks_all_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe()
    run_dir = _archived_run(tmp_path)
    _patch_source_hashes(probe, run_dir, monkeypatch)
    output_dir = tmp_path / "probe-output"
    _patch_managed_venv(probe, tmp_path, monkeypatch)
    calls: list[str] = []

    def _parse(model_path, **kwargs):
        assert kwargs.get("end_node_names") is None
        case_id = Path(model_path).parent.name
        calls.append(case_id)
        with Path(model_path).open("ab") as handle:
            handle.write(b"mutation")
        boundary = int(case_id[1:])
        return _result(
            ok=False,
            error=f"'base_conv{BASE_CONV[boundary]}' is not in list",
        )

    monkeypatch.setattr(probe, "hailo_parse_check_auto", _parse)
    rc = probe.main(
        ["--run-dir", str(run_dir), "--output-dir", str(output_dir)]
    )

    assert rc == 2
    assert calls == list(CASE_IDS)
    verdict = json.loads((output_dir / "parse_verdict.json").read_text())
    assert verdict["status"] == "FAIL"
    assert verdict["source_mutation_detected"] is True
    assert verdict["parser_attempt_count"] == 4
    assert verdict["retry_case_count"] == 0
    assert all(not row["retry_attempted"] for row in verdict["cases"])


def test_direct_probe_rejects_an_unbound_source_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe()
    run_dir = _archived_run(tmp_path)
    output_dir = tmp_path / "probe-output"
    _patch_managed_venv(probe, tmp_path, monkeypatch)
    calls: list[object] = []
    monkeypatch.setattr(
        probe,
        "hailo_parse_check_auto",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    rc = probe.main(
        ["--run-dir", str(run_dir), "--output-dir", str(output_dir)]
    )

    assert rc == 2
    assert calls == []
    verdict = json.loads((output_dir / "parse_verdict.json").read_text())
    assert verdict["status"] == "PREFLIGHT_FAIL"
    assert "SHA-256 mismatch" in verdict["preflight_errors"][0]


def test_direct_probe_preflights_all_cases_before_parser_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe()
    run_dir = _archived_run(tmp_path, omit_case="b104")
    _patch_source_hashes(probe, run_dir, monkeypatch)
    output_dir = tmp_path / "probe-output"
    _patch_managed_venv(probe, tmp_path, monkeypatch)
    calls: list[object] = []
    monkeypatch.setattr(
        probe,
        "hailo_parse_check_auto",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    rc = probe.main(
        ["--run-dir", str(run_dir), "--output-dir", str(output_dir)]
    )

    assert rc == 2
    assert calls == []
    verdict = json.loads((output_dir / "parse_verdict.json").read_text())
    assert verdict["status"] == "PREFLIGHT_FAIL"
    assert verdict["parser_started"] is False
