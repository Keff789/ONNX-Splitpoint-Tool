from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_v2775_yolo26_archived_part1.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("v2775_compat_direct_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    for case_id in ("b066", "b088", "b104", "b199"):
        if case_id == omit_case:
            continue
        case_dir = root / case_id
        case_dir.mkdir(parents=True)
        model_name = f"yolo26s_part1_hailo_identity_b{int(case_id[1:])}.onnx"
        (case_dir / model_name).write_bytes(case_id.encode("ascii"))
        manifest = {
            "hailo": {
                "part1_accel_model": model_name,
                "hefs": {
                    "hailo10": {
                        "part1_error": "'base_conv14' is not in list",
                        "part1_base_conv_resolution": {
                            "attempted": True,
                            "end_node_names": [f"explicit_{case_id}"],
                        },
                    }
                },
            }
        }
        (case_dir / "split_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
    return run_dir


def _patch_managed_venv(
    probe,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    managed = tmp_path / "managed"
    python = managed / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    onnxsim = python.parent / "onnxsim"
    onnxsim.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    onnxsim.chmod(0o755)
    monkeypatch.setattr(
        probe,
        "_resolve_managed_venv_python",
        lambda **_kwargs: ("hailo10-test", python, managed / "bin/activate"),
    )
    return onnxsim


def test_direct_probe_parses_exact_four_and_preserves_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe()
    run_dir = _archived_run(tmp_path)
    output_dir = tmp_path / "probe-output"
    onnxsim = _patch_managed_venv(probe, tmp_path, monkeypatch)
    calls: list[tuple[str, list[str]]] = []

    def _fake_parse(model_path, **kwargs):
        calls.append((Path(model_path).parent.name, list(kwargs["end_node_names"])))
        case_out = Path(kwargs["outdir"])
        case_out.mkdir(parents=True)
        har = case_out / "parsed.har"
        har.write_bytes(b"parsed")
        return SimpleNamespace(
            ok=True,
            elapsed_s=0.01,
            backend="venv",
            error=None,
            har_path=str(har),
            fixed_onnx_path=str(case_out / "fixed.onnx"),
        )

    monkeypatch.setattr(probe, "hailo_parse_check_auto", _fake_parse)

    rc = probe.main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    assert [case_id for case_id, _nodes in calls] == [
        "b066",
        "b088",
        "b104",
        "b199",
    ]
    verdict = json.loads((output_dir / "parse_verdict.json").read_text())
    assert verdict["status"] == "PASS"
    assert verdict["passed_cases"] == 4
    assert verdict["source_mutation_detected"] is False
    assert verdict["onnxsim_path"] == str(onnxsim.resolve())


def test_direct_probe_preflights_all_cases_before_parser_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe()
    run_dir = _archived_run(tmp_path, omit_case="b104")
    output_dir = tmp_path / "probe-output"
    _patch_managed_venv(probe, tmp_path, monkeypatch)
    calls: list[object] = []
    monkeypatch.setattr(
        probe,
        "hailo_parse_check_auto",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    rc = probe.main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 2
    assert calls == []
    verdict = json.loads((output_dir / "parse_verdict.json").read_text())
    assert verdict["status"] == "PREFLIGHT_FAIL"
    assert verdict["parser_started"] is False
