from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.benchmark.model_preparation import (
    PreparationRuntimeOptions,
    _probe_full_hailo,
)
from onnx_splitpoint_tool.hailo_build_context import make_build_evidence_context


ROOT = Path(__file__).resolve().parents[1]


def test_preparation_binds_actual_full_variant_and_preserves_endpoint_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onnx_splitpoint_tool import hailo_backend

    captured = []

    def builder(source, **kwargs):
        captured.append((Path(source), kwargs))
        return SimpleNamespace(ok=True, error=None, hef_path=None)

    monkeypatch.setattr(hailo_backend, "hailo_build_hef_auto", builder)
    for variant, payload in (("current", b"original model"), ("exported", b"different export")):
        source = tmp_path / f"{variant}.onnx"
        source.write_bytes(payload)
        for end_nodes in (None, ["raw_head_0", "raw_head_1"]):
            result = _probe_full_hailo(
                source, variant, runtime=PreparationRuntimeOptions(),
                screening_dir=tmp_path / "screening", task="detection",
                end_node_names=end_nodes,
            )
            assert result[0] is True
            supplied_source, kwargs = captured[-1]
            context = kwargs["build_evidence_context"]
            assert supplied_source == source
            assert context["full_source_onnx_path"] == str(source)
            assert context["full_source_onnx_sha256"] == hashlib.sha256(payload).hexdigest()
            assert context["model_id"] == variant
            assert context["stage"] == "full"
            assert context["split_manifest"] == {}
            assert "boundary" not in context
            assert "identity_error" not in context
            assert kwargs["end_node_names"] == end_nodes
            assert kwargs["task"] == "detection"
    assert captured[0][1]["build_evidence_context"] != captured[2][1]["build_evidence_context"]


def _gui_build_call(source_name: str) -> ast.Call:
    """Run the real export call without constructing Tk or its worker thread."""
    tree = ast.parse((ROOT / "onnx_splitpoint_tool/gui_app.py").read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hailo_build_hef_auto"
        and node.args and isinstance(node.args[0], ast.Name)
        and node.args[0].id == source_name
    ]
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize("stage,source_name", [
    ("full", "full_model_src"), ("part1", "p1_path"), ("part2", "p2_path"),
])
def test_gui_export_supplies_full_model_and_real_split_contract(
    tmp_path: Path, stage: str, source_name: str,
) -> None:
    full = tmp_path / "yolo26s.onnx"
    part1 = tmp_path / "part1.onnx"
    part2 = tmp_path / "part2.onnx"
    full.write_bytes(b"full model identity")
    part1.write_bytes(b"part1 compiler input")
    part2.write_bytes(b"part2 compiler input")
    manifest = {
        "boundary": 364,
        "cut_tensors": ["concat22_output"],
        "part1_cut_names": ["out0"],
        "part2_cut_names": ["in0"],
        "cut": {"full_names": ["concat22_output"], "part1_names": ["out0"], "part2_names": ["in0"]},
        "io": {"orig_inputs": ["images"], "part2_external_inputs": []},
    }
    captured = {}

    def builder(source, **kwargs):
        captured.update(source=source, **kwargs)
        return SimpleNamespace(ok=False, error="injected known negative")

    environment = dict(
        hailo_build_hef_auto=builder,
        make_build_evidence_context=make_build_evidence_context,
        full_model_src=str(full), p1_path=str(part1), p2_path=str(part2),
        base="yolo26s", b=364, manifest_out=manifest,
        hef_backend="venv", hw_arch="hailo8", hef_fixup=True,
        hef_opt_level=1, hef_calib_dir=str(tmp_path / "calib"),
        hef_calib_count=64, hef_calib_bs=8, hef_force=False, hef_keep=True,
        hef_wsl_distro=None, hef_wsl_venv="auto", hef_timeout_s=3600,
        _hef_on_log=lambda *_args: None, hailo_image_task="detection",
        out_full=str(tmp_path / "full"), out_p1=str(tmp_path / "p1"),
        out_p2=str(tmp_path / "p2"),
    )
    expression = ast.Expression(body=_gui_build_call(source_name))
    result = eval(compile(ast.fix_missing_locations(expression), "<gui-export-build>", "eval"), environment)
    assert result.error == "injected known negative"
    assert captured["source"] == environment[source_name]
    context = captured["build_evidence_context"]
    assert context["full_source_onnx_path"] == str(full)
    assert context["full_source_onnx_sha256"] == hashlib.sha256(full.read_bytes()).hexdigest()
    assert context["model_id"] == "yolo26s"
    assert context["stage"] == stage
    assert "identity_error" not in context
    if stage == "full":
        # The legacy full net_name contains b364, but its build has no cut.
        assert "b364" in captured["net_name"]
        assert "boundary" not in context
        assert context["split_manifest"] == {}
    else:
        assert context["boundary"] == 364
        assert context["split_manifest"] == manifest
        manifest["cut_tensors"].append("later_target_metadata")
        assert context["split_manifest"]["cut_tensors"] == ["concat22_output"]
    if stage == "part2":
        assert captured["activation_part1_onnx"] == str(part1)


def test_preparation_does_not_substitute_missing_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onnx_splitpoint_tool import hailo_backend

    captured = {}

    def builder(_source, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ok=False, error="missing full source")

    monkeypatch.setattr(hailo_backend, "hailo_build_hef_auto", builder)
    result = _probe_full_hailo(
        tmp_path / "missing.onnx", "current", runtime=PreparationRuntimeOptions(),
        screening_dir=tmp_path / "screening", task="classification",
    )
    assert result[0] is False
    context = captured["build_evidence_context"]
    assert context["identity_error"]
    assert "full_source_onnx_sha256" not in context
