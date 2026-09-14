from __future__ import annotations
from pathlib import Path
import json
import tempfile

from onnx_splitpoint_tool.native_full_quality import (
    normalise_evaluation_profile, resolve_native_energy_enabled, resolve_native_full_plan,
    save_normalised_yaml, load_and_normalise_yaml,
)
from onnx_splitpoint_tool.reporting_quality_decomposition import build_quality_decomposition, write_quality_decomposition


def profile():
    return {
        "hardware_run_profiles": {
            "Hailo-8 Full": True,
            "Hailo-8 -> TensorRT": True,
            "Hailo-10H Full": True,
            "Hailo-10H -> TensorRT": True,
            "DEEPX Full": True,
            "DEEPX -> TensorRT": True,
            "TensorRT Full": True,
        },
        "native_producers": {"enabled": True, "energy": {"enabled": True}},
        "execution_preset": {"overrides": {"native_enabled": True, "energy_enabled": False}},
        "energy": {"enabled": False},
    }


def test_native_full_is_derived_per_physical_producer():
    p=profile(); plan=resolve_native_full_plan(p)
    assert plan.enabled
    assert plan.backends_by_producer == {
        "hailo8": ("hailo8", "tensorrt"),
        "hailo10h": ("hailo10h", "tensorrt"),
        "deepx": ("deepx", "tensorrt"),
    }


def test_native_energy_modern_key_wins_and_roundtrips(tmp_path: Path):
    p=profile(); normalise_evaluation_profile(p)
    assert resolve_native_energy_enabled(p) is True
    assert p["energy"]["enabled"] is False
    assert p["energy"]["requested_native_energy"] is True
    assert p["native_producers"]["energy"]["enabled"] is True
    assert p["native_producers"]["energy"]["mode"] == "measure"
    assert p["native_producers"]["energy"].get("include_split_rows", True) is True
    assert p["native_producers"]["energy"].get("include_full_baselines", True) is True
    assert p["execution_preset"]["overrides"]["energy_enabled"] is True
    path=tmp_path/"p.yaml"; save_normalised_yaml(path,p); q=load_and_normalise_yaml(path)
    assert resolve_native_energy_enabled(q) is True


def _row(model, variant, backend, top1, top5, **extra):
    base={"model":model,"task":"classification","variant":variant,"backend":backend,
          "top1":top1,"top5":top5,"dataset_id":"d","preprocessing_hash":"p",
          "decoder_hash":"d0","contract_comparable":True}
    base.update(extra); return base


def test_quality_decomposition_separates_vendor_and_split_loss(tmp_path: Path):
    rows=[
      _row("m","canonical_full_onnx","ort_cpu",.80,.95),
      _row("m","full","hailo8",.78,.94),
      _row("m","split","hailo8",.77,.93,case_id="b1"),
    ]
    refs,decomp=build_quality_decomposition(rows)
    assert len(refs)==2 and len(decomp)==1
    assert abs(decomp[0]["vendor_loss_Top1"] + .02) < 1e-9
    assert abs(decomp[0]["split_extra_loss_Top1"] + .01) < 1e-9
    result=write_quality_decomposition(tmp_path,rows)
    assert result["loss_decomposition_count"]==1
    assert (tmp_path/"task_quality_loss_decomposition.csv").exists()
    assert (tmp_path/"thesis_tables/task_quality_loss_decomposition.tex").exists()


def test_contract_mismatch_blocks_delta():
    rows=[
      _row("m","canonical_full_onnx","ort_cpu",.8,.9),
      _row("m","full","hailo8",.7,.8,preprocessing_hash="other"),
      _row("m","split","hailo8",.7,.8,case_id="b1"),
    ]
    _,d=build_quality_decomposition(rows)
    assert d[0]["status"]=="contract_incomparable"
    assert d[0]["vendor_loss_Top1"] is None
