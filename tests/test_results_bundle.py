from __future__ import annotations

import tarfile
from pathlib import Path

from onnx_splitpoint_tool.benchmark.results_bundle import (
    collect_results_from_suite,
    create_results_bundle,
)


def test_results_bundle_contains_postprocess_artifacts(tmp_path: Path) -> None:
    # Create a tiny fake suite with one case and a couple of expected artifacts.
    suite_dir = tmp_path / "suite"
    case_dir = suite_dir / "b001"
    out_dir = case_dir / "results_cpu"
    out_dir.mkdir(parents=True)

    # Minimal expected artifacts.
    (out_dir / "validation_report.json").write_text("{}", encoding="utf-8")
    (out_dir / "detections_full.json").write_text("{\"detections\": []}", encoding="utf-8")

    # Collect into a results dir and then bundle.
    results_dir = tmp_path / "results"
    collect_results_from_suite(suite_dir, results_dir)
    tar_path = tmp_path / "results_bundle.tar.gz"
    create_results_bundle(results_dir, tar_path)

    assert tar_path.is_file()

    with tarfile.open(tar_path, "r:gz") as tf:
        names = tf.getnames()

    # We used arcname='.' so paths will be relative to '.' in the archive.
    assert any(name.endswith("b001/results_cpu/validation_report.json") for name in names)
    assert any(name.endswith("b001/results_cpu/detections_full.json") for name in names)
