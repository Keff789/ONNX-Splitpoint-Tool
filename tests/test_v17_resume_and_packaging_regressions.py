from __future__ import annotations

import json
import tarfile
from pathlib import Path


def test_generation_state_roundtrip_and_find_latest_resumable_set(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.benchmark.generation_state import (
        find_latest_resumable_set,
        init_state,
        update_state,
    )

    root = tmp_path / "sets"
    root.mkdir()

    old_dir = root / "yolov7_benchmark_20260319_010101"
    new_dir = root / "yolov7_benchmark_20260319_020202"
    old_dir.mkdir()
    new_dir.mkdir()

    state_old = init_state(
        model_name="yolov7",
        model_source="/models/yolov7.onnx",
        requested_cases=10,
        ranked_candidates=[80, 116],
        candidate_search_pool=[80, 116, 135],
        hef_full_policy="end",
        run_mode="new",
    )
    state_new = init_state(
        model_name="yolov7",
        model_source="/models/yolov7.onnx",
        requested_cases=10,
        ranked_candidates=[80, 116],
        candidate_search_pool=[80, 116, 135],
        hef_full_policy="end",
        run_mode="new",
    )

    update_state(old_dir / "generation_state.json", state_old, status="partial", generated_cases=3)
    update_state(new_dir / "generation_state.json", state_new, status="running", generated_cases=5)

    latest = find_latest_resumable_set(root, "yolov7")
    assert latest == new_dir

    payload = json.loads((new_dir / "generation_state.json").read_text(encoding="utf-8"))
    assert payload["status"] == "running"
    assert payload["requested_cases"] == 10
    assert payload["generated_cases"] == 5
    assert payload["ranked_candidates"] == [80, 116]
    assert payload["candidate_search_pool"] == [80, 116, 135]



def test_remote_minimal_bundle_patterns_keep_runtime_files_but_drop_heavy_exports(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.remote.bundle import build_suite_bundle, remote_minimal_bundle_patterns

    suite = tmp_path / "suite"
    (suite / "schemas").mkdir(parents=True)
    (suite / "splitpoint_runners" / "harness").mkdir(parents=True)
    (suite / "models").mkdir(parents=True)
    (suite / "analysis_plots").mkdir(parents=True)
    (suite / "b080" / "resources" / "test_images").mkdir(parents=True)
    (suite / "b080" / "hailo" / "hailo8" / "part1").mkdir(parents=True)

    (suite / "benchmark_plan.json").write_text("{}", encoding="utf-8")
    (suite / "benchmark_set.json").write_text("{}", encoding="utf-8")
    (suite / "benchmark_suite.py").write_text("print('ok')\n", encoding="utf-8")
    (suite / "schemas" / "benchmark_plan.schema.json").write_text("{}", encoding="utf-8")
    (suite / "splitpoint_runners" / "harness" / "yolo.py").write_text("# runner\n", encoding="utf-8")
    (suite / "models" / "yolov7.onnx").write_text("onnx", encoding="utf-8")
    (suite / "analysis_plots" / "analysis_overview.pdf").write_text("heavy", encoding="utf-8")
    (suite / "b080" / "split_manifest.json").write_text("{}", encoding="utf-8")
    (suite / "b080" / "run_split_onnxruntime.py").write_text("print('case')\n", encoding="utf-8")
    (suite / "b080" / "yolov7_part1_b80.onnx").write_text("onnx", encoding="utf-8")
    (suite / "b080" / "resources" / "test_images" / "test_image_coco.png").write_text("img", encoding="utf-8")
    (suite / "b080" / "split_context_b80.pdf").write_text("heavy", encoding="utf-8")
    (suite / "b080" / "hailo" / "hailo8" / "part1" / "compiled.hef").write_text("hef", encoding="utf-8")

    includes, excludes = remote_minimal_bundle_patterns()
    out = tmp_path / "bundle.tar.gz"
    build_suite_bundle(suite, out, includes=includes, excludes=excludes)

    with tarfile.open(out, "r:gz") as tf:
        names = set(tf.getnames())

    assert "benchmark_plan.json" in names
    assert "benchmark_set.json" in names
    assert "benchmark_suite.py" in names
    assert "schemas/benchmark_plan.schema.json" in names
    assert "splitpoint_runners/harness/yolo.py" in names
    assert "models/yolov7.onnx" in names
    assert "b080/split_manifest.json" in names
    assert "b080/run_split_onnxruntime.py" in names
    assert "b080/yolov7_part1_b80.onnx" in names
    assert "b080/resources/test_images/test_image_coco.png" in names
    assert "b080/hailo/hailo8/part1/compiled.hef" in names

    assert "analysis_plots/analysis_overview.pdf" not in names
    assert "b080/split_context_b80.pdf" not in names



def test_runner_templates_expose_quiet_progress_flags() -> None:
    runner = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    suite = Path("onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt").read_text(encoding="utf-8")

    assert "--verbose-runs" in runner
    assert "--progress-every" in runner
    assert "verbose_runs=args.verbose_runs" in runner
    assert "progress_every=args.progress_every" in runner

    assert "--verbose-runs" in suite
    assert "--progress-every" in suite
    assert "progress_every=args.progress_every" in suite



def test_resources_utils_can_read_bundled_templates_from_source_tree() -> None:
    from onnx_splitpoint_tool.resources_utils import persistent_resource_path, read_text

    text = read_text("resources", "templates", "run_split_onnxruntime.py.txt")
    schemas_root = persistent_resource_path("resources", "schemas")

    assert text.startswith("#!/usr/bin/env python3")
    assert schemas_root.is_dir()
    assert (schemas_root / "benchmark_plan.schema.json").exists()
