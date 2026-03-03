from __future__ import annotations

from pathlib import Path


def test_write_runner_skeleton_drops_test_images_into_resources(tmp_path: Path) -> None:
    """Runner exports should include both bundled test images under resources/test_images."""

    from onnx_splitpoint_tool import split_export_runners as ser

    out_dir = tmp_path / "export"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal args: we don't care about the runner script itself here.
    ser.write_runner_skeleton_onnxruntime(out_dir=str(out_dir))

    img_dir = out_dir / "resources" / "test_images"
    assert (img_dir / "test_image_coco.png").is_file()
    assert (img_dir / "test_image_imagenet.png").is_file()


def test_benchmark_suite_deduplicates_test_images_to_suite_root(tmp_path: Path) -> None:
    """Benchmark suites should place the test images once at the suite root."""

    from onnx_splitpoint_tool import split_export_runners as ser

    suite_root = tmp_path / "suite"
    suite_root.mkdir(parents=True, exist_ok=True)
    (suite_root / "benchmark_set.json").write_text("{}", encoding="utf-8")

    case_dir = suite_root / "b000"
    case_dir.mkdir(parents=True, exist_ok=True)

    ser.write_runner_skeleton_onnxruntime(out_dir=str(case_dir))

    root_img_dir = suite_root / "resources" / "test_images"
    assert (root_img_dir / "test_image_coco.png").is_file()
    assert (root_img_dir / "test_image_imagenet.png").is_file()

    # Case dir should not need its own copy (allowing de-duplication).
    assert not (case_dir / "resources" / "test_images").exists()
