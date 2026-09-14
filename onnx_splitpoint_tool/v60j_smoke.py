"""Fast hardware-independent smoke tests for v60j complete dataset provisioning."""
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from . import __version__
from . import dataset_provisioning as dp
from .v60i_smoke import run_smoke as run_v60i_smoke
from .workflow.runner import WORKFLOW_VERSION


def _result(name: str, ok: bool, detail: str = "", **extra: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail, **extra}


def _check_imagenet_calibration_template() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v60j-imagenet-template-") as temp:
        job = dp._write_imagenet_calibration_export_kernel(
            Path(temp) / "kernel",
            username="smoke-user",
            competition=dp.IMAGENET_KAGGLE_COMPETITION,
            calibration_items=1000,
            seed=dp.DEFAULT_SEED,
            chunk_mib=1,
        )
        code = Path(job["code_file"]).read_text(encoding="utf-8")
        metadata = json.loads(Path(job["metadata"]).read_text(encoding="utf-8"))
        tokens = (
            "class_stratified_deterministic_hash",
            "ILSVRC2012_img_train_calibration.tar",
            "imagenet_train_calibration_export_manifest.json",
        )
        ok = all(token in code for token in tokens) and metadata.get("is_private") is True
        return _result(
            "imagenet_private_calibration_export",
            ok,
            f"kernel_ref={job['kernel_ref']} private={metadata.get('is_private')}",
        )


def _check_coco_subset_materialisation() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v60j-coco-subset-") as temp:
        root = Path(temp)
        archive = root / "train2017.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for index in range(3):
                zf.writestr(f"train2017/{index}.jpg", f"image-{index}".encode())
        annotations = root / "instances_train2017.json"
        annotations.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": index, "file_name": f"{index}.jpg", "width": 1, "height": 1}
                        for index in range(3)
                    ],
                    "annotations": [],
                    "categories": [],
                }
            ),
            encoding="utf-8",
        )
        result = dp._create_coco_calibration_subset_from_archive(
            archive=archive,
            annotations=annotations,
            coco_root=root / "coco2017",
            calibration_items=2,
            seed=11,
        )
        files = list(Path(result["root"]).glob("*.jpg"))
        ok = len(files) == 2 and Path(result["annotations"]).is_file() and Path(result["selection"]).is_file()
        return _result(
            "coco_materialised_calibration_subset",
            ok,
            f"selected={len(files)} population={result['population_count']}",
        )


def run_smoke() -> dict[str, Any]:
    base = run_v60i_smoke()
    checks = list(base.get("checks") or [])
    for function in (_check_imagenet_calibration_template, _check_coco_subset_materialisation):
        try:
            checks.append(function())
        except Exception as exc:
            checks.append(_result(function.__name__.lstrip("_"), False, f"{type(exc).__name__}: {exc}"))
    passed = sum(1 for check in checks if check.get("ok"))
    return {
        "schema": "onnx-splitpoint/v60j-smoke-report",
        "schema_version": 1,
        "tool_version": __version__,
        "workflow_version": WORKFLOW_VERSION,
        "status": "ok" if passed == len(checks) else "failed",
        "passed": passed,
        "failed": len(checks) - passed,
        "checks": checks,
        "hardware_coverage": False,
        "network_coverage": False,
        "note": "This smoke checks local dataset/evidence plumbing; Kaggle, COCO download and accelerator hardware still require targeted integration runs.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fast v60j dataset/evidence smoke tests")
    parser.add_argument("--json", dest="json_path", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    report = run_smoke()
    if args.json_path:
        output = Path(args.json_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not args.quiet:
        for check in report["checks"]:
            marker = "PASS" if check.get("ok") else "FAIL"
            print(f"[{marker}] {check.get('name')}: {check.get('detail')}")
    print(
        f"v60j smoke: {report['status']} "
        f"({report['passed']} passed, {report['failed']} failed) "
        f"tool={report['tool_version']} workflow={report['workflow_version']}"
    )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
