"""Fast hardware-independent smoke tests for v60k registry reuse and recovery."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock

from . import __version__
from . import dataset_provisioning as dp
from .v60j_smoke import run_smoke as run_v60j_smoke
from .workflow.runner import WORKFLOW_VERSION


def _result(name: str, ok: bool, detail: str = "", **extra: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail, **extra}


def _make_manifest_pair(root: Path, task: str) -> tuple[Path, Path]:
    if task == "classification":
        cal_root = root / "cls_cal" / "n00000001"
        val_root = root / "cls_val" / "n00000001"
        cal_root.mkdir(parents=True)
        val_root.mkdir(parents=True)
        (cal_root / "a.JPEG").write_bytes(b"cal")
        (val_root / "b.JPEG").write_bytes(b"val")
        calibration = dp.create_dataset_manifest(
            task="classification",
            role="calibration",
            dataset_id="smoke-cls-cal",
            split="train",
            root=cal_root.parent,
            output=root / "manifests" / "imagenet_train_calibration_manifest.json",
            max_items=0,
        )
        validation = dp.create_dataset_manifest(
            task="classification",
            role="validation",
            dataset_id="smoke-cls-val",
            split="val",
            root=val_root.parent,
            output=root / "manifests" / "imagenet_val_manifest.json",
            max_items=0,
        )
    else:
        raise ValueError(task)
    return calibration, validation


def _check_task_scoped_readiness() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v60k-readiness-") as temp:
        root = Path(temp)
        calibration, validation = _make_manifest_pair(root, "classification")
        registry = dp.load_registry(root / "dataset_registry.json")
        cp = json.loads(calibration.read_text(encoding="utf-8"))
        vp = json.loads(validation.read_text(encoding="utf-8"))
        dp._register_dataset(
            registry,
            "imagenet_calibration",
            root=Path(cp["root"]),
            source="smoke",
        )
        dp._register_dataset(
            registry,
            "imagenet_validation",
            root=Path(vp["root"]),
            source="smoke",
        )
        registry["manifests"] = {
            "classification_calibration": str(calibration),
            "classification_validation": str(validation),
        }
        registry_path = dp.save_registry(registry, root / "dataset_registry.json")
        status = dp.registry_status(registry_path, verify_manifests=True)
        ok = (
            status["ready_for_classification_profile"] is True
            and status["ready_for_detection_profile"] is False
            and status["ready_for_final_profile"] is False
        )
        return _result(
            "task_scoped_registry_readiness",
            ok,
            "classification ready independently of missing detection assets",
        )


def _check_coco_network_free_reuse() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v60k-coco-reuse-") as temp:
        root = Path(temp)
        fixtures = root / "fixtures"
        fixtures.mkdir(parents=True)
        val_zip = fixtures / "val2017.zip"
        train_zip = fixtures / "train2017.zip"
        ann_zip = fixtures / "annotations_trainval2017.zip"
        with zipfile.ZipFile(val_zip, "w") as zf:
            zf.writestr("val2017/val.jpg", b"val")
        with zipfile.ZipFile(train_zip, "w") as zf:
            zf.writestr("train2017/a.jpg", b"a")
            zf.writestr("train2017/b.jpg", b"b")
        val_ann = {
            "images": [{"id": 10, "file_name": "val.jpg", "width": 1, "height": 1}],
            "annotations": [],
            "categories": [{"id": 1, "name": "object"}],
        }
        train_ann = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 1, "height": 1},
                {"id": 2, "file_name": "b.jpg", "width": 1, "height": 1},
            ],
            "annotations": [],
            "categories": [{"id": 1, "name": "object"}],
        }
        with zipfile.ZipFile(ann_zip, "w") as zf:
            zf.writestr("annotations/instances_val2017.json", json.dumps(val_ann))
            zf.writestr("annotations/instances_train2017.json", json.dumps(train_ann))
        mapping = {
            "val2017.zip": val_zip,
            "train2017.zip": train_zip,
            "annotations_trainval2017.zip": ann_zip,
        }

        def copy_download(spec: Any, destination: Any, **_: Any) -> Path:
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(mapping[str(spec.get("filename"))], destination)
            return destination

        campaign = root / "campaign"
        with mock.patch.object(dp, "_download_verified_archive", side_effect=copy_download):
            dp.provision_coco2017(
                root=campaign,
                include_train=True,
                calibration_items=1,
                seed=19,
                retain_train_archive=False,
            )
        with mock.patch.object(
            dp,
            "_download_verified_archive",
            side_effect=AssertionError("network access attempted"),
        ) as download:
            reused = dp.provision_coco2017(
                root=campaign,
                include_train=True,
                calibration_items=1,
                seed=19,
                retain_train_archive=False,
            )
        ok = (
            download.call_count == 0
            and reused.get("train_archive_downloaded") is False
            and (reused.get("calibration_subset") or {}).get("status") == "reused"
            and not (campaign / "downloads" / "train2017.zip").exists()
        )
        return _result(
            "coco_network_free_subset_reuse",
            ok,
            "existing selected subset reused before any archive request",
        )


def _check_registry_repair() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v60k-repair-") as temp:
        root = Path(temp)
        calibration, validation = _make_manifest_pair(root, "classification")
        result = dp.repair_dataset_registry(
            root=root,
            registry_path=root / "dataset_registry.json",
            verify_manifests=True,
        )
        ok = (
            result.get("network_used") is False
            and result["readiness"]["ready_for_classification_profile"] is True
            and Path(result["registry"]).is_file()
        )
        return _result(
            "network_free_registry_repair",
            ok,
            f"recovered={len(result.get('recovered') or {})}",
        )


def run_smoke() -> dict[str, Any]:
    base = run_v60j_smoke()
    checks = list(base.get("checks") or [])
    for function in (_check_task_scoped_readiness, _check_coco_network_free_reuse, _check_registry_repair):
        try:
            checks.append(function())
        except Exception as exc:
            checks.append(
                _result(
                    function.__name__.lstrip("_"),
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
            )
    passed = sum(1 for check in checks if check.get("ok"))
    return {
        "schema": "onnx-splitpoint/v60k-smoke-report",
        "schema_version": 1,
        "tool_version": __version__,
        "workflow_version": WORKFLOW_VERSION,
        "status": "ok" if passed == len(checks) else "failed",
        "passed": passed,
        "failed": len(checks) - passed,
        "checks": checks,
        "hardware_coverage": False,
        "network_coverage": False,
        "note": (
            "The smoke verifies task-scoped readiness and network-free registry "
            "recovery; real dataset downloads and accelerator hardware remain external."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fast v60k registry/dataset smoke tests")
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
        f"v60k smoke: {report['status']} "
        f"({report['passed']} passed, {report['failed']} failed) "
        f"tool={report['tool_version']} workflow={report['workflow_version']}"
    )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

# v60m: development profiles keep screening validation unless final validation
# was explicitly selected; calibration manifests may still auto-bind.
from onnx_splitpoint_tool.v60m_policy import install_dataset_binding_guards as _v60m_install_dataset_binding_guards
_v60m_install_dataset_binding_guards(globals())
