from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from onnx_splitpoint_tool import dataset_provisioning as dp


class CompleteDatasetProvisioningV60jTests(unittest.TestCase):
    def _write_zip(self, path: Path, members: dict[str, bytes]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
            for name, data in members.items():
                zf.writestr(name, data)
        return path

    def test_coco_one_click_materialises_only_selected_train_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures = root / "fixtures"
            val_zip = self._write_zip(
                fixtures / "val2017.zip",
                {"val2017/val_a.jpg": b"validation-image"},
            )
            train_members = {
                f"train2017/train_{index}.jpg": f"train-image-{index}".encode()
                for index in range(5)
            }
            train_zip = self._write_zip(fixtures / "train2017.zip", train_members)
            val_annotations = {
                "images": [{"id": 101, "file_name": "val_a.jpg", "width": 4, "height": 4}],
                "annotations": [],
                "categories": [{"id": 1, "name": "object"}],
            }
            train_annotations = {
                "images": [
                    {"id": index, "file_name": f"train_{index}.jpg", "width": 4, "height": 4}
                    for index in range(5)
                ],
                "annotations": [
                    {"id": index, "image_id": index, "category_id": 1, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}
                    for index in range(5)
                ],
                "categories": [{"id": 1, "name": "object"}],
            }
            ann_zip = self._write_zip(
                fixtures / "annotations_trainval2017.zip",
                {
                    "annotations/instances_val2017.json": json.dumps(val_annotations).encode(),
                    "annotations/instances_train2017.json": json.dumps(train_annotations).encode(),
                },
            )

            def fake_download(spec, destination, **kwargs):
                name = str(spec.get("filename"))
                return {
                    "val2017.zip": val_zip,
                    "train2017.zip": train_zip,
                    "annotations_trainval2017.zip": ann_zip,
                }[name]

            with mock.patch.object(dp, "_download_verified_archive", side_effect=fake_download):
                result = dp.provision_coco2017(
                    root=root / "campaign",
                    include_train=True,
                    calibration_items=2,
                    seed=17,
                    retain_train_archive=True,
                )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["provisioning_scope"], "validation_and_calibration")
            registry = dp.load_registry(result["registry"])
            calibration = Path(registry["datasets"]["coco2017_calibration"]["root"])
            self.assertTrue(calibration.is_dir())
            self.assertEqual(len(list(calibration.glob("*.jpg"))), 2)
            self.assertFalse((root / "campaign" / "coco2017" / "train2017").exists())
            calibration_manifest = json.loads(
                Path(result["manifests"]["detection_calibration"]).read_text(encoding="utf-8")
            )
            self.assertEqual(calibration_manifest["item_count"], 2)
            self.assertEqual(
                calibration_manifest["provisioning_selection"]["strategy"],
                "deterministic_hash",
            )
            self.assertTrue(result["calibration_validation_disjointness"]["requested_task_ok"])

    def test_generated_imagenet_calibration_kernel_is_deterministic_and_compact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input" / "unexpected-alias" / "ILSVRC" / "Data" / "CLS-LOC" / "train"
            for class_index in range(3):
                class_dir = source / f"n{class_index:08d}"
                class_dir.mkdir(parents=True)
                for image_index in range(3):
                    (class_dir / f"image_{image_index}.JPEG").write_bytes(
                        f"{class_index}-{image_index}".encode()
                    )
            kernel = root / "kernel"
            job = dp._write_imagenet_calibration_export_kernel(
                kernel,
                username="unit-test-user",
                competition=dp.IMAGENET_KAGGLE_COMPETITION,
                calibration_items=4,
                seed=123,
                chunk_mib=1,
            )
            output = root / "output"
            env = dict(os.environ)
            env["ONNX_SPLITPOINT_KAGGLE_INPUT_ROOT"] = str(root / "input")
            env["ONNX_SPLITPOINT_KAGGLE_WORKING_DIR"] = str(output)
            proc = subprocess.run(
                [sys.executable, job["code_file"]],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout)
            verified = dp._verify_and_reassemble_imagenet_calibration_export(output)
            self.assertEqual(verified["image_count"], 4)
            self.assertEqual(verified["class_count"], 3)
            selection = json.loads(Path(verified["selection"]).read_text(encoding="utf-8"))
            self.assertEqual(selection["selected_count"], 4)
            self.assertEqual(selection["seed"], 123)
            self.assertEqual(len(selection["items"]), 4)
            self.assertTrue(Path(verified["archive"]).is_file())


    def test_host_installs_calibration_export_and_updates_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input" / "ILSVRC" / "Data" / "CLS-LOC" / "train"
            for class_index in range(2):
                class_dir = source / f"n{class_index:08d}"
                class_dir.mkdir(parents=True)
                for image_index in range(2):
                    (class_dir / f"image_{image_index}.JPEG").write_bytes(
                        f"{class_index}-{image_index}".encode()
                    )
            generated = root / "generated-output"
            kernel_seed = root / "seed-kernel"
            seed_job = dp._write_imagenet_calibration_export_kernel(
                kernel_seed,
                username="user",
                competition=dp.IMAGENET_KAGGLE_COMPETITION,
                calibration_items=2,
                seed=5,
                chunk_mib=1,
            )
            env = dict(os.environ)
            env["ONNX_SPLITPOINT_KAGGLE_INPUT_ROOT"] = str(root / "input")
            env["ONNX_SPLITPOINT_KAGGLE_WORKING_DIR"] = str(generated)
            proc = subprocess.run(
                [sys.executable, seed_job["code_file"]],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout)

            val_root = root / "campaign" / "imagenet2012" / "prepared" / "val_by_wnid" / "n99999999"
            val_root.mkdir(parents=True)
            (val_root / "val.JPEG").write_bytes(b"different-validation-image")
            registry = root / "campaign" / "dataset_registry.json"
            dp.register_imagenet(
                train_root=None,
                val_root=val_root.parent,
                calibration_items=2,
                seed=5,
                registry_path=registry,
            )

            push_proc = subprocess.CompletedProcess(
                ["kaggle", "kernels", "push"],
                0,
                stdout="Kernel version 1 successfully pushed.",
            )

            def fake_download(command, kernel_ref, destination, *, log=None):
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(generated, destination)
                return sorted(path for path in destination.rglob("*") if path.is_file())

            with mock.patch.object(dp, "_run_kaggle", return_value=push_proc), mock.patch.object(
                dp, "_wait_for_kaggle_kernel", return_value={"status": "complete"}
            ), mock.patch.object(
                dp, "_download_kaggle_kernel_output", side_effect=fake_download
            ):
                result = dp._provision_imagenet_calibration_via_private_kernel(
                    command=["kaggle"],
                    competition=dp.IMAGENET_KAGGLE_COMPETITION,
                    username="user",
                    kernel_slug="unit-calibration",
                    base=root / "campaign",
                    download_dir=root / "campaign" / "downloads" / "imagenet_kaggle",
                    calibration_items=2,
                    seed=5,
                    registry_path=registry,
                    kernel_timeout_s=60,
                    kernel_poll_interval_s=0.01,
                    log=None,
                )
            self.assertEqual(result["status"], "ok")
            reg = dp.load_registry(registry)
            self.assertTrue(Path(reg["datasets"]["imagenet_calibration"]["root"]).is_dir())
            manifest = Path(reg["manifests"]["classification_calibration"])
            self.assertTrue(manifest.is_file())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["item_count"], 2)
            self.assertTrue(result["calibration_validation_disjointness"]["requested_task_ok"])

    def test_complete_imagenet_reuses_validation_then_requests_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            val = root / "imagenet2012" / "prepared" / "val_by_wnid" / "n00000001"
            val.mkdir(parents=True)
            (val / "val.JPEG").write_bytes(b"val")
            registry = root / "dataset_registry.json"
            dp.register_imagenet(
                train_root=None,
                val_root=val.parent,
                calibration_items=2,
                seed=9,
                registry_path=registry,
            )

            def fake_calibration(**kwargs):
                reg = dp.load_registry(registry)
                calib = root / "imagenet2012" / "prepared" / "train_calibration_n2_s9" / "n00000001"
                calib.mkdir(parents=True)
                (calib / "train.JPEG").write_bytes(b"train")
                (calib.parent / "n00000002").mkdir()
                (calib.parent / "n00000002" / "train2.JPEG").write_bytes(b"train2")
                dp._register_dataset(reg, "imagenet_calibration", root=calib.parent, source="test")
                manifest = dp.create_dataset_manifest(
                    task="classification",
                    role="calibration",
                    dataset_id="test",
                    split="train",
                    root=calib.parent,
                    output=root / "manifests" / "imagenet_train_calibration_manifest.json",
                    max_items=0,
                    selection_seed=9,
                )
                dp._annotate_dataset_manifest(
                    manifest,
                    provisioning_selection={"seed": 9, "selected_items": 2},
                )
                reg.setdefault("manifests", {})["classification_calibration"] = str(manifest)
                dp.save_registry(reg, registry)
                return {"status": "ok", "kernel_ref": "user/calibration", "registry": str(registry)}

            with mock.patch.object(dp, "resolve_kaggle_cli", return_value=(["kaggle"], "2.2")), mock.patch.object(
                dp, "_kaggle_competition_files", return_value=(["probe"], "")
            ), mock.patch.object(
                dp, "_ensure_kaggle_competition_rules_accepted", return_value={"rules_accepted": True}
            ), mock.patch.object(
                dp, "resolve_kaggle_username", return_value=("user", "explicit")
            ), mock.patch.object(
                dp, "_provision_imagenet_calibration_via_private_kernel", side_effect=fake_calibration
            ) as calibration_call:
                result = dp.provision_imagenet_complete(
                    root=root,
                    accept_terms=True,
                    include_calibration=True,
                    calibration_items=2,
                    seed=9,
                    registry_path=registry,
                    kaggle_username="user",
                )
            self.assertEqual(result["validation"]["status"], "reused")
            self.assertEqual(result["calibration"]["status"], "ok")
            calibration_call.assert_called_once()
            self.assertIn("classification_calibration", result["manifests"])


    def test_registry_status_is_ready_only_with_four_verified_disjoint_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests_dir = root / "manifests"
            registry = dp.load_registry(root / "dataset_registry.json")
            roles = {
                "classification_calibration": ("classification", "calibration", "cls-cal", b"a"),
                "classification_validation": ("classification", "validation", "cls-val", b"b"),
                "detection_calibration": ("detection", "calibration", "det-cal", b"c"),
                "detection_validation": ("detection", "validation", "det-val", b"d"),
            }
            for key, (task, role, dirname, content) in roles.items():
                data_root = root / dirname
                if task == "classification":
                    data_root = data_root / f"class-{role}"
                data_root.mkdir(parents=True)
                image = data_root / f"{role}.jpg"
                image.write_bytes(content)
                manifest = dp.create_dataset_manifest(
                    task=task,
                    role=role,
                    dataset_id=key,
                    split=role,
                    root=(data_root.parent if task == "classification" else data_root),
                    output=manifests_dir / f"{key}.json",
                    max_items=0,
                    selection_seed=1,
                )
                registry.setdefault("manifests", {})[key] = str(manifest)
            dp.save_registry(registry, root / "dataset_registry.json")
            status = dp.registry_status(root / "dataset_registry.json", verify_manifests=True)
            self.assertTrue(status["ready_for_final_profile"])
            self.assertTrue(status["calibration_validation_disjointness"]["ok"])

    def test_cli_defaults_to_complete_imagenet_unless_skipped(self) -> None:
        parser = dp.build_parser()
        ns = parser.parse_args(
            [
                "provision-imagenet-kaggle",
                "--accept-terms",
                "--kaggle-username",
                "example-user",
            ]
        )
        self.assertFalse(ns.skip_calibration)
        self.assertEqual(
            ns.calibration_kernel_slug,
            dp.IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG,
        )


if __name__ == "__main__":
    unittest.main()
