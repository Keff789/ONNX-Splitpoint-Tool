from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from onnx_splitpoint_tool import dataset_provisioning as dp
from onnx_splitpoint_tool.gui.dataset_dialogs import format_registry_status_line


class RegistryReuseRepairV60kTests(unittest.TestCase):
    def _write_zip(self, path: Path, members: dict[str, bytes]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
            for name, data in members.items():
                zf.writestr(name, data)
        return path

    def _coco_fixtures(self, root: Path) -> dict[str, Path]:
        fixtures = root / "fixtures"
        val_zip = self._write_zip(
            fixtures / "val2017.zip",
            {"val2017/val_a.jpg": b"validation-image"},
        )
        train_zip = self._write_zip(
            fixtures / "train2017.zip",
            {
                f"train2017/train_{index}.jpg": f"train-image-{index}".encode()
                for index in range(5)
            },
        )
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
                {
                    "id": index,
                    "image_id": index,
                    "category_id": 1,
                    "bbox": [0, 0, 1, 1],
                    "area": 1,
                    "iscrowd": 0,
                }
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
        return {
            "val2017.zip": val_zip,
            "train2017.zip": train_zip,
            "annotations_trainval2017.zip": ann_zip,
        }

    def _copy_download(self, fixtures: dict[str, Path]):
        def fake_download(spec, destination, **kwargs):
            source = fixtures[str(spec.get("filename"))]
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            return destination

        return fake_download

    def test_task_readiness_is_reported_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures = self._coco_fixtures(root)
            campaign = root / "campaign"
            with mock.patch.object(
                dp,
                "_download_verified_archive",
                side_effect=self._copy_download(fixtures),
            ):
                result = dp.provision_coco2017(
                    root=campaign,
                    include_train=True,
                    calibration_items=2,
                    seed=17,
                    retain_train_archive=False,
                )

            status = dp.registry_status(result["registry"], verify_manifests=True)
            self.assertTrue(status["ready_for_detection_profile"])
            self.assertFalse(status["ready_for_classification_profile"])
            self.assertFalse(status["ready_for_final_profile"])
            self.assertEqual(
                status["task_readiness"]["detection"]["calibration_validation_disjointness"]["status"],
                "pass",
            )
            line = format_registry_status_line(status)
            self.assertIn("COCO: ready", line)
            self.assertIn("ImageNet: incomplete", line)

    def test_second_coco_provision_reuses_subset_without_train_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures = self._coco_fixtures(root)
            campaign = root / "campaign"
            with mock.patch.object(
                dp,
                "_download_verified_archive",
                side_effect=self._copy_download(fixtures),
            ):
                first = dp.provision_coco2017(
                    root=campaign,
                    include_train=True,
                    calibration_items=2,
                    seed=17,
                    retain_train_archive=False,
                )
            self.assertTrue(first["train_archive_downloaded"])
            self.assertTrue(first["train_archive_removed"])
            self.assertFalse((campaign / "downloads" / "train2017.zip").exists())

            with mock.patch.object(
                dp,
                "_download_verified_archive",
                side_effect=AssertionError("no archive download expected"),
            ) as download:
                second = dp.provision_coco2017(
                    root=campaign,
                    include_train=True,
                    calibration_items=2,
                    seed=17,
                    retain_train_archive=False,
                )
            download.assert_not_called()
            self.assertFalse(second["train_archive_downloaded"])
            self.assertEqual(second["calibration_subset"]["status"], "reused")
            self.assertTrue(second["task_readiness"]["ready"])

    def test_coco_manifest_can_be_rebuilt_from_materialised_subset_without_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures = self._coco_fixtures(root)
            campaign = root / "campaign"
            with mock.patch.object(
                dp,
                "_download_verified_archive",
                side_effect=self._copy_download(fixtures),
            ):
                first = dp.provision_coco2017(
                    root=campaign,
                    include_train=True,
                    calibration_items=2,
                    seed=17,
                    retain_train_archive=False,
                )

            manifest = Path(first["manifests"]["detection_calibration"])
            self.assertTrue(manifest.is_file())
            manifest.unlink()
            registry_path = Path(first["registry"])
            registry = dp.load_registry(registry_path)
            registry.setdefault("manifests", {}).pop("detection_calibration", None)
            registry.setdefault("datasets", {}).pop("coco2017_calibration", None)
            dp.save_registry(registry, registry_path)

            with mock.patch.object(
                dp,
                "_download_verified_archive",
                side_effect=AssertionError("no archive download expected"),
            ) as download:
                second = dp.provision_coco2017(
                    root=campaign,
                    include_train=True,
                    calibration_items=2,
                    seed=17,
                    retain_train_archive=False,
                )
            download.assert_not_called()
            self.assertEqual(second["calibration_subset"]["status"], "recovered")
            self.assertTrue(Path(second["manifests"]["detection_calibration"]).is_file())
            self.assertTrue(second["task_readiness"]["ready"])
            self.assertFalse((campaign / "downloads" / "train2017.zip").exists())

    def test_registry_repair_uses_manifests_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures = self._coco_fixtures(root)
            campaign = root / "campaign"
            with mock.patch.object(
                dp,
                "_download_verified_archive",
                side_effect=self._copy_download(fixtures),
            ):
                result = dp.provision_coco2017(
                    root=campaign,
                    include_train=True,
                    calibration_items=2,
                    seed=17,
                    retain_train_archive=False,
                )

            registry_path = Path(result["registry"])
            registry_path.write_text(
                json.dumps(
                    {
                        "schema": dp.REGISTRY_SCHEMA,
                        "schema_version": dp.REGISTRY_VERSION,
                        "root": str(campaign),
                        "datasets": {},
                        "manifests": {},
                        "settings": {},
                    }
                ),
                encoding="utf-8",
            )
            repaired = dp.repair_dataset_registry(
                root=campaign,
                registry_path=registry_path,
                verify_manifests=True,
            )
            self.assertFalse(repaired["network_used"])
            self.assertEqual(
                repaired["recovered"]["detection_calibration"]["status"],
                "recovered",
            )
            self.assertTrue(
                repaired["readiness"]["ready_for_detection_profile"]
            )
            self.assertFalse((campaign / "downloads" / "train2017.zip").exists())


if __name__ == "__main__":
    unittest.main()
