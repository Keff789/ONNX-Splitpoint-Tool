from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from onnx_splitpoint_tool import dataset_provisioning as dp


class ImageNetValidationExportV60fTests(unittest.TestCase):
    def test_username_normalisation_and_explicit_resolution(self) -> None:
        self.assertEqual(dp._normalise_kaggle_username("@Example_User"), "Example_User")
        self.assertEqual(
            dp._normalise_kaggle_username("https://www.kaggle.com/example-user"),
            "example-user",
        )
        username, source = dp.resolve_kaggle_username("example-user")
        self.assertEqual(username, "example-user")
        self.assertEqual(source, "explicit")
        with self.assertRaises(ValueError):
            dp._normalise_kaggle_username("bad user")

    def test_kernel_status_parser(self) -> None:
        self.assertEqual(
            dp._parse_kaggle_kernel_status('owner/job has status "complete"')[0],
            "complete",
        )
        self.assertEqual(
            dp._parse_kaggle_kernel_status('owner/job has status "running"')[0],
            "running",
        )
        self.assertEqual(dp._parse_kaggle_kernel_status("FAILED")[0], "error")

    def test_generated_private_kernel_packages_only_validation_and_is_importable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_root = root / "input"
            val = input_root / "ILSVRC" / "Data" / "CLS-LOC" / "val"
            ann = input_root / "ILSVRC" / "Annotations" / "CLS-LOC" / "val"
            val.mkdir(parents=True)
            ann.mkdir(parents=True)
            mapping = input_root / "LOC_synset_mapping.txt"
            mapping.write_text("n00000001 first\nn00000002 second\n", encoding="utf-8")

            rows = [
                ("ILSVRC2012_val_00000001", "n00000001"),
                ("ILSVRC2012_val_00000002", "n00000002"),
            ]
            for stem, wnid in rows:
                (val / f"{stem}.JPEG").write_bytes((stem + "-image").encode())
                (ann / f"{stem}.xml").write_text(
                    "<annotation><object><name>"
                    + wnid
                    + "</name><bndbox><xmin>1</xmin><ymin>2</ymin>"
                    "<xmax>3</xmax><ymax>4</ymax></bndbox></object></annotation>",
                    encoding="utf-8",
                )

            kernel_dir = root / "kernel"
            job = dp._write_imagenet_export_kernel(
                kernel_dir,
                username="unit-test-user",
                competition="imagenet-object-localization-challenge",
                expected_image_count=2,
                chunk_mib=1,
            )
            self.assertEqual(
                job["kernel_ref"],
                f"unit-test-user/{dp.IMAGENET_EXPORT_KERNEL_SLUG}",
            )
            metadata = json.loads(Path(job["metadata"]).read_text(encoding="utf-8"))
            self.assertTrue(metadata["is_private"])
            self.assertFalse(metadata["enable_internet"])
            self.assertEqual(
                metadata["title"].replace(" ", "-"),
                dp.IMAGENET_EXPORT_KERNEL_SLUG,
            )
            self.assertEqual(
                metadata["competition_sources"],
                ["imagenet-object-localization-challenge"],
            )

            output_dir = root / "output"
            output_dir.mkdir()
            env = dict(os.environ)
            env["ONNX_SPLITPOINT_KAGGLE_INPUT_ROOT"] = str(input_root)
            env["ONNX_SPLITPOINT_KAGGLE_WORKING_DIR"] = str(output_dir)
            proc = subprocess.run(
                [sys.executable, job["code_file"]],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout)
            export = dp._verify_and_reassemble_imagenet_export(output_dir)
            self.assertEqual(export["image_count"], 2)
            self.assertTrue(Path(export["archive"]).is_file())
            self.assertTrue(Path(export["solution"]).is_file())

            registry = root / "campaign" / "dataset_registry.json"
            result = dp.import_imagenet(
                source=export["archive"],
                root=root / "campaign",
                validation_solution=export["solution"],
                labels=export["labels"],
                registry_path=registry,
                link_mode="copy",
            )
            self.assertEqual(result["status"], "ok")
            manifest = json.loads(
                Path(result["manifests"]["classification_validation"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["item_count"], 2)
            self.assertEqual(
                {item["class_name"] for item in manifest["items"]},
                {"n00000001", "n00000002"},
            )
            part_paths = [Path(path) for path in export["part_paths"]]
            self.assertTrue(all(path.is_file() for path in part_paths))
            self.assertEqual(dp._remove_imagenet_export_parts(export), len(part_paths))
            self.assertTrue(all(not path.exists() for path in part_paths))

    def test_smart_validation_mode_falls_back_to_private_kernel(self) -> None:
        expected = {
            "status": "ok",
            "registry": "/tmp/registry.json",
            "kernel_ref": "user/job",
        }
        with mock.patch.object(
            dp, "resolve_kaggle_cli", return_value=(["kaggle"], "Kaggle CLI 2.2.3")
        ), mock.patch.object(
            dp, "kaggle_cli_status", return_value={"authentication_configured_hint": True}
        ), mock.patch.object(
            dp,
            "_kaggle_competition_files",
            return_value=(
                ["ILSVRC/Annotations/CLS-LOC/train/n01440764/example.xml"],
                "Next Page Token = token",
            ),
        ), mock.patch.object(
            dp, "resolve_kaggle_username", return_value=("user", "explicit")
        ), mock.patch.object(
            dp,
            "_ensure_kaggle_competition_rules_accepted",
            return_value={"download_access": True, "rules_accepted": True},
        ), mock.patch.object(
            dp, "_provision_imagenet_via_private_kernel", return_value=expected
        ) as provision, mock.patch.object(
            dp, "_record_imagenet_provisioning_settings"
        ):
            result = dp.provision_imagenet_kaggle(
                root="/tmp/datasets",
                accept_terms=True,
                download_mode="validation_only",
                kaggle_username="user",
            )
        self.assertEqual(result, expected)
        provision.assert_called_once()
        self.assertEqual(provision.call_args.kwargs["username"], "user")

    def test_cli_exposes_private_kernel_mode_and_owner(self) -> None:
        parser = dp.build_parser()
        ns = parser.parse_args(
            [
                "provision-imagenet-kaggle",
                "--accept-terms",
                "--download-mode",
                "validation_via_private_kernel",
                "--kaggle-username",
                "example-user",
                "--kernel-slug",
                "custom-export",
            ]
        )
        self.assertEqual(ns.download_mode, "validation_via_private_kernel")
        self.assertEqual(ns.kaggle_username, "example-user")
        self.assertEqual(ns.kernel_slug, "custom-export")


if __name__ == "__main__":
    unittest.main()
