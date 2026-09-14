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


class KaggleMountAndDiagnosticsV60hTests(unittest.TestCase):
    def _write_sample_tree(self, source_root: Path) -> None:
        val = source_root / "ILSVRC" / "Data" / "CLS-LOC" / "val"
        ann = source_root / "ILSVRC" / "Annotations" / "CLS-LOC" / "val"
        val.mkdir(parents=True)
        ann.mkdir(parents=True)
        stem = "ILSVRC2012_val_00000001"
        (val / f"{stem}.JPEG").write_bytes(b"image")
        (ann / f"{stem}.xml").write_text(
            "<annotation><object><name>n00000001</name>"
            "<bndbox><xmin>1</xmin><ymin>2</ymin>"
            "<xmax>3</xmax><ymax>4</ymax></bndbox>"
            "</object></annotation>",
            encoding="utf-8",
        )
        (source_root / "LOC_synset_mapping.txt").write_text(
            "n00000001 sample\n", encoding="utf-8"
        )

    def test_enum_qualified_kernel_status_is_normalised(self) -> None:
        self.assertEqual(
            dp._parse_kaggle_kernel_status(
                'owner/job has status "KernelWorkerStatus.ERROR"'
            )[0],
            "error",
        )
        self.assertEqual(
            dp._parse_kaggle_kernel_status(
                'owner/job has status "KernelWorkerStatus.RUNNING"'
            )[0],
            "running",
        )
        self.assertEqual(
            dp._parse_kaggle_kernel_status(
                'owner/job has status "KernelWorkerStatus.COMPLETE"'
            )[0],
            "complete",
        )

    def test_generated_kernel_discovers_unexpected_mount_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_parent = root / "kaggle-input"
            source_root = input_parent / "unexpected-source-alias" / "nested-copy"
            self._write_sample_tree(source_root)
            output_dir = root / "output"
            kernel_dir = root / "kernel"
            job = dp._write_imagenet_export_kernel(
                kernel_dir,
                username="unit-test-user",
                competition=dp.IMAGENET_KAGGLE_COMPETITION,
                expected_image_count=1,
                chunk_mib=1,
            )
            env = dict(os.environ)
            # Intentionally point at a parent that does not itself contain
            # ILSVRC. The generated exporter must discover the source alias.
            env["ONNX_SPLITPOINT_KAGGLE_INPUT_ROOT"] = str(input_parent)
            env["ONNX_SPLITPOINT_KAGGLE_WORKING_DIR"] = str(output_dir)
            proc = subprocess.run(
                [sys.executable, str(job["code_file"])],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout)
            manifest = json.loads(
                (output_dir / "imagenet_val_export_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["image_count"], 1)
            self.assertIn("unexpected-source-alias", manifest["validation_image_source"])
            self.assertTrue((output_dir / "imagenet_input_discovery.json").is_file())

    def test_generated_kernel_writes_error_and_mount_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            empty_input = root / "empty-input"
            empty_input.mkdir()
            output_dir = root / "output"
            kernel_dir = root / "kernel"
            job = dp._write_imagenet_export_kernel(
                kernel_dir,
                username="unit-test-user",
                competition=dp.IMAGENET_KAGGLE_COMPETITION,
                expected_image_count=1,
                chunk_mib=1,
            )
            env = dict(os.environ)
            env["ONNX_SPLITPOINT_KAGGLE_INPUT_ROOT"] = str(empty_input)
            env["ONNX_SPLITPOINT_KAGGLE_WORKING_DIR"] = str(output_dir)
            proc = subprocess.run(
                [sys.executable, str(job["code_file"])],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            error_json = output_dir / "imagenet_export_error.json"
            error_txt = output_dir / "imagenet_export_error.txt"
            self.assertTrue(error_json.is_file(), proc.stdout)
            self.assertTrue(error_txt.is_file(), proc.stdout)
            payload = json.loads(error_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["exception_type"], "FileNotFoundError")
            self.assertIn("input_inventory", payload)
            self.assertIn("Could not discover", payload["message"])

    def test_failed_kernel_stops_immediately_and_downloads_log(self) -> None:
        status_proc = subprocess.CompletedProcess(
            ["kaggle", "kernels", "status"],
            0,
            stdout='owner/job has status "KernelWorkerStatus.ERROR"',
        )
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "diagnostics"

            def fake_download(command, kernel_ref, output_dir, *, log=None):
                output_dir.mkdir(parents=True, exist_ok=True)
                log_path = output_dir / "job.log"
                log_path.write_text(
                    json.dumps(
                        [
                            {
                                "stream_name": "stderr",
                                "time": 1.0,
                                "data": "Traceback (most recent call last):\\n",
                            },
                            {
                                "stream_name": "stderr",
                                "time": 1.1,
                                "data": "FileNotFoundError: mount missing\\n",
                            },
                        ]
                    ),
                    encoding="utf-8",
                )
                return [log_path]

            with mock.patch.object(
                dp, "_run_kaggle", return_value=status_proc
            ) as run, mock.patch.object(
                dp, "_download_kaggle_kernel_output", side_effect=fake_download
            ) as download:
                with self.assertRaises(RuntimeError) as ctx:
                    dp._wait_for_kaggle_kernel(
                        ["kaggle"],
                        "owner/job",
                        timeout_s=30,
                        poll_interval_s=0.01,
                        failure_output_dir=destination,
                    )
            self.assertEqual(run.call_count, 1)
            self.assertEqual(download.call_count, 1)
            self.assertIn("FileNotFoundError: mount missing", str(ctx.exception))
            self.assertTrue(
                (destination / "kaggle_imagenet_export_failure_summary.txt").is_file()
            )
            self.assertTrue(
                (destination / "kaggle_imagenet_export_failure.json").is_file()
            )

    def test_cli_exposes_diagnostic_collection(self) -> None:
        ns = dp.build_parser().parse_args(
            [
                "collect-imagenet-kernel-diagnostics",
                "--kaggle-username",
                "example-user",
                "--kernel-slug",
                "example-kernel",
                "--output-dir",
                "/tmp/example-output",
            ]
        )
        self.assertEqual(ns.command, "collect-imagenet-kernel-diagnostics")
        self.assertEqual(ns.kaggle_username, "example-user")
        self.assertEqual(ns.kernel_slug, "example-kernel")



if __name__ == "__main__":
    unittest.main()
