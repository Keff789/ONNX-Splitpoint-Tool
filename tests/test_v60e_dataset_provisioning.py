from __future__ import annotations

import hashlib
import io
import ssl
import tempfile
import unittest
import urllib.error
import tarfile
import zipfile
from pathlib import Path
from unittest import mock

from onnx_splitpoint_tool import dataset_provisioning as dp


class _FakeResponse:
    def __init__(self, payload: bytes, *, status: int = 200) -> None:
        self._stream = io.BytesIO(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def _tiny_zip() -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("payload.txt", "verified")
    return out.getvalue()


class DatasetProvisioningV60eTests(unittest.TestCase):
    def test_coco_auto_falls_back_without_disabling_tls_and_verifies_digest(self) -> None:
        payload = _tiny_zip()
        digest = hashlib.md5(payload).hexdigest()
        spec = {
            "urls": [
                "https://images.cocodataset.org/test.zip",
                "http://images.cocodataset.org/test.zip",
            ],
            "md5": digest,
        }
        calls: list[str] = []

        def fake_urlopen(req, timeout=0):
            url = req.full_url
            calls.append(url)
            if url.startswith("https://"):
                cert_error = ssl.SSLCertVerificationError(
                    1, "certificate verify failed: Hostname mismatch"
                )
                raise urllib.error.URLError(cert_error)
            return _FakeResponse(payload)

        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            dp.urllib.request, "urlopen", side_effect=fake_urlopen
        ):
            destination = Path(temp) / "test.zip"
            logs: list[str] = []
            result = dp._download_verified_archive(
                spec, destination, policy="auto", log=logs.append
            )
            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(calls[0].split(":", 1)[0], "https")
            self.assertEqual(calls[1].split(":", 1)[0], "http")
            self.assertTrue(any("certificate is not bypassed" in line for line in logs))
            self.assertTrue(any("verified test.zip" in line for line in logs))

    def test_plain_http_source_is_rejected_without_pinned_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(dp.DatasetDownloadError) as ctx:
                dp._download_verified_archive(
                    {"urls": ["http://example.invalid/archive.zip"], "md5": ""},
                    Path(temp) / "archive.zip",
                    policy="auto",
                )
            self.assertIn("no pinned digest", str(ctx.exception))

    def test_https_only_policy_never_selects_plain_http(self) -> None:
        urls = dp._source_urls(
            {
                "urls": [
                    "https://example.invalid/a.zip",
                    "http://example.invalid/a.zip",
                ]
            },
            "https_only",
        )
        self.assertEqual(urls, ["https://example.invalid/a.zip"])

    def test_imagenet_missing_cli_error_names_active_interpreter(self) -> None:
        with mock.patch.object(dp, "resolve_kaggle_cli", return_value=(None, "")):
            with self.assertRaises(RuntimeError) as ctx:
                dp.provision_imagenet_kaggle(accept_terms=True)
        text = str(ctx.exception)
        self.assertIn(dp.sys.executable, text)
        self.assertIn("Install/repair dataset support", text)

    def test_full_imagenet_download_requires_explicit_large_download_consent(self) -> None:
        with mock.patch.object(
            dp, "resolve_kaggle_cli", return_value=(["kaggle"], "Kaggle API 1.7")
        ):
            with self.assertRaises(PermissionError):
                dp.provision_imagenet_kaggle(
                    accept_terms=True,
                    download_mode="full_competition",
                    allow_large_download=False,
                )

    def test_validation_only_requires_username_for_private_export_when_no_direct_tar(self) -> None:
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
            dp, "resolve_kaggle_username", return_value=("", "unresolved")
        ):
            with self.assertRaises(RuntimeError) as ctx:
                dp.provision_imagenet_kaggle(
                    accept_terms=True,
                    download_mode="validation_only",
                )
        text = str(ctx.exception)
        self.assertIn("competition access are valid", text)
        self.assertIn("private Kaggle CPU kernel", text)
        self.assertNotIn("Authentication required", text)

    def test_gui_contains_one_click_dependency_install_and_safe_modes(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "onnx_splitpoint_tool"
            / "gui"
            / "dataset_dialogs.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Install/repair dataset support", source)
        self.assertIn("validation_only", source)
        self.assertIn("validation_via_private_kernel", source)
        self.assertIn("Public Kaggle username", source)
        self.assertIn("COCO_DOWNLOAD_POLICIES", source)
        self.assertIn("Validation solution CSV", source)
        self.assertIn("Check Kaggle auth/rules/access", source)

    def test_kaggle_module_cli_fallback_is_considered(self) -> None:
        commands = dp._candidate_kaggle_commands()
        flattened = [" ".join(command) for command in commands]
        # This assertion is conditional because the test environment does not
        # necessarily have the optional package installed.
        with mock.patch.object(dp.importlib.util, "find_spec", return_value=object()):
            commands = dp._candidate_kaggle_commands()
        flattened = [" ".join(command) for command in commands]
        self.assertTrue(any("-m kaggle.cli" in command for command in flattened))


    def test_coco_default_source_order_avoids_broken_custom_host_first(self) -> None:
        urls = dp._source_urls(dp.COCO_ARCHIVES["val2017"], "auto")
        self.assertGreaterEqual(len(urls), 3)
        self.assertEqual(
            urls[0],
            "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip",
        )
        self.assertIn("https://images.cocodataset.org/zips/val2017.zip", urls)
        self.assertTrue(urls[-1].startswith("http://"))

    def test_custom_coco_mirror_is_first_and_keeps_relative_path(self) -> None:
        urls = dp._source_urls(
            dp.COCO_ARCHIVES["annotations"],
            "auto",
            mirror_base="https://mirror.example.invalid/coco",
        )
        self.assertEqual(
            urls[0],
            "https://mirror.example.invalid/coco/annotations/annotations_trainval2017.zip",
        )

    def test_dependency_installer_targets_active_gui_interpreter(self) -> None:
        commands: list[list[str]] = []

        class FakeProcess:
            def __init__(self, cmd, **kwargs) -> None:
                commands.append(list(cmd))
                self.stdout = io.StringIO("installed\n")

            def wait(self) -> int:
                return 0

        probe = dp.subprocess.CompletedProcess(
            args=[dp.sys.executable, "-m", "pip", "--version"],
            returncode=0,
            stdout="pip 25",
        )
        with mock.patch.object(dp.subprocess, "run", return_value=probe), mock.patch.object(
            dp.subprocess, "Popen", side_effect=FakeProcess
        ), mock.patch.object(
            dp, "kaggle_cli_status", return_value={"available": True}
        ):
            result = dp.install_optional_dataset_dependencies()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(commands[0][0], dp.sys.executable)
        self.assertEqual(commands[0][1:4], ["-m", "pip", "install"])
        self.assertIn("pycocotools>=2.0.7", commands[0])
        self.assertIn("kaggle>=1.6", commands[0])

    def test_dataset_cli_exposes_safe_download_and_install_options(self) -> None:
        parser = dp.build_parser()
        ns = parser.parse_args(
            [
                "provision-coco",
                "--download-policy",
                "https_only",
                "--mirror-base",
                "https://mirror.example.invalid/coco",
            ]
        )
        self.assertEqual(ns.download_policy, "https_only")
        self.assertEqual(ns.mirror_base, "https://mirror.example.invalid/coco")
        deps = parser.parse_args(["install-deps", "--no-coco"])
        self.assertTrue(deps.no_coco)

    def test_authorised_validation_tar_plus_solution_is_importable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir = root / "raw"
            source_dir.mkdir()
            images = {
                "ILSVRC2012_val_00000001.JPEG": b"image-one",
                "ILSVRC2012_val_00000002.JPEG": b"image-two",
            }
            archive = root / "ILSVRC2012_img_val.tar"
            with tarfile.open(archive, "w") as tf:
                for name, payload in images.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    tf.addfile(info, io.BytesIO(payload))
            solution = root / "LOC_val_solution.csv"
            solution.write_text(
                "ImageId,PredictionString\n"
                "ILSVRC2012_val_00000001,n00000001 1 1 2 2\n"
                "ILSVRC2012_val_00000002,n00000002 1 1 2 2\n",
                encoding="utf-8",
            )
            registry = root / "campaign" / "dataset_registry.json"
            result = dp.import_imagenet(
                source=archive,
                root=root / "campaign",
                validation_solution=solution,
                registry_path=registry,
                link_mode="copy",
            )
            self.assertEqual(result["status"], "ok")
            self.assertTrue(registry.is_file())
            manifest = Path(result["manifests"]["classification_validation"])
            self.assertEqual(manifest.parent, registry.parent / "manifests")
            payload = dp.json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["item_count"], 2)
            self.assertEqual(
                {row["class_name"] for row in payload["items"]},
                {"n00000001", "n00000002"},
            )


if __name__ == "__main__":
    unittest.main()
