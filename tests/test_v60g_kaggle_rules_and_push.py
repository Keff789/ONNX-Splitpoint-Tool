from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from onnx_splitpoint_tool import dataset_provisioning as dp


class KaggleRulesAndPushV60gTests(unittest.TestCase):
    def test_push_error_in_stdout_fails_even_with_zero_return_code(self) -> None:
        proc = subprocess.CompletedProcess(
            ["kaggle", "kernels", "push"],
            0,
            stdout=(
                "Kernel push error: You must accept this competition's rules before "
                "you'll be able to add it as a datasource: "
                "imagenet-object-localization-challenge"
            ),
        )
        with self.assertRaises(PermissionError) as ctx:
            dp._assert_kaggle_kernel_push_succeeded(
                proc,
                competition=dp.IMAGENET_KAGGLE_COMPETITION,
                kernel_ref="owner/example",
            )
        message = str(ctx.exception)
        self.assertIn("Join Competition", message)
        self.assertIn(dp.IMAGENET_KAGGLE_RULES_URL, message)

    def test_private_kernel_does_not_poll_after_failed_push(self) -> None:
        failed_push = subprocess.CompletedProcess(
            ["kaggle", "kernels", "push"],
            0,
            stdout="Kernel push error: You must accept this competition's rules",
        )
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            dp, "_run_kaggle", return_value=failed_push
        ), mock.patch.object(dp, "_wait_for_kaggle_kernel") as wait:
            with self.assertRaises(PermissionError):
                dp._provision_imagenet_via_private_kernel(
                    command=["kaggle"],
                    competition=dp.IMAGENET_KAGGLE_COMPETITION,
                    username="unit-user",
                    kernel_slug="unit-kernel",
                    base=Path(temp) / "datasets",
                    download_dir=Path(temp) / "downloads",
                    calibration_items=10,
                    seed=1,
                    link_mode="copy",
                    registry_path=Path(temp) / "registry.json",
                    kernel_timeout_s=10,
                    kernel_poll_interval_s=0.01,
                    log=None,
                )
        wait.assert_not_called()

    def test_status_poll_stops_on_inaccessible_kernel(self) -> None:
        denied = subprocess.CompletedProcess(
            ["kaggle", "kernels", "status"],
            0,
            stdout=(
                "Cannot access kernel 'owner/job' "
                "(Permission 'kernels.get' was denied)."
            ),
        )
        with mock.patch.object(dp, "_run_kaggle", return_value=denied):
            with self.assertRaises(RuntimeError) as ctx:
                dp._wait_for_kaggle_kernel(
                    ["kaggle"],
                    "owner/job",
                    timeout_s=10,
                    poll_interval_s=0.01,
                )
        self.assertIn("Status polling has been stopped immediately", str(ctx.exception))

    def test_cli_status_distinguishes_listing_from_rules_acceptance(self) -> None:
        listing = subprocess.CompletedProcess(
            ["kaggle", "competitions", "files"],
            0,
            stdout=(
                "name,size,creationDate\n"
                "ILSVRC/Annotations/CLS-LOC/train/n01440764/example.xml,483,2022-01-01\n"
            ),
        )
        with mock.patch.object(
            dp, "resolve_kaggle_cli", return_value=(["kaggle"], "Kaggle CLI 2.2.3")
        ), mock.patch.object(dp, "_run_kaggle", return_value=listing), mock.patch.object(
            dp,
            "_probe_kaggle_competition_download_access",
            return_value={
                "checked": True,
                "download_access": False,
                "rules_accepted": False,
                "probe_file": "example.xml",
                "output": "You must accept this competition's rules",
            },
        ):
            status = dp.kaggle_cli_status(check_access=True)
        self.assertTrue(status["competition_listing_access"])
        self.assertTrue(status["competition_access"])
        self.assertFalse(status["competition_download_access"])
        self.assertFalse(status["competition_rules_accepted"])
        self.assertIn("action_required", status)

    def test_rules_probe_success(self) -> None:
        successful = subprocess.CompletedProcess(
            ["kaggle", "competitions", "download"],
            0,
            stdout="Downloaded example.xml",
        )

        def fake_run(command, args, **kwargs):
            destination = Path(args[args.index("-p") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "example.xml").write_text("ok", encoding="utf-8")
            return successful

        with mock.patch.object(dp, "_run_kaggle", side_effect=fake_run):
            result = dp._probe_kaggle_competition_download_access(
                ["kaggle"],
                dp.IMAGENET_KAGGLE_COMPETITION,
                "ILSVRC/Annotations/example.xml",
            )
        self.assertTrue(result["download_access"])
        self.assertTrue(result["rules_accepted"])


if __name__ == "__main__":
    unittest.main()
