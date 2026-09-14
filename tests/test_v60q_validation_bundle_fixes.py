from __future__ import annotations

import json
import os
import subprocess
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from onnx_splitpoint_tool.benchmark.classification_validation_presets import (
    provision_classification_validation_source_to_suite,
)
from onnx_splitpoint_tool.benchmark.validation_assets import (
    provision_detection_validation_source_to_suite,
    resolve_detection_validation_source,
)
from onnx_splitpoint_tool.campaign import create_dataset_manifest
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.workflow.deepx_build_binding import _infer_deepx_calibration_dir
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.benchmark.suite_refresh import _normalize_suite_validation_payloads
from onnx_splitpoint_tool.benchmark.remote_run import _exclusive_suite_refresh_guard, _remote_cache_population_guard
from onnx_splitpoint_tool.remote.bundle import (
    build_suite_bundle,
    remote_minimal_bundle_patterns,
)


class V60QValidationBundleFixTests(unittest.TestCase):
    def _classification_source(self, root: Path, count: int = 10) -> tuple[Path, Path]:
        images = root / "imagenet_val"
        for idx in range(count):
            cls = images / f"n{idx % 4:08d}"
            cls.mkdir(parents=True, exist_ok=True)
            (cls / f"image_{idx:03d}.JPEG").write_bytes((f"classification-{idx}" * 17).encode())
        manifest = root / "imagenet_val_manifest.json"
        create_dataset_manifest(
            task="classification",
            role="validation",
            dataset_id="imagenet-test",
            split="val",
            root=images,
            output=manifest,
            max_items=0,
        )
        return images, manifest

    def _detection_source(self, root: Path, count: int = 6) -> tuple[Path, Path, Path]:
        images = root / "coco_val"
        images.mkdir(parents=True)
        image_rows = []
        annotations = []
        for idx in range(1, count + 1):
            name = f"{idx:012d}.jpg"
            (images / name).write_bytes((f"detection-{idx}" * 19).encode())
            image_rows.append({"id": idx, "file_name": name, "width": 64, "height": 64})
            annotations.append({"id": idx, "image_id": idx, "category_id": 1, "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0})
        ann = root / "instances_val.json"
        ann.write_text(json.dumps({"images": image_rows, "annotations": annotations, "categories": [{"id": 1, "name": "object"}]}), encoding="utf-8")
        manifest = root / "coco_val_manifest.json"
        create_dataset_manifest(
            task="detection",
            role="validation",
            dataset_id="coco-test",
            split="val",
            root=images,
            annotations=ann,
            output=manifest,
            max_items=0,
        )
        return images, ann, manifest

    def test_classification_manifest_materialises_only_requested_subset_and_reuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images, manifest = self._classification_source(root)
            suite = root / "suite"
            suite.mkdir()
            rel = provision_classification_validation_source_to_suite(
                suite,
                str(images),
                max_images=4,
                manifest_path=str(manifest),
                selection_seed=123,
            )
            self.assertIsNotNone(rel)
            out_root = suite / str(rel)
            self.assertTrue(out_root.is_dir())
            out_manifest = out_root / "manifest.json"
            payload = json.loads(out_manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["selection"]["requested_images"], 4)
            self.assertEqual(payload["selection"]["selected_images"], 4)
            self.assertEqual(len(payload["samples"]), 4)
            for sample in payload["samples"]:
                self.assertTrue((out_root / sample["image"]).is_file())
            before = out_manifest.stat().st_mtime_ns
            rel2 = provision_classification_validation_source_to_suite(
                suite,
                str(images),
                max_images=4,
                manifest_path=str(manifest),
                selection_seed=123,
            )
            self.assertEqual(rel2, rel)
            self.assertEqual(out_manifest.stat().st_mtime_ns, before)

    def test_detection_manifest_materialises_requested_subset_with_filtered_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images, _ann, manifest = self._detection_source(root)
            suite = root / "suite"
            suite.mkdir()
            rel = provision_detection_validation_source_to_suite(
                suite,
                str(images),
                max_images=3,
                manifest_path=str(manifest),
                selection_seed=456,
            )
            self.assertIsNotNone(rel)
            out = suite / str(rel)
            payload = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["selection"]["requested_images"], 3)
            self.assertEqual(payload["selection"]["selected_images"], 3)
            self.assertEqual(len(payload["samples"]), 3)
            self.assertEqual(len(list(out.glob("*.jpg"))), 3)
            # Three per-image sidecars plus the two suite-level JSON files.
            self.assertEqual(len([p for p in out.glob("*.json") if p.name not in {"manifest.json", "instances_subset.json"}]), 3)
            coco = json.loads((out / "instances_subset.json").read_text(encoding="utf-8"))
            self.assertEqual(len(coco["images"]), 3)
            self.assertEqual(len(coco["annotations"]), 3)

    def test_explicit_detection_path_is_not_canonicalised_to_coco50(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "custom_detection"
            source.mkdir()
            self.assertEqual(resolve_detection_validation_source(str(source)), source.resolve())

    def test_suite_refresh_rewrites_full_manifest_source_to_exact_smoke_subset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images, manifest = self._classification_source(root, count=12)
            suite = root / "suite"
            suite.mkdir()
            plan = {"runs": [{
                "id": "ort_cpu",
                "benchmark_task": "classification",
                "validation_images": str(images),
                "validation_manifest": str(manifest),
                "validation_max_images": 50000,
                "mini_classification_eval": True,
            }]}
            (suite / "benchmark_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (suite / "benchmark_set.json").write_text(json.dumps({"model_name": "resnet50", "plan": plan}), encoding="utf-8")
            result = _normalize_suite_validation_payloads(
                suite,
                benchmark_set_json=suite / "benchmark_set.json",
                validation_images=str(images),
                validation_max_images=4,
                validation_reference_mode="auto",
                mini_coco_ap50=False,
                benchmark_task="classification",
                mini_classification_eval=True,
                log=None,
            )
            self.assertEqual(result["validation_max_images"], 4)
            effective = str(result["validation_images"] or "")
            self.assertIn("_n4_", effective)
            updated = json.loads((suite / "benchmark_plan.json").read_text(encoding="utf-8"))["runs"][0]
            self.assertEqual(updated["validation_max_images"], 4)
            self.assertEqual(updated["validation_images"], effective)
            payload = json.loads((suite / effective / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["samples"]), 4)


    def test_model_resolution_prefers_explicit_canonical_source_over_old_run_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            canonical = root / "models" / "resnet50.onnx"
            canonical.parent.mkdir(parents=True)
            canonical.write_bytes(b"canonical-model")
            old_run_model = root / "EvaluationRuns" / "old-run" / "models" / "resnet50.onnx"
            old_run_model.parent.mkdir(parents=True)
            old_run_model.write_bytes(b"old-run-copy")

            runner = object.__new__(EvaluationWorkflowRunner)
            runner.options = type("Options", (), {
                "models_root": "",
                "include_reserve": False,
                "only_model": "",
            })()
            runner.profile_payload = {
                "models": [{
                    "id": "resnet50",
                    "onnx": str(canonical),
                    "resolved_path": str(old_run_model),
                }]
            }
            runner.warnings = []

            rows = runner._resolve_model_rows()
            self.assertEqual(len(rows), 1)
            self.assertEqual(Path(rows[0]["resolved_path"]), canonical.resolve())
            self.assertTrue(rows[0]["resolved"])

    def test_deepx_calibration_manifest_is_task_specific_even_with_generic_legacy_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cls = root / "imagenet_calib"
            det = root / "coco_calib"
            cls.mkdir(); det.mkdir()
            (cls / "n00000001").mkdir()
            (cls / "n00000001" / "a.jpg").write_bytes(b"cls")
            (det / "a.jpg").write_bytes(b"det")
            cls_manifest = root / "cls.json"
            det_manifest = root / "det.json"
            cls_manifest.write_text(json.dumps({"root": str(cls)}), encoding="utf-8")
            det_manifest.write_text(json.dumps({"root": str(det)}), encoding="utf-8")
            profile = {
                "campaign": {
                    "dataset_manifests": {
                        "classification": {"calibration": str(cls_manifest)},
                        "detection": {"calibration": str(det_manifest)},
                    }
                }
            }
            cfg = {"calib_dir": str(det)}
            cls_out = _infer_deepx_calibration_dir(
                row={"id": "resnet50", "task": "classification"},
                model_id="resnet50", cfg=cfg, profile_payload=profile,
            )
            det_out = _infer_deepx_calibration_dir(
                row={"id": "yolo26s", "task": "detection"},
                model_id="yolo26s", cfg=cfg, profile_payload=profile,
            )
            self.assertEqual(Path(cls_out), cls.resolve())
            self.assertEqual(Path(det_out), det.resolve())

    def _minimal_suite(self, root: Path) -> tuple[Path, Path, Path]:
        suite = root / "suite"
        suite.mkdir()
        (suite / "benchmark_plan.json").write_text('{"runs": []}', encoding="utf-8")
        (suite / "benchmark_set.json").write_text('{"cases": []}', encoding="utf-8")
        (suite / "benchmark_suite.py").write_text("print('ok')\n", encoding="utf-8")
        selected = suite / "resources" / "validation" / "classification" / "smoke_n4_s1"
        selected.mkdir(parents=True)
        for idx in range(4):
            (selected / f"sample_{idx}.JPEG").write_bytes(b"selected" + bytes([idx]))
        (selected / "manifest.json").write_text('{"samples": []}', encoding="utf-8")
        stale = suite / "resources" / "validation" / "classification" / "full_imagenet_50000"
        stale.mkdir(parents=True)
        for idx in range(120):
            (stale / f"stale_{idx:04d}.JPEG").write_bytes(b"stale")
        return suite, selected, stale

    def test_bundle_uses_exact_validation_root_and_excludes_stale_dataset_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            suite, selected, stale = self._minimal_suite(root)
            includes, excludes = remote_minimal_bundle_patterns()
            self.assertNotIn("resources/validation/**", includes)
            includes.append(f"{selected.relative_to(suite).as_posix()}/**")
            out = suite / "dist" / "suite_bundle.tar.gz"
            stats = build_suite_bundle(suite, out, includes=includes, excludes=excludes)
            self.assertFalse(stats.reused)
            manifest = json.loads(Path(stats.manifest_path).read_text(encoding="utf-8"))
            rels = {row["rel"] for row in manifest["files"]}
            self.assertTrue(any(rel.startswith(selected.relative_to(suite).as_posix() + "/") for rel in rels))
            self.assertFalse(any(rel.startswith(stale.relative_to(suite).as_posix() + "/") for rel in rels))
            with tarfile.open(out, "r:gz") as tf:
                names = set(tf.getnames())
            self.assertFalse(any(name.startswith(stale.relative_to(suite).as_posix() + "/") for name in names))

    def test_parallel_setup_workers_share_one_local_bundle_build(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            suite, selected, _stale = self._minimal_suite(root)
            includes, excludes = remote_minimal_bundle_patterns()
            includes.append(f"{selected.relative_to(suite).as_posix()}/**")
            out = suite / "dist" / "suite_bundle.tar.gz"
            results = []
            failures = []
            lock = threading.Lock()

            def worker() -> None:
                try:
                    result = build_suite_bundle(suite, out, includes=includes, excludes=excludes)
                    with lock:
                        results.append(result)
                except Exception as exc:  # pragma: no cover - surfaced below
                    with lock:
                        failures.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
            self.assertFalse(failures, failures)
            self.assertEqual(len(results), 3)
            self.assertEqual(sum(1 for row in results if not row.reused), 1)
            self.assertEqual(sum(1 for row in results if row.reused), 2)
            self.assertEqual(len({row.sha256 for row in results}), 1)


    def test_parallel_setup_workers_serialize_shared_suite_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            suite = Path(td) / "suite"
            suite.mkdir()
            active = 0
            max_active = 0
            guard = threading.Lock()
            barrier = threading.Barrier(3)

            def worker() -> None:
                nonlocal active, max_active
                barrier.wait()
                with _exclusive_suite_refresh_guard(suite):
                    with guard:
                        active += 1
                        max_active = max(max_active, active)
                    import time
                    time.sleep(0.03)
                    with guard:
                        active -= 1

            threads = [threading.Thread(target=worker) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(max_active, 1)

    def test_remote_suite_cache_population_is_atomic_across_workers(self) -> None:
        class LocalShellTransport:
            def run(self, command: str, timeout: int | None = None):
                completed = subprocess.run(
                    ["bash", "-lc", command],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
                return completed.returncode, completed.stdout

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_root = root / "remote_cache"
            cache_root.mkdir()
            lock_dir = cache_root / ".population.lock"
            cached_bundle = cache_root / "suite_bundle.tar.gz"
            bundle_ready = cache_root / "BUNDLE_READY"
            cached_suite = cache_root / "suite"
            suite_ready = cache_root / "SUITE_READY"
            ready_test = (
                f"test -s {cached_bundle} "
                f"-a -f {bundle_ready} "
                f"-a -d {cached_suite} "
                f"-a -f {suite_ready}"
            )
            transport = LocalShellTransport()
            ownership: list[bool] = []
            failures: list[BaseException] = []
            mutex = threading.Lock()
            start = threading.Barrier(3)

            def worker() -> None:
                try:
                    start.wait(timeout=5)
                    with _remote_cache_population_guard(
                        transport=transport,
                        lock_dir=str(lock_dir),
                        ready_test_cmd=ready_test,
                        log=lambda _message: None,
                        timeout_s=60,
                    ) as owner:
                        with mutex:
                            ownership.append(bool(owner))
                        if owner:
                            # Simulate one upload plus extraction transaction.
                            cached_bundle.write_bytes(b"bundle")
                            bundle_ready.touch()
                            cached_suite.mkdir(exist_ok=True)
                            (cached_suite / "benchmark_plan.json").write_text("{}", encoding="utf-8")
                            suite_ready.touch()
                except BaseException as exc:  # pragma: no cover - asserted below
                    with mutex:
                        failures.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(3)]
            real_sleep = __import__("time").sleep
            with mock.patch("onnx_splitpoint_tool.benchmark.remote_run.time.sleep", side_effect=lambda _seconds: real_sleep(0.01)):
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)

            self.assertFalse(any(thread.is_alive() for thread in threads), "cache population workers did not finish")
            self.assertFalse(failures, failures)
            self.assertEqual(len(ownership), 3)
            self.assertEqual(sum(ownership), 1)
            self.assertEqual(sum(not value for value in ownership), 2)
            self.assertTrue(cached_bundle.is_file())
            self.assertTrue(suite_ready.is_file())
            self.assertFalse(lock_dir.exists())

    def test_relaxed_bundle_skips_per_file_hashes_but_strict_bundle_records_them(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            suite, selected, _stale = self._minimal_suite(root)
            includes, excludes = remote_minimal_bundle_patterns()
            includes.append(f"{selected.relative_to(suite).as_posix()}/**")
            relaxed = suite / "dist" / "relaxed.tar.gz"
            with mock.patch.dict(os.environ, {"ONNX_SPLITPOINT_INTEGRITY_MODE": "fast"}, clear=False):
                r = build_suite_bundle(suite, relaxed, includes=includes, excludes=excludes)
            rm = json.loads(Path(r.manifest_path).read_text(encoding="utf-8"))
            self.assertFalse(rm["per_file_hashes"])
            self.assertTrue(all(not row.get("sha256") for row in rm["files"]))
            self.assertEqual(rm["gzip_level"], 1)

            strict = suite / "dist" / "strict.tar.gz"
            with mock.patch.dict(os.environ, {"ONNX_SPLITPOINT_INTEGRITY_MODE": "strict"}, clear=False):
                q = build_suite_bundle(suite, strict, includes=includes, excludes=excludes)
            qm = json.loads(Path(q.manifest_path).read_text(encoding="utf-8"))
            self.assertTrue(qm["per_file_hashes"])
            self.assertTrue(all(len(str(row.get("sha256") or "")) == 64 for row in qm["files"]))

    def test_deterministic_bundle_hash_survives_different_source_mtimes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hashes = []
            for index in range(2):
                suite = root / f"suite_{index}"
                suite.mkdir()
                model = suite / "model.onnx"
                model.write_bytes(b"same-model-content")
                os.utime(model, ns=(1_000_000_000 + index * 9_000_000_000, 1_000_000_000 + index * 9_000_000_000))
                includes, excludes = remote_minimal_bundle_patterns()
                previous = os.environ.get("ONNX_SPLITPOINT_INTEGRITY_MODE")
                os.environ["ONNX_SPLITPOINT_INTEGRITY_MODE"] = "fast"
                try:
                    stats = build_suite_bundle(suite, suite / "dist" / "suite_bundle.tar.gz", includes=includes, excludes=excludes)
                finally:
                    if previous is None:
                        os.environ.pop("ONNX_SPLITPOINT_INTEGRITY_MODE", None)
                    else:
                        os.environ["ONNX_SPLITPOINT_INTEGRITY_MODE"] = previous
                manifest = json.loads(Path(stats.manifest_path).read_text(encoding="utf-8"))
                self.assertTrue(manifest["deterministic_archive"])
                hashes.append(stats.sha256)
            self.assertEqual(hashes[0], hashes[1])

    def test_effective_plan_exposes_exact_subset_embedding_budget(self) -> None:
        profile = {
            "model_suite": {"primary": [
                {"id": "resnet50", "task": "classification", "enabled": True},
                {"id": "yolo26s", "task": "detection", "enabled": True},
            ]},
            "run_profiles": [{"id": "hailo8", "enabled": True}],
            "selection_policy": {"max_accepted_cases_per_model": 1},
            "execution_preset": {
                "id": "smoke",
                "label": "Smoke",
                "snapshot": {
                    "data": {"validation_items": {"classification": 16, "detection": 12}},
                    "runtime": {"benchmark": {}, "native": {}},
                    "quality": {},
                    "build": {"hailo": {}},
                },
                "overrides": {},
            },
        }
        plan = build_effective_execution_plan(profile)
        self.assertEqual(plan["validation_embedding_policy"], "exact_run_mode_subset")
        self.assertEqual(plan["estimated_validation_files_total"], 16 + 1 + (2 * 12) + 2)


if __name__ == "__main__":
    unittest.main()
