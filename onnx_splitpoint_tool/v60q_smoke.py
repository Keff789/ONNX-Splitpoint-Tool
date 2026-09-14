from __future__ import annotations

"""Fast, hardware-free smoke checks for v60q validation-subset and shared-bundle rules."""

import argparse
import json
import os
import tarfile
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

from .benchmark.remote_run import (
    _exclusive_suite_refresh_guard,
    _extract_run_ids_from_add_args,
    _filter_benchmark_plan_for_run_ids,
    _remote_suite_cache_paths,
)
from .benchmark.classification_validation_presets import provision_classification_validation_source_to_suite
from .benchmark.validation_assets import provision_detection_validation_source_to_suite
from .remote.bundle import build_suite_bundle, remote_minimal_bundle_patterns
from .workflow.deepx_build_binding import _infer_deepx_calibration_dir
from .campaign import create_dataset_manifest, verify_dataset_manifest
from .execution_plan import build_effective_execution_plan
from .run_modes import apply_run_mode, default_run_modes_config
from .v60m_policy import normalize_profile
from .workflow.legacy_benchmarkset_binding import _infer_run_switches
from .workflow.execution_binding import _energy_enabled_for_profile
from .workflow.results import _best_full_latency
from .workflow.hailo_remote_binding import _infer_backend
from .workflow.runner import EvaluationWorkflowRunner, _benchmark_stage_status_v60p, _campaign_stage_status_v60p


def _base_profile(mode: str = "standard", *, native: bool = True, energy: bool = False) -> dict[str, Any]:
    return {
        "name": "v60q_smoke",
        "selection_policy": {
            "max_accepted_cases_per_model": 1,
            "preferred_shortlist": 5,
            "min_gap": 1,
            "candidate_search_pool": "auto",
            "selection_strategy": "stratified_windows",
        },
        "model_suite": {
            "primary": [
                {"id": "resnet50", "task": "classification", "evaluation_role": "development", "enabled": True},
                {"id": "yolo26s", "task": "detection", "evaluation_role": "development", "enabled": True},
            ]
        },
        "run_profiles": [
            {"id": "hailo8", "full": "hailo8", "stage1": "hailo8", "stage2": "hailo8", "enabled": True},
            {"id": "hailo8_to_trt", "stage1": "hailo8", "stage2": "tensorrt", "enabled": True},
            {"id": "deepx_m1", "full": "deepx_m1", "stage1": "deepx_m1", "stage2": "deepx_m1", "enabled": True},
        ],
        "execution_preset": {
            "id": mode,
            "follow_tool_config": False,
            "snapshot": default_run_modes_config()["modes"][mode],
            "overrides": {"native_enabled": native, "energy_enabled": energy},
        },
    }


def _run_checks() -> list[dict[str, Any]]:
    checks: list[tuple[str, Callable[[], None]]] = []

    def check(name: str):
        def deco(fn: Callable[[], None]) -> Callable[[], None]:
            checks.append((name, fn))
            return fn
        return deco

    @check("generic_energy_is_hard_disabled")
    def _() -> None:
        profile, _audit = apply_run_mode(_base_profile("final", native=True, energy=True))
        assert profile["energy"]["enabled"] is False
        assert profile["energy"]["generic_enabled"] is False
        assert profile["energy"]["measurement_path"] == "native_only"
        assert _energy_enabled_for_profile(object(), profile) is False

    @check("native_energy_follows_native_and_energy_switches")
    def _() -> None:
        enabled, _ = apply_run_mode(_base_profile("final", native=True, energy=True))
        native_off, _ = apply_run_mode(_base_profile("final", native=False, energy=True))
        energy_off, _ = apply_run_mode(_base_profile("final", native=True, energy=False))
        assert enabled["native_producers"]["energy"]["enabled"] is True
        assert native_off["native_producers"]["energy"]["enabled"] is False
        assert energy_off["native_producers"]["energy"]["enabled"] is False

    @check("native_energy_survives_profile_normalization")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("final", native=True, energy=True))
        normalized, audit = normalize_profile(profile)
        assert normalized["native_producers"]["energy"]["enabled"] is True
        assert audit["energy"]["master_enabled"] is True

    @check("unused_hailo_part2_build_is_not_requested")
    def _() -> None:
        profile = _base_profile("standard", native=True, energy=False)
        switches = _infer_run_switches(profile, [])
        assert switches["hailo_part1_requested"] is True
        assert switches["hailo_part2_requested"] is False

    @check("standard_uses_sampled_integrity_and_500_items")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("standard", native=True, energy=False))
        assert profile["integrity_policy"]["mode"] == "relaxed"
        assert profile["integrity_policy"]["dataset_sample_size"] > 0
        assert profile["validation_execution"]["max_items"] == {"classification": 500, "detection": 500}
        assert profile["hailo_build"]["calib_count"] == 500
        assert profile["hailo_build"]["calibration_storage"] == "memmap"

    @check("final_quality_uses_standard_integrity")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("final", native=True, energy=True))
        assert profile["integrity_policy"]["mode"] == "relaxed"
        assert profile["integrity_policy"]["dataset_sample_size"] == 24
        assert profile["campaign"]["mode"] == "development"

    @check("sampled_manifest_verification")
    def _() -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "images"
            root.mkdir()
            for idx in range(20):
                (root / f"n{idx:04d}.jpg").write_bytes((f"sample-{idx}" * 4).encode("utf-8"))
            manifest_path = Path(temp) / "manifest.json"
            create_dataset_manifest(
                task="classification", role="validation", dataset_id="v60p", split="val",
                root=root, output=manifest_path,
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            result = verify_dataset_manifest(payload, verification_mode="sampled", sample_size=5)
            assert result["ok"] is True
            assert result["checked_item_count"] == 5

    @check("setup_local_run_id_batching")
    def _() -> None:
        ids = _extract_run_ids_from_add_args("--run-ids hailo8,hailo8_to_trt")
        assert ids == ["hailo8", "hailo8_to_trt"]
        plan = {"runs": [{"id": "hailo8"}, {"id": "hailo8_to_trt"}, {"id": "deepx_m1"}]}
        filtered = _filter_benchmark_plan_for_run_ids(plan, ids)
        assert [row["id"] for row in filtered["runs"]] == ids

    @check("effective_plan_exposes_batching_and_native_energy")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("standard", native=True, energy=True))
        plan = build_effective_execution_plan(profile)
        assert plan["generic_energy_enabled"] is False
        assert plan["native_energy_enabled"] is True
        assert plan["batched_remote_dispatches_total"] < plan["generic_rows_total"]
        assert plan["uploads_per_model_setup"] == 1

    @check("tensorrt_full_uses_its_own_timing")
    def _() -> None:
        row = {"latency_ms": 100.0, "timings": {"composed": {"mean_ms": 100.0}, "full": {"mean_ms": 9.0}}}
        total, raw, _e2e, status = _best_full_latency(row)
        assert total == 9.0 and raw == 9.0 and status == "full"

    @check("smoke_mode_is_small_registry_bound_and_timeout_safe")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("smoke", native=False, energy=False))
        snap = profile["execution_preset"]["snapshot"]
        assert snap["data"]["use_final_dataset_registry"] is True
        assert snap["data"]["validation_items"] == {"classification": 16, "detection": 12}
        assert snap["quality"]["bootstrap_repetitions"] == 25
        assert snap["runtime"]["benchmark"]["timeout_s"] == 3600

    @check("hailo10_path_is_not_misclassified_as_hailo8")
    def _() -> None:
        assert _infer_backend(Path("suite/b052/hailo/hailo10/part1/compiled.hef")) == "hailo10h"
        assert _infer_backend(Path("suite/b052/hailo/hailo8/part1/compiled.hef")) == "hailo8"

    @check("development_readiness_is_green")
    def _() -> None:
        assert _campaign_stage_status_v60p("development_ready") == "ok"
        assert _campaign_stage_status_v60p("blocked") == "warn"

    @check("partial_remote_matrix_stays_partial_with_rows")
    def _() -> None:
        assert _benchmark_stage_status_v60p(normalized_row_count=3, executor_status="partial", executor_metrics={}) == "partial"
        assert _benchmark_stage_status_v60p(normalized_row_count=3, executor_status="ok", executor_metrics={}) == "ok"

    @check("runtime_template_contains_safe_bootstrap_shortcuts")
    def _() -> None:
        source = (Path(__file__).parent / "resources" / "templates" / "run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
        assert "point_estimate_below_non_inferiority_margin" in source
        assert "candidate_reference_identical" in source
        assert "bootstrap_repetitions_requested" in source
        assert "paired_detection_cached_matching_v1" in source

    @check("native_outputs_are_tagged_and_merged")
    def _() -> None:
        resources = Path(__file__).parent / "resources" / "remote_scripts"
        runner = (resources / "native_producer_e2e_eval_runner.py").read_text(encoding="utf-8")
        report = (resources / "native_producer_final_report.py").read_text(encoding="utf-8")
        assert "--analysis-tag" in runner
        assert "native_hailo10h_producer_e2e_eval*.json" in report

    @check("classification_smoke_materialises_only_requested_items")
    def _() -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "imagenet"
            for idx in range(8):
                cls = images / f"n{idx % 2:08d}"
                cls.mkdir(parents=True, exist_ok=True)
                (cls / f"{idx}.JPEG").write_bytes(f"image-{idx}".encode())
            manifest = root / "manifest.json"
            create_dataset_manifest(task="classification", role="validation", dataset_id="v60q", split="val", root=images, output=manifest)
            suite = root / "suite"
            suite.mkdir()
            rel = provision_classification_validation_source_to_suite(
                suite, str(images), max_images=3, manifest_path=str(manifest), selection_seed=7
            )
            assert rel
            payload = json.loads((suite / rel).read_text(encoding="utf-8"))
            assert payload["selection"]["selected_images"] == 3
            assert len(payload["samples"]) == 3

    @check("detection_smoke_materialises_only_requested_items")
    def _() -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "coco"
            images.mkdir()
            image_rows = []
            annotations = []
            for idx in range(1, 6):
                name = f"{idx:012d}.jpg"
                (images / name).write_bytes(f"coco-{idx}".encode())
                image_rows.append({"id": idx, "file_name": name, "width": 32, "height": 32})
                annotations.append({"id": idx, "image_id": idx, "category_id": 1, "bbox": [0, 0, 1, 1]})
            ann = root / "instances.json"
            ann.write_text(json.dumps({"images": image_rows, "annotations": annotations, "categories": [{"id": 1, "name": "x"}]}), encoding="utf-8")
            manifest = root / "manifest.json"
            create_dataset_manifest(task="detection", role="validation", dataset_id="v60q", split="val", root=images, annotations=ann, output=manifest)
            suite = root / "suite"
            suite.mkdir()
            rel = provision_detection_validation_source_to_suite(
                suite, str(images), max_images=2, manifest_path=str(manifest), selection_seed=8
            )
            assert rel
            subset = suite / rel
            assert len(list(subset.glob("*.jpg"))) == 2
            assert len(json.loads((subset / "instances_subset.json").read_text(encoding="utf-8"))["images"]) == 2

    @check("remote_bundle_has_no_broad_validation_allowlist")
    def _() -> None:
        includes, _excludes = remote_minimal_bundle_patterns()
        assert "resources/validation/**" not in includes

    @check("exact_validation_root_excludes_stale_full_tree")
    def _() -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite = root / "suite"
            suite.mkdir()
            (suite / "benchmark_plan.json").write_text("{}", encoding="utf-8")
            selected = suite / "resources" / "validation" / "classification" / "smoke_n2"
            selected.mkdir(parents=True)
            (selected / "a.JPEG").write_bytes(b"a")
            stale = suite / "resources" / "validation" / "classification" / "full_50000"
            stale.mkdir(parents=True)
            for idx in range(20):
                (stale / f"{idx}.JPEG").write_bytes(b"stale")
            includes, excludes = remote_minimal_bundle_patterns()
            includes.append(f"{selected.relative_to(suite).as_posix()}/**")
            previous_integrity = os.environ.get("ONNX_SPLITPOINT_INTEGRITY_MODE")
            os.environ["ONNX_SPLITPOINT_INTEGRITY_MODE"] = "fast"
            try:
                stats = build_suite_bundle(suite, suite / "dist" / "suite_bundle.tar.gz", includes=includes, excludes=excludes)
            finally:
                if previous_integrity is None:
                    os.environ.pop("ONNX_SPLITPOINT_INTEGRITY_MODE", None)
                else:
                    os.environ["ONNX_SPLITPOINT_INTEGRITY_MODE"] = previous_integrity
            manifest = json.loads(Path(stats.manifest_path).read_text(encoding="utf-8"))
            rels = [str(row.get("rel") or "") for row in manifest["files"]]
            assert any("smoke_n2/a.JPEG" in rel for rel in rels)
            assert not any("full_50000" in rel for rel in rels)
            assert manifest["per_file_hashes"] is False
            assert manifest["gzip_level"] == 1

    @check("parallel_setup_workers_share_one_bundle_build")
    def _() -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite = root / "suite"
            suite.mkdir()
            (suite / "benchmark_plan.json").write_text("{}", encoding="utf-8")
            includes, excludes = remote_minimal_bundle_patterns()
            out = suite / "dist" / "suite_bundle.tar.gz"
            rows = []
            lock = threading.Lock()
            def worker() -> None:
                result = build_suite_bundle(suite, out, includes=includes, excludes=excludes)
                with lock:
                    rows.append(result)
            threads = [threading.Thread(target=worker) for _ in range(3)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            assert len(rows) == 3
            assert sum(not row.reused for row in rows) == 1
            assert len({row.sha256 for row in rows}) == 1


    @check("parallel_suite_refresh_is_serialized")
    def _() -> None:
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp) / "suite"
            suite.mkdir()
            active = 0
            max_active = 0
            counter_lock = threading.Lock()
            start = threading.Barrier(3)

            def worker() -> None:
                nonlocal active, max_active
                start.wait()
                with _exclusive_suite_refresh_guard(suite):
                    with counter_lock:
                        active += 1
                        max_active = max(max_active, active)
                    import time
                    time.sleep(0.02)
                    with counter_lock:
                        active -= 1

            threads = [threading.Thread(target=worker) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
            assert all(not thread.is_alive() for thread in threads)
            assert max_active == 1

    @check("remote_suite_cache_population_is_atomic")
    def _() -> None:
        paths = _remote_suite_cache_paths("/tmp/remote", "resnet50-suite", "abc123")
        assert paths["population_lock"].endswith("/.population.lock")
        source = (Path(__file__).parent / "benchmark" / "remote_run.py").read_text(encoding="utf-8")
        assert "def _remote_cache_population_guard" in source
        assert "mkdir {qlock}" in source
        assert "population owner uploading" in source
        assert "Remote suite-cache population finished without complete ready markers" in source


    @check("explicit_model_source_beats_stale_old_run_resolution")
    def _() -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = root / "models" / "resnet50.onnx"
            canonical.parent.mkdir(parents=True)
            canonical.write_bytes(b"canonical")
            stale = root / "EvaluationRuns" / "old" / "models" / "resnet50.onnx"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"stale")
            runner = object.__new__(EvaluationWorkflowRunner)
            runner.options = type("Options", (), {"models_root": "", "include_reserve": False, "only_model": ""})()
            runner.profile_payload = {"models": [{"id": "resnet50", "onnx": str(canonical), "resolved_path": str(stale)}]}
            runner.warnings = []
            rows = runner._resolve_model_rows()
            assert len(rows) == 1
            assert Path(rows[0]["resolved_path"]) == canonical.resolve()

    @check("deepx_calibration_is_bound_per_model_task")
    def _() -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cls = root / "imagenet_calib"
            det = root / "coco_calib"
            (cls / "n00000001").mkdir(parents=True)
            det.mkdir()
            (cls / "n00000001" / "a.jpg").write_bytes(b"cls")
            (det / "a.jpg").write_bytes(b"det")
            cls_manifest = root / "cls.json"
            det_manifest = root / "det.json"
            cls_manifest.write_text(json.dumps({"root": str(cls)}), encoding="utf-8")
            det_manifest.write_text(json.dumps({"root": str(det)}), encoding="utf-8")
            profile = {"campaign": {"dataset_manifests": {
                "classification": {"calibration": str(cls_manifest)},
                "detection": {"calibration": str(det_manifest)},
            }}}
            cfg = {"calib_dir": str(det)}
            assert Path(_infer_deepx_calibration_dir(row={"task": "classification", "id": "resnet50"}, model_id="resnet50", cfg=cfg, profile_payload=profile)) == cls.resolve()
            assert Path(_infer_deepx_calibration_dir(row={"task": "detection", "id": "yolo26s"}, model_id="yolo26s", cfg=cfg, profile_payload=profile)) == det.resolve()

    @check("deterministic_bundle_hash_is_stable_across_mtimes")
    def _() -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hashes = []
            for idx in range(2):
                suite = root / f"suite_{idx}"
                suite.mkdir()
                model = suite / "model.onnx"
                model.write_bytes(b"stable")
                os.utime(model, ns=(1_000_000_000 + idx * 1_000_000_000, 1_000_000_000 + idx * 1_000_000_000))
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
                hashes.append(stats.sha256)
            assert hashes[0] == hashes[1]

    @check("execution_plan_exposes_exact_validation_embedding")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("smoke", native=False, energy=False))
        plan = build_effective_execution_plan(profile)
        assert plan["validation_embedding_policy"] == "exact_run_mode_subset"
        assert plan["estimated_validation_files_total"] == 43

    results: list[dict[str, Any]] = []
    for name, fn in checks:
        try:
            fn()
            results.append({"name": name, "status": "pass"})
        except Exception as exc:
            results.append({"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onnx-splitpoint-smoke-v60q", description="Run fast v60q validation-subset/shared-bundle checks without hardware.")
    parser.add_argument("--json", default="", help="Optional JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    results = _run_checks()
    passed = sum(row["status"] == "pass" for row in results)
    failed = len(results) - passed
    payload = {
        "schema": "onnx-splitpoint/v60q-smoke",
        "status": "ok" if failed == 0 else "failed",
        "passed": passed,
        "failed": failed,
        "checks": results,
    }
    if str(args.json or "").strip():
        path = Path(args.json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in results:
        suffix = "" if row["status"] == "pass" else f" — {row.get('error', '')}"
        print(f"[{row['status'].upper()}] {row['name']}{suffix}")
    print(f"v60q smoke: {'ok' if failed == 0 else 'FAILED'} ({passed} passed, {failed} failed)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
