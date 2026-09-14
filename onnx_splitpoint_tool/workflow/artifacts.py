from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import sys
from importlib import metadata as importlib_metadata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .checkpoints import atomic_write_json
from ..filesystem_admission import require_write_target

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

try:
    from .. import (
        __build_contract_version__ as TOOL_BUILD_CONTRACT_VERSION,
        __build_features__ as TOOL_BUILD_FEATURES,
        __build_id__ as TOOL_BUILD_ID,
        __development_lineage__ as TOOL_LINEAGE,
        __release__ as TOOL_RELEASE,
        __version__ as TOOL_VERSION,
    )
except Exception:  # pragma: no cover
    TOOL_VERSION = "unknown"
    TOOL_RELEASE = "unknown"
    TOOL_LINEAGE = "unknown"
    TOOL_BUILD_ID = "unknown"
    TOOL_BUILD_CONTRACT_VERSION = 0
    TOOL_BUILD_FEATURES = ()

STAGE_RESULT_SCHEMA_VERSION = 1
EVALUATION_RUN_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now_iso() -> str:
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")
        except Exception:
            pass
    return utc_now_iso()


def timestamp_for_run() -> str:
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%d_%H%M%S")
        except Exception:
            pass
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def slugify(value: Any, *, fallback: str = "item") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text).strip("_")
    return text or fallback


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    return "sha256:" + sha256_payload(payload)


def sha256_file_uncached(
    path: str | Path, chunk_size: int = 1024 * 1024, *,
    on_chunk: Callable[[int], None] | None = None,
) -> Optional[str]:
    """Read every byte directly; terminal integrity never consults the cache."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("sha256_chunk_size_must_be_positive_integer")
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
            if on_chunk is not None:
                on_chunk(len(chunk))
    return "sha256:" + h.hexdigest()


sha256_file_uncached._v60m_uncached = True  # type: ignore[attr-defined]
sha256_file_uncached._v60m_hash_prefix = "sha256:"  # type: ignore[attr-defined]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    return sha256_file_uncached(path, chunk_size)


# The development hash cache in ``v60m_policy`` must preserve this public
# serialization contract.  Other wrapped helpers intentionally return bare
# hexadecimal digests, so the prefix is declared on the function itself.
sha256_file._v60m_hash_prefix = "sha256:"  # type: ignore[attr-defined]


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")
    return p


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(text), encoding="utf-8")
    return p


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [dict(r or {}) for r in rows]
    if fieldnames is None:
        keys: list[str] = []
        for row in data:
            for key in row.keys():
                if str(key) not in keys:
                    keys.append(str(key))
        fieldnames = keys
    fields = list(fieldnames or [])
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in data:
            writer.writerow({
                key: (
                    json.dumps(
                        row.get(key),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    if isinstance(row.get(key), (Mapping, list, tuple))
                    else "" if row.get(key) is None
                    else row.get(key, "")
                )
                for key in fields
            })
    return p


def relpath(path: str | Path, root: str | Path) -> str:
    p = Path(path)
    r = Path(root)
    try:
        return str(p.resolve().relative_to(r.resolve())).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def file_fingerprint(path: str | Path | None, *, hash_file: bool = True) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False, "sha256": "", "size_bytes": None, "mtime": ""}
    p = Path(path).expanduser()
    exists = p.is_file()
    return {
        "path": str(p),
        "exists": bool(exists),
        "sha256": sha256_file(p) if (exists and hash_file) else "",
        "size_bytes": int(p.stat().st_size) if exists else None,
        "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds") if exists else "",
    }


def package_build_snapshot() -> dict[str, Any]:
    """Return an install/source identity that is stronger than a version label.

    The identity covers every package-owned source/resource file and separately
    records a compact digest of the claim-critical subset.  For wheel installs,
    pip's immutable archive hash is preferred over the environment-specific
    raw ``RECORD`` file.  The static build ID and feature contract are present
    in both source-tree and installed execution.
    """
    package_root = Path(__file__).resolve().parents[1]
    # Keep labels package-relative and select only package-owned files.  This
    # makes the semantic code digest identical for a source checkout and an
    # installed distribution while still covering every claim-critical 2.65
    # path, including the runtime-contract fixes carried forward from 2.62.1.
    critical_files = (
        package_root / "__init__.py",
        package_root / "benchmark" / "accuracy_gate.py",
        package_root / "benchmark" / "accuracy_gates.py",
        package_root / "benchmark" / "classification_validation_presets.py",
        package_root / "benchmark" / "evaluation_profiles.py",
        package_root / "benchmark" / "hailo_policy.py",
        package_root / "benchmark" / "model_preparation.py",
        package_root / "benchmark" / "remote_run.py",
        package_root / "benchmark" / "services.py",
        package_root / "benchmark" / "suite_refresh.py",
        package_root / "benchmark" / "validation_assets.py",
        package_root / "energy" / "collector.py",
        package_root / "energy" / "config.py",
        package_root / "energy" / "metrics.py",
        package_root / "deepx" / "config.py",
        package_root / "deepx" / "compiler.py",
        package_root / "deepx" / "env_status.py",
        package_root / "deepx" / "preprocessing_ab.py",
        package_root / "deepx" / "calibration_size_canary.py",
        package_root / "hailo_backend.py",
        package_root / "config_values.py",
        package_root / "force_build_admission.py",
        package_root / "build_dispatch_policy.py",
        package_root / "hailo_compiler_context.py",
        package_root / "cuda_probe.py",
        package_root / "cache_verify_policy.py",
        package_root / "campaign.py",
        package_root / "execution_plan.py",
        package_root / "filesystem_admission.py",
        package_root / "gui" / "benchmark_workflow.py",
        package_root / "gui" / "controller.py",
        package_root / "gui" / "app.py",
        package_root / "gui" / "profile_editor.py",
        package_root / "gui" / "panels" / "panel_evaluation_workflow.py",
        package_root / "gui" / "panels" / "panel_validate.py",
        package_root / "gui_app.py",
        package_root / "native_command_contract.py",
        package_root / "native_execution_contract.py",
        package_root / "native_performance_identity.py",
        package_root / "native_trt_utils.py",
        package_root / "remote_runtime_closure.py",
        package_root / "hailo_full_contract_promotion.py",
        package_root / "native_energy_quality_admission.py",
        package_root / "native_energy_reporting.py",
        package_root / "resume_artifact_contract.py",
        package_root / "resume_artifact_rehydration.py",
        package_root / "resume_hailo8_source_recovery.py",
        package_root / "resume_remote_rehydration.py",
        package_root / "resume_cohort_preflight.py",
        package_root / "resume_preparation.py",
        package_root / "native_output_endpoint.py",
        package_root / "resources_utils.py",
        package_root / "runners" / "_types.py",
        package_root / "runners" / "harness" / "yolo.py",
        package_root / "runners" / "backends" / "base.py",
        package_root / "management_reference.py",
        package_root / "native_progress.py",
        package_root / "process_control.py",
        package_root / "protocol_freeze.py",
        package_root / "preprocessing_contract.py",
        package_root / "v60m_policy.py",
        package_root / "quality_cache.py",
        package_root / "quality_metrics.py",
        package_root / "quality_replay.py",
        package_root / "quality_service.py",
        package_root / "quality_lifecycle.py",
        package_root / "trt_quality_chain.py",
        package_root / "run_modes.py",
        package_root / "split_export_graph.py",
        package_root / "split_export_runners.py",
        package_root / "validation" / "accuracy_gates.py",
        package_root / "validation" / "host_postprocess.py",
        package_root / "v264_smoke.py",
        package_root / "v265_smoke.py",
        package_root / "v266_smoke.py",
        package_root / "v267_smoke.py",
        package_root / "v268_smoke.py",
        package_root / "v269_smoke.py",
        package_root / "v269a_smoke.py",
        package_root / "v269b_smoke.py",
        package_root / "v269c_smoke.py",
        package_root / "v269d_smoke.py",
        package_root / "v269e_smoke.py",
        package_root / "v269f_smoke.py",
        package_root / "v270_smoke.py",
        package_root / "v270a_smoke.py",
        package_root / "v270b_smoke.py",
        package_root / "v270c_smoke.py",
        package_root / "v270d_smoke.py",
        package_root / "v270e_smoke.py",
        package_root / "v270f_smoke.py",
        package_root / "v270g_smoke.py",
        package_root / "v270h_smoke.py",
        package_root / "v270i_smoke.py",
        package_root / "v270j_smoke.py",
        package_root / "v270k_smoke.py",
        package_root / "v270l_smoke.py",
        package_root / "v270m_smoke.py",
        package_root / "v271_smoke.py",
        package_root / "v272_smoke.py",
        package_root / "v273_smoke.py",
        package_root / "v27547_smoke.py",
        package_root / "native_detection_postprocess.py",
        package_root / "validation" / "official_coco.py",
        package_root / "native_detection_diagnostics.py",
        package_root / "native_split_quality.py",
        package_root / "native_split_quality_authority.py",
        package_root / "native_performance_reporting.py",
        package_root / "native_fastpath" / "CMakeLists.txt",
        package_root / "native_fastpath" / "hailo_trt_fifo_fastpath.cpp",
        package_root / "resources" / "native_fastpath" / "CMakeLists.txt",
        package_root / "resources" / "native_fastpath" / "hailo_trt_fifo_runner.cpp",
        package_root / "resources" / "native_hailo_trt_fifo" / "CMakeLists.txt",
        package_root / "resources" / "native_hailo_trt_fifo" / "native_hailo_trt_fifo.cpp",
        package_root / "resources" / "schemas" / "evaluation_profile.schema.json",
        package_root / "workflow" / "cross_runner_reporting.py",
        package_root / "workflow" / "checkpoints.py",
        package_root / "workflow" / "contracts.py",
        package_root / "workflow" / "start_snapshot.py",
        package_root / "workflow" / "profile_options.py",
        package_root / "workflow" / "run_evaluation.py",
        package_root / "workflow" / "hardware_matrix.py",
        package_root / "workflow" / "benchmark_binding.py",
        package_root / "workflow" / "full_only_quality_canary.py",
        package_root / "workflow" / "legacy_benchmarkset_binding.py",
        package_root / "workflow" / "native_energy_preflight.py",
        package_root / "workflow" / "runner.py",
        package_root / "workflow" / "run_control.py",
        package_root / "workflow" / "results.py",
        package_root / "workflow" / "result_context.py",
        package_root / "workflow" / "execution_binding.py",
        package_root / "workflow" / "evidence_status.py",
        package_root / "workflow" / "jobs.py",
        package_root / "workflow" / "scientific_reporting.py",
        package_root / "workflow" / "setup_local_trt_dispatch.py",
        package_root / "workflow" / "artifacts.py",
        package_root / "workflow" / "analysis_pack.py",
        package_root / "workflow" / "debug_pack_policy.py",
        package_root / "workflow" / "debug_pack.py",
        package_root / "workflow" / "run_discovery.py",
        package_root / "workflow" / "deepx_build_binding.py",
        package_root / "workflow" / "zip_utils.py",
        package_root / "remote" / "ssh_transport.py",
        package_root / "window_method_validation_probe.py",
        package_root / "runners" / "backends" / "hailo_backend.py",
        package_root / "runners" / "backends" / "hailo_utils.py",
        package_root / "runners" / "native_split_quality_runtime.py",
        package_root / "runners" / "native_full_input.py",
        package_root / "resources" / "templates" / "benchmark_suite.py.txt",
        package_root / "resources" / "templates" / "run_split_onnxruntime.py.txt",
        package_root / "resources" / "remote_scripts" / "energy_measurement_cli.py",
        package_root / "resources" / "remote_scripts" / "materialize_cache_verify_native_split_binding.py",
        package_root / "resources" / "remote_scripts" / "native_deepx_full_energy_hotloop.py",
        package_root / "resources" / "remote_scripts" / "native_deepx_trt_e2e_from_benchmarkset.py",
        package_root / "resources" / "remote_scripts" / "native_fifo_eval_runner.py",
        package_root / "resources" / "remote_scripts" / "native_fifo_smoke_matrix.py",
        package_root / "resources" / "remote_scripts" / "native_fifo_capability_report.py",
        package_root / "resources" / "remote_scripts" / "native_full_baseline_eval_runner.py",
        package_root / "resources" / "remote_scripts" / "native_full_semantic_dump.py",
        package_root / "resources" / "remote_scripts" / "native_hailo10_trt_e2e_from_benchmarkset.py",
        package_root / "resources" / "remote_scripts" / "native_hailo_trt_fifo_from_benchmarkset.py",
        package_root / "resources" / "remote_scripts" / "native_host_telemetry.py",
        package_root / "resources" / "remote_scripts" / "native_producer_e2e_eval_runner.py",
        package_root / "resources" / "remote_scripts" / "native_producer_energy_plan.py",
        package_root / "resources" / "remote_scripts" / "native_split_energy_preflight.py",
        package_root / "resources" / "remote_scripts" / "native_trt_from_benchmarkset.py",
        package_root / "resources" / "remote_scripts" / "native_trt_full_completed_hotloop.py",
        package_root / "resources" / "remote_scripts" / "native_producer_final_report.py",
        package_root / "resources" / "remote_scripts" / "native_producer_validate_visualize.py",
        package_root / "resources" / "remote_scripts" / "native_yolo_full_self_reference_probe.py",
        package_root / "resources" / "remote_scripts" / "run_native_producer_energy_from_summary.py",
        package_root / "resources" / "remote_scripts" / "run_and_report_work_units.py",
        package_root / "resources" / "remote_scripts" / "run_window_method_validation_probe.py",
        package_root / "resources" / "remote_scripts" / "run_evalrun_native_producer_variants.py",
        package_root / "resources" / "remote_scripts" / "smoke_hailo10_hef_runner.py",
        package_root / "resources" / "remote_scripts" / "smoke_hailo10_full_from_benchmarkset.py",
        package_root / "resources" / "remote_scripts" / "update_evalset_native_producers.py",
        package_root / "resources" / "remote_scripts" / "validate_output_dumps.py",
    )
    package_hashes: dict[str, str] = {}
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        digest = sha256_file(path)
        if digest:
            package_hashes[str(path.relative_to(package_root)).replace("\\", "/")] = digest
    package_digest = sha256_json({
        label: digest for label, digest in sorted(package_hashes.items())
    })
    critical_hashes = {
        label: package_hashes[label]
        for path in critical_files
        for label in (str(path.relative_to(package_root)).replace("\\", "/"),)
        if label in package_hashes
    }
    critical_digest = sha256_json({
        label: digest for label, digest in sorted(critical_hashes.items())
    })
    payload: dict[str, Any] = {
        "schema": "onnx-splitpoint/package-build-identity",
        "schema_version": 1,
        "package_version": TOOL_VERSION,
        "release": TOOL_RELEASE,
        "lineage": TOOL_LINEAGE,
        "build_id": TOOL_BUILD_ID,
        "build_contract_version": int(TOOL_BUILD_CONTRACT_VERSION),
        "features": list(TOOL_BUILD_FEATURES),
        "package_root": str(package_root),
        "package_content_sha256": package_digest,
        "package_file_count": len(package_hashes),
        "critical_module_sha256": critical_hashes,
        "critical_code_digest_sha256": critical_digest,
        "critical_module_count": len(critical_hashes),
        "critical_module_expected_count": len(critical_files),
        "critical_module_set_complete": len(critical_hashes) == len(critical_files),
        "distribution_archive_sha256": "",
        "distribution_owned_files_sha256": "",
        "distribution_owned_file_count": 0,
        "distribution_record_sha256": "",
        "distribution_record_status": "not_installed_from_matching_distribution",
    }
    try:
        distribution = importlib_metadata.distribution("onnx-splitpoint-tool")
        installed_init = Path(distribution.locate_file("onnx_splitpoint_tool/__init__.py")).resolve()
        if installed_init == (package_root / "__init__.py").resolve():
            # Prefer the immutable archive hash pip records for a wheel.  Do
            # not hash raw installed RECORD: it contains venv-relative console
            # wrappers and generated pyc entries, so the same wheel would get
            # a different identity in another environment.
            try:
                direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
            except Exception:
                direct_url = {}
            archive_info = direct_url.get("archive_info") if isinstance(direct_url, Mapping) else {}
            archive_info = archive_info if isinstance(archive_info, Mapping) else {}
            archive_hashes = archive_info.get("hashes")
            archive_hashes = archive_hashes if isinstance(archive_hashes, Mapping) else {}
            archive_hash = str(
                archive_info.get("hash")
                or archive_hashes.get("sha256")
                or ""
            ).strip().lower()
            if archive_hash.startswith("sha256="):
                archive_hash = "sha256:" + archive_hash.split("=", 1)[1]
            elif archive_hash and not archive_hash.startswith("sha256:"):
                archive_hash = "sha256:" + archive_hash
            payload["distribution_archive_sha256"] = archive_hash

            owned_hashes: dict[str, str] = {}
            for package_path in list(distribution.files or []):
                label = str(package_path).replace("\\", "/")
                if not label.startswith("onnx_splitpoint_tool/"):
                    continue
                if "/__pycache__/" in label or label.endswith((".pyc", ".pyo")):
                    continue
                located = Path(distribution.locate_file(package_path))
                digest = sha256_file(located)
                if digest:
                    owned_hashes[label] = digest
            if owned_hashes:
                owned_digest = sha256_json({
                    label: digest for label, digest in sorted(owned_hashes.items())
                })
                payload["distribution_owned_files_sha256"] = owned_digest
                payload["distribution_owned_file_count"] = len(owned_hashes)
                # Backward-compatible field name; semantics are explicitly
                # canonical package-owned content, not the raw RECORD bytes.
                payload["distribution_record_sha256"] = owned_digest
                payload["distribution_record_status"] = "canonical_package_owned_files"
            else:
                payload["distribution_record_status"] = "package_owned_files_missing"
    except Exception as exc:  # pragma: no cover - installation dependent
        payload["distribution_record_status"] = f"unavailable:{type(exc).__name__}"
    return payload


def environment_snapshot() -> dict[str, Any]:
    return {
        "tool_version": TOOL_VERSION,
        "tool_build": package_build_snapshot(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "created_at": now_iso(),
    }


def file_record(path: str | Path, *, root: str | Path, kind: str, producer_stage: str, model_id: str | None = None, hash_fn: Callable[[Path], Optional[str]] | None = None) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": relpath(p, root),
        "kind": kind,
        "producer_stage": producer_stage,
        "model_id": model_id,
        "sha256": (sha256_file if hash_fn is None else hash_fn)(p),
        "size_bytes": int(p.stat().st_size) if p.is_file() else None,
        "created_at": now_iso(),
    }


@dataclass(frozen=True)
class EvaluationRunPaths:
    run_root: Path
    reports_dir: Path
    run_stages_dir: Path

    def model_root(self, model_id: str) -> Path:
        return self.run_root / "models" / slugify(model_id, fallback="model")

    def model_subdir(self, model_id: str, name: str) -> Path:
        return self.model_root(model_id) / name

    def stage_result_path(self, stage: str, model_id: Optional[str] = None) -> Path:
        if model_id:
            return self.model_root(model_id) / "stages" / stage / "stage_result.json"
        return self.run_stages_dir / stage / "stage_result.json"


def ensure_run_layout(run_root: str | Path, model_ids: Iterable[str] = ()) -> EvaluationRunPaths:
    root = Path(run_root)
    reports = root / "reports"
    stages = root / "stages"
    for p in (root, reports, reports / "tables", reports / "figures", stages):
        p.mkdir(parents=True, exist_ok=True)
    paths = EvaluationRunPaths(root, reports, stages)
    for mid in model_ids:
        base = paths.model_root(mid)
        for sub in ("analysis", "model_preparation", "benchmark_set", "benchmark_results", "validation", "hardware", "full_baselines", "stages", "logs"):
            (base / sub).mkdir(parents=True, exist_ok=True)
    return paths

# v49a compatibility aliases used by the formal workflow runner.
def safe_token(value: Any, default: str = "item") -> str:
    return slugify(value, fallback=default)


def local_timestamp() -> str:
    return timestamp_for_run()


def artifact_exists(root: str | Path, rel_or_abs: str | Path) -> bool:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = Path(root) / p
    return p.is_file()


def artifact_record(path: str | Path, *, root: str | Path, kind: str, producer_stage: str, model_id: str | None = None) -> dict[str, Any]:
    return file_record(path, root=root, kind=kind, producer_stage=producer_stage, model_id=model_id)


def artifact_relpath(root: str | Path, path: str | Path) -> str:
    return relpath(path, root)


class EvaluationStageHandle:
    def __init__(self, ctx: "EvaluationRunContext", stage: str, model_id: str | None = None, inputs: Mapping[str, Any] | None = None, metadata: Mapping[str, Any] | None = None) -> None:
        self.ctx = ctx
        self.stage = str(stage)
        self.model_id = str(model_id) if model_id else None
        self.inputs = dict(inputs or {})
        self.metadata = dict(metadata or {})
        self.started_at = utc_now_iso()

    def finish(self, status: str, message: str = "", outputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self.ctx.write_stage_result(
            self.stage,
            model_id=self.model_id,
            status=status,
            started_at=self.started_at,
            message=message,
            inputs=self.inputs,
            outputs=dict(outputs or {}),
            metadata=self.metadata,
        )


class EvaluationRunContext:
    def __init__(self, *, output_root: str | Path, profile_id: str, profile_path: str = "", profile_payload: Mapping[str, Any] | None = None, run_id: str | None = None, resume: bool = False) -> None:
        if resume:
            raise RuntimeError(
                "Legacy EvaluationRunContext resume is disabled because it has "
                "no single-writer/exact-contract admission. Resume through "
                "EvaluationWorkflowRunner instead; historical runs remain unchanged."
            )
        self.output_root = Path(output_root).expanduser().resolve()
        require_write_target(
            self.output_root,
            operation="EvaluationRun context",
            minimum_free_bytes=1024 * 1024,
            minimum_free_inodes=32,
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.profile_id = slugify(profile_id, fallback="evaluation_profile")
        self.profile_path = str(profile_path or "")
        self.profile_payload = dict(profile_payload or {})
        if run_id:
            self.run_id = slugify(run_id, fallback=f"{self.profile_id}_{timestamp_for_run()}")
        elif resume:
            from .run_discovery import inspect_evaluation_run

            matches = sorted(
                [
                    p
                    for p in self.output_root.glob(f"{self.profile_id}_*")
                    if p.is_dir()
                    and inspect_evaluation_run(p).resumable
                    and str(
                        (read_json(p / "run_manifest.json", default={}) or {}).get(
                            "profile_id"
                        )
                        or ""
                    )
                    == self.profile_id
                ],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            self.run_id = matches[0].name if matches else f"{self.profile_id}_{timestamp_for_run()}"
        else:
            self.run_id = f"{self.profile_id}_{timestamp_for_run()}"
        self.run_dir = self.output_root / self.run_id
        self.paths = ensure_run_layout(self.run_dir)
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.artifact_index_path = self.run_dir / "artifact_index.json"
        self.manifest = read_json(self.manifest_path, default={}) if resume else {}
        if not isinstance(self.manifest, Mapping) or not self.manifest:
            self.manifest = {
                "schema": "onnx-splitpoint/evaluation-run-manifest",
                "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
                "workflow_version": "v51a-profile-persistence-runtime-gates",
                "tool_version": TOOL_VERSION,
                "profile_id": self.profile_id,
                "profile_path": self.profile_path,
                "profile_hash": sha256_json(self.profile_payload),
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
                "status": "created",
                "message": "",
                "models": {},
                "root_stages": {},
                "output_contracts": [],
                "reports": [],
            }
        self.artifact_index = read_json(self.artifact_index_path, default={}) if resume else {}
        if not isinstance(self.artifact_index, Mapping) or not self.artifact_index:
            self.artifact_index = {"schema": "onnx-splitpoint/artifact-index", "schema_version": 1, "run_id": self.run_id, "created_at": utc_now_iso(), "updated_at": utc_now_iso(), "artifacts": []}
        profile_copy = self.run_dir / "profile.yaml"
        if self.profile_path and Path(self.profile_path).is_file():
            try:
                profile_copy.write_text(Path(self.profile_path).read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                profile_copy.write_text(stable_json_dumps(self.profile_payload), encoding="utf-8")
        else:
            profile_copy.write_text(stable_json_dumps(self.profile_payload), encoding="utf-8")
        self._add_artifact(profile_copy, kind="profile", producer_stage="resolve_profile")
        self.save()

    def save(self) -> None:
        self.manifest["updated_at"] = utc_now_iso()
        self.artifact_index["updated_at"] = utc_now_iso()
        write_json(self.manifest_path, self.manifest)
        write_json(self.artifact_index_path, self.artifact_index)

    def set_status(self, status: str, message: str = "") -> None:
        self.manifest["status"] = str(status)
        if message:
            self.manifest["message"] = str(message)
        self.save()

    def stage(self, stage: str, model_id: str | None = None, inputs: Mapping[str, Any] | None = None, metadata: Mapping[str, Any] | None = None) -> EvaluationStageHandle:
        return EvaluationStageHandle(self, stage, model_id=model_id, inputs=inputs, metadata=metadata)

    def register_model(self, model_id: str, **kwargs: Any) -> Path:
        mid = slugify(model_id, fallback="model")
        ensure_run_layout(self.run_dir, [mid])
        model_dir = self.paths.model_root(mid)
        model_path = kwargs.get("model_path")
        manifest = {
            "schema": "onnx-splitpoint/model-manifest",
            "schema_version": 1,
            "model_id": mid,
            "model_path": str(model_path or ""),
            "task": kwargs.get("task"),
            "family": kwargs.get("family"),
            "tier": kwargs.get("tier"),
            "entry": dict(kwargs.get("entry") or {}),
            "export_metadata": dict(kwargs.get("export_metadata") or {}),
            "model_sha256": sha256_file(model_path) if kwargs.get("include_sha256") and model_path else None,
            "registered_at": utc_now_iso(),
        }
        p = write_json(model_dir / "model_manifest.json", manifest)
        self.manifest.setdefault("models", {})[mid] = {**manifest, "manifest_path": relpath(p, self.run_dir), "stages": {}}
        self._add_artifact(p, kind="model_manifest", producer_stage="resolve_model", model_id=mid)
        self.save()
        return model_dir

    def write_stage_result(self, stage: str, *, model_id: str | None = None, status: str = "success", started_at: str = "", message: str = "", inputs: Mapping[str, Any] | None = None, outputs: Mapping[str, Any] | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        mid = slugify(model_id, fallback="") if model_id else None
        status_norm = {"success": "ok", "ok": "ok", "failed": "failed", "failure": "failed", "planned": "planned", "skipped": "skipped", "warn": "warn", "partial": "partial", "cancelled": "cancelled", "canceled": "cancelled"}.get(str(status).lower(), str(status))
        lifecycle_state = (
            "cancelled" if status_norm == "cancelled"
            else "failed" if status_norm == "failed"
            else "running" if status_norm == "planned"
            else "completed"
        )
        artifacts = []
        for v in dict(outputs or {}).values():
            if isinstance(v, (str, Path)) and Path(str(v)).is_file():
                artifacts.append(relpath(v, self.run_dir))
        payload = {
            "schema": "onnx-splitpoint/evaluation-stage-result",
            "schema_version": STAGE_RESULT_SCHEMA_VERSION,
            "stage": stage,
            "model_id": mid,
            "status": status_norm,
            "state": lifecycle_state,
            "complete": lifecycle_state not in {"running", "cancelled"},
            "started_at": started_at or utc_now_iso(),
            "finished_at": utc_now_iso(),
            "message": message,
            "inputs": dict(inputs or {}),
            "outputs": dict(outputs or {}),
            "metadata": dict(metadata or {}),
            "artifacts": artifacts,
            "input_hash": sha256_json({"stage": stage, "model_id": mid, "inputs": dict(inputs or {}), "profile_hash": self.manifest.get("profile_hash")}),
            "output_hash": sha256_json({"status": status_norm, "outputs": dict(outputs or {})}),
        }
        path = self.paths.stage_result_path(stage, model_id=mid)
        payload["stage_result_path"] = relpath(path, self.run_dir)
        atomic_write_json(path, payload)
        if mid:
            self.manifest.setdefault("models", {}).setdefault(mid, {"model_id": mid, "stages": {}}).setdefault("stages", {})[stage] = payload
        else:
            self.manifest.setdefault("root_stages", {})[stage] = payload
        self._add_artifact(path, kind="stage_result", producer_stage=stage, model_id=mid)
        for a in artifacts:
            self._add_artifact(self.run_dir / a, kind="stage_artifact", producer_stage=stage, model_id=mid)
        self.save()
        return payload

    def write_analysis_artifacts(self, model_id: str, *, analysis: Mapping[str, Any], picks: Sequence[Any], params: Any = None, prediction_metadata: Mapping[str, Any] | None = None) -> dict[str, str]:
        mid = slugify(model_id, fallback="model")
        adir = self.paths.model_subdir(mid, "analysis")
        candidates = []
        for rank, p in enumerate(list(picks or []), 1):
            boundary = int(p) if isinstance(p, int) or str(p).isdigit() else p
            candidates.append({"rank": rank, "case_id": f"b{boundary}" if isinstance(boundary, int) else str(boundary), "split_index": boundary, "boundary": boundary, "source": "gui_analysis_picks"})
        p_analysis = write_json(adir / "analysis.json", {"schema_version": 1, "model_id": mid, **dict(analysis or {})})
        p_rank = write_json(adir / "candidate_ranking.json", {"schema_version": 1, "model_id": mid, "candidates": candidates, "top_candidates": candidates})
        p_pred = write_json(adir / "prediction.json", {"schema_version": 1, "model_id": mid, "source_candidate_ranking": "candidate_ranking.json", "selection_strategy": "gui_analysis_topk", "candidates": candidates, "top_candidates": candidates, "metadata": dict(prediction_metadata or {}), "params": str(params or "")})
        for p in (p_analysis, p_rank, p_pred):
            self._add_artifact(p, kind="analysis_artifact", producer_stage="analyze_model", model_id=mid)
        self.save()
        return {"analysis_json": str(p_analysis), "candidate_ranking_json": str(p_rank), "prediction_json": str(p_pred)}

    def write_final_candidate_plan(self, model_id: str, *, suite_dir: str | Path, generation_result: Mapping[str, Any] | None = None) -> dict[str, str]:
        """Mirror the real benchmark-generator decision surface into the workflow bundle.

        The original plan requires that benchmark generation is traceable back to
        `prediction.json` / `final_candidate_plan.json`.  The GUI generator still
        owns the heavy splitting/Hailo build work, but this method now writes a
        workflow-local candidate plan, benchmark generation plan and decisions
        file from the generator callback summary.
        """
        mid = slugify(model_id, fallback="model")
        bdir = self.paths.model_subdir(mid, "benchmark_set")
        pred_path = self.paths.model_subdir(mid, "analysis") / "prediction.json"
        pred = read_json(pred_path, default={}) or {}
        selected = list(pred.get("top_candidates") or pred.get("candidates") or [])
        gen = dict(generation_result or {})
        summary = dict(gen.get("summary") or {}) if isinstance(gen.get("summary"), Mapping) else {}
        suite_dir_path = Path(suite_dir).expanduser() if str(suite_dir or "").strip() else Path("")
        suite_json = suite_dir_path / "benchmark_set.json" if suite_dir_path else Path("")
        suite_payload = read_json(suite_json, default={}) if suite_json.is_file() else {}

        def _case_id_from_boundary(value: Any) -> str:
            try:
                return f"b{int(value)}"
            except Exception:
                s = str(value or "").strip()
                return s if s.startswith("b") else f"b{s}"

        planned_by_boundary: dict[str, dict[str, Any]] = {}
        for cand in selected:
            if not isinstance(cand, Mapping):
                continue
            boundary = cand.get("boundary", cand.get("split_index", cand.get("case_id", "")))
            case_id = str(cand.get("case_id") or _case_id_from_boundary(boundary))
            planned_by_boundary[case_id.lstrip("b")] = dict(cand)
            planned_by_boundary[case_id] = dict(cand)

        accepted_boundaries = list(summary.get("accepted_boundaries") or summary.get("accepted_hailo_boundaries") or [])
        accepted_cases = []
        policy_backfills = []
        for rank, boundary in enumerate(accepted_boundaries, start=1):
            case_id = _case_id_from_boundary(boundary)
            src = planned_by_boundary.get(str(boundary)) or planned_by_boundary.get(case_id) or {}
            row = {
                "case_id": case_id,
                "boundary": boundary,
                "accepted_rank": rank,
                "source_rank": src.get("rank", ""),
                "source": src.get("source", "generator_summary"),
                "origin": "prediction_plan" if src else "generator_policy_or_backfill",
            }
            accepted_cases.append(row)
            if not src:
                policy_backfills.append({**row, "reason": "accepted by generator but not present in workflow prediction top candidates"})

        issue_groups = [dict(x or {}) for x in list(summary.get("issue_groups") or []) if isinstance(x, Mapping)]
        rejected_cases = []
        for grp in issue_groups:
            title = str(grp.get("title") or "").lower()
            if "reject" not in title and str(grp.get("kind") or "").lower() != "rejected":
                continue
            rejected_cases.append({
                "error_class": "compile_failed" if "hailo" in title or "hef" in title else "unknown_error",
                "count": int(grp.get("count") or 0),
                "reason": str(grp.get("title") or "Rejected"),
                "examples": list(grp.get("examples") or []),
                "sample_detail": str(grp.get("sample_detail") or ""),
            })

        plan_adjustments = [dict(x or {}) for x in list(summary.get("plan_adjustments") or []) if isinstance(x, Mapping)]
        policy_promotions = []
        for item in plan_adjustments:
            policy_promotions.append({
                "kind": str(item.get("kind") or item.get("policy") or "plan_adjustment"),
                "boundary": item.get("boundary", item.get("candidate", "")),
                "reason": str(item.get("reason") or item.get("message") or item),
                "raw": item,
            })

        payload = {
            "schema": "onnx-splitpoint/final-candidate-plan",
            "schema_version": 1,
            "model_id": mid,
            "source_prediction_json": relpath(pred_path, self.run_dir),
            "suite_dir": str(suite_dir_path),
            "benchmark_set_json": str(suite_json) if suite_json.is_file() else "",
            "generation_status": str(gen.get("status") or summary.get("final_status") or ""),
            "generation_message": str(gen.get("message") or ""),
            "selected_candidates": selected,
            "accepted_cases": accepted_cases,
            "policy_promotions": policy_promotions,
            "policy_backfills": policy_backfills,
            "excluded_candidates": [],
            "methodological_note": "The selected_candidates list comes from prediction.json; accepted_cases/policy_* reflect the real generator outcome.",
        }
        p_plan = write_json(bdir / "final_candidate_plan.json", payload)

        benchmark_plan = {
            "schema": "onnx-splitpoint/benchmark-generation-plan",
            "schema_version": 1,
            "model_id": mid,
            "source_prediction_json": relpath(pred_path, self.run_dir),
            "source_final_candidate_plan_json": relpath(p_plan, self.run_dir),
            "planned_candidates": selected,
            "suite_dir": str(suite_dir_path),
            "benchmark_set_json": str(suite_json) if suite_json.is_file() else "",
            "generator_status": str(gen.get("status") or summary.get("final_status") or ""),
        }
        p_bench_plan = write_json(bdir / "benchmark_plan.json", benchmark_plan)

        decisions = {
            "schema": "onnx-splitpoint/benchmark-generation-decisions",
            "schema_version": 1,
            "model_id": mid,
            "source_final_candidate_plan_json": relpath(p_plan, self.run_dir),
            "suite_dir": str(suite_dir_path),
            "status": str(gen.get("status") or summary.get("final_status") or ""),
            "accepted_cases": accepted_cases,
            "rejected_cases": rejected_cases,
            "policy_promotions": policy_promotions,
            "policy_backfills": policy_backfills,
            "accepted_count": int(summary.get("accepted_count") or len(accepted_cases) or 0),
            "requested_cases": int(summary.get("requested_cases") or 0),
            "shortfall": int(summary.get("shortfall") or 0),
            "backfilled_cases_count": int(summary.get("backfilled_cases_count") or len(policy_backfills) or 0),
            "generator_summary": summary,
        }
        p_decisions = write_json(bdir / "generation_decisions.json", decisions)

        # Keep a local copy of benchmark_set.json if the real generator produced one.
        out_paths = {
            "final_candidate_plan_json": str(p_plan),
            "benchmark_plan_json": str(p_bench_plan),
            "generation_decisions_json": str(p_decisions),
        }
        if suite_json.is_file():
            local_suite_json = bdir / "benchmark_set.json"
            try:
                local_suite_json.write_text(suite_json.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                write_json(local_suite_json, suite_payload if isinstance(suite_payload, Mapping) else {"source": str(suite_json)})
            out_paths["benchmark_set_json"] = str(local_suite_json)
            self._add_artifact(local_suite_json, kind="benchmark_set", producer_stage="generate_benchmark_set", model_id=mid)

        for path in (p_plan, p_bench_plan, p_decisions):
            self._add_artifact(path, kind="benchmark_generation", producer_stage="generate_benchmark_set", model_id=mid)
        self.save()
        return out_paths

    def write_summary_reports(self, *, model_results: Sequence[Mapping[str, Any]], missing_models: Sequence[str] = ()) -> dict[str, str]:
        reports = self.paths.reports_dir
        rows = [dict(r or {}) for r in model_results]
        p_summary = write_csv(reports / "summary.csv", [{"run_id": self.run_id, "profile_id": self.profile_id, "model_count": len(rows), "missing_models": ",".join(missing_models), "status": self.manifest.get("status", "")}])
        p_model = write_csv(reports / "model_summary.csv", rows)
        p_pred = write_csv(reports / "prediction_vs_benchmark.csv", [{"model_id": r.get("model_id", ""), "status": "pending_benchmark_ingestion"} for r in rows])
        p_hw = write_csv(reports / "hardware_summary.csv", [])
        p_rank = write_csv(reports / "splitpoint_ranking.csv", [])
        p_text = write_text(reports / "thesis_section.md", f"# Evaluation Workflow Run {self.run_id}\n\nProfile: `{self.profile_id}`\n\nGenerated by the v49b GUI Evaluation Workflow adapter. Benchmark generation can now be started directly from the dedicated Evaluation Workflow tab; measured benchmark metrics are populated after remote/local execution results are ingested.\n")
        out = {"summary_csv": str(p_summary), "model_summary_csv": str(p_model), "prediction_vs_benchmark_csv": str(p_pred), "hardware_summary_csv": str(p_hw), "splitpoint_ranking_csv": str(p_rank), "thesis_section_md": str(p_text)}
        self.manifest["reports"] = [relpath(v, self.run_dir) for v in out.values()]
        for p in (p_summary, p_model, p_pred, p_hw, p_rank, p_text):
            self._add_artifact(Path(p), kind="report", producer_stage="generate_report")
        self.save()
        return out

    def _add_artifact(self, path: str | Path, *, kind: str, producer_stage: str, model_id: str | None = None) -> None:
        rec = file_record(path, root=self.run_dir, kind=kind, producer_stage=producer_stage, model_id=model_id)
        arts = [dict(x or {}) for x in list(self.artifact_index.get("artifacts") or [])]
        arts = [x for x in arts if x.get("path") != rec.get("path")]
        arts.append(rec)
        self.artifact_index["artifacts"] = arts


def create_evaluation_run_context(**kwargs: Any) -> EvaluationRunContext:
    return EvaluationRunContext(**kwargs)

# v60m: unchanged files use a stat-keyed SHA-256 cache in development;
# final campaigns still re-hash strictly.
from onnx_splitpoint_tool.v60m_policy import install_hash_wrappers as _v60m_install_hash_wrappers
_v60m_install_hash_wrappers(globals())
