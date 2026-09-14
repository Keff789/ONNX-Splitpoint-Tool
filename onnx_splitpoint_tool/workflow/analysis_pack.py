from __future__ import annotations

"""Create a compact, thesis-oriented analysis bundle for an EvaluationRun.

The v60 pack consumes the canonical scientific-report contract.  It deliberately
omits the former claim-table/dashboard duplicates and keeps raw execution logs in
a separate provenance area.  Large model artefacts, energy traces and vendor
binaries remain in the EvaluationRun/debug pack and are not copied here.
"""

import csv
from contextlib import contextmanager
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from .zip_utils import (
    iter_safe_pack_files,
    require_safe_pack_directory,
    require_safe_pack_source,
    temporary_zip_path,
    verify_zip_archive,
)


CANONICAL_RESULT_FILES = (
    "row_eligibility.csv",
    "task_quality.csv",
    "task_quality_reference_comparison.csv",
    "task_quality_reference_comparison.json",
    "task_quality_reference_comparison.md",
    "task_quality_loss_decomposition.csv",
    "task_quality_loss_decomposition.json",
    "task_quality_loss_decomposition.md",
    "quality_evidence_v61a.json",
    "quality_evidence_v60z.json",
    "performance_results.csv",
    "performance_observations.csv",
    "performance_observations.json",
    "performance_cohorts.csv",
    "performance_cohorts.json",
    "performance_cohort_summary.json",
    "endpoint_lifecycle_ledger.csv",
    "endpoint_lifecycle_ledger.json",
    "endpoint_lifecycle_summary.json",
    "energy_results.csv",
    "screening_performance_observations.csv",
    "screening_performance_observations.json",
    "screening_energy_observations.csv",
    "screening_energy_observations.json",
    "claim_exclusion_summary.json",
    "claim_exclusion_summary.md",
    "claim_exclusion_details.csv",
    "claim_exclusion_grouped_counts.csv",
    "claim_exclusion_dimension_counts.csv",
    "native_energy_observations.csv",
    "native_energy_observations.json",
    "native_energy_pair_comparison.csv",
    "native_energy_pair_comparison.json",
    "native_energy_pair_comparison.md",
    "native_performance_matrix.json",
    "native_performance_matrix.md",
    "native_performance_observations.csv",
    "native_performance_observations.json",
    "native_energy_ab_aggregates.csv",
    "native_energy_ab_aggregates.json",
    "native_energy_ab_aggregates.md",
    "ranking_method_comparison.csv",
    "ranking_method_macro.csv",
    "ranking_method_cohort_sensitivity.csv",
    "ranking_method_cohort_sensitivity.json",
    "ranking_method_cohort_sensitivity.md",
    "cross_runner_candidate_pairs.csv",
    "cross_runner_candidate_pairs.json",
    "cross_runner_ranking_validation.csv",
    "cross_runner_ranking_validation.json",
    "cross_runner_ranking_macro.json",
    "cross_runner_ranking.md",
)
CANONICAL_TABLE_FILES = (
    "task_quality_gates.tex",
    "task_quality_reference_comparison.tex",
    "task_quality_loss_decomposition.tex",
    "ranking_method_comparison.tex",
    "ranking_method_cohort_sensitivity.tex",
    "cross_runner_ranking_validation.tex",
    "performance_results.tex",
    "performance_observations.tex",
    "energy_results.tex",
    "screening_performance_observations.tex",
    "screening_energy_observations.tex",
    "native_energy_observations.tex",
    "native_energy_pair_comparison.tex",
    "native_performance_observations.tex",
    "native_energy_ab_aggregates.tex",
)


def _read_csv(path: Path, *, allowed_root: Path | None = None) -> List[Dict[str, Any]]:
    try:
        if allowed_root is not None:
            path = require_safe_pack_source(path, allowed_root)
        with path.open("r", newline="", encoding="utf-8") as fh:
            return [dict(r) for r in csv.DictReader(fh)]
    except Exception:
        return []


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> str:
    if not fieldnames:
        seen: List[str] = []
        for row in rows:
            for key in row:
                if str(key) not in seen:
                    seen.append(str(key))
        fieldnames = seen
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fieldnames or []), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return buf.getvalue()


def _safe_name(value: Any, default: str = "item") -> str:
    raw = str(value or default).strip() or default
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)[:120] or default


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _small_file(path: Path, max_bytes: int) -> bool:
    try:
        return path.is_file() and int(path.stat().st_size) <= int(max_bytes)
    except Exception:
        return False


def _write_file(
    zf: zipfile.ZipFile,
    src: Path,
    arcname: str,
    written: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
    *,
    max_bytes: int = 8 * 1024 * 1024,
    allowed_root: Path,
) -> None:
    try:
        src = require_safe_pack_source(src, allowed_root)
        if not src.is_file():
            skipped.append({"path": arcname, "reason": "missing"})
            return
        if not _small_file(src, max_bytes):
            skipped.append({"path": arcname, "reason": "too_large", "size_bytes": int(src.stat().st_size)})
            return
        data = src.read_bytes()
        zf.writestr(arcname, data)
        written.append({"path": arcname, "size_bytes": len(data), "sha256": _sha256_bytes(data)})
    except Exception as exc:
        skipped.append({"path": arcname, "reason": f"{type(exc).__name__}: {exc}"})


def _write_text(zf: zipfile.ZipFile, arcname: str, text: str, written: List[Dict[str, Any]]) -> None:
    data = text.encode("utf-8")
    zf.writestr(arcname, data)
    written.append({"path": arcname, "size_bytes": len(data), "sha256": _sha256_bytes(data)})


@contextmanager
def _temporary_analysis_zip(path: Path):
    """Open one unpublished ZIP candidate and remove it on every build error."""

    try:
        with zipfile.ZipFile(
            path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            yield archive
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_tail(
    zf: zipfile.ZipFile,
    src: Path,
    arcname: str,
    written: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
    *,
    max_bytes: int = 1_000_000,
    allowed_root: Path,
) -> None:
    try:
        src = require_safe_pack_source(src, allowed_root)
        if not src.is_file():
            return
        size = int(src.stat().st_size)
        with src.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read(max_bytes)
        prefix = b"" if size <= len(data) else f"[tail only: last {len(data)} of {size} bytes]\n".encode()
        payload = prefix + data
        zf.writestr(arcname, payload)
        written.append({"path": arcname, "size_bytes": len(payload), "sha256": _sha256_bytes(payload), "source_tail_of": str(src)})
    except Exception as exc:
        skipped.append({"path": arcname, "reason": f"tail_failed: {type(exc).__name__}: {exc}"})


def _model_summary(model_id: str, eligibility: Sequence[Mapping[str, Any]], quality: Sequence[Mapping[str, Any]], ranking: Sequence[Mapping[str, Any]]) -> str:
    q_status: Dict[str, int] = {}
    for row in quality:
        key = str(row.get("task_quality_status") or "unavailable")
        q_status[key] = q_status.get(key, 0) + 1
    eligible = sum(str(row.get("ranking_eligible") or "").lower() in {"true", "1", "yes"} for row in eligibility)
    lines = [
        f"# {model_id}",
        "",
        f"Rows: **{len(eligibility)}**  ",
        f"Ranking-eligible rows: **{eligible}**  ",
        "Task-quality decisions: " + (", ".join(f"`{k}`={v}" for k, v in sorted(q_status.items())) or "none"),
        "",
        "The CSV files in this folder are filtered views of the canonical report. A row is suitable for a thesis claim only when the corresponding eligibility field is true and the validation tier is `final`.",
        "",
    ]
    if ranking:
        lines.extend(["## Ranking-method comparison", "", "| Method | Direction | Runner | n | Universe | Frozen | Spearman | Kendall | Hit@5 | Elite R@5 | Regret@5 | Status |", "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|"])
        for row in ranking:
            lines.append("| " + " | ".join(str(row.get(k) or "") for k in ("method_label", "direction", "runner_regime", "paired_candidate_count", "candidate_universe_complete", "predictions_frozen", "spearman_rho", "kendall_tau_b", "hit_at_5", "elite_recall_at_5_q3", "regret_at_5", "status")) + " |")
        lines.append("")
    return "\n".join(lines)


def create_analysis_pack(
    run_dir: Path,
    out_zip: Optional[Path] = None,
    *,
    tool_version: str = "",
    materialize_missing_report: bool = False,
) -> Dict[str, Any]:
    original_run_dir = Path(run_dir).expanduser()
    if original_run_dir.is_symlink():
        raise RuntimeError(f"Analysis pack source must not be a symlink: {original_run_dir}")
    run_dir = original_run_dir.resolve(strict=True)
    reports = run_dir / "reports"
    scientific = reports / "scientific"
    run_id = run_dir.name or time.strftime("%Y%m%d_%H%M%S")
    if out_zip is None:
        configured = str(os.environ.get("ONNX_SPLITPOINT_EXPORT_DIR", "") or "").strip()
        export_root = Path(configured).expanduser() if configured else Path.home() / "Downloads"
        out_zip = export_root / f"{run_id}_analysis_pack.zip"
    else:
        out_zip = Path(out_zip)
    from ..filesystem_admission import (
        require_output_outside_source,
        require_write_target,
    )
    from .run_discovery import inspect_evaluation_run

    require_output_outside_source(
        run_dir,
        out_zip,
        operation="Analysis pack export",
    )
    if not materialize_missing_report:
        inspection = inspect_evaluation_run(run_dir)
        if not inspection.analysis_ready:
            raise RuntimeError(
                "The selected EvaluationRun is not eligible for a read-only "
                "analysis pack: " + ",".join(inspection.reason_codes)
            )

    require_write_target(
        out_zip.parent,
        operation="Analysis pack export",
        minimum_free_bytes=16 * 1024 * 1024,
        minimum_free_inodes=16,
    )
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary_out_zip = temporary_zip_path(out_zip)

    # Reporting is part of the workflow.  For older or manually copied runs,
    # materialise the canonical report before creating the pack instead of
    # falling back to the removed claim-table files.
    if not (scientific / "scientific_report.json").is_file():
        if not materialize_missing_report:
            raise RuntimeError(
                "The selected EvaluationRun has no canonical scientific report. "
                "Read-only pack mode will not create or delete files inside the "
                f"source run: {run_dir}"
            )
        try:
            from .scientific_reporting import build_scientific_reports
            build_scientific_reports(run_dir, tool_version=tool_version, cleanup_legacy=True)
        except Exception as exc:
            raise RuntimeError(f"Could not create canonical scientific report for {run_dir}: {exc}") from exc

    report_payload: Dict[str, Any] = {}
    try:
        report_path = require_safe_pack_source(
            scientific / "scientific_report.json", run_dir
        )
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        report_payload = {}

    eligibility_rows = _read_csv(scientific / "row_eligibility.csv", allowed_root=run_dir)
    quality_rows = _read_csv(scientific / "task_quality.csv", allowed_root=run_dir)
    performance_rows = _read_csv(scientific / "performance_results.csv", allowed_root=run_dir)
    performance_observation_rows = _read_csv(
        scientific / "performance_observations.csv", allowed_root=run_dir,
    )
    energy_rows = _read_csv(scientific / "energy_results.csv", allowed_root=run_dir)
    native_performance_rows = _read_csv(scientific / "native_performance_observations.csv", allowed_root=run_dir)
    native_ab_rows = _read_csv(scientific / "native_energy_ab_aggregates.csv", allowed_root=run_dir)
    ranking_rows = _read_csv(scientific / "ranking_method_comparison.csv", allowed_root=run_dir)
    if not materialize_missing_report and (
        not eligibility_rows or not quality_rows
    ):
        raise RuntimeError(
            "The selected EvaluationRun has no complete canonical row "
            "eligibility/task-quality tables; no analysis ZIP was opened."
        )
    model_root = run_dir / "models"
    model_id_set = {str(r.get("model_id") or "") for r in eligibility_rows if str(r.get("model_id") or "")}
    # A prospective hold-out can pause after freeze/approval but before any
    # report row exists. Include model directories and enabled profile entries
    # so that this pre-execution evidence is not silently omitted.
    safe_model_root: Path | None = None
    if model_root.exists() or model_root.is_symlink():
        try:
            safe_model_root = require_safe_pack_directory(model_root, run_dir)
        except (OSError, RuntimeError, ValueError):
            safe_model_root = None
    if safe_model_root is not None:
        for path in safe_model_root.iterdir():
            if not path.name or path.is_symlink():
                continue
            try:
                require_safe_pack_directory(path, run_dir)
            except (OSError, RuntimeError, ValueError):
                continue
            model_id_set.add(path.name)
    profile_path = run_dir / "profile.yaml"
    if profile_path.exists() or profile_path.is_symlink():
        try:
            safe_profile_path = require_safe_pack_source(profile_path, run_dir)
            profile_payload = yaml.safe_load(safe_profile_path.read_text(encoding="utf-8")) or {}
            suite = profile_payload.get("model_suite") if isinstance(profile_payload, Mapping) and isinstance(profile_payload.get("model_suite"), Mapping) else {}
            for tier in ("primary", "reserve"):
                for item in list(suite.get(tier) or []):
                    if not isinstance(item, Mapping) or item.get("enabled") is False:
                        continue
                    model_id = str(item.get("id") or "").strip()
                    if model_id:
                        model_id_set.add(model_id)
        except Exception as exc:
            raise RuntimeError(
                "Analysis pack refused unsafe or unreadable profile.yaml"
            ) from exc
    model_ids = sorted(model_id_set)
    model_archive_names: Dict[str, str] = {}
    archive_name_owners: Dict[str, str] = {}
    for model_id in model_ids:
        archive_name = _safe_name(model_id)
        previous_owner = archive_name_owners.get(archive_name)
        if previous_owner is not None and previous_owner != model_id:
            raise RuntimeError(
                "Analysis pack model-id archive name collision: "
                f"{previous_owner!r} and {model_id!r} both map to "
                f"{archive_name!r}."
            )
        archive_name_owners[archive_name] = model_id
        model_archive_names[model_id] = archive_name

    run_manifest_payload: Dict[str, Any] = {}
    try:
        run_manifest_payload = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except Exception:
        run_manifest_payload = {}
    source_identity = {
        "schema": "onnx-splitpoint/pack-source-identity",
        "schema_version": 1,
        "run_id": str(run_manifest_payload.get("run_id") or run_id),
        "run_dir": str(run_dir),
        "tool_version_recorded": str(run_manifest_payload.get("tool_version") or ""),
        "workflow_version_recorded": str(run_manifest_payload.get("workflow_version") or ""),
        "run_status_recorded": str(run_manifest_payload.get("status") or ""),
        "pack_tool_version": str(tool_version or ""),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    written: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    readme = f"""# ONNX Splitpoint Scientific Analysis Pack

Run: `{run_id}`

This bundle contains the current versioned canonical reporting artefacts used for thesis writing.

- `00_overview/`: human- and machine-readable scientific report.
- `01_results/`: canonical row eligibility, task quality, performance, energy, ranking-method and Generic-to-Native transfer tables.
- `02_thesis_tables/`: LaTeX tables generated from the same canonical rows.
- `03_figures/`: thesis-ready PDF and PNG figures.
- `04_model_details/`: per-model filtered tables plus prediction/model manifests.
- `99_provenance/`: profile, run/report manifests and compact log tails.
- `99_provenance/host_telemetry/`: Native-host power mode, configured clocks,
  governors and thermal state captured immediately before and after execution.

The former `claim_table_*`, `thesis_metrics.*`, per-tag benchmark summaries and overlapping dashboard tables are intentionally absent. Raw traces and large vendor artefacts stay in the EvaluationRun/debug pack.

A final claim requires a `final` task-quality tier, a passed non-inferiority gate and the relevant eligibility flag. Ranking/Top-k metrics are only claimable for an auditable, score-independent candidate universe with predictions frozen before measurement.
Native energy observations and Split/Full pairs are packaged explicitly. Pair deltas are emitted only when semantic, task, model, preprocessing/decode, setup, physical-scope, calibrated-window and exact-work-unit contracts match.
The complete setup-specific Native performance matrix is retained as development/screening evidence. Measured FPS is separate from theoretical cycle-model rates. Compact A/B repeat aggregates include valid/requested n, means, confidence intervals and completeness status; raw captures remain outside this pack.
"""

    with _temporary_analysis_zip(temporary_out_zip) as zf:
        _write_text(zf, "README.md", readme, written)
        _write_text(zf, "00_overview/pack_source_identity.json", json.dumps(source_identity, indent=2, ensure_ascii=False) + "\n", written)
        for name in ("scientific_report.md", "scientific_report.json", "report_manifest.json", "report_cleanup.json"):
            _write_file(zf, scientific / name, f"00_overview/{name}", written, skipped, allowed_root=run_dir)
        for name in CANONICAL_RESULT_FILES:
            _write_file(zf, scientific / name, f"01_results/{name}", written, skipped, allowed_root=run_dir)
        for name in CANONICAL_TABLE_FILES:
            _write_file(zf, scientific / "thesis_tables" / name, f"02_thesis_tables/{name}", written, skipped, allowed_root=run_dir)
        figures_dir = scientific / "figures"
        try:
            figure_files = iter_safe_pack_files(figures_dir, run_dir)
        except (OSError, RuntimeError, ValueError):
            figure_files = ()
            if figures_dir.exists() or figures_dir.is_symlink():
                skipped.append({
                    "path": "03_figures/",
                    "reason": "unsafe_optional_source_root",
                })
        for path in sorted(
            p for p in figure_files if p.suffix.lower() in {".pdf", ".png"}
        ):
            _write_file(zf, path, f"03_figures/{path.relative_to(figures_dir).as_posix()}", written, skipped, max_bytes=12 * 1024 * 1024, allowed_root=run_dir)

        # Per-model views are derived from the canonical CSVs, not from a second
        # claim-selection implementation.
        for model_id in model_ids:
            base = f"04_model_details/{model_archive_names[model_id]}"
            e_rows = [r for r in eligibility_rows if str(r.get("model_id") or "") == model_id]
            q_rows = [r for r in quality_rows if str(r.get("model_id") or "") == model_id]
            p_rows = [r for r in performance_rows if str(r.get("model_id") or "") == model_id]
            po_rows = [
                r for r in performance_observation_rows
                if str(r.get("model_id") or "") == model_id
            ]
            n_rows = [r for r in energy_rows if str(r.get("model_id") or "") == model_id]
            np_rows = [r for r in native_performance_rows if str(r.get("model_id") or r.get("model") or "") == model_id]
            ab_rows = [r for r in native_ab_rows if str(r.get("model_id") or r.get("model") or "") == model_id]
            r_rows = [r for r in ranking_rows if str(r.get("model_id") or "") == model_id]
            _write_text(zf, f"{base}/README.md", _model_summary(model_id, e_rows, q_rows, r_rows), written)
            for filename, rows in (
                ("row_eligibility.csv", e_rows),
                ("task_quality.csv", q_rows),
                ("performance_results.csv", p_rows),
                ("performance_observations.csv", po_rows),
                ("energy_results.csv", n_rows),
                ("native_performance_observations.csv", np_rows),
                ("native_energy_ab_aggregates.csv", ab_rows),
                ("ranking_method_comparison.csv", r_rows),
            ):
                _write_text(zf, f"{base}/{filename}", _csv_text(rows), written)
            model_dir = model_root / model_id
            try:
                safe_model_dir = require_safe_pack_directory(model_dir, run_dir)
            except (OSError, RuntimeError, ValueError):
                safe_model_dir = None
            for rel in (
                "model_manifest.json",
                "analysis/prediction.json",
                "analysis/predictions_frozen.csv",
                "analysis/holdout_predictions_frozen.csv",
                "analysis/ranking_predictions_frozen.csv",
                "analysis/holdout_ranking_predictions_frozen.csv",
                "analysis/prediction_freeze_manifest.json",
                "analysis/prediction_freeze_conflict.json",
                "analysis/prediction_freeze_approval.json",
                "analysis/prediction_freeze_approval_verification.json",
                "analysis/prediction_freeze_approval_request.md",
                "analysis/candidate_universe_manifest.json",
                "analysis/candidate_universe.csv",
                "analysis/audit_plan.json",
                "analysis/final_candidate_plan.json",
                "analysis/model_facts.json",
            ):
                if safe_model_dir is not None:
                    _write_file(zf, safe_model_dir / rel, f"{base}/{rel}", written, skipped, allowed_root=run_dir)

        for name in (
            "run_manifest.json",
            "profile.yaml",
            "profile_source.yaml",
            "profile_start_snapshot.json",
            "profile_resolution.json",
            "effective_execution_plan.json",
            "artifact_index.json",
        ):
            _write_file(zf, run_dir / name, f"99_provenance/{name}", written, skipped, allowed_root=run_dir)
        for name in (
            "results_bundle_manifest.json", "run_status_summary.json",
            "native_evidence_status.json", "energy_status.json",
        ):
            _write_file(zf, reports / name, f"99_provenance/{name}", written, skipped, allowed_root=run_dir)
        for name in (
            "native_expected_matrix.json",
            "native_expected_matrix_missing.csv",
            "native_stage_concise_summary.json",
            "native_stage_concise_summary.csv",
            "native_host_telemetry_summary.json",
            "native_energy_model_hash_map.json",
        ):
            _write_file(zf, reports / name, f"99_provenance/reports/{name}", written, skipped, allowed_root=run_dir)
        for energy_plan_root in (
            reports / "native_energy_plan",
            reports / "native_energy_measurements" / "plan",
        ):
            for name in (
                "native_producer_energy_plan.json",
                "native_producer_energy_plan.md",
            ):
                path = energy_plan_root / name
                relative = path.relative_to(run_dir).as_posix()
                _write_file(
                    zf,
                    path,
                    f"99_provenance/{relative}",
                    written,
                    skipped,
                    allowed_root=run_dir,
                )
        native_root = run_dir / "native_producers"
        try:
            native_files = iter_safe_pack_files(native_root, run_dir)
        except (OSError, RuntimeError, ValueError):
            native_files = ()
            if native_root.exists() or native_root.is_symlink():
                skipped.append({
                    "path": "99_provenance/host_telemetry/",
                    "reason": "unsafe_optional_source_root",
                })
        for path in sorted(
            p
            for p in native_files
            if p.suffix.lower() == ".json" and p.parent.name == "host_telemetry"
        ):
            relative = path.relative_to(native_root)
            _write_file(
                zf,
                path,
                f"99_provenance/host_telemetry/{relative.parts[0]}/{path.name}",
                written,
                skipped,
                allowed_root=run_dir,
            )
        for name in ("central_quality_summary.json", "central_quality_summary.csv"):
            _write_file(
                zf,
                run_dir / "quality_management" / name,
                f"99_provenance/quality_management/{name}",
                written,
                skipped,
                allowed_root=run_dir,
            )
        stages_root = run_dir / "stages"
        try:
            stage_files = iter_safe_pack_files(stages_root, run_dir)
        except (OSError, RuntimeError, ValueError):
            stage_files = ()
            if stages_root.exists() or stages_root.is_symlink():
                skipped.append({
                    "path": "99_provenance/stages/",
                    "reason": "unsafe_optional_source_root",
                })
        for path in sorted(
            p for p in stage_files if p.name == "stage_result.json"
        ):
            _write_file(
                zf,
                path,
                f"99_provenance/{path.relative_to(run_dir).as_posix()}",
                written,
                skipped,
                allowed_root=run_dir,
            )
        _write_tail(zf, run_dir / "evaluation_workflow.log", "99_provenance/evaluation_workflow_tail.log", written, skipped, allowed_root=run_dir)

        manifest = {
            "schema": "onnx-splitpoint/scientific-analysis-pack-manifest",
            "schema_version": 2,
            "source_admission_mode": (
                "explicit_materialized_legacy_compatibility"
                if materialize_missing_report
                else "read_only_completed_measurement_contract"
            ),
            "tool_version": tool_version,
            "run_id": run_id,
            "source_identity": source_identity,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "scientific_report_schema": report_payload.get("schema"),
            "model_count": len(model_ids),
            "row_count": len(eligibility_rows),
            "task_quality_row_count": len(quality_rows),
            "performance_observation_count": len(
                performance_observation_rows
            ),
            "ranking_method_group_count": len(ranking_rows),
            "native_performance_observation_count": len(native_performance_rows),
            "native_energy_ab_aggregate_count": len(native_ab_rows),
            "file_count": len(written),
            "skipped_count": len(skipped),
            "files": written,
            "skipped": skipped[:300],
        }
        _write_text(zf, "analysis_pack_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", written)

    strict_required_archive_members = {
        "analysis_pack_manifest.json",
        "00_overview/scientific_report.json",
        "00_overview/report_manifest.json",
        "01_results/row_eligibility.csv",
        "01_results/task_quality.csv",
        "01_results/performance_results.csv",
        "01_results/energy_results.csv",
        "99_provenance/run_manifest.json",
        "99_provenance/results_bundle_manifest.json",
        "99_provenance/run_status_summary.json",
    }
    required_archive_members = (
        {
            "analysis_pack_manifest.json",
            "00_overview/scientific_report.json",
            "99_provenance/run_manifest.json",
        }
        if materialize_missing_report
        else strict_required_archive_members
    )
    try:
        try:
            verify_zip_archive(
                temporary_out_zip,
                required_members=required_archive_members,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Analysis pack verification failed for {temporary_out_zip}: {exc}"
            ) from exc
        os.replace(temporary_out_zip, out_zip)
    finally:
        temporary_out_zip.unlink(missing_ok=True)

    return {
        "zip_path": str(out_zip),
        "file_count": len(written),
        "skipped_count": len(skipped),
        "model_count": len(model_ids),
        "row_count": len(eligibility_rows),
        "ranking_method_group_count": len(ranking_rows),
    }
