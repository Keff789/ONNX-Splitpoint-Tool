#!/usr/bin/env python3
from __future__ import annotations

"""Rebuild Native supplement reports locally without rerunning hardware.

The source EvaluationRun is read-only.  A new sibling projection root is
published atomically only after the 24-row Native report, exact Central Quality
binding and the final claim projection have all completed.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onnx_splitpoint_tool.validation.accuracy_gates import (  # noqa: E402
    AccuracyGatePolicy,
)


FROZEN_V27549_DECISIONS = {
    "pass": 9,
    "fail": 11,
    "inconclusive": 4,
}

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read the EvalRun profile")
        value = yaml.safe_load(text) or {}
    else:
        value = json.loads(text)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Expected a mapping in {path}")
    return dict(value)


def _stable_performance_fingerprint(
    row: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Identity plus hardware observations that report replay may not change."""

    def text(value: Any) -> str:
        return str(value or "").strip().lower()

    def number(value: Any) -> float | None:
        try:
            return float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    def integer(value: Any) -> int | None:
        try:
            return int(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    return (
        text(row.get("backend")),
        text(row.get("model") or row.get("model_id")),
        text(row.get("case") or row.get("case_id")),
        text(row.get("precision")),
        text(row.get("setup_id")),
        text(row.get("comparison_backend")),
        row.get("ok") is True,
        number(row.get("fps_makespan")),
        number(row.get("fps_median")),
        integer(row.get("repetition_count_requested")),
        integer(row.get("repetition_count_attempted")),
        integer(row.get("repetition_count_valid")),
        text(row.get("repetition_status")),
        text(row.get("repetition_aggregation")),
        text(row.get("repetition_claim_identity_status")),
    )


def _stable_performance_multiset(
    rows: list[Mapping[str, Any]],
) -> Counter[tuple[Any, ...]]:
    return Counter(_stable_performance_fingerprint(row) for row in rows)


def _run(command: list[str], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): "
            f"{' '.join(command)}\n{completed.stderr[-4000:]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except Exception:
        payload = {"stdout": completed.stdout[-4000:]}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _copy_if_file(source: Path, destination: Path) -> None:
    if source.is_file() and not source.is_symlink():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _rebase_published_text_paths(
    root: Path,
    *,
    staging_root: Path,
    published_root: Path,
) -> None:
    """Replace staging-root paths before the atomic directory publish.

    The Native validator intentionally emits absolute artifact paths.  During
    an offline replay those paths initially point below the temporary staging
    directory.  ``os.replace`` publishes the whole directory atomically, but
    cannot update strings already embedded in JSON/CSV/Markdown reports.  Keep
    the existing absolute-path contract and rebase only this replay's staging
    prefix to the final projection root.
    """

    staging_prefix = str(staging_root)
    published_prefix = str(published_root)
    text_suffixes = {".json", ".csv", ".md"}
    text_artifacts = sorted(
        path for path in root.rglob("*")
        if (
            path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in text_suffixes
        )
    )
    for path in text_artifacts:
        content = path.read_text(encoding="utf-8")
        if staging_prefix not in content:
            continue
        path.write_text(
            content.replace(staging_prefix, published_prefix),
            encoding="utf-8",
        )

    remaining = [
        str(path.relative_to(root))
        for path in text_artifacts
        if staging_prefix in path.read_text(encoding="utf-8")
    ]
    if remaining:
        raise RuntimeError(
            "Native replay output still references its temporary staging "
            f"directory: {remaining[:10]}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", required=True, type=Path,
        help="Completed Native supplement EvaluationRun; used read-only.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "New projection root. Defaults to a sibling named "
            "<run>_native_reprojected_v276."
        ),
    )
    parser.add_argument(
        "--expected-rows", type=int, default=24,
        help="Frozen Native supplement denominator (default: 24).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_arg = args.run_dir.expanduser()
    if source_arg.is_symlink():
        raise SystemExit("Native replay source must not be a symlink")
    source = source_arg.resolve(strict=True)
    if not source.is_dir():
        raise SystemExit(f"Native replay source is not a directory: {source}")
    destination = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else source.parent / f"{source.name}_native_reprojected_v276"
    )
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        raise SystemExit("Native replay output must be outside the source run")
    if destination.exists() or destination.is_symlink():
        raise SystemExit(
            f"Native replay output already exists: {destination}"
        )

    profile_path = source / "profile.yaml"
    profile = _read_mapping(profile_path)
    quality_gate = profile.get("quality_gate")
    if not isinstance(quality_gate, Mapping) or not quality_gate:
        raise SystemExit("EvalRun profile has no authoritative quality_gate")
    central_quality = (
        source / "quality_management" / "central_quality_summary.json"
    )
    expected_matrix = source / "reports" / "native_expected_matrix.json"
    stage_path = source / "reports" / "native_producer_stage.json"
    native_root = source / "native_producers"
    roots = sorted(
        path for path in native_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    ) if native_root.is_dir() else []
    required = [
        source / "run_manifest.json", profile_path, central_quality,
        expected_matrix, stage_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if not roots:
        missing.append(str(native_root / "<variant-root>"))
    if missing:
        raise SystemExit(
            "Native replay source is incomplete:\n- " + "\n- ".join(missing)
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.partial-",
        dir=str(destination.parent),
    ))
    try:
        reports = temporary / "reports"
        reports.mkdir(parents=True)
        _copy_if_file(source / "run_manifest.json", temporary / "run_manifest.json")
        _copy_if_file(profile_path, temporary / "profile.yaml")
        _copy_if_file(stage_path, reports / "native_producer_stage.json")
        _copy_if_file(
            source / "reports" / "native_producer_stage_config.json",
            reports / "native_producer_stage_config.json",
        )
        _copy_if_file(expected_matrix, reports / "native_expected_matrix.json")

        source_summary = source / "reports" / "native_producer_summary.json"
        if not source_summary.is_file():
            source_summary = (
                source / "reports" / "native_producer_combined_summary.json"
            )
        if not source_summary.is_file():
            raise RuntimeError(
                "Native replay source has no preserved performance summary"
            )
        # The coordinator already aggregated and committed all hardware rows
        # before its later checkpoint/reporting failure.  That committed
        # summary is the replay input; rebuilding it from optional child
        # analysis sidecars can silently lose backends whose compact pack no
        # longer contains those redundant sidecars.
        source_performance = _read_mapping(source_summary)
        source_native_rows = [
            row for row in list(source_performance.get("rows") or [])
            if isinstance(row, Mapping)
        ]
        expected_rows = int(args.expected_rows)
        matrix = _read_mapping(expected_matrix)
        if not (
            int(matrix.get("expected_row_count") or -1) == expected_rows
            and int(matrix.get("present_expected_row_count") or -1)
            == expected_rows
            and int(matrix.get("successful_expected_row_count") or -1)
            == expected_rows
            and matrix.get("row_presence_complete") is True
            and matrix.get("execution_success_complete") is True
        ):
            raise RuntimeError(
                "Native expected matrix does not attest the complete bounded "
                f"supplement denominator ({expected_rows})"
            )
        if len(source_native_rows) != expected_rows:
            raise RuntimeError(
                f"Native replay row count is {len(source_native_rows)}, "
                f"expected {expected_rows}"
            )
        if any(row.get("ok") is not True for row in source_native_rows):
            raise RuntimeError(
                "At least one preserved Native performance row is not ok"
            )
        for extension in ("json", "csv", "md"):
            source_file = source_summary.with_suffix(f".{extension}")
            if not source_file.is_file():
                source_file = (
                    source / "reports"
                    / f"native_producer_summary.{extension}"
                )
            if source_file.is_file():
                shutil.copy2(
                    source_file,
                    reports / f"native_producer_summary.{extension}",
                )
                shutil.copy2(
                    source_file,
                    reports / f"native_producer_combined_summary.{extension}",
                )

        validation_dir = reports / "native_validation"
        validation_command = [
            sys.executable, "-B",
            str(PROJECT_ROOT / "scripts" / "native_producer_validate_visualize.py"),
            # Keep source EvaluationRun context for model/reference lookup while
            # writing every derived artifact exclusively below ``temporary``.
            "--summary", str(source_summary),
            "--out-dir", str(validation_dir),
            "--quality-gate-json", json.dumps(
                dict(quality_gate), sort_keys=True, separators=(",", ":"),
            ),
            "--central-quality-summary", str(central_quality),
        ]
        for root in roots:
            validation_command.extend(["--root", str(root)])
        validation_result = _run(validation_command, cwd=PROJECT_ROOT)
        quality_summary = (
            validation_dir / "native_producer_validation_summary.json"
        )
        if not quality_summary.is_file():
            raise RuntimeError("Native validation replay produced no summary")

        native_payload = _read_mapping(
            reports / "native_producer_summary.json"
        )
        validation_payload = _read_mapping(quality_summary)
        native_rows = [
            row for row in list(native_payload.get("rows") or [])
            if isinstance(row, Mapping)
        ]
        validation_rows = [
            row for row in list(validation_payload.get("rows") or [])
            if isinstance(row, Mapping)
        ]
        if len(native_rows) != expected_rows:
            raise RuntimeError(
                f"Native replay row count is {len(native_rows)}, expected "
                f"{expected_rows}"
            )
        if any(row.get("ok") is not True for row in native_rows):
            raise RuntimeError("At least one preserved Native performance row is not ok")
        if len(validation_rows) != expected_rows:
            raise RuntimeError(
                "Native validation replay did not retain every performance row"
            )
        exact_quality = sum(
            row.get("central_quality_evidence_verified") is True
            for row in validation_rows
        )
        expected_policy_sha = AccuracyGatePolicy.from_mapping(
            dict(quality_gate)
        ).sha256()
        invalid_quality_rows = [
            {
                "backend": row.get("backend"),
                "model": row.get("model"),
                "case": row.get("case"),
                "binding_status": row.get(
                    "central_quality_binding_status"
                ),
            }
            for row in validation_rows
            if not (
                row.get("central_quality_binding_status")
                == "exact_identity_match"
                and row.get("central_quality_evidence_verified") is True
                and row.get("precision_quality_binding_verified") is True
                and str(row.get("accuracy_gate_policy_sha256") or "")
                == expected_policy_sha
                and str(row.get("task_quality_policy_sha256") or "")
                == expected_policy_sha
            )
        ]
        if exact_quality != expected_rows or invalid_quality_rows:
            raise RuntimeError(
                "Central Quality replay did not produce one exact policy-bound "
                f"match for every row: exact={exact_quality}/{expected_rows}; "
                f"invalid={invalid_quality_rows[:5]}"
            )
        decisions = {"pass": 0, "fail": 0, "inconclusive": 0}
        for row in validation_rows:
            status = str(
                row.get("task_quality_status")
                or (row.get("task_quality_gate") or {}).get("decision")
                or ""
            ).strip().lower()
            if status in decisions:
                decisions[status] += 1
        decision_count = sum(decisions.values())
        if decision_count != expected_rows:
            raise RuntimeError(
                "Central Quality replay did not retain one terminal decision "
                f"per row: decisions={decision_count}/{expected_rows}; "
                f"distribution={decisions}"
            )
        if (
            expected_rows == 24
            and decisions != FROZEN_V27549_DECISIONS
        ):
            raise RuntimeError(
                "Central Quality replay drifted from the frozen v2.75.49 "
                f"supplement decisions: {decisions} != "
                f"{FROZEN_V27549_DECISIONS}"
            )

        # Rebuild the canonical quality-gated performance summary just as the
        # normal coordinator does after Native validation.  Earlier replay
        # releases copied the ungated performance summary to both canonical
        # names and published the exact validation rows only as a sidecar.
        # Cross-runner reporting could read that sidecar, but downstream
        # consumers of ``native_producer_combined_summary.json`` saw
        # ``quality_evidence=not_provided`` and 24 missing bindings.  The
        # canonical final reporter performs the existing exact, fail-closed
        # identity/provenance join without touching any source artifact.
        final_report_command = [
            sys.executable, "-B",
            str(PROJECT_ROOT / "scripts" / "native_producer_final_report.py"),
        ]
        for root in roots:
            final_report_command.extend(["--root", str(root)])
        final_report_command.extend([
            "--recursive",
            "--out-dir", str(reports),
            "--quality-summary", str(quality_summary),
            "--expected-matrix", str(
                reports / "native_expected_matrix.json"
            ),
        ])
        final_report_result = _run(
            final_report_command, cwd=PROJECT_ROOT,
        )
        combined_summary = (
            reports / "native_producer_combined_summary.json"
        )
        if not combined_summary.is_file():
            raise RuntimeError(
                "Quality-gated Native replay produced no combined summary"
            )
        combined_payload = _read_mapping(combined_summary)
        combined_rows = [
            row for row in list(combined_payload.get("rows") or [])
            if isinstance(row, Mapping)
        ]
        quality_evidence = combined_payload.get("quality_evidence")
        if not isinstance(quality_evidence, Mapping):
            quality_evidence = {}
        preserved_performance = _stable_performance_multiset(native_rows)
        rebuilt_performance = _stable_performance_multiset(combined_rows)
        if rebuilt_performance != preserved_performance:
            source_only = list(
                (preserved_performance - rebuilt_performance).elements()
            )[:3]
            rebuilt_only = list(
                (rebuilt_performance - preserved_performance).elements()
            )[:3]
            raise RuntimeError(
                "Quality-gated Native replay changed preserved performance "
                "identity or measurements: "
                f"source_only={source_only}; rebuilt_only={rebuilt_only}"
            )
        if (
            len(combined_rows) != expected_rows
            or any(row.get("ok") is not True for row in combined_rows)
            or quality_evidence.get("status") != "loaded"
            or quality_evidence.get("schema_valid") is not True
            or int(quality_evidence.get("quality_row_count", -1))
            != expected_rows
            or int(quality_evidence.get("unique_identity_count", -1))
            != expected_rows
            or int(quality_evidence.get("duplicate_identity_count", -1))
            != 0
            or int(
                quality_evidence.get(
                    "verified_performance_row_count", -1,
                )
            ) != expected_rows
            or int(
                quality_evidence.get(
                    "missing_performance_row_count", -1,
                )
            ) != 0
            or int(
                quality_evidence.get(
                    "failed_performance_row_count", -1,
                )
            ) != 0
            or int(
                quality_evidence.get(
                    "ambiguous_performance_row_count", -1,
                )
            ) != 0
        ):
            raise RuntimeError(
                "Quality-gated Native replay did not retain one exact "
                "validation binding for every preserved performance row: "
                f"rows={len(combined_rows)}/{expected_rows}; "
                f"quality_evidence={dict(quality_evidence)}"
            )
        for extension in ("json", "csv", "md"):
            source_file = (
                reports / f"native_producer_combined_summary.{extension}"
            )
            if source_file.is_file():
                shutil.copy2(
                    source_file,
                    reports / f"native_producer_summary.{extension}",
                )
        native_rows = combined_rows

        technical_error_count = sum(
            bool(row.get("error"))
            or str(row.get("status") or "").strip().lower()
            in {"error", "diagnostic_technical_error", "missing_dump"}
            or row.get("runtime_executable") is False
            or row.get("tensor_ok") is False
            for row in validation_rows
        )
        claim_eligible_count = sum(
            row.get("eligible_for_ranking") is True
            for row in validation_rows
        )
        status_payload = {
            "schema": "onnx-splitpoint/native-supplement-offline-replay",
            "schema_version": 1,
            "status": "ok",
            "source_run_dir": str(source),
            "source_mutated": False,
            "native_performance_row_count": len(native_rows),
            "native_performance_ok_count": sum(
                row.get("ok") is True for row in native_rows
            ),
            "native_validation_row_count": len(validation_rows),
            "central_quality_exact_binding_count": exact_quality,
            "central_quality_decisions": decisions,
            "central_quality_decision_count": decision_count,
            "technical_error_count": technical_error_count,
            "claim_eligible_count": claim_eligible_count,
            "scientific_pass": bool(
                technical_error_count == 0
                and decisions["fail"] == 0
                and decisions["inconclusive"] == 0
            ),
            "scientific_pass_note": (
                "Offline projection completed successfully; scientific_pass "
                "is evaluated separately from replay success."
            ),
            "validation": validation_result,
            "quality_gated_final_report": final_report_result,
            "source_performance_summary": str(source_summary),
        }
        (reports / "native_replay_status.json").write_text(
            json.dumps(status_payload, indent=2), encoding="utf-8"
        )
        _rebase_published_text_paths(
            temporary,
            staging_root=temporary,
            published_root=destination,
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps({
        "ok": True,
        "status": "ok",
        "source_mutated": False,
        "output_dir": str(destination),
        "native_performance_row_count": expected_rows,
        "central_quality_exact_binding_count": exact_quality,
        "central_quality_decisions": decisions,
        "technical_error_count": technical_error_count,
        "claim_eligible_count": claim_eligible_count,
        "scientific_pass": bool(
            technical_error_count == 0
            and decisions["fail"] == 0
            and decisions["inconclusive"] == 0
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
