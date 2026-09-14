from __future__ import annotations

"""Integration entry points used by the profile resolver and scientific reporter."""

from pathlib import Path
from typing import Any, Mapping, MutableMapping
import json

from .native_full_quality import normalise_evaluation_profile
from .reporting_quality_decomposition import write_quality_decomposition
from .validation.official_coco_orchestrator import run_official_coco_for_run
from .native_energy_reporting import collect_native_energy, build_native_energy_pairs


def prepare_profile_for_execution(profile: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    return normalise_evaluation_profile(profile)


def augment_scientific_report(
    *,
    run_dir: str | Path,
    report_dir: str | Path,
    profile: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    run = Path(run_dir)
    report = Path(report_dir)
    official_cfg = profile.get("official_coco_evaluation") if isinstance(profile, Mapping) else {}
    if not isinstance(official_cfg, Mapping):
        official_cfg = {}
    enabled = bool(official_cfg.get("enabled", False))
    required = bool(official_cfg.get("required_for_final", False))
    annotations = official_cfg.get("annotations") or official_cfg.get("local_annotations")
    coco = {"status": "disabled"}
    if enabled and annotations:
        coco = run_official_coco_for_run(
            run, annotations=annotations, enabled=True, required=required,
            output_dir=report / "official_coco",
        )
    elif enabled:
        coco = {"status": "required_missing" if required else "unavailable", "reason": "annotations_not_configured"}
    quality = write_quality_decomposition(report, rows)
    native_rows = collect_native_energy(run)
    native_energy = {"row_count": len(native_rows), "pair_count": len(build_native_energy_pairs(native_rows))}
    summary = {"official_coco": coco, "quality_decomposition": quality, "native_energy": native_energy}
    (report / "quality_evidence_v61a.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (report / "quality_evidence_v60z.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
