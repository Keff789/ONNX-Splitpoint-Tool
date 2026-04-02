from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_BOUNDARY_DIR_RE = re.compile(r"^b(\d+)$")



def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default



def _boundary_from_folder(name: str) -> Optional[int]:
    m = _BOUNDARY_DIR_RE.match(str(name).strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None



def _ensure_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []



def _rel_path(path: Path, start: Path) -> str:
    try:
        return Path(path.relative_to(start)).as_posix()
    except Exception:
        try:
            return Path(path).as_posix()
        except Exception:
            return str(path)



def _extract_model_path(manifest: Dict[str, Any], key: str) -> str:
    if key in manifest and manifest.get(key):
        return str(manifest.get(key) or "")
    models = manifest.get("models") if isinstance(manifest.get("models"), dict) else {}
    entry = models.get(key) if isinstance(models, dict) else None
    if isinstance(entry, dict) and entry.get("path"):
        return str(entry.get("path") or "")
    return ""



def _inspect_case_dir(case_dir: Path, suite_dir: Path) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    manifest_path = case_dir / "split_manifest.json"
    if not manifest_path.exists():
        return None, [f"{case_dir.name}: missing split_manifest.json"]

    manifest = _read_json(manifest_path, default={}) or {}
    if not isinstance(manifest, dict):
        return None, [f"{case_dir.name}: invalid split_manifest.json"]

    boundary = manifest.get("boundary")
    try:
        boundary = int(boundary)
    except Exception:
        boundary = _boundary_from_folder(case_dir.name)
    if boundary is None:
        return None, [f"{case_dir.name}: could not determine boundary"]

    part1_rel = _extract_model_path(manifest, "part1")
    part2_rel = _extract_model_path(manifest, "part2")
    missing: List[str] = []
    for rel in (part1_rel, part2_rel):
        if not rel:
            missing.append("model path missing in manifest")
            continue
        model_path = case_dir / rel
        if not model_path.exists():
            missing.append(rel)
    if missing:
        return None, [f"b{boundary}: incomplete case artifacts ({', '.join(missing)})"]

    entry: Dict[str, Any] = {
        "boundary": int(boundary),
        "folder": case_dir.name,
        "case_dir": case_dir.name,
        "manifest": manifest_path.name,
        "predicted": manifest.get("predicted") if isinstance(manifest.get("predicted"), dict) else {},
    }
    if manifest.get("hailo_parse_check") is not None:
        entry["hailo_parse_check"] = manifest.get("hailo_parse_check")
    for key in (
        "hailo_parse_ok",
        "hailo_parse_symbol",
        "hailo_parse_message",
        "hailo_parse_target",
        "hailo_parse_backend",
        "hailo_parse_cache_hit",
    ):
        if key in manifest:
            entry[key] = manifest.get(key)
    return entry, warnings



def _inspect_rejected_dir(case_dir: Path, suite_dir: Path) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    manifest_path = case_dir / "split_manifest.json"
    manifest = _read_json(manifest_path, default={}) if manifest_path.exists() else {}
    rejection = {}
    if isinstance(manifest, dict):
        rejection = manifest.get("benchmark_rejection") if isinstance(manifest.get("benchmark_rejection"), dict) else {}

    boundary = None
    if isinstance(rejection, dict):
        try:
            boundary = int(rejection.get("boundary"))
        except Exception:
            boundary = None
    if boundary is None:
        boundary = _boundary_from_folder(case_dir.name)
    if boundary is None:
        return None, [f"{case_dir.name}: could not determine rejected boundary"]

    record: Dict[str, Any] = {
        "status": "rejected",
        "boundary": int(boundary),
        "folder": case_dir.name,
        "archive_dir": _rel_path(case_dir, suite_dir),
    }
    if isinstance(rejection, dict):
        record.update({k: v for k, v in rejection.items() if v not in (None, "")})
    if manifest_path.exists():
        record.setdefault("manifest", manifest_path.name)
    return record, warnings



def _canonical_case_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(entry)
    try:
        out["boundary"] = int(out.get("boundary"))
    except Exception:
        pass
    folder = str(out.get("folder") or out.get("case_dir") or "").strip()
    if folder:
        out["folder"] = folder
        out["case_dir"] = folder
    manifest = str(out.get("manifest") or "split_manifest.json").strip() or "split_manifest.json"
    out["manifest"] = manifest
    return out



def _load_benchmark_fallbacks(out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    bench = _read_json(out_dir / "benchmark_set.json", default={}) or {}
    if not isinstance(bench, dict):
        return [], []
    cases = [dict(x) for x in _ensure_list(bench.get("cases")) if isinstance(x, dict)]
    discards = [dict(x) for x in _ensure_list(bench.get("discarded_cases")) if isinstance(x, dict)]
    return cases, discards


@dataclass
class ResumeIntegrityReport:
    repaired_state: Dict[str, Any]
    warnings: List[str]
    inferred_cases: int = 0
    dropped_cases: int = 0
    inferred_rejections: int = 0
    dropped_rejections: int = 0

    @property
    def changed(self) -> bool:
        return any(
            int(getattr(self, attr, 0) or 0) > 0
            for attr in ("inferred_cases", "dropped_cases", "inferred_rejections", "dropped_rejections")
        )

    def summary(self) -> str:
        parts: List[str] = []
        if self.inferred_cases:
            parts.append(f"Recovered accepted cases from disk: {self.inferred_cases}")
        if self.dropped_cases:
            parts.append(f"Dropped stale accepted entries: {self.dropped_cases}")
        if self.inferred_rejections:
            parts.append(f"Recovered rejected cases from disk: {self.inferred_rejections}")
        if self.dropped_rejections:
            parts.append(f"Dropped stale rejected entries: {self.dropped_rejections}")
        if self.warnings:
            parts.append("")
            parts.append("Warnings:")
            parts.extend([f"  - {w}" for w in self.warnings[:10]])
            if len(self.warnings) > 10:
                parts.append(f"  ... and {len(self.warnings) - 10} more")
        return "\n".join(parts).strip()



def reconcile_generation_state(out_dir: Path, state: Optional[Dict[str, Any]]) -> ResumeIntegrityReport:
    out_dir = Path(out_dir)
    raw_state: Dict[str, Any] = dict(state or {}) if isinstance(state, dict) else {}
    warnings: List[str] = []

    fallback_cases, fallback_discards = _load_benchmark_fallbacks(out_dir)
    raw_cases = [dict(x) for x in _ensure_list(raw_state.get("case_entries") or raw_state.get("cases")) if isinstance(x, dict)]
    raw_discards = [dict(x) for x in _ensure_list(raw_state.get("discarded_case_entries") or raw_state.get("discarded_cases")) if isinstance(x, dict)]
    if not raw_cases and fallback_cases:
        raw_cases = fallback_cases
    if not raw_discards and fallback_discards:
        raw_discards = fallback_discards

    accepted: Dict[int, Dict[str, Any]] = {}
    dropped_cases = 0
    for entry in raw_cases:
        try:
            boundary = int(entry.get("boundary"))
        except Exception:
            dropped_cases += 1
            warnings.append("Accepted state entry without valid boundary was ignored")
            continue
        folder = str(entry.get("folder") or entry.get("case_dir") or f"b{boundary}").strip()
        case_dir = out_dir / folder
        inspected, case_warnings = _inspect_case_dir(case_dir, out_dir)
        warnings.extend(case_warnings)
        if inspected is None:
            dropped_cases += 1
            continue
        merged = _canonical_case_entry({**entry, **inspected})
        accepted[boundary] = merged

    inferred_cases = 0
    for child in sorted(out_dir.iterdir()) if out_dir.exists() else []:
        if not child.is_dir() or child.name == "_rejected_cases":
            continue
        boundary = _boundary_from_folder(child.name)
        if boundary is None or boundary in accepted:
            continue
        inspected, case_warnings = _inspect_case_dir(child, out_dir)
        warnings.extend(case_warnings)
        if inspected is None:
            continue
        accepted[boundary] = _canonical_case_entry(inspected)
        inferred_cases += 1

    rejected: Dict[int, Dict[str, Any]] = {}
    dropped_rejections = 0
    for entry in raw_discards:
        try:
            boundary = int(entry.get("boundary"))
        except Exception:
            dropped_rejections += 1
            warnings.append("Rejected state entry without valid boundary was ignored")
            continue
        folder = str(entry.get("folder") or entry.get("case_dir") or f"b{boundary}").strip()
        archive_dir = out_dir / "_rejected_cases" / folder
        live_dir = out_dir / folder
        manifest_backed = archive_dir if archive_dir.exists() else (live_dir if live_dir.exists() else None)
        if manifest_backed is not None:
            inspected, reject_warnings = _inspect_rejected_dir(manifest_backed, out_dir)
            warnings.extend(reject_warnings)
            if inspected is not None:
                merged = dict(entry)
                merged.update(inspected)
                rejected[boundary] = merged
                continue
        if folder:
            warnings.append(f"b{boundary}: rejected entry has no archived artifacts; keeping metadata only")
            merged = dict(entry)
            merged["folder"] = folder
            rejected[boundary] = merged
        else:
            dropped_rejections += 1

    inferred_rejections = 0
    reject_root = out_dir / "_rejected_cases"
    if reject_root.exists():
        for child in sorted(reject_root.iterdir()):
            if not child.is_dir():
                continue
            boundary = _boundary_from_folder(child.name)
            if boundary is None or boundary in rejected:
                continue
            inspected, reject_warnings = _inspect_rejected_dir(child, out_dir)
            warnings.extend(reject_warnings)
            if inspected is None:
                continue
            rejected[boundary] = dict(inspected)
            inferred_rejections += 1

    repaired = dict(raw_state)
    cases_list = [accepted[k] for k in sorted(accepted)]
    rejected_list = [rejected[k] for k in sorted(rejected)]
    accepted_boundaries = sorted(accepted)
    discarded_boundaries = sorted(rejected)
    completed = sorted(set(accepted_boundaries) | set(discarded_boundaries))
    requested = repaired.get("requested_cases")
    try:
        requested_i = int(requested)
    except Exception:
        requested_i = len(cases_list)

    repaired.update(
        {
            "case_entries": cases_list,
            "cases": cases_list,
            "discarded_case_entries": rejected_list,
            "discarded_cases": rejected_list,
            "accepted_boundaries": accepted_boundaries,
            "discarded_boundaries": discarded_boundaries,
            "completed_boundaries": completed,
            "generated_cases": len(cases_list),
            "discarded_cases_count": len(rejected_list),
            "shortfall": max(0, requested_i - len(cases_list)),
            "resume_integrity": {
                "warnings": list(warnings),
                "inferred_cases": int(inferred_cases),
                "dropped_cases": int(dropped_cases),
                "inferred_rejections": int(inferred_rejections),
                "dropped_rejections": int(dropped_rejections),
            },
        }
    )
    return ResumeIntegrityReport(
        repaired_state=repaired,
        warnings=warnings,
        inferred_cases=inferred_cases,
        dropped_cases=dropped_cases,
        inferred_rejections=inferred_rejections,
        dropped_rejections=dropped_rejections,
    )
