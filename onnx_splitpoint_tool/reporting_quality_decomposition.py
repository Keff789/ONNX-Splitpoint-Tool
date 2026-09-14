from __future__ import annotations

from .quality_result_contract import UNCERTAINTY_FIELDS, project_quality_result, project_quality_component, project_flat_quality_uncertainty

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
import csv
import json
import math


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _num(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except Exception:
            continue
    return None


def _task(row: Mapping[str, Any]) -> str:
    value = _norm(row.get("task") or row.get("benchmark_task") or row.get("task_type"))
    if "detect" in value or value in {"coco", "object_detection"}:
        return "detection"
    if "class" in value or value in {"imagenet", "classification"}:
        return "classification"
    # Infer from available metrics.
    if any(k in row for k in ("ap", "ap50", "coco_ap", "metric_ap")):
        return "detection"
    return "classification"


def _variant(row: Mapping[str, Any]) -> str:
    value = _norm(row.get("variant") or row.get("row_variant") or row.get("execution_variant") or row.get("kind"))
    if "canonical" in value or ("onnx" in value and "full" in value):
        return "canonical_full_onnx"
    if value in {"full", "vendor_full", "native_full"} or value.endswith("_full"):
        return "vendor_full"
    if "split" in value or "composed" in value or "to_tensorrt" in value:
        return "split"
    # Common boolean fields.
    if row.get("is_reference"):
        return "canonical_full_onnx"
    return value or "unknown"


def _backend(row: Mapping[str, Any]) -> str:
    value = _norm(row.get("backend") or row.get("producer_backend") or row.get("backend_pair") or row.get("provider"))
    for backend in ("hailo10h", "hailo8", "deepx", "tensorrt", "ort_cpu", "ort_cuda"):
        if backend in value:
            return backend
    if "hailo10" in value:
        return "hailo10h"
    return value or "unknown"


def _contract_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(k) for k in (
        "dataset_manifest_sha256", "validation_dataset_manifest_sha256",
        "validation_dataset_sha256", "dataset_manifest", "dataset_id",
        "validation_dataset", "preprocessing_hash",
        "preprocessing_contract_hash", "preprocessing_contract_sha256",
        "decoder_hash", "decoder_contract_hash", "decoder_contract_sha256",
        "nms_hash", "nms_contract_sha256", "postprocessing_hash",
        "quality_policy_sha256", "task_quality_policy_sha256",
        "score_threshold", "nms_iou_threshold", "max_detections",
        "validation_item_ids_hash", "validation_count",
    ))


def _contracts_comparable(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    sa, sb = _contract_signature(a), _contract_signature(b)
    compared = False
    for x, y in zip(sa, sb):
        if x in (None, "") or y in (None, ""):
            continue
        compared = True
        if str(x) != str(y):
            return False
    # Missing metadata does not prove equality.  Runtime rows can explicitly
    # opt into comparability after the profile/manifest gate has checked it.
    explicit = a.get("contract_comparable") is True and b.get("contract_comparable") is True
    return compared or explicit


def _metrics(row: Mapping[str, Any]) -> dict[str, float | None]:
    if _task(row) == "detection":
        return {
            "AP": _num(row, "official_ap", "coco_ap", "ap", "primary_metric_value", "task_metric"),
            "AP50": _num(row, "official_ap50", "ap50", "guardrail_ap50"),
            "AP75": _num(row, "official_ap75", "ap75", "guardrail_ap75"),
        }
    return {
        "Top1": _num(row, "top1", "top1_accuracy", "accuracy_top1", "primary_metric_value", "task_metric"),
        "Top5": _num(row, "top5", "top5_accuracy", "accuracy_top5", "guardrail_top5"),
    }


def _setup(row: Mapping[str, Any]) -> str:
    return str(
        row.get("setup_id")
        or row.get("source_setup_id")
        or row.get("measurement_setup_id")
        or ""
    ).strip()


def _metric_label(value: Any, task: str) -> str:
    token = _norm(value)
    if "ap50" in token:
        return "AP50"
    if "ap75" in token:
        return "AP75"
    if "top1" in token:
        return "Top1"
    if "top5" in token:
        return "Top5"
    if task == "detection" and token in {
        "ap", "map", "coco_ap", "official_ap", "primary",
    }:
        return "AP"
    if task == "classification" and token in {
        "accuracy", "primary", "primary_metric",
    }:
        return "Top1"
    return token.upper() if token else ("AP" if task == "detection" else "Top1")


def _paired_quality_components(
    row: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Read paired candidate/reference components without deriving evidence."""

    row = project_flat_quality_uncertainty(project_quality_result(row))
    task = _task(row)
    components: dict[str, dict[str, Any]] = {}
    primary = row.get("primary")
    if isinstance(primary, Mapping):
        component = dict(primary)
    else:
        component = {
            "metric": row.get("task_quality_metric"),
            "candidate": row.get("task_quality_candidate"),
            "reference": row.get("task_quality_reference"),
            "delta": row.get("task_quality_delta"),
            "ci_low": row.get("task_quality_ci_low"),
            "ci_high": row.get("task_quality_ci_high"),
            "margin": row.get("task_quality_margin"),
            "decision": row.get("task_quality_decision"),
            "status": row.get("task_quality_status"),
            # Canonical scientific-report rows flatten the primary component.
            # Retain its statistical provenance here instead of accidentally
            # showing only the nested guardrail skip reason in the reference
            # comparison exports.
            "n": (
                row.get("validation_evaluated_count")
                or row.get("task_quality_n")
            ),
            "bootstrap_repetitions_requested": row.get(
                "task_quality_bootstrap_repetitions_requested"
            ),
            "bootstrap_repetitions": row.get(
                "task_quality_bootstrap_repetitions"
            ),
            "bootstrap_engine": row.get("task_quality_bootstrap_engine"),
            "bootstrap_skipped_reason": row.get(
                "task_quality_bootstrap_skipped_reason"
            ),
            "bootstrap_elapsed_s": row.get(
                "task_quality_bootstrap_elapsed_s"
            ),
            **{field: row.get(f"task_quality_{field}") for field in UNCERTAINTY_FIELDS},
        }
    if any(component.get(key) not in (None, "") for key in (
        "candidate", "reference", "delta",
    )):
        components[_metric_label(component.get("metric"), task)] = component

    guardrails = row.get("guardrails")
    if isinstance(guardrails, Mapping):
        for name, raw in guardrails.items():
            if not isinstance(raw, Mapping):
                continue
            component = dict(raw)
            component.setdefault("metric", name)
            components[_metric_label(component.get("metric") or name, task)] = component
    return components


def _paired_contracts_comparable(
    first: Mapping[str, Any], second: Mapping[str, Any],
) -> bool:
    return not _paired_contract_conflicts(first, second)


def _paired_contract_conflicts(
    first: Mapping[str, Any], second: Mapping[str, Any],
) -> list[str]:
    conflicts: list[str] = []
    first_reference = str(first.get("reference_identity") or "").strip()
    second_reference = str(second.get("reference_identity") or "").strip()
    if not first_reference or first_reference != second_reference:
        conflicts.append("reference_identity_missing_or_mismatch")
    if _task(first) != _task(second):
        conflicts.append("task_mismatch")
    first_signature = _contract_signature(first)
    second_signature = _contract_signature(second)
    contract_fields = (
        "dataset_manifest_sha256", "validation_dataset_manifest_sha256",
        "validation_dataset_sha256", "dataset_manifest", "dataset_id",
        "validation_dataset", "preprocessing_hash",
        "preprocessing_contract_hash", "preprocessing_contract_sha256",
        "decoder_hash", "decoder_contract_hash", "decoder_contract_sha256",
        "nms_hash", "nms_contract_sha256", "postprocessing_hash",
        "quality_policy_sha256", "task_quality_policy_sha256",
        "score_threshold", "nms_iou_threshold", "max_detections",
        "validation_item_ids_hash", "validation_count",
    )
    for field, left, right in zip(
        contract_fields, first_signature, second_signature,
    ):
        if left in (None, "") or right in (None, ""):
            continue
        if str(left) != str(right):
            conflicts.append(f"{field}_mismatch")
    return conflicts


def _build_paired_quality_decomposition(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project central paired results directly against their bound reference."""

    paired = [row for row in rows if _paired_quality_components(row)]
    reference_rows: list[dict[str, Any]] = []
    for row in paired:
        components = _paired_quality_components(row)
        reference_bound = bool(str(row.get("reference_identity") or "").strip())
        complete = all(
            _num(component, "candidate") is not None
            and _num(component, "reference") is not None
            for component in components.values()
        )
        comparison_status = (
            "ok" if reference_bound and complete
            else "reference_identity_missing" if not reference_bound
            else "paired_metric_incomplete"
        )
        quality_decision = _norm(
            row.get("task_quality_decision")
            or row.get("task_quality_status")
            or row.get("decision")
        )
        if quality_decision not in {"pass", "fail", "inconclusive"}:
            quality_decision = "not_evaluated"
        record: dict[str, Any] = {
            "model": str(row.get("model") or row.get("model_id") or "unknown"),
            "task": _task(row),
            "backend": _backend(row),
            "setup_id": _setup(row),
            "variant": _variant(row),
            "case_id": row.get("case_id") or row.get("boundary") or "",
            "reference_identity": row.get("reference_identity") or "",
            "reference_comparison_source": "paired_central_quality_result",
            # ``status`` historically meant only that candidate/reference
            # values were available, which rendered quality FAIL rows as
            # ``ok``. Keep that transport fact explicitly while making the
            # primary status/decision scientifically meaningful.
            "comparison_status": comparison_status,
            "quality_decision": quality_decision,
            "status": (
                quality_decision
                if comparison_status == "ok"
                else comparison_status
            ),
        }
        for name, component in sorted(components.items()):
            candidate = _num(component, "candidate")
            reference = _num(component, "reference")
            delta = _num(component, "delta")
            if delta is None and candidate is not None and reference is not None:
                delta = candidate - reference
            record[f"reference_{name}"] = reference
            record[f"row_{name}"] = candidate
            record[f"delta_vs_full_onnx_{name}"] = delta
            record[f"decision_{name}"] = component.get("decision")
            for field in ("ci_low", "ci_high", *UNCERTAINTY_FIELDS):
                record[f"{field}_{name}"] = component.get(field)
            record[f"bootstrap_skipped_reason_{name}"] = component.get(
                "bootstrap_skipped_reason"
            )
        reference_rows.append(record)

    full_by_identity: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in paired:
        if _variant(row) != "vendor_full":
            continue
        key = (
            str(row.get("model") or row.get("model_id") or "unknown"),
            _backend(row),
            _setup(row),
            str(row.get("reference_identity") or "").strip(),
        )
        full_by_identity.setdefault(key, row)

    decomposition: list[dict[str, Any]] = []
    for split in (row for row in paired if _variant(row) == "split"):
        model = str(split.get("model") or split.get("model_id") or "unknown")
        backend = _backend(split)
        reference_identity = str(split.get("reference_identity") or "").strip()
        key = (model, backend, _setup(split), reference_identity)
        vendor = full_by_identity.get(key)
        record: dict[str, Any] = {
            "model": model,
            "task": _task(split),
            "backend": backend,
            "setup_id": _setup(split),
            "case_id": split.get("case_id") or split.get("boundary") or "",
            "reference_identity": reference_identity,
            "decomposition_source": "paired_central_quality_results",
        }
        if vendor is None:
            record["status"] = "missing_vendor_full"
            decomposition.append(record)
            continue
        contract_conflicts = _paired_contract_conflicts(vendor, split)
        comparable = not contract_conflicts
        record["status"] = "ok" if comparable else "contract_incomparable"
        record["contract_comparison_conflicts"] = contract_conflicts
        full_components = _paired_quality_components(vendor)
        split_components = _paired_quality_components(split)
        for name in sorted(set(full_components) | set(split_components)):
            full_component = full_components.get(name, {})
            split_component = split_components.get(name, {})
            reference = _num(split_component, "reference")
            if reference is None:
                reference = _num(full_component, "reference")
            vendor_value = _num(full_component, "candidate")
            split_value = _num(split_component, "candidate")
            record[f"full_onnx_{name}"] = reference
            record[f"vendor_full_{name}"] = vendor_value
            record[f"split_{name}"] = split_value
            record[f"vendor_loss_{name}"] = (
                vendor_value - reference
                if comparable and vendor_value is not None and reference is not None
                else None
            )
            record[f"split_extra_loss_{name}"] = (
                split_value - vendor_value
                if comparable and split_value is not None and vendor_value is not None
                else None
            )
            record[f"total_split_loss_{name}"] = (
                split_value - reference
                if comparable and split_value is not None and reference is not None
                else None
            )
        decomposition.append(record)
    return reference_rows, decomposition


def build_quality_decomposition(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = [dict(r) for r in rows]
    if any(_paired_quality_components(row) for row in source):
        return _build_paired_quality_decomposition(source)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        by_model[str(row.get("model") or row.get("model_id") or row.get("model_name") or "unknown")].append(row)

    reference_rows: list[dict[str, Any]] = []
    decomposition: list[dict[str, Any]] = []
    for model, model_rows in sorted(by_model.items()):
        refs = [r for r in model_rows if _variant(r) == "canonical_full_onnx"]
        if not refs:
            continue
        reference = refs[0]
        ref_metrics = _metrics(reference)
        for row in model_rows:
            variant = _variant(row)
            if row is reference or variant == "canonical_full_onnx":
                continue
            task = _task(row)
            metrics = _metrics(row)
            comparable = _contracts_comparable(reference, row)
            rec: dict[str, Any] = {
                "model": model, "task": task, "backend": _backend(row), "variant": variant,
                "case_id": row.get("case_id") or row.get("boundary") or row.get("boundary_id") or "",
                "status": "ok" if comparable else "contract_incomparable",
            }
            for name, value in metrics.items():
                ref_value = ref_metrics.get(name)
                rec[f"reference_{name}"] = ref_value
                rec[f"row_{name}"] = value
                rec[f"delta_vs_full_onnx_{name}"] = (value - ref_value) if comparable and value is not None and ref_value is not None else None
            reference_rows.append(rec)

        full_by_backend: dict[str, dict[str, Any]] = {}
        for r in model_rows:
            if _variant(r) == "vendor_full":
                full_by_backend.setdefault(_backend(r), r)
        for split in (r for r in model_rows if _variant(r) == "split"):
            backend = _backend(split)
            vendor = full_by_backend.get(backend)
            if vendor is None:
                decomposition.append({
                    "model": model, "task": _task(split), "backend": backend,
                    "case_id": split.get("case_id") or split.get("boundary") or "",
                    "status": "missing_vendor_full",
                })
                continue
            comparable = _contracts_comparable(reference, vendor) and _contracts_comparable(reference, split) and _contracts_comparable(vendor, split)
            rec = {
                "model": model, "task": _task(split), "backend": backend,
                "case_id": split.get("case_id") or split.get("boundary") or "",
                "status": "ok" if comparable else "contract_incomparable",
            }
            ref_m, vendor_m, split_m = _metrics(reference), _metrics(vendor), _metrics(split)
            for name in sorted(set(ref_m) | set(vendor_m) | set(split_m)):
                a, b, c = ref_m.get(name), vendor_m.get(name), split_m.get(name)
                rec[f"full_onnx_{name}"] = a
                rec[f"vendor_full_{name}"] = b
                rec[f"split_{name}"] = c
                rec[f"vendor_loss_{name}"] = (b - a) if comparable and a is not None and b is not None else None
                rec[f"split_extra_loss_{name}"] = (c - b) if comparable and b is not None and c is not None else None
                rec[f"total_split_loss_{name}"] = (c - a) if comparable and a is not None and c is not None else None
            decomposition.append(rec)
    return reference_rows, decomposition


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo comparable rows were available.\n", encoding="utf-8")
        return
    has_quality_decision = any("quality_decision" in row for row in rows)
    preferred_names = [
        "model", "task", "backend", "variant", "case_id",
    ]
    preferred_names.extend(
        ["quality_decision", "comparison_status"]
        if has_quality_decision else ["status"]
    )
    preferred = [
        key for key in preferred_names if any(key in row for row in rows)
    ]
    metric_cols = [k for k in sorted({k for r in rows for k in r}) if "delta" in k or "loss" in k]
    decision_cols = [
        key for key in sorted({key for row in rows for key in row})
        if key.startswith("decision_")
    ]
    bootstrap_cols = [
        key for key in sorted({key for row in rows for key in row})
        if key.startswith("bootstrap_skipped_reason_")
    ]
    cols = preferred + metric_cols + decision_cols + bootstrap_cols
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(c) is None else str(row.get(c)) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tex(path: Path, rows: list[dict[str, Any]], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "backend", "case_id", "quality_decision", "comparison_status"]
    lines = ["% Auto-generated from the canonical scientific report.", "\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{lllll}", "\\toprule", "Model & Backend & Case & Decision & Comparison \\\\", "\\midrule"]
    for row in rows:
        vals = [str(row.get(c, "")).replace("_", "\\_") for c in cols]
        lines.append(" & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_decomposition(report_dir: str | Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    refs, decomposition = build_quality_decomposition(rows)
    artefacts = {
        "task_quality_reference_comparison": refs,
        "task_quality_loss_decomposition": decomposition,
    }
    for stem, data in artefacts.items():
        (out / f"{stem}.json").write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        _write_csv(out / f"{stem}.csv", data)
        _write_md(out / f"{stem}.md", stem.replace("_", " ").title(), data)
    tex = out / "thesis_tables"
    _write_tex(tex / "task_quality_reference_comparison.tex", refs,
               "Task-quality comparison against the canonical Full-ONNX reference.",
               "tab:task-quality-reference-comparison")
    _write_tex(tex / "task_quality_loss_decomposition.tex", decomposition,
               "Decomposition of vendor-full and additional split-induced task-quality drift.",
               "tab:task-quality-loss-decomposition")
    return {"reference_comparison_count": len(refs), "loss_decomposition_count": len(decomposition)}
