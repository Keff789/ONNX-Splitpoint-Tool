"""Bounded, graph-only planning for the post-planned THESIS20 YOLO cohort.

This is an additive work list, not a replacement ranker or build admission.
It reads existing identities and exact negative evidence; no runtime, compiler,
remote, or quality operation is performed here. The normal writers still admit
each new artifact and result under their own contracts.
"""
from __future__ import annotations

import copy
import csv
import json
import re
from pathlib import Path
from typing import Any

from .build_evidence import (
    BUILD_INDEX_SCHEMA, canonical_build_key_from_hailo_v3_payload,
    validate_build_evidence_index,
)
from .build_evidence_store import default_build_evidence_root
from .core_analysis import analyze_model
from .generic_completion_plan import SETUPS, validate_completion_scope
from .onnx_utils import elemtype_from_vi, shape_from_vi, value_info_map
from .runners.task_completion import valid_split_completion_repetitions
from .split_export_graph import cut_tensors_for_boundary, part2_external_inputs_for_boundary
from .workflow.artifacts import sha256_file

COHORT_ID = "yolo_min3_capability_augmentation"
BASE_CASES = {
    "yolo11l": ["b003", "b062"], "yolo26m": ["b003", "b038", "b043"],
    "yolo26s": ["b003", "b021"], "yolov7_paper": ["b009"],
}
PAPER_RUN = "paper_comparison_yolov7_b066_hailo8_native_full_energy_b500_20260827_110251"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def graph_candidates(primary_run: str | Path, model_id: str) -> dict:
    """Use the existing Analyzer and exact export backward slice, never timing."""
    root = Path(primary_run) / "models" / model_id
    manifest_path = root / "model_manifest.json"
    manifest = _read(manifest_path)
    expected = str(manifest.get("observed_model_sha256") or manifest.get("model_sha256") or "").removeprefix("sha256:")
    graph_path = root / "benchmark_set" / "legacy_suite" / "models" / f"{model_id}.onnx"
    if not graph_path.is_file():
        raise ValueError(f"augmentation archived original graph missing: {graph_path}")
    if not expected or str(sha256_file(graph_path) or "").removeprefix("sha256:") != expected:
        raise ValueError(f"augmentation original graph identity mismatch: {model_id}")
    analysis = analyze_model(str(graph_path))
    model, nodes, order = (analysis[k] for k in ("model", "nodes", "order"))
    shapes = value_info_map(model)
    candidates = []
    # The normal analysis excludes the first and final boundary positions.
    for boundary in range(1, len(order) - 1):
        cut = cut_tensors_for_boundary(order, nodes, boundary)
        external = part2_external_inputs_for_boundary(model, order, nodes, boundary)
        if len(external) != 1:
            continue
        node = nodes[order[boundary]]
        inputs = [{"name": name,
                   "shape": shape_from_vi(shapes[name]) if name in shapes else None,
                   "elemtype": elemtype_from_vi(shapes[name]) if name in shapes else None}
                  for name in external]
        candidates.append({"boundary": boundary, "case_id": f"b{boundary:03d}",
            "topological_node_index": order[boundary], "node_name": node.name, "node_op": node.op_type,
            "cut_tensors": cut, "part2_external_inputs": external, "inputs": inputs,
            "part2_input_count": len(external), "part2_input_count_source": "exact_export_backward_slice",
            "selection_eligible": True, "exclusions": []})
    return {"model_id": model_id, "source_graph": str(graph_path),
            "model_manifest": str(manifest_path), "model_sha256": expected,
            "node_count": len(nodes), "original_outputs": [o.name for o in model.graph.output],
            "candidates": candidates}


def _negative_records() -> list[dict]:
    records = []
    for path in sorted(default_build_evidence_root().glob("*.json")):
        data = _read(path)
        if not (data.get("schema") == BUILD_INDEX_SCHEMA or "index_payload_sha256" in data):
            continue
        for record in validate_build_evidence_index(data)["records"]:
            if record.get("reusable") is True and record.get("deterministic") is True and record.get("state") in {
                    "COMPILE_INFEASIBLE", "PARSER_UNSUPPORTED", "RUNTIME_UNSUPPORTED"}:
                records.append({"source": str(path), "record": record})
    return records


def _apply_known_negatives(graph: dict, run: Path, records: list[dict]) -> None:
    """Only negatives of this exact graph and unchanged existing build recipe.

    This candidate exclusion is not an exact cache HIT claim: the regular build
    boundary retains its complete source/compiler/key checks before any dispatch.
    No negative is generalized to adjacent boundaries or another architecture.
    """
    by_boundary = {r["boundary"]: r for r in graph["candidates"]}
    for item in records:
        record = item["record"]
        key = record["key"]
        if key.get("full_source_onnx_sha256") != graph["model_sha256"]:
            continue
        recipe, calibration = key["recipe"], key["calibration"]
        match = re.fullmatch(re.escape(graph["model_id"]) + r"_part1_b0*(\d+)", recipe["net_name"])
        if not match or int(match[1]) not in by_boundary:
            continue
        arch = key["hw_arch"]
        folder = "hailo10" if arch == "hailo10h" else arch
        receipt_path = (run / "models" / graph["model_id"] / "benchmark_set" / "legacy_suite"
                        / BASE_CASES[graph["model_id"]][0] / "hailo" / folder / "part1" / "hailo_hef_build_receipt.json")
        if not receipt_path.is_file():
            raise ValueError(f"augmentation cannot compare negative build recipe: {receipt_path}")
        receipt = _read(receipt_path)
        reference_key = canonical_build_key_from_hailo_v3_payload(
            receipt["cache_payload"], builder_source_onnx_sha256=receipt["source_onnx_sha256"],
            full_source_onnx_sha256=graph["model_sha256"],
            boundary_endpoint_contract_sha256=key["boundary_endpoint_contract_sha256"])
        same = (reference_key["compiler_version"] == key["compiler_version"]
                and reference_key["calibration"] == calibration
                and reference_key["preprocessing_contract_sha256"] == key["preprocessing_contract_sha256"]
                and {k: v for k, v in reference_key["recipe"].items() if k != "net_name"}
                == {k: v for k, v in recipe.items() if k != "net_name"})
        candidate = by_boundary[int(match[1])]
        evidence = {"reason": "exact_historical_compile_infeasibility_for_unchanged_recipe",
            "backend": arch, "state": record["state"], "source": item["source"],
            "record_sha256": record["record_sha256"], "key_sha256": record["key_sha256"],
            "boundary_endpoint_contract_sha256": key["boundary_endpoint_contract_sha256"],
            "build_recipe_comparison_source": str(receipt_path), "unchanged_recipe": same,
            "original_evidence": record["evidence_origin"]}
        if not same:
            raise ValueError(f"augmentation negative build recipe requires explicit review: {graph['model_id']}/{candidate['case_id']}/{arch}")
        candidate["selection_eligible"] = False
        candidate["exclusions"].append(evidence)


def _paper_anchor(graph: dict) -> dict:
    root = Path.home() / "onnx-splitpoint-results" / "results" / "paper" / PAPER_RUN
    run_path = root / "run_manifest.json"
    binding_path = root / "reports/native_quality_first/orin_nx_hailo8_01/native_split_quality_binding_set.json"
    if not run_path.is_file() or not binding_path.is_file():
        return {"status": "historical_identity_unavailable", "run_manifest": str(run_path), "binding_set": str(binding_path)}
    historical = _read(run_path)["models"]["yolov7_paper"]["model_sha256"].removeprefix("sha256:")
    binding = _read(binding_path)["bindings_by_model_case_backend"]["yolov7_paper|b066|hailo8_to_trt"]
    candidate = next((r for r in graph["candidates"] if r["boundary"] == 66), None)
    metadata = binding["boundary_metadata_payload"]["boundary_tensor"]
    boundary = binding["boundary_contract"]
    match = bool(candidate and historical == graph["model_sha256"]
        and candidate["part2_external_inputs"] == [metadata["name"]]
        and candidate["inputs"][0]["shape"] == metadata["canonical_part2_shape"]
        and boundary["boundary_tensor_name"] == metadata["name"]
        and boundary["boundary_tensor_shape"] == metadata["shape"]
        and boundary["precision"] == "uint8_dequant_fp16"
        and boundary["boundary_layout"] == "memory_nhwc_to_nchw")
    return {"status": "same_original_graph_and_boundary" if match else "historical_identity_mismatch",
        "run_manifest": str(run_path), "binding_set": str(binding_path),
        "historical_model_sha256": historical, "binding_sha256": binding["binding_sha256"],
        "historical_boundary_contract": boundary, "historical_boundary_tensor": metadata,
        "historical_performance_reused": False, "historical_quality_reused": False,
        "claim": "Graph/endpoint identity only; historical timing, power and B500 quality are not new measurements."}


def choose_boundaries(candidates: list[dict], included: list[int], count: int) -> list[dict]:
    """Farthest-first graph coverage with the lower boundary index as tie-break."""
    current = list(included)
    chosen = []
    for _ in range(count):
        pool = [r for r in candidates if r.get("selection_eligible") is True and r["boundary"] not in current]
        if not pool:
            break
        row = min(pool, key=lambda r: (-min(abs(r["boundary"] - b) for b in current), r["boundary"]))
        selected = copy.deepcopy(row)
        selected["minimum_topological_distance"] = min(abs(row["boundary"] - b) for b in current)
        selected["distance_reference_boundaries"] = list(current)
        selected["selection_reason"] = "maximum_minimum_topological_distance_then_lower_boundary_index"
        chosen.append(selected)
        current.append(row["boundary"])
    return chosen


def _actual_compile_failure(row: dict, observation: dict, suite: Path, stage_path: Path) -> dict:
    """Verify a newly recorded compiler failure without changing its dispatch."""
    from .build_evidence import (
        BuildEvidenceError, boundary_endpoint_contract_sha256,
        canonical_build_key_from_hailo_v3_payload, validate_build_evidence_record,
    )
    from .build_evidence_store import BuildEvidenceStore
    from .native_job_identity import failed_native_result, native_comparison
    count = observation.get("compiler_dispatch_count")
    if (observation.get("status") != "failed" or observation.get("readiness") != "blocked"
            or type(count) is not int or count < 1 or observation.get("identity_conflicts")
            or observation.get("job_kind", "artifact") != "artifact"
            or observation.get("cache_hit") is True):
        return {}
    try:
        case = suite / row["case_id"]
        manifest_path = case / "split_manifest.json"
        manifest = _read(manifest_path)
        request_path = Path(str(observation.get("request") or "")).resolve()
        if request_path.name != "deferred_hailo_build.json" or case.resolve() not in request_path.parents:
            return {}
        request = _read(request_path)
        args = request["kwargs"]
        output = Path(args["outdir"]).resolve()
        if (output != request_path.parent or output.name != "part1" or request.get("status") != "failed"
                or (request.get("build_observation") or {}).get("compiler_dispatch_count") != count):
            return {}
        family = native_comparison(row["backend"])
        if native_comparison(args.get("hw_arch")) != family or family not in {"hailo8", "hailo10h"}:
            return {}
        arch = "hailo10" if family == "hailo10h" else "hailo8"
        hefs = manifest.get("hailo", {}).get("hefs", {})
        summary = (hefs.get(arch) or hefs.get(family) or {}).get("part1_build") or {}
        info = summary.get("build_evidence") or {}
        context = info.get("context") or {}
        if (summary.get("ok") is not False or summary.get("skipped") is not False
                or summary.get("timed_out") is not False or summary.get("cache_hit") is not False
                or info.get("recorded") is not True or info.get("negative_evidence_hit") is not False
                or info.get("compiler_dispatch_allowed") is not True
                or info.get("recorded_state") not in {"COMPILE_INFEASIBLE", "PARSER_UNSUPPORTED"}
                or context != args.get("build_evidence_context")
                or context.get("model_id") != row["model_id"] or context.get("stage") != "part1"
                or str(context.get("boundary")).zfill(3) != row["case_id"][1:]
                or any(block.get(field) not in (None, "", row["setup_id"])
                       for block in (observation, context) for field in ("setup_id", "measurement_setup_id", "expected_setup_id"))
                or context.get("full_source_onnx_sha256") != row["model_sha256"].removeprefix("sha256:")
                or manifest.get("part2_external_inputs") != row["part2_external_inputs"]
                or manifest.get("cut_tensors") != row["cut_tensors"]):
            return {}
        payload = info["cache_payload_v3"]
        source = Path(request["source_onnx"]).resolve()
        if case.resolve() not in source.parents or str(source) != str(summary.get("source_onnx_path")):
            return {}
        endpoint = boundary_endpoint_contract_sha256(stage="part1", cache_payload=payload, split_manifest=manifest)
        key = canonical_build_key_from_hailo_v3_payload(payload,
            builder_source_onnx_sha256=str(sha256_file(source)).removeprefix("sha256:"),
            full_source_onnx_sha256=context["full_source_onnx_sha256"],
            boundary_endpoint_contract_sha256=endpoint, expected_cache_key=info["cache_key_v3"])
        if key != info.get("key") or key["recipe"]["net_name"] != args.get("net_name"):
            return {}
        decision = BuildEvidenceStore().lookup(key)
        if decision.status != "HIT" or decision.reusable is not True or decision.state != info["recorded_state"]:
            return {}
        record = validate_build_evidence_record(decision.record or {})
        origin = record["evidence_origin"]
        phase = origin.get("phase")
        if (record["record_sha256"] != info.get("record_sha256")
                or origin.get("source") != "live_hailo_build" or origin.get("net_name") != args["net_name"]
                or Path(origin.get("outdir", "")).resolve() != output
                or phase not in {"translate", "parse", "parser", "optimize", "optimization", "compile", "compilation", "mapping"}):
            return {}
        terminal_path = output / "hailo_attempt_receipts/terminal_attempt.json"
        terminal = _read(terminal_path)
        immutable_path = Path(terminal["immutable_receipt"]).resolve()
        if immutable_path.parent != terminal_path.parent.resolve():
            return {}
        immutable = _read(immutable_path)
        if (str(sha256_file(immutable_path)).removeprefix("sha256:") != terminal["immutable_receipt_sha256"]
                or immutable != {k: v for k, v in terminal.items() if k not in {"immutable_receipt", "immutable_receipt_sha256"}}
                or terminal.get("terminal") is not True or terminal.get("invocation_status") != "returned"
                or terminal.get("timed_out") is not False or terminal.get("semantic_status") not in {"failed", "unsupported"}
                or terminal.get("compiler_phase") != phase or terminal.get("outdir") != str(output)
                or terminal.get("source_onnx_sha256") != key["builder_source_onnx_sha256"]
                or terminal.get("details", {}).get("build_evidence") != info):
            return {}
        return failed_native_result({**row, "backend": row.get("direction") or row["backend"]},
            failure_stage="build_backend_artifacts", failure_reason=record["state"].lower(),
            status="failed", technical_terminal_kind="actual_compile_failed", case_key=row["case_key"],
            selection_role="primary", compiler_dispatched=True, compiler_dispatch_count=count,
            runtime_started=False, measurement_started=False, upstream_build_observation=copy.deepcopy(observation),
            technical_terminal_source=str(stage_path), current_split_manifest=str(manifest_path),
            actual_compile_failure={"record": record, "original_recorded_build_evidence": info,
                "request": str(request_path), "terminal_attempt": str(terminal_path),
                "immutable_attempt": str(immutable_path), "source": "existing_exact_build_evidence"})
    except (OSError, ValueError, TypeError, KeyError, BuildEvidenceError):
        return {}


def collect_technical_terminals(plan: dict, augmentation_run: str | Path) -> list[dict]:
    """Read exact normal build terminals for the immutable primary identities.

    No log classification, compiler call, new negative admission or reserve
    dispatch occurs here. A generic build failure is deliberately unresolved.
    """
    from .build_evidence import boundary_endpoint_contract_sha256
    from .native_job_identity import (
        known_build_exclusion, native_comparison, native_identity_key,
        native_job_prerequisite,
    )
    root = Path(augmentation_run).resolve()
    if root != Path(plan["augmentation_run"]).resolve():
        raise ValueError("augmentation technical terminal run mismatch")
    terminals = []
    for row in plan["rows"]:
        if row.get("selection_role") != "primary":
            continue
        model = root / "models" / row["model_id"]
        suite = model / "benchmark_set/legacy_suite"
        stage_path = model / "stages/build_backend_artifacts/stage_result.json"
        if not stage_path.is_file():
            continue
        stage = _read(stage_path)
        readiness = (stage.get("details") or {}).get("deferred_build_readiness") or {}
        selected = [value for value in readiness.get("blocked_jobs", []) if isinstance(value, dict)
            and value.get("model_id") == row["model_id"]
            and value.get("boundary") == row["case_id"] and value.get("stage") == "part1"
            and native_comparison(value.get("backend")) == native_comparison(row["backend"])]
        # Ambiguous or conflicting presentations must not pick a convenient row.
        if len(selected) != 1:
            continue
        if (any(selected[0].get(field) is True for field in (
                "runtime_started", "measurement_started", "execution_started", "runtime_success", "actual_ok"))
                or any(selected[0].get(field) not in (None, 0) for field in (
                    "repetition_count_attempted", "repetition_count_valid", "completed_frames"))):
            continue
        job = {**row, "backend": row.get("direction") or row["backend"]}
        observed = native_job_prerequisite(job, suite_dir=suite, model_dir=model)
        proof = known_build_exclusion(observed)
        if not proof:
            cold = _actual_compile_failure(row, selected[0], suite, stage_path)
            if cold:
                terminals.append(cold)
            continue
        if native_identity_key(observed) != native_identity_key(job):
            continue
        decision = proof["build_evidence"]
        context = (selected[0].get("build_evidence") or {}).get("context") or {}
        if any(block.get(field) not in (None, "", row["setup_id"])
               for block in (selected[0], context)
               for field in ("setup_id", "measurement_setup_id", "expected_setup_id")):
            continue
        if decision["key"]["full_source_onnx_sha256"] != row["model_sha256"].removeprefix("sha256:"):
            continue
        manifest_path = suite / row["case_id"] / "split_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read(manifest_path)
        if (manifest.get("part2_external_inputs") != row["part2_external_inputs"]
                or manifest.get("cut_tensors") != row["cut_tensors"]):
            continue
        current_endpoint = boundary_endpoint_contract_sha256(stage="part1",
            cache_payload=decision["cache_payload_v3"], split_manifest=manifest)
        if current_endpoint != decision["key"]["boundary_endpoint_contract_sha256"]:
            continue
        terminals.append({**observed, "case_key": row["case_key"],
            "technical_terminal_kind": "known_build_nonstart",
            "selection_role": "primary", "build_exclusion": proof,
            "technical_terminal_source": str(stage_path), "current_split_manifest": str(manifest_path)})
    return terminals


def propose_reserve_activation(plan: dict, technical_terminals: list[dict]) -> dict:
    """Read-only standby accounting; this never changes the accepted scope."""
    from .native_job_identity import known_build_exclusion, native_identity_key
    primary = {row["case_key"]: row for row in plan["rows"] if row.get("selection_role") == "primary"}
    groups: dict[tuple[str, str], list[dict]] = {}
    seen = set()
    for terminal in technical_terminals:
        row = primary.get(terminal.get("case_key"))
        proof = known_build_exclusion(terminal)
        if not row or not proof:
            continue
        job = {**row, "backend": row.get("direction") or row["backend"]}
        if (native_identity_key(job) != native_identity_key(terminal)
                or proof["build_evidence"]["key"]["full_source_onnx_sha256"] != row["model_sha256"].removeprefix("sha256:")):
            continue
        if row["case_key"] in seen:
            raise ValueError("augmentation duplicate technical terminal")
        seen.add(row["case_key"])
        groups.setdefault((row["model_id"], row["setup_id"]), []).append({**terminal, "build_exclusion": proof})
    proposals = []
    for (model, setup), negatives in sorted(groups.items()):
        reserves = [row for row in plan["rows"] if row.get("selection_role") == "reserve"
                    and row["model_id"] == model and row["setup_id"] == setup]
        if len(reserves) != 1:
            continue
        proposals.append({"model_id": model, "setup_id": setup, "case_key": reserves[0]["case_key"],
            "case_id": reserves[0]["case_id"], "activated": False, "dispatch_allowed": False,
            "reason": "exact_primary_build_nonstart_standby_candidate_only",
            "negative_primary_case_keys": [r["case_key"] for r in negatives],
            "existing_build_key_sha256s": [r["build_exclusion"]["build_evidence"]["key_sha256"] for r in negatives]})
    return {"status": "standby_only_no_dispatch", "executable_reserve_budget": 0,
        "accepted_primary_scope_modified": False, "proposals": proposals,
        "limitation": "Later technical failures remain terminal shortfalls; no reserve is appended to the sealed accepted cases."}


def build_augmentation_plan(primary_run: str | Path, completion_root: str | Path, *, output_root: str | Path | None = None) -> dict:
    run, completion = Path(primary_run).resolve(), Path(completion_root).resolve()
    terminal = _read(completion / "terminal-status.json")
    if terminal.get("status") != "complete" or terminal.get("exitcode") != 0:
        raise ValueError("augmentation requires the completed Generic base")
    summary = terminal.get("summary") or {}
    if summary.get("valid_repetitions") != 576 or summary.get("completed_task_count") != 576000 or summary.get("status_counts", {}).get("completed") != 192:
        raise ValueError("augmentation completed base counts differ")
    source_plan = completion / "plan/execution-plan.json"
    base = _read(source_plan)
    if Path(base["original_run"]).resolve() != run:
        raise ValueError("augmentation base run identity mismatch")
    validate_completion_scope(base["rows"])
    records = _negative_records()
    models, rows = {}, []
    for model_id, expected_cases in BASE_CASES.items():
        base_rows = [r for r in base["rows"] if r["model_id"] == model_id]
        for row in base_rows:
            observed = _read(Path(row["result_path"]))
            if observed.get("status") != "ok" or len(valid_split_completion_repetitions(observed, row["runtime_config"])) != 3:
                raise ValueError(f"augmentation completed base result invalid: {row['case_key']}")
        for setup in SETUPS:
            if sorted(r["case_id"] for r in base_rows if r["setup_id"] == setup) != expected_cases:
                raise ValueError(f"augmentation original boundary set changed: {model_id}/{setup}")
        if model_id == "yolo26m":
            models[model_id] = {"base_cases": expected_cases, "primary_cases": [], "reserve_cases": [],
                                "candidates": [], "reason": "already_three_boundaries_no_new_work"}
            continue
        graph = graph_candidates(run, model_id)
        if any(r["model_sha256"].removeprefix("sha256:") != graph["model_sha256"] for r in base_rows):
            raise ValueError(f"augmentation Completion model identity mismatch: {model_id}")
        _apply_known_negatives(graph, run, records)
        included = [int(c[1:]) for c in expected_cases]
        primary = []
        if model_id == "yolov7_paper":
            graph["paper_anchor"] = _paper_anchor(graph)
            for boundary in (44, 66):
                candidate = next((r for r in graph["candidates"] if r["boundary"] == boundary and r["selection_eligible"]), None)
                if boundary == 66 and graph["paper_anchor"]["status"] != "same_original_graph_and_boundary":
                    raise ValueError("augmentation paper anchor identity requires review")
                if candidate:
                    primary.append({**copy.deepcopy(candidate), "selection_reason": "explicit_original_graph_anchor"})
                    included.append(boundary)
        primary.extend(choose_boundaries(graph["candidates"], included, 3 - len(included)))
        reserve = choose_boundaries(graph["candidates"], [*included, *(r["boundary"] for r in primary)], 1)
        graph.update(base_cases=expected_cases, primary_cases=[r["case_id"] for r in primary],
                     reserve_cases=[r["case_id"] for r in reserve], selection=primary + reserve)
        models[model_id] = graph
        for role, selected in (("primary", primary), ("reserve", reserve)):
            for candidate in selected:
                for setup in sorted(SETUPS):
                    template = min((r for r in base_rows if r["setup_id"] == setup), key=lambda r: r["case_id"])
                    identity = {**template["identity"], "case_id": candidate["case_id"]}
                    rows.append({**identity, "identity": identity, "backend": SETUPS[setup], "task": "detection",
                        "selection_role": role, "activated": role == "primary",
                        "activation": "planned" if role == "primary" else "standby_technical_shortfall_only",
                        "model_sha256": graph["model_sha256"], "source_graph": graph["source_graph"],
                        "boundary": candidate["boundary"], "cut_tensors": candidate["cut_tensors"],
                        "part2_external_inputs": candidate["part2_external_inputs"], "boundary_inputs": candidate["inputs"],
                        "selection_reason": candidate["selection_reason"], "case_key": "|".join(identity[k] for k in ("model_id", "case_id", "direction", "precision", "setup_id", "comparison_backend", "output_endpoint_id")),
                        "base_template_case": template["case_id"], "base_template_case_key": template["case_key"],
                        "base_execution_plan": str(source_plan), "base_completion_result": template["result_path"],
                        "base_manifest": template["original_manifest"], "base_native_source": template["native_source"],
                        "base_quality_request": template["generic_original_quality_request"],
                        "input_contract_template": copy.deepcopy(template["input_binding"]),
                        "completion_contract_template": copy.deepcopy(template["completion_contract"]),
                        "measurement_boundary": template["measurement_boundary"],
                        "new_quality_binding_required": True, "new_runtime_artifact_binding_required": True,
                        "quality_status": "pending_new_source", "warmup_count": 100, "task_count": 1000, "repetitions": 3})
    primary_rows = [r for r in rows if r["selection_role"] == "primary"]
    # One fixed primary per device; deterministic contract coverage, no outcomes.
    starts = {"orin_nx_deepx_m1_01": "yolov7_paper", "orin_nx_hailo8_01": "yolo11l", "orin_nx_hailo10_01": "yolo26s"}
    start_keys = [min((r for r in primary_rows if r["setup_id"] == setup and r["model_id"] == model), key=lambda r: r["boundary"])["case_key"] for setup, model in starts.items()]
    if len(primary_rows) > 12 or len(rows) > 21 or len({r["case_key"] for r in rows}) != len(rows):
        raise ValueError("augmentation bounded cohort identity conflict")
    result = {"schema": "onnx-splitpoint/yolo-min3-augmentation-plan", "schema_version": 1,
        "cohort_id": COHORT_ID, "status": "draft_pending_build_policy_review", "frozen": False,
        "original_run": str(run), "base_completion_root": str(completion), "base_execution_plan": str(source_plan),
        "base_scientific_reports": str(Path.home() / "Reports/thesis20-auswertung-20261002_110429/scientific-corrected-v2"),
        "analysis_scope": "post_planned_capability_graph_augmentation_not_holdout", "evaluation_role": "development",
        "selection_policy": "explicit_yolov7_anchors_then_maximum_minimum_topological_distance_lower_index_tie",
        "reserve_activation_policy": "only_proven_technical_shortfall_never_accuracy_fps_energy_or_claim",
        "models": models, "rows": rows, "primary_count": len(primary_rows), "reserve_count": len(rows)-len(primary_rows),
        "start_case_keys": start_keys, "force_rebuild": False,
        "requires_normal_artifact_quality_native_and_energy_admission": True}
    if output_root is not None:
        out = Path(output_root).resolve()
        if out == run or run in out.parents or out == completion or completion in out.parents:
            raise ValueError("augmentation plan output must be outside preserved bases")
        out.mkdir(parents=True, exist_ok=True)
        report_root = out.parent if out.name == "plan" else out
        suffix = report_root.name.removeprefix("thesis20-yolo-min3-augmentation-")
        result["augmentation_run"] = str(report_root / "runs" / f"yolo_min3_{suffix}")
        (out / "augmentation-plan-draft.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        fields = ["model_id", "case_id", "setup_id", "backend", "direction", "precision", "selection_role", "activation", "selection_reason"]
        with (out / "augmentation-cases-draft.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
    return result
