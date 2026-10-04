"""Reuse the closed campaign's CPU reference without executing or relabelling it.

The augmentation owns new producer requests. Only their CPU comparison source
comes from another run, through the ordinary original-run reference reader.
"""
from __future__ import annotations

import concurrent.futures
import copy
import json
from pathlib import Path
import threading

from .workflow.runner import EvaluationWorkflowRunner


def _quality_key(row):
    from .native_split_quality import canonical_native_split_backend
    return (str(row.get("model_id") or row.get("model") or ""),
            str(row.get("case_id") or row.get("case") or "").lower(),
            str(row.get("setup_id") or row.get("source_setup_id")
                or row.get("expected_setup_id") or ""),
            canonical_native_split_backend(row.get("source_run_id") or row.get("run_id")
                or row.get("direction") or row.get("backend")))


def augmentation_quality_source_rows(runner, supplement_plan):
    """Add missing candidate work without reclassifying preserved performance.

    Existing requests are reopened through the regular loader. A retained valid
    candidate needs local statistics only; corrupt evidence never becomes an
    excuse to repeat inference. The original generic rows are not modified.
    """
    from .quality_service import quality_request_from_manifest
    expected = runner._augmentation_expected_quality_rows()
    existing = {}
    for model in sorted({key[0] for key in expected}):
        root = runner.run_dir / "models" / model / "benchmark_results"
        for path in sorted(root.rglob("task_quality_inputs/*_request.json")):
            # Only ordinary canonical imports (including normal attempts) own
            # requests; raw remote diagnostic mirrors are checked by the queue.
            if "quality_inputs" not in path.relative_to(root).parts:
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            runner._validate_quality_request_model_binding(model, manifest)
            identity = runner._quality_request_identity(path, model_id=model, manifest=manifest)
            if identity.get("identity_valid") is not True:
                raise ValueError("augmentation retained quality request identity invalid: " + str(path))
            key = _quality_key(identity)
            if key not in expected:
                raise ValueError("augmentation retained quality request outside frozen scope: " + str(key))
            _records, status, encoded = runner._augmentation_original_reference(model)
            quality_request_from_manifest(path, reference_artifact={
                "path": status["reference_path"], "sha256": status["reference_sha256"],
                "size_bytes": status["reference_size_bytes"]}, reference_artifact_bytes=encoded)
            existing.setdefault(key, []).append(str(path.relative_to(runner.run_dir)))
    completed = {str(row["path"]) for row in supplement_plan.get("completed_quality_requests", [])}
    generic = {_quality_key(row): row for row in supplement_plan.get("generic_rows", [])}
    if not set(expected).issubset(generic):
        raise ValueError("augmentation quality scope lacks its original Generic identity")
    rows = []
    for key, original in expected.items():
        row = copy.deepcopy(generic[key])
        sources = existing.get(key, [])
        row.update(stage="quality_source", precision=original["precision"],
            action="preserve" if sources and all(p in completed for p in sources)
                else "local_statistics" if sources else "execute",
            source_requests=sources, performance_action="preserve",
            reason="missing_new_candidate_records" if not sources else "retained_candidate_records")
        rows.append(row)
    return rows


class _OriginalReferenceReader(EvaluationWorkflowRunner):
    """Read-only view: no workflow constructor, writers, executors or dispatch."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_id = run_dir.name
        self._management_reference_futures = {}
        self._management_reference_source_contracts = {}
        self._management_services_lock = threading.RLock()
        self._management_admission_closed = False
        self._emit_log = lambda *_: None


class OriginalCPUReferenceMixin:
    """Use before EvaluationWorkflowRunner in the augmentation's MRO.

    The adapter must set ``augmentation_reference_run`` to the explicitly bound
    closed EvaluationRun. Active-run path guards remain unchanged everywhere
    else. The normal quality loader subsequently checks the new request's model,
    dataset, preprocessing, decoder, NMS and image pairing against these bytes.
    """

    def _augmentation_expected_quality_rows(self):
        from .yolo_augmentation_plan import collect_technical_terminals
        plan = self.augmentation_plan
        excluded = {_quality_key(row) for row in collect_technical_terminals(plan, self.run_dir)}
        return {_quality_key(row): row for row in plan["rows"]
                if row.get("selection_role") == "primary" and _quality_key(row) not in excluded}

    def _validate_quality_request_model_binding(self, model_id, manifest):
        super()._validate_quality_request_model_binding(model_id, manifest)
        # The ordinary portable binding validator still verifies all artifacts,
        # duplicate fields and hashes. This adds only the frozen cohort/budget.
        from .native_split_quality import validate_native_split_quality_binding
        binding, status = validate_native_split_quality_binding(manifest.get("native_split_quality_binding"))
        if binding is None or manifest.get("native_split_quality_binding_required") is not True:
            raise ValueError("augmentation quality requires the new split binding: " + status)
        selection = binding["preselection"]
        key = _quality_key(selection)
        expected = self._augmentation_expected_quality_rows().get(key)
        if (expected is None or key[0] != model_id or binding.get("eval_run_id") != self.run_id
                or manifest.get("eval_run_id") != self.run_id
                or selection.get("precision") != expected["precision"]):
            raise ValueError("augmentation quality request differs from the frozen new identity")
        statistics = manifest.get("statistics") or {}
        if (manifest.get("task") != "detection" or manifest.get("variant") != "composed"
                or manifest.get("record_count") != 5000 or manifest.get("reference_record_count") != 5000
                or statistics.get("bootstrap_repetitions") != 1000 or statistics.get("seed") != 20260710):
            raise ValueError("augmentation quality request requires exact N5000/B1000/seed20260710")

    def _quality_obligation_errors(self, results):
        from .trt_quality_chain import TensorRTQualityChainError, split_binding_set_from_central_quality_summary
        summary = {"schema": "onnx-splitpoint/central-quality-summary", "schema_version": 1,
                   "results": list(results)}
        errors = []
        for key, row in self._augmentation_expected_quality_rows().items():
            try:
                split_binding_set_from_central_quality_summary(summary, eval_run_id=self.run_id,
                    setup_id=row["setup_id"], selections=[dict(row, backend=row["direction"])])
            except TensorRTQualityChainError as exc:
                errors.append("|".join(key) + ": " + str(exc))
        return errors

    def _merge_supplement_quality_metadata(self, kept, new):
        """Attach the new candidate descriptor, retaining all old measurement fields.

        Both request identity and the old/new runtime artifacts are verified;
        neither a similar row nor a successful statistical decision is enough.
        Structural/numerical gates are deliberately not upgraded from a later
        candidate invocation. The central consumer derives Quality decisions.
        """
        from .native_split_quality import validate_native_split_quality_binding
        from .quality_service import quality_request_from_manifest
        from .workflow.artifacts import sha256_file
        expected = self._augmentation_expected_quality_rows()
        candidates = {}
        for row in new:
            key = _quality_key(row)
            if key in candidates:
                raise ValueError("augmentation duplicate candidate metadata: " + str(key))
            candidates[key] = row
        output = []
        for original in kept:
            row = copy.deepcopy(original)
            key = _quality_key(row)
            candidate = candidates.get(key)
            if candidate is None or key not in expected:
                output.append(row)
                continue
            identity = self._quality_request_identity_for_row_variant(candidate, "composed")
            request_sha = self._quality_sha256_token(identity.get("source_request_sha256"))
            if not request_sha or identity.get("request_descriptor_present") is not True:
                raise ValueError("augmentation candidate lacks its normal Quality request descriptor")
            model = key[0]
            result_root = self.run_dir / "models" / model / "benchmark_results"
            matched = []
            for path in result_root.rglob("task_quality_inputs/*_request.json"):
                if "quality_inputs" not in path.relative_to(result_root).parts:
                    continue
                if self._quality_sha256_token(sha256_file(path)) != request_sha:
                    continue
                if path.is_symlink() or path.resolve() != path.absolute():
                    raise ValueError("augmentation Quality request is not an ordinary local file")
                manifest = json.loads(path.read_text(encoding="utf-8"))
                self._validate_quality_request_model_binding(model, manifest)
                actual = self._quality_request_identity(path, model_id=model, manifest=manifest)
                if actual.get("identity_valid") is not True or _quality_key(actual) != key:
                    raise ValueError("augmentation candidate descriptor differs from its imported request")
                if any(identity.get(field) != actual.get(field) for field in
                       ("endpoint_contract_hash", "runtime_precision_identity")):
                    raise ValueError("augmentation candidate descriptor endpoint/precision drift")
                _records, status, encoded = self._augmentation_original_reference(model)
                quality_request_from_manifest(path, reference_artifact={
                    "path": status["reference_path"], "sha256": status["reference_sha256"],
                    "size_bytes": status["reference_size_bytes"]}, reference_artifact_bytes=encoded)
                matched.append((path, actual))
            if not matched:
                raise ValueError("augmentation candidate request is absent from the normal local import")
            path, actual = matched[0]
            producer = actual["native_split_quality_binding"]
            # preserve_metadata archives metadata, not the downloaded raw tree.
            # These original remote_runs remain at their canonical path; new
            # downloads live under attempts/<session> and are never eligible.
            previous_root = self.run_dir / "models" / model / "benchmark_results"
            prior = []
            for binding_path in previous_root.glob("remote_runs/**/task_quality_inputs/native_split_quality_binding.json"):
                value = json.loads(binding_path.read_text(encoding="utf-8"))
                if _quality_key(value.get("preselection") or {}) != key:
                    continue
                if binding_path.is_symlink() or binding_path.resolve() != binding_path.absolute():
                    raise ValueError("augmentation original runtime binding is not an ordinary local file")
                original_index = getattr(self, "_supplement_source_index", None)
                if original_index is not None:
                    from .workflow.supplement import _verify_original_binding
                    _verify_original_binding(binding_path, binding_path.relative_to(self.run_dir).as_posix(), original_index)
                verified, reason = validate_native_split_quality_binding(value)
                if verified is None:
                    raise ValueError("augmentation original runtime binding invalid: " + reason)
                prior.append(verified)
            if not prior:
                raise ValueError("augmentation original measured runtime binding is missing")
            for previous in prior:
                if previous.get("eval_run_id") != self.run_id:
                    raise ValueError("augmentation original runtime binding belongs to another run")
                for role in ("part1_runtime", "source_part2_onnx", "build_part2_onnx", "engine"):
                    if any(previous["artifacts"][role][field] != producer["artifacts"][role][field]
                           for field in ("sha256", "size_bytes")):
                        raise ValueError("augmentation candidate runtime differs from the preserved measurement: " + role)
                for field in ("precision", "boundary_tensor_name", "boundary_tensor_shape", "boundary_tensor_dtype",
                              "boundary_layout", "boundary_transform", "dequant_scale", "dequant_zero_point"):
                    if previous["preselection"].get(field) != producer["preselection"].get(field):
                        raise ValueError("augmentation candidate boundary differs from the preserved measurement: " + field)
            old_identity = self._quality_request_identity_for_row_variant(original, "composed")
            old_sha = self._quality_sha256_token(old_identity.get("source_request_sha256"))
            if old_sha and old_sha != request_sha:
                raise ValueError("augmentation cannot replace an already bound original Quality request")
            for field in ("task_quality_gate", "task_quality_gates_by_variant", "quality_request_identities_by_variant",
                          "quality_source_variant", "quality_source_run_id", "quality_source_setup_ids",
                          "quality_evaluation_pending", "task_quality_policy", "task_quality_policy_sha256"):
                if field in candidate:
                    row[field] = copy.deepcopy(candidate[field])
            row["supplement_quality_source"] = {
                "source_request": str(path.relative_to(self.run_dir)), "source_request_sha256": request_sha,
                "candidate_report_source": candidate.get("source_path"),
                "preserved_performance_source": original.get("source_path"),
                "runtime_artifacts_match_preserved_measurement": True,
                "performance_fields_preserved": True,
            }
            output.append(row)
        return output

    def _augmentation_reference_reader(self):
        raw = getattr(self, "augmentation_reference_run", None)
        if not raw:
            raise ValueError("augmentation CPU reference source run is missing")
        source = Path(raw).expanduser()
        if not source.is_absolute() or source.resolve(strict=True) != source:
            raise ValueError("augmentation CPU reference source must be a canonical original run")
        current = Path(self.run_dir).resolve()
        if source == current or source in current.parents or current in source.parents:
            raise ValueError("augmentation and original CPU reference runs must be separate")
        manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("run_id") != source.name or str(self.run_id) == source.name:
            raise ValueError("augmentation CPU reference original run identity mismatch")
        reader = getattr(self, "_augmentation_original_reference_reader", None)
        if reader is None:
            reader = _OriginalReferenceReader(source)
            self._augmentation_original_reference_reader = reader
        elif reader.run_dir != source:
            raise ValueError("augmentation CPU reference source run changed")
        return reader

    def _augmentation_original_reference(self, model_id):
        if model_id not in {"yolo11l", "yolo26s", "yolov7_paper"}:
            raise ValueError("augmentation CPU reference model is outside the authorized cohort")
        reader = self._augmentation_reference_reader()
        reader._restore_management_cpu_reference(model_id)
        records, status, encoded = reader._management_reference_records(model_id)
        contract = status.get("quality_contract") or {}
        expected_model = self._canonical_model_sha256_for_quality(model_id)
        if (not expected_model or self._quality_sha256_token(
                (contract.get("model") or {}).get("sha256")) != expected_model):
            raise ValueError("augmentation CPU reference differs from the new model identity")
        if (status.get("task") != "detection" or len(records) != 5000
                or (contract.get("dataset") or {}).get("image_count") != 5000):
            raise ValueError("augmentation requires the original N5000 detection reference")
        # This annotates the new result only. The status file, source contract,
        # canonical reference path and reference payload retain old identities.
        status = dict(status, reference_reuse_mode="explicit_original_run",
                      source_evaluation_run=str(reader.run_dir),
                      source_evaluation_run_id=reader.run_id)
        return records, status, encoded

    def _schedule_management_cpu_reference(self, model_id, suite_dir):
        # Intentionally never call the generator: the new suite's source
        # contract is different and would schedule a fresh CPU Full inference.
        with self._management_services_lock:
            if self._management_admission_closed:
                return
            _records, status, _encoded = self._augmentation_original_reference(model_id)
            if model_id not in self._management_reference_futures:
                future = concurrent.futures.Future()
                future.set_result(status)
                self._management_reference_futures[model_id] = future
                self._management_reference_source_contracts[model_id] = self._quality_sha256_token(
                    status["source_contract_sha256"])

    def _restore_management_cpu_reference(self, model_id):
        # Reentry reads the same explicit source; it must not look for or create
        # an imported reference under the new run's identity.
        self._schedule_management_cpu_reference(model_id, None)

    def _management_reference_records(self, model_id):
        if model_id not in self._management_reference_futures:
            raise RuntimeError("augmentation CPU reference was not admitted for " + model_id)
        records, status, encoded = self._augmentation_original_reference(model_id)
        admitted = self._management_reference_futures[model_id].result()
        if any(status.get(key) != admitted.get(key) for key in (
                "reference_path", "reference_sha256", "reference_size_bytes",
                "source_contract_sha256", "source_evaluation_run_id")):
            raise ValueError("augmentation CPU reference changed after admission")
        return records, status, encoded
