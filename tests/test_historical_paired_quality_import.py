"""Synthetic paired DeepX imports; original declarations are never rewritten."""
from copy import deepcopy
from dataclasses import replace

import pytest

from onnx_splitpoint_tool import native_detection_postprocess as native
from onnx_splitpoint_tool.native_command_contract import seal_native_command_contract
from onnx_splitpoint_tool.workflow import historical_quality_import as subject
from test_historical_quality_import import Fixture, OLD_IMPLEMENTATION


class PairedFixture(Fixture):
    def __init__(self, tmp_path, monkeypatch):
        super().__init__(tmp_path, monkeypatch)
        self.identity.update(direction="deepx_to_trt", comparison_backend="deepx",
                             setup_id="orin_nx_deepx_m1_01", precision="float32_layout_fp16")
        self.uid = "generic|yolo26s|DeepX|deepx_to_trt|b321"
        self.target.update(generic_uid=self.uid, prior_generic_sourceversion_reuse_admitted=False,
                           prior_paired_reuse_admitted=True, admission_mode="reused_prior_paired",
                           observation_mode="original_paired_completion")
        self.config["identity"] = {**self.identity, "comparison_backend": "deepx_m1"}
        self.qrow["identity"] = self.config["identity"]
        input_contract = {"name": "images", "dtype": "uint8", "layout": "HWC", "shape": [64, 64, 3],
                          "embedded_numeric_path": "dxcom_div255_detection_default"}
        self.metadata = {"backend": "deepx_to_trt", "boundary_tensor_count": 1,
            "model_id": "yolo26s", "case_id": "b321", "setup_id": "orin_nx_deepx_m1_01",
            "boundary_transform": "identity", "boundary_layout": "as_input", "metadata_sha256": "6" * 64,
            "boundary_tensor": {"runtime_name": "feature", "dtype": "float32",
                                "canonical_part2_shape": [1, 128, 16, 16],
                                "deepx_input_contract": input_contract}}
        self.artifacts["boundary_metadata"] = self.save("boundary", self.metadata)
        self.artifacts["engine_build_receipt"] = self.save("engine_receipt", {"synthetic": True})
        self.receipt = {"ok": True, "dxnn_sha256": self.artifacts["part1_runtime"]["sha256"],
                        "source_onnx_sha256": self.artifacts["source_part1_onnx"]["sha256"]}
        self.artifacts["part1_build_receipt"] = self.save("part1_receipt", self.receipt)
        self.command.update(backend="deepx_to_trt", comparison_backend="deepx",
                            setup_id=self.identity["setup_id"], precision=self.identity["precision"],
                            runner="scripts/native_deepx_trt_e2e_from_benchmarkset.py")
        self.command["runtime_options"]["producer_impl"] = "deepx_native_fifo"
        self.command["artifacts"]["dxnn"] = self.command["artifacts"].pop("hef")
        self.command["artifacts"]["prepared_input"] = {"path": "/original/input.npy", "sha256": "a" * 64}
        for role in ("boundary_metadata", "source_part2_onnx", "build_part2_onnx", "engine_build_receipt"):
            self.command["artifacts"][role] = self.artifacts[role]
        self.command = seal_native_command_contract(self.command)
        self.native.update(native_command_contract=self.command,
            native_command_contract_sha256=self.command["contract_sha256"],
            setup_id=self.identity["setup_id"], precision=self.identity["precision"],
            completed_frames=2, measurement_boundary="workers_ready_to_last_completed_task_frame",
            fifo_payload_ownership="owned_no_alias_to_deepx_runtime_output", boundary_copy_count=1,
            boundary_copy_policy="one_owned_contiguous_typed_copy_before_fifo_enqueue",
            strict_quality_boundary={"status": "exact_runtime_boundary_verified", "output_count": 1,
                "runtime_name": "feature", "shape": [1, 128, 16, 16], "dtype": "float32",
                "binding_boundary_metadata_sha256": "6" * 64},
            input_selection={"selection_method": "quality_bound_exact_input_contract_no_probe_or_fallback",
                "selected_shape": [64, 64, 3], "selected_dtype": "uint8", "source_input_contract": input_contract},
            completion_runtime_source_binding={"before": {"/native/native_detection_postprocess.py": OLD_IMPLEMENTATION},
                "after": {"/native/native_detection_postprocess.py": OLD_IMPLEMENTATION}, "unchanged": True})
        self.native["timing_input_binding"].update(canonical_rgb_sha256="7" * 64,
            sequence="single_exact_prepared_feed_repeated", preprocessing_timed=False)
        self.native["timing_input_binding"].pop("executed_rgb_matches_common_preprocessing")
        self.native.pop("completed_work_units_status")
        self.prepared["format"] = "npy"
        rgb = {"dtype": "uint8", "shape": [64, 64, 3], "tensor_sha256": "7" * 64}
        self.proof["completed_quality_binding"]["representative_input_comparison"] = {
            "prepared_tensor": rgb, "actual_runtime_input": {"images": deepcopy(rgb)}}
        implementation = native._implementation_artifacts()["native_detection_postprocess"]["sha256"]
        self.source_map = {"/original/splitpoint_runners/native_detection_postprocess.py": implementation}
        for rep in self.fps["repetitions"]:
            rep["runtime_source_binding"] = {"before": self.source_map, "after": self.source_map, "unchanged": True}
        self.original.update(native_command_contract=self.command, native_measurement_boundary={
            "task": "detection", "completion_mode": self.old["completion_mode"],
            "native_command_contract_sha256": self.command["contract_sha256"],
            "measured_task_count": 2, "warmup_count": 1, "repetitions": 3,
            "process_repetition_policy": "separate_processes",
            "timer_start": "workers_ready_before_first_prepared_input_admission",
            "timer_end": "last_required_task_completion_after_synchronized_host_output_and_task_postprocess",
            "preprocessing_included": False, "host_device_transfers_included": True,
            "handoff_included": True, "output_materialization_included": True,
            "pipeline_fill_included": True, "pipeline_drain_included": True,
            "warmup_fully_completed_and_drained_before_timer": True})
        self.original["runtime_config"]["runtime_source_files"] = [
            {"path": k, "sha256": v} for k, v in self.source_map.items()]
        self.prior = {"identity": self.identity, "model_sha256": self.target["model_sha256"],
                      "expected_runtime_artifacts": self.artifacts}
        self.cache = {"backend": "deepx", "model_id": "yolo26s", "case_id": "b321",
            "source_part1_onnx": {"declared_sha256": self.artifacts["source_part1_onnx"]["sha256"],
                                  "size_bytes": self.artifacts["source_part1_onnx"]["size_bytes"]},
            "part1_receipt": self.artifacts["part1_build_receipt"]}
        self.target["actual_artifact_authority"] = {
            "authority_kind": "prior_normal_product_single_input_plan_identity", "source_identity": self.identity,
            "bound_runtime_artifacts": self.artifacts, "source_row_index": 0}

    def bind(self):
        self.artifacts["boundary_metadata"] = self.save("boundary", self.metadata)
        self.artifacts["part1_build_receipt"] = self.save("part1_receipt", self.receipt)
        self.cache["part1_receipt"] = self.artifacts["part1_build_receipt"]
        original_native = self.save("native", self.native)
        self.original["fresh_native_cohort"] = {"rows": [{
            "raw_report": original_native["path"], "raw_report_sha256": original_native["sha256"],
            "raw_report_size_bytes": original_native["size_bytes"]}]}
        binding = super().bind()
        self.target["actual_artifact_authority"]["source_plan"] = self.save("prior_plan", {"rows": [self.prior]})
        self.target["source_authorities"].update(prior_paired_ledger_id="A|" + self.uid,
            cache_inventory={**self.save("cache", {"rows": [self.cache]}),
                "selector": {"backend": "deepx", "model_id": "yolo26s", "case_id": "b321"},
                "canonical_row_sha256": native.canonical_json_sha256(self.cache)})
        return binding


def test_paired_import_keeps_declared_executed_contracts_and_419_other_rows(tmp_path, monkeypatch):
    f = PairedFixture(tmp_path, monkeypatch)
    binding = f.bind()
    others = [{"generic_uid": "other-" + str(i), "fps": [i]} for i in range(419)]
    original = deepcopy([f.target, *others])
    rows = subject.import_historical_operator_quality(original, {f.uid: binding})
    assert rows[1:] == original[1:] and original[0] == f.target
    provenance = rows[0]["historical_completed_operator_quality_relation"]["runtime_reporting_provenance"]
    assert provenance["fps_declared_postprocessor_sha256"] == OLD_IMPLEMENTATION
    assert provenance["native_actual_postprocessor_sha256"] == OLD_IMPLEMENTATION
    assert provenance["fps_actual_postprocessor_sha256"] == provenance["quality_postprocessor_sha256"]
    assert provenance["original_contract_rewritten"] is False
    for field in ("effective_generic_fps", "effective_repeats", "attempts", "source_authorities", "full_product_identity"):
        assert rows[0][field] == original[0][field]
    assert "source_request_sha256" not in original[0]


@pytest.mark.parametrize("fault", ["unadmitted", "backend", "prior_row", "cache_source", "receipt",
    "boundary_dtype", "boundary_identity", "owned_fifo", "copy_count", "input_shape", "input_encoding",
    "rgb", "native_runtime", "generic_runtime", "native_count", "native_method"])
def test_paired_original_authority_runtime_input_and_owned_capture_fail_closed(tmp_path, monkeypatch, fault):
    f = PairedFixture(tmp_path, monkeypatch)
    if fault == "unadmitted": f.target["prior_paired_reuse_admitted"] = False
    elif fault == "backend": f.identity["comparison_backend"] = "hailo10h"
    elif fault == "prior_row": f.prior["model_sha256"] = "9" * 64
    elif fault == "cache_source": f.cache["source_part1_onnx"]["declared_sha256"] = "9" * 64
    elif fault == "receipt": f.receipt["dxnn_sha256"] = "9" * 64
    elif fault == "boundary_dtype": f.native["strict_quality_boundary"]["dtype"] = "float16"
    elif fault == "boundary_identity": f.metadata["boundary_transform"] = "cast"
    elif fault == "owned_fifo": f.native["fifo_payload_ownership"] = "borrowed"
    elif fault == "copy_count": f.native["boundary_copy_count"] = 0
    elif fault == "input_shape": f.native["input_selection"]["selected_shape"] = [1, 3, 64, 64]
    elif fault == "input_encoding": f.native["input_selection"]["source_input_contract"] = {"dtype": "float32"}
    elif fault == "rgb": f.native["timing_input_binding"]["canonical_rgb_sha256"] = "9" * 64
    elif fault == "native_runtime": f.native["completion_runtime_source_binding"]["unchanged"] = False
    elif fault == "generic_runtime":
        f.source_map["/original/splitpoint_runners/native_detection_postprocess.py"] = OLD_IMPLEMENTATION
    elif fault == "native_count": f.native["completed_frames"] = 1
    elif fault == "native_method": f.original["native_measurement_boundary"]["preprocessing_included"] = True
    with pytest.raises(ValueError):
        f.run()


@pytest.mark.parametrize("fault", ["same_path_new_body", "missing_pin", "ambiguous_pin", "wrong_size"])
def test_original_native_cohort_body_pin_cannot_be_replaced_by_caller_descriptor(tmp_path, monkeypatch, fault):
    f = PairedFixture(tmp_path, monkeypatch)
    binding = f.bind()
    if fault == "same_path_new_body":
        replacement = {**f.native, "unbound_replacement": True}
        changed = f.save("native", replacement)
        assert changed["path"] == binding.native_record["path"]
        assert changed["sha256"] != binding.native_record["sha256"]
        binding = replace(binding, native_record=changed)
    else:
        selected = f.original["fresh_native_cohort"]["rows"][0]
        if fault == "missing_pin": selected.pop("raw_report_sha256")
        elif fault == "ambiguous_pin": f.original["fresh_native_cohort"]["rows"].append(deepcopy(selected))
        elif fault == "wrong_size": selected["raw_report_size_bytes"] += 1
        changed = f.save("plan", f.plan)
        f.target["source_authorities"]["reused_original_plan"] = changed
        binding = replace(binding, fps_plan=changed)
    with pytest.raises(ValueError, match="original Native report|SHA identity"):
        subject.import_historical_operator_quality([f.target], {f.uid: binding})
