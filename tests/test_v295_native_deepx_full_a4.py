"""Exercise the actual shipped Full worker; only the physical SDK is simulated."""
from __future__ import annotations

import json
import sys
import types
import hashlib
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from onnx_splitpoint_tool.runners.harness.yolo import YOLOV7_PAPER_ONNX_SHA256
from test_v27926_deepx_full_decoded_pre_nms import _decoded_outputs, _suite


def _sdk(monkeypatch, contract, outputs, *, wait_failure=False):
    """Borrowed outputs are overwritten immediately when their slot is reused."""
    observed = SimpleNamespace(engines=[], submits=[], waits=[], serial=0)

    class Engine:
        def __init__(self, _path):
            self.jobs = {}
            self.next_job = 0
            observed.engines.append(self)

        def get_input_tensors_info(self):
            return [{"name": contract["input"]["name"], "shape": [1, *contract["input"]["shape"]], "dtype": np.uint8}]

        def get_output_tensors_info(self):
            return [{"name": f"sdk_head_{index}", "shape": list(value.shape), "dtype": value.dtype}
                    for index, value in enumerate(outputs)]

        def _values(self, feeds):
            if contract.get("task") == "classification":
                offset = int(feeds[0].flat[0])
                return [(value + offset).astype(value.dtype) for value in outputs]
            return outputs

        def run(self, feeds):
            # The serial structural probe uses the same explicit SDK batch.
            assert feeds[0].shape == (1, *contract["input"]["shape"])
            observed.serial += 1
            return [value.copy() for value in self._values(feeds)]

        def run_async(self, feeds, *, user_arg, output_buffer):
            assert feeds[0].shape == (1, *contract["input"]["shape"])
            assert feeds[0].dtype == np.uint8
            buffers, offset = [], 0
            assert output_buffer.nbytes == sum(value.nbytes for value in outputs)
            for source in outputs:
                buffers.append(np.ndarray(source.shape, dtype=source.dtype, buffer=output_buffer, offset=offset))
                offset += source.nbytes
            for target in buffers:
                target.fill(0)
            job = self.next_job
            self.next_job += 1
            self.jobs[job] = (user_arg, buffers, feeds)
            observed.submits.append((job, user_arg, len(self.jobs)))
            return job

        def wait(self, job):
            observed.waits.append(job)
            if wait_failure and (type(wait_failure) is bool or len(observed.waits) > int(wait_failure)):
                raise RuntimeError("synthetic wait failure")
            _, buffers, feeds = self.jobs.pop(job)
            for target, source in zip(buffers, self._values(feeds)):
                np.copyto(target, source)
            return buffers

    dx = types.ModuleType("dx_engine")
    dx.InferenceEngine = Engine
    monkeypatch.setitem(sys.modules, "dx_engine", dx)
    package = types.ModuleType("splitpoint_runners")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "splitpoint_runners", package)
    import onnx_splitpoint_tool.runners.native_full_input as input_module
    import onnx_splitpoint_tool.native_detection_postprocess as postprocess
    import onnx_splitpoint_tool.native_output_endpoint as endpoint
    for name, module in {"native_full_input": input_module, "native_detection_postprocess": postprocess,
                         "native_output_endpoint": endpoint}.items():
        monkeypatch.setitem(sys.modules, "splitpoint_runners." + name, module)
    return observed


def _prepare(tmp_path, monkeypatch, model, *, wait_failure=False):
    task = "detection" if "yolo" in model else "classification"
    bn6 = model in {"yolo26m", "yolo26s"}
    raw = model == "yolov7_paper"
    if task == "classification":
        outputs = [np.arange(1000, dtype=np.float32)[None], np.array([[3, 9]], dtype=np.int32)]
    elif bn6:
        # Overlap deliberately survives: the source already completed NMS.
        outputs = [np.array([[[10, 10, 20, 20, .9, 0], [11, 11, 21, 21, .8, 0]]], dtype=np.float32)]
    elif raw:
        outputs = [np.full((1, 3, size, size, 85), -20, dtype=np.float32) for size in (80, 40, 20)]
    else:
        outputs = [_decoded_outputs()]
    contract = {
        "model_id": model, "task": task,
        "input": {"name": "images", "shape": [640, 640, 3] if task == "detection" else [32, 32, 3], "dtype": "uint8", "layout": "HWC",
                  "normalization": "embedded_dxcom_preprocessing", "color_space": "RGB",
                  "preprocess_mode": "letterbox" if task == "detection" else "resize",
                  "letterbox_pad_value": 114 if task == "detection" else 0},
        "outputs": [{"name": "logits" if task == "classification" and index == 0 else f"endpoint_{index}",
                     "shape": list(value.shape), "dtype": str(value.dtype)} for index, value in enumerate(outputs)],
    }
    if task == "detection":
        contract.update(contract_family="decoded_nms" if bn6 else "raw_head" if raw else "decoded_pre_nms",
                        endpoint_mode="decoded" if bn6 else "raw_detection_head" if raw else "decoded_pre_nms",
                        host_tail_required=not bn6)
        if raw:
            contract["source_onnx_sha256"] = YOLOV7_PAPER_ONNX_SHA256
        authority = {
            "model_id": model, "backend": "deepx_m1", "variant": "full", "task": task,
            "contract_status": "recorded", "endpoint_mode": contract["endpoint_mode"],
            "host_tail_required": not bn6, "postprocessing_required": not bn6,
        }
        if bn6:
            authority["source_coordinate_space"] = "model_input_letterbox_xyxy_pixels"
        (tmp_path / "output_contracts.json").write_text(json.dumps({"model_id": model, "task": task, "contracts": [authority]}))
    image = tmp_path / "timing.png"
    Image.new("RGB", (32, 32), (17, 43, 71)).save(image)
    dxnn = tmp_path / "existing.dxnn"
    dxnn.write_bytes(b"synthetic SDK handle; no compiler")
    observed = _sdk(monkeypatch, contract, outputs, wait_failure=wait_failure)
    suite = _suite()
    monkeypatch.setattr(suite, "_deepx_input_size_from_contract", lambda *_args, **_kw: (contract["input"]["shape"][0], contract))
    monkeypatch.setattr(suite, "_deepx_find_prepared_feed_image", lambda *_: (image, "fixed_test_image"))
    return suite, contract, outputs, dxnn, observed


@pytest.mark.parametrize("model", ["mobilenet_v3_large", "regnet_x_1_6gf", "resnet50", "yolo11l", "yolo26m", "yolo26s", "yolov7_paper"])
@pytest.mark.parametrize("warmup", [0, 3])
def test_actual_full_a4_all_families_preserve_completion(tmp_path, monkeypatch, model, warmup):
    suite, contract, outputs, dxnn, observed = _prepare(tmp_path, monkeypatch, model)
    args = SimpleNamespace(runs=5, warmup=warmup, deepx_runtime_mode="A4",
                           native_full_postprocess_runtime_mode="auto")
    result = suite._run_deepx_prepared_feed_benchmark(
        tmp_path, dxnn, {"model_id": model, "benchmark_task": contract["task"]}, args, tmp_path,
    )
    assert result["status"] == "ok", result
    assert len(observed.engines) == 1
    assert result["completed_frames"] == result["postprocess_completed_frames"] == 5
    assert result["submitted_count"] == 5
    assert result["actual_pending_max"] == 4
    assert result["accepted_work_drained"] is True
    assert result["warmup_completed_count"] == warmup
    assert all(not engine.jobs for engine in observed.engines)
    assert result["runtime_input_shapes"] == [[1, *contract["input"]["shape"]]]
    assert len(result["runtime_output_names"]) == len(outputs)
    assert result["endpoint_output_names"] == [item["name"] for item in contract["outputs"]]
    assert result["request_latency"]["status"] == "complete"
    assert result["request_latency"]["count"] == 5
    assert "generic_completion_evidence" not in result
    if contract["task"] == "classification":
        assert result["classification_topk"] == {"top1": [999], "top5": [[999, 998, 997, 996, 995]]}
    else:
        assert result["completed_task_result_artifact_saved"] is True
        if model in {"yolo26m", "yolo26s"}:
            assert result["frozen_decoded_nms_normalization_result"]["detection_count"] == 2
            assert result["frozen_decoded_nms_normalization_contract"]["host_nms_applied"] is False


def test_actual_full_wait_failure_preserves_unresolved_runtime(tmp_path, monkeypatch):
    suite, contract, _, dxnn, _ = _prepare(tmp_path, monkeypatch, "mobilenet_v3_large", wait_failure=True)
    result = suite._run_deepx_prepared_feed_benchmark(
        tmp_path, dxnn, {"model_id": "mobilenet_v3_large", "benchmark_task": contract["task"]},
        SimpleNamespace(runs=5, warmup=0, deepx_runtime_mode="A4"), tmp_path,
    )
    assert result["status"] == "runtime_failed", result
    assert result["unsafe_runtime_state"] is True
    assert result["accepted_work_drained"] is False
    assert result["completed_frames"] == 0


def test_actual_full_partial_warmup_is_never_reported_as_measured(tmp_path, monkeypatch):
    suite, contract, _, dxnn, _ = _prepare(tmp_path, monkeypatch, "mobilenet_v3_large", wait_failure=1)
    result = suite._run_deepx_prepared_feed_benchmark(
        tmp_path, dxnn, {"model_id": "mobilenet_v3_large", "benchmark_task": contract["task"]},
        SimpleNamespace(runs=5, warmup=3, deepx_runtime_mode="A4"), tmp_path,
    )
    assert result["status"] == "runtime_failed", result
    assert result["completed_frames"] == result["submitted_count"] == 0
    assert result["warmup_completed_count"] == 1
    assert result["failed_phase_counts"]["phase"] == "warmup"


def test_actual_full_rejects_unbound_extra_head_before_submit(tmp_path, monkeypatch):
    suite, contract, _, dxnn, observed = _prepare(tmp_path, monkeypatch, "resnet50")
    contract["outputs"].pop()
    result = suite._run_deepx_prepared_feed_benchmark(
        tmp_path, dxnn, {"model_id": "resnet50", "benchmark_task": "classification"},
        SimpleNamespace(runs=5, warmup=0, deepx_runtime_mode="A4"), tmp_path,
    )
    assert result["status"] == "engine_init_failed", result
    assert "endpoint_output_order_invalid" in result["error"]
    assert observed.submits == []


@pytest.mark.parametrize("model", ["mobilenet_v3_large", "regnet_x_1_6gf", "resnet50", "yolo11l", "yolo26m", "yolo26s", "yolov7_paper"])
def test_actual_full_numerical_sequence_compares_every_head_and_endpoint(tmp_path, monkeypatch, model):
    suite, contract, _, dxnn, observed = _prepare(tmp_path, monkeypatch, model)
    paths = [tmp_path / "timing.png"]
    for index, size in enumerate([(39, 29), (27, 43)]):
        path = tmp_path / f"mixed_{index}.png"
        Image.new("RGB", size, (80 + index, 23, 49)).save(path)
        paths.append(path)
    manifest = tmp_path / "numerical_images.json"
    manifest.write_text(json.dumps({"images": [path.name for path in paths]}))
    result = suite._run_deepx_prepared_feed_benchmark(
        tmp_path, dxnn, {"model_id": model, "benchmark_task": contract["task"]},
        SimpleNamespace(runs=5, warmup=0, deepx_runtime_mode="A4", native_full_postprocess_runtime_mode="auto",
                        deepx_numerical_images=str(manifest), deepx_numerical_only=True),
        tmp_path,
    )
    assert result["status"] == "ok", result
    validation = result["numerical_validation"]
    assert validation["status"] == "passed"
    assert validation["serial_reference_count"] == validation["mixed_completed_count"] == 3
    assert validation["all_outputs_compared"] is validation["completed_endpoints_compared"] is True
    assert validation["accepted_work_drained"] is True
    assert validation["actual_pending_max"] == 3
    assert result["numerical_smoke_completed_count"] == 5
    assert len(observed.submits) == 8


@pytest.mark.parametrize("duration,warmup", [(0.0, 0), (0.0, 100), (.01, 0)])
def test_actual_full_energy_a4_counts_every_completion_after_drain(tmp_path, monkeypatch, duration, warmup):
    from scripts import native_deepx_full_energy_hotloop as hotloop
    from test_v275_deepx_sealed_runtime_input import _sealed_input_fixture

    manifest, _, _, feed = _sealed_input_fixture(tmp_path)
    metadata = json.loads(manifest.read_text())
    tensor = Path(metadata["runtime_input_file"])
    dxnn = tmp_path / "existing.dxnn"
    dxnn.write_bytes(b"existing artifact, no compiler")
    outputs = [np.array([[0, 7, 2, 3, 4, 1]], dtype=np.float32), np.array([[1, 2]], dtype=np.int16)]
    observed = _sdk(monkeypatch, {"input": {"name": metadata["runtime_input_name"], "shape": list(feed.shape[1:])}}, outputs)
    runner = Path(hotloop.__file__).resolve()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    now = time.time_ns()
    preflight = tmp_path / "preflight.json"
    proof = dict(schema=hotloop.PREFLIGHT_SCHEMA, schema_version=1, ok=True, nonce="a4_full_test",
                 command_contract_sha256="a" * 64, artifact_verification_status="pass", host=socket.gethostname(),
                 created_at_unix_ns=now, expires_at_unix_ns=now + 60_000_000_000,
                 verified_artifact_sha256=dict(hotloop_runner=sha(runner), dxnn=sha(dxnn), runtime_input_tensor=sha(tensor)))
    proof["attestation_sha256"] = hotloop._canonical_json_sha256(proof)
    preflight.write_text(json.dumps(proof))
    options = {
        "dxnn": dxnn, "prepared-input-file": tensor, "prepared-feed-contract-version": hotloop.CONTRACT_VERSION,
        "expected-prepared-input-path": tensor, "expected-prepared-input-root": tmp_path,
        "expected-prepared-input-sha256": sha(tensor), "expected-prepared-input-bytes": metadata["runtime_input_bytes"],
        "expected-prepared-input-name": metadata["runtime_input_name"],
        "expected-prepared-input-shape-json": json.dumps(metadata["runtime_input_shape"]),
        "expected-prepared-input-dtype": metadata["runtime_input_dtype"], "expected-prepared-input-layout": metadata["runtime_input_layout"],
        "runtime-preprocessing-identity-json": json.dumps(metadata["runtime_preprocessing_identity"]),
        "expected-runtime-preprocessing-sha256": metadata["runtime_preprocessing_sha256"],
        "runtime-numeric-input-identity-json": json.dumps(metadata["runtime_numeric_input_identity"]),
        "expected-runtime-numeric-input-sha256": metadata["runtime_numeric_input_sha256"], "original-image-wh-json": "[2,2]",
        "task": "classification", "frames": 5, "warmup": warmup, "duration-s": duration, "json-out": tmp_path / "energy.json",
        "expected-runner-sha256": sha(runner), "expected-runner-path": runner, "expected-runner-root": runner.parent,
        "expected-dxnn-sha256": sha(dxnn), "expected-dxnn-path": dxnn, "expected-dxnn-root": tmp_path,
        "source-contract-sha256": "a" * 64, "preflight-attestation": preflight, "preflight-nonce": "a4_full_test",
        "deepx-runtime-mode": "A4", "endpoint-output-names-json": '["logits","auxiliary"]',
    }
    monkeypatch.setattr(sys, "argv", [str(runner)] + [part for key, value in options.items() for part in ["--" + key, str(value)]])
    assert hotloop.main() == 0
    result = json.loads((tmp_path / "energy.json").read_text())
    completed = result["completed_work_units"]
    assert result["deepx_runtime_mode"] == "A4"
    assert result["actual_pending_max"] == 4
    assert result["accepted_work_drained"] is True
    assert completed == result["submitted_count"] == result["postprocess_completed_frames"]
    assert result["warmup_completed_count"] == warmup
    assert len(observed.submits) == completed + warmup
    assert all(not engine.jobs for engine in observed.engines)
    assert result["classification_topk"] == {"top1": [1], "top5": [[1, 4, 3, 2, 5]]}
    if duration:
        assert result["measurement_control"] == "duration_admission_then_complete_drain"
        assert result["admission_stopped_at_deadline"] is True
    else:
        assert completed == 5


def test_full_summary_preserves_observed_native_policy_only_for_deepx(tmp_path):
    from scripts import native_producer_final_report as report
    rows = []
    for backend in ("native_full_deepx", "native_full_hailo8"):
        rows.append({"backend": backend, "model": "mobilenet_v3_large", "ok": True, "outer_makespan_verified": True,
                     "fps_makespan": 100, "task": "classification", "deepx_runtime_mode": "A4", "deepx_max_inflight": 4,
                     "actual_pending_max": 3, "accepted_work_drained": True, "deepx_async_observation": {"observed_max_inflight": 3}})
    destination = tmp_path / "analysis_tables/native_full_baseline_eval.json"
    destination.parent.mkdir()
    destination.write_text(json.dumps({"rows": rows}))
    projected = report._rows_from_native_full(tmp_path)
    assert projected[0]["deepx_runtime_mode"] == "A4"
    assert projected[0]["actual_pending_max"] == 3
    assert projected[0]["accepted_work_drained"] is True
    assert "deepx_runtime_mode" not in projected[1]


def test_outer_full_failure_retains_unresolved_sdk_ownership(tmp_path, monkeypatch):
    from scripts import native_full_baseline_eval_runner as runner
    from test_v27930_native_full_semantic_merge import prepare_runner_case

    case = prepare_runner_case(tmp_path, monkeypatch)
    semantic, blocked = runner._deepx_full_series_preflight(case.root, case.model, case.ns)
    assert blocked is None
    case.ns.deepx_runtime_mode = "A4"

    def failed_process(_command, **_kwargs):
        result = {"run_id": "deepx_m1_full", "backend": "deepx_m1", "variant": "full", "runtime_ok": False,
                  "deepx_prepared_feed_benchmark": {
                      "status": "runtime_failed", "deepx_runtime_mode": "A4", "deepx_max_inflight": 4,
                      "unsafe_runtime_state": True, "accepted_work_drained": False,
                      "deepx_async_observation": {"pending": 0, "unreleased": 1},
                      "unsafe_runtime_reasons": ["wait_unresolved"],
                  }}
        (case.root / "benchmark_results_deepx_m1_full_auto.json").write_text(json.dumps([result]))
        return {"rc": 5, "timed_out": False, "stdout": "", "stderr": "wait_unresolved"}

    monkeypatch.setattr(runner, "_run", failed_process)
    result = runner._generic_full_via_suite(case.root, case.model, "native_full_deepx", "deepx_m1_full", case.ns,
                                           prepared_input_manifest=Path(semantic["input_manifest"]))
    assert result["ok"] is False
    assert result["unsafe_runtime_state"] is True
    assert result["pending"] == 0 and result["unreleased"] == 1
    assert result["accepted_work_drained"] is False


@pytest.mark.parametrize("mode", [None, "S1", "A4"])
def test_serial_semantic_oracle_adds_only_the_a4_sdk_batch_axis(tmp_path, monkeypatch, mode):
    from test_v27930_deepx_semantic_pre_nms import prepare_semantic_case, load_semantic_module

    case = prepare_semantic_case(tmp_path, monkeypatch)
    observed = []

    class Engine:
        def __init__(self, path):
            assert Path(path) == case.full / "model.dxnn"

        def get_input_tensors_info(self):
            assert mode == "A4"
            return [{"name": "images", "shape": [1, 640, 640, 3], "dtype": np.uint8}]

        def get_output_tensors_info(self):
            assert mode == "A4"
            return [{"name": f"sdk_{index}", "shape": list(value.shape), "dtype": value.dtype}
                    for index, value in enumerate(case.values)]

        def run(self, feeds):
            assert feeds[0].shape == ((1, 640, 640, 3) if mode == "A4" else (640, 640, 3))
            observed.append(feeds[0].copy())
            return case.values

        def run_async(self, *_args, **_kwargs):
            raise AssertionError("the independent reference must remain serial")

        def wait(self, *_args, **_kwargs):
            raise AssertionError("the independent reference must remain serial")

    monkeypatch.setattr(sys.modules["dx_engine"], "InferenceEngine", Engine)
    module = load_semantic_module()
    result = module._run_deepx(case.root, case.image, case.root / "semantic", case.model, case.task,
                               case.setup_id, "deepx", **({"deepx_runtime_mode": mode} if mode is not None else {}))
    assert result["ok"] is True
    assert len(observed) == 1
    metadata = json.loads(Path(result["input_manifest"]).read_text())
    assert metadata["runtime_input_shape"] == [640, 640, 3]
    runtime_file = Path(metadata["runtime_input_file"])
    if not runtime_file.is_absolute():
        runtime_file = Path(result["input_manifest"]).parent / runtime_file
    assert runtime_file.read_bytes() == observed[0].tobytes()
    if mode == "A4":
        assert result["sdk_runtime_input_shapes"] == [[1, 640, 640, 3]]
        assert result["serial_reference_calls_started"] == result["serial_reference_calls_completed"] == 1
        assert result["reference_api"] == "serial_engine_run"
    else:
        assert "sdk_runtime_input_shapes" not in result


@pytest.mark.parametrize("mode", ["S1", "A4"])
def test_semantic_child_mode_option_is_only_added_for_explicit_a4(tmp_path, monkeypatch, mode):
    from scripts import native_full_baseline_eval_runner as runner
    from test_v27930_native_full_semantic_merge import prepare_runner_case

    case = prepare_runner_case(tmp_path, monkeypatch)
    case.ns.deepx_runtime_mode = mode
    calls = []
    original = runner._run

    def capture(command, **kwargs):
        calls.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(runner, "_run", capture)
    semantic, blocked = runner._deepx_full_series_preflight(case.root, case.model, case.ns)
    assert blocked is None
    assert semantic["ok"] is True
    assert len(calls) == 1
    if mode == "A4":
        assert calls[0][calls[0].index("--deepx-runtime-mode") + 1] == "A4"
    else:
        assert "--deepx-runtime-mode" not in calls[0]


def test_serial_numerical_sdk_exception_cannot_prove_idle_ownership(tmp_path, monkeypatch):
    suite, contract, _, dxnn, _ = _prepare(tmp_path, monkeypatch, "mobilenet_v3_large")

    def fail_serial(_self, _feeds):
        raise RuntimeError("serial SDK return was not observed")

    monkeypatch.setattr(sys.modules["dx_engine"].InferenceEngine, "run", fail_serial)
    result = suite._run_deepx_prepared_feed_benchmark(
        tmp_path, dxnn, {"model_id": "mobilenet_v3_large", "benchmark_task": "classification"},
        SimpleNamespace(runs=5, warmup=0, deepx_runtime_mode="A4",
                        deepx_numerical_images=json.dumps({"images": [str(tmp_path / "timing.png")]})), tmp_path,
    )
    assert result["status"] == "runtime_failed"
    assert result["unsafe_runtime_state"] is True
    assert result["serial_runtime_calls_started"] == 1
    assert result["serial_runtime_calls_completed"] == 0
    assert result["deepx_async_observation"]["unsafe_runtime_state"] is True
    assert result["numerical_validation"]["serial_reference_count"] == 0
