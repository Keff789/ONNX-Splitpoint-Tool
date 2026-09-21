"""Physical hardware/subprocesses simulated; all repaired functions are real."""
from __future__ import annotations
import copy
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import native_full_baseline_eval_runner as runner
from onnx_splitpoint_tool.runners.native_full_input import prepare_and_seal_deepx_native_full_input
from onnx_splitpoint_tool.native_detection_postprocess import build_frozen_postprocess_contract, FrozenDetectionPostprocessor
from test_v27930_deepx_semantic_pre_nms import prepare_semantic_case, run_semantic_case, write_json, load_semantic_module
ROOT = Path(__file__).resolve().parents[1]

def prepare_runner_case(tmp_path, monkeypatch, *, model="yolo11l", task="detection", output=None):
    case = prepare_semantic_case(tmp_path, monkeypatch, output, model=model, task=task)
    package = types.ModuleType("splitpoint_runners"); package.__path__ = []
    monkeypatch.setitem(sys.modules, "splitpoint_runners", package)
    import onnx_splitpoint_tool.native_detection_postprocess as pp
    import onnx_splitpoint_tool.native_output_endpoint as ep
    import onnx_splitpoint_tool.runners.native_full_input as fi
    for name, module in {"native_detection_postprocess": pp, "native_output_endpoint": ep, "native_full_input": fi}.items():
        monkeypatch.setitem(sys.modules, "splitpoint_runners." + name, module)
    prepare_and_seal_deepx_native_full_input(image_path=case.image, input_contract=case.contract, task=case.task,
        out_dir=case.root / "results/deepx_m1_full/prepared_input", model=case.model, setup_id=case.setup_id, comparison_backend="deepx")
    (case.root / "b003").mkdir()
    (case.root / "b003/run_split_onnxruntime.py").write_text("# simulated external process\n")
    source = case.root / "models" / (case.model + ".onnx")
    source.parent.mkdir()
    source.write_bytes(b"synthetic source model; the accelerator boundary is simulated")
    case.contract["source_onnx_sha256"] = runner._sha256_file(source)
    write_json(case.full / "output_contract.json", case.contract)
    source_relative = "models/" + case.model + ".onnx"
    write_json(case.root / "benchmark_set.json", {
        "schema": "onnx-splitpoint/benchmark-set", "schema_version": 2,
        "model_name": case.model, "model": source_relative, "cases": ["b003"],
        "artifact_manifest": {"schema": "onnx-splitpoint/benchmark-set", "schema_version": 2,
                              "files": {"models": [source_relative]}, "counts": {"models": 1}},
    })
    case.ns = SimpleNamespace(setup_id=case.setup_id, comparison_backend="deepx", repetitions=3,
        image_map_data={case.model: {"b003": str(case.image)}}, dump_outputs=True,
        trt_precision="fp16", timeout=60, frames=3, warmup=0, duration_s=0, comparison_precision="uint8_cast_fp16")
    case.processes = []
    template = ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
    suite = types.ModuleType("v27930_suite_merge"); suite.__file__ = str(template)
    monkeypatch.setitem(sys.modules, suite.__name__, suite)
    exec(compile(template.read_text(), str(template), "exec"), suite.__dict__)
    semantic_module = load_semantic_module()
    monkeypatch.setattr(runner, "_suite_python_env", lambda *_: (sys.executable, {}, []))
    def external_process(command, **kwargs):
        kind = "semantic" if "native_full_semantic_dump.py" in str(command[1]) else "performance"
        case.processes.append(kind)
        def option(name): return command[command.index(name) + 1]
        if kind == "semantic":
            result = run_semantic_case(case, semantic_module, Path(option("--out-dir")), Path(option("--prepared-input-manifest")))
            write_json(Path(option("--json-out")), dict(result, ok=True))
            if getattr(case, "semantic_mutation", None):
                path = Path(result["output_manifest"])
                payload = json.loads(path.read_text())
                case.semantic_mutation(payload)
                write_json(path, payload)
        else:
            result = suite._run_deepx_prepared_feed_benchmark(case.root, case.full / "model.dxnn",
                {"benchmark_task": case.task, "model_id": case.model, "setup_id": case.setup_id},
                SimpleNamespace(runs=3, warmup=0, energy_measurement_only=True, throughput_frames=3, throughput_warmup_frames=0,
                    prepared_feed_image=str(case.image), prepared_input_manifest=option("--prepared-input-manifest"),
                    quality_evidence_model_id=case.model, quality_evidence_setup_id=case.setup_id), case.root / "results/deepx_m1_full")
            assert result["status"] == "ok", result
            write_json(case.root / "benchmark_results_deepx_m1_full_auto.json", [{"run_id": "deepx_m1_full", "backend": "deepx_m1", "variant": "full",
                "model_id": case.model, "setup_id": case.setup_id, "runtime_ok": True, "dxnn_path": str(case.full / "model.dxnn"),
                "execution_precision": "int8", "deepx_prepared_feed_benchmark": result}])
        return {"rc": 0, "timed_out": False, "stdout": "simulated physical runtime", "stderr": "", "cmd": command}
    monkeypatch.setattr(runner, "_run", external_process)
    return case

def measured_and_semantic(tmp_path, monkeypatch):
    case = prepare_runner_case(tmp_path, monkeypatch)
    semantic, blocked = runner._deepx_full_series_preflight(case.root, case.model, case.ns)
    assert blocked is None, blocked
    performance = runner._generic_full_via_suite(case.root, case.model, "native_full_deepx", "deepx_m1_full", case.ns,
        prepared_input_manifest=Path(semantic["input_manifest"]))
    assert performance["ok"] is True, performance
    performance["runtime_success"] = True
    return case, performance, semantic

def attach(case, row, semantic, module=runner):
    return module._attach_semantic_dump(row, case.root, case.model, "native_full_deepx", "deepx_m1_full", case.ns, precomputed_result=semantic)

def owned(row):
    return {k: copy.deepcopy(v) for k, v in row.items() if k.startswith(("frozen_", "completed_", "postprocess_")) or k in {
        "report", "runtime_success", "frames", "returncode", "timed_out", "host_postprocess_frozen", "normalization_frozen",
        "fps_makespan", "latency_mean_ms", "deepx_prepared_feed_benchmark"}}

def test_m01_genuine_semantic_and_suite_measurement_merge(tmp_path, monkeypatch):
    case, row, semantic = measured_and_semantic(tmp_path, monkeypatch)
    from onnx_splitpoint_tool.native_detection_postprocess import build_completed_detection_endpoint_attestation
    semantic["completed_task_endpoint_attestation"] = build_completed_detection_endpoint_attestation(
        semantic["frozen_host_postprocess_contract"], semantic["frozen_host_postprocess_result"],
        completed_frames=1, postprocess_completed_frames=1, source_endpoint_contract_hash=semantic["endpoint_contract_hash"])
    semantic.update(frames=1, completed_frames=1, postprocess_completed_frames=1, fps_makespan=9999., latency_mean_ms=.001)
    before = owned(row); result = attach(case, row, semantic)
    assert result["ok"] is True, result
    assert result["stage"] == "decoded_pre_nms"
    assert result["postprocess_completed_frames"] == 3
    assert result["completed_task_endpoint_attestation"]["completed_frames"] == 3
    assert owned(result) == before
    assert case.processes == ["semantic", "performance"]
    from onnx_splitpoint_tool.native_rate_endpoints import rate_endpoint_fields
    projected = rate_endpoint_fields(runner._aggregate_full_repetitions([result], requested=1))
    assert projected["completed_task_fps"] == result["fps_makespan"]
    assert projected["completed_task_measurement_times_s"] == [result["measured_makespan_s"]]

@pytest.mark.parametrize("mutation", ["empty_contract", "empty_result", "false", "different_contract", "bad_image", "bad_endpoint", "source_false", "wrong_stage", "wrong_signature"])
def test_m02_m04_failure_preserves_measurements(tmp_path, monkeypatch, mutation):
    case, row, semantic = measured_and_semantic(tmp_path, monkeypatch); before = owned(row)
    row["claim_ok"] = True
    if mutation == "empty_contract": semantic.update(frozen_host_postprocess_contract={}, frozen_host_postprocess_contract_sha256="")
    elif mutation == "empty_result": semantic["frozen_host_postprocess_result"] = {}
    elif mutation == "false": semantic["host_postprocess_frozen"] = False
    elif mutation == "different_contract":
        changed = build_frozen_postprocess_contract(model_id=case.model, outputs={"model_outputs": case.values[0]}, input_hw=[640, 640],
            original_wh=[641, 480], source_contract_family="decoded_pre_nms")
        semantic.update(frozen_host_postprocess_contract=changed, frozen_host_postprocess_contract_sha256=changed["contract_sha256"],
            frozen_host_postprocess_result=FrozenDetectionPostprocessor(changed).process({"model_outputs": case.values[0]}))
    elif mutation == "bad_image": semantic["input_image_sha256"] = "b" * 64
    elif mutation == "source_false": semantic["output_endpoint_attestation"]["attested"] = False
    elif mutation == "wrong_stage": semantic["stage"] = "raw_head"
    elif mutation == "wrong_signature": semantic["tensor_signature"] = {}
    else: semantic["endpoint_contract_hash"] = "b" * 64
    result = attach(case, row, semantic)
    assert result["ok"] is False and result["failure_reason"]
    assert result["claim_ok"] is False
    assert owned(result) == before
    assert result["runtime_success"] is True and result["returncode"] == 0

@pytest.mark.parametrize("missing", [False, True])
def test_m05_precomputed_wins_over_existing_manifest(tmp_path, monkeypatch, missing):
    case, row, semantic = measured_and_semantic(tmp_path, monkeypatch)
    row["output_dump_manifest"] = semantic["output_dump_manifest"]
    semantic.update(ok=False, failure_reason="explicit_semantic_failure")
    if missing: semantic.pop("frozen_host_postprocess_contract")
    result = attach(case, row, semantic)
    assert result["ok"] is False and result["semantic_dump_status"] == "failed"
    assert case.processes == ["semantic", "performance"]

def test_m05_manifest_only_performs_checked_preparation(tmp_path, monkeypatch):
    case, row, semantic = measured_and_semantic(tmp_path, monkeypatch)
    row["output_dump_manifest"] = semantic["output_dump_manifest"]
    result = runner._attach_semantic_dump(row, case.root, case.model, "native_full_deepx", "deepx_m1_full", case.ns)
    assert result["ok"] is True, result
    assert case.processes == ["semantic", "performance", "semantic"]

@pytest.mark.parametrize("bad", [False, True])
def test_m06_top_level_evidence_is_independent(tmp_path, monkeypatch, bad):
    case, row, semantic = measured_and_semantic(tmp_path, monkeypatch)
    row.pop("deepx_prepared_feed_benchmark"); before = owned(row)
    if bad: semantic.update(frozen_host_postprocess_contract={}, frozen_host_postprocess_contract_sha256="")
    result = attach(case, row, semantic)
    assert result["ok"] is (not bad), result
    assert owned(result) == before

def test_m06_explicit_false_performance_not_hidden_by_nested_true(tmp_path, monkeypatch):
    case, row, semantic = measured_and_semantic(tmp_path, monkeypatch); row["host_postprocess_frozen"] = False
    result = attach(case, row, semantic)
    assert result["ok"] is False and result["host_postprocess_frozen"] is False
    assert "host_postprocess_frozen" in result["semantic_performance_projection_conflict_fields"]

def test_m07_trt_receives_following_hotloop_contract(tmp_path, monkeypatch):
    case, _, semantic = measured_and_semantic(tmp_path, monkeypatch)
    result = runner._attach_semantic_dump({"ok": True, "frames": 3000, "fps_makespan": 50., "runtime_success": True},
        case.root, case.model, "native_full_tensorrt", "ort_tensorrt", case.ns, precomputed_result=semantic)
    assert result["ok"] is True and result["host_postprocess_frozen"] is True
    assert result["frozen_host_postprocess_contract"] == semantic["frozen_host_postprocess_contract"]
    assert result["frames"] == 3000

@pytest.mark.parametrize("task", ["classification", "direct_bn6"])
def test_m07_classification_and_direct_normalization_keep_owners(tmp_path, monkeypatch, task):
    import numpy as np
    from test_v27930_deepx_semantic_pre_nms import _set_endpoint
    output = np.linspace(-2, 2, 1000, dtype=np.float32)[None, :] if task == "classification" else np.array([[[10, 150, 120, 250, .9, 0]]], dtype=np.float32)
    case = prepare_runner_case(tmp_path, monkeypatch, model="resnet50" if task == "classification" else "yolo26s",
        task="classification" if task == "classification" else "detection", output=output)
    _set_endpoint(case, "classification_logits" if task == "classification" else "decoded_nms")
    if task == "direct_bn6":
        declared = json.loads((case.root / "output_contracts.json").read_text())
        declared["contracts"][0]["endpoint_mode"] = "decoded"
        write_json(case.root / "output_contracts.json", declared)
    semantic, blocked = runner._deepx_full_series_preflight(case.root, case.model, case.ns)
    assert blocked is None, blocked
    row = runner._generic_full_via_suite(case.root, case.model, "native_full_deepx", "deepx_m1_full", case.ns,
        prepared_input_manifest=Path(semantic["input_manifest"]))
    assert row["ok"] is True, row
    row["runtime_success"] = True
    before = owned(row)
    result = attach(case, row, semantic)
    assert result["ok"] is True, result
    assert owned(result) == before
    from onnx_splitpoint_tool.native_rate_endpoints import rate_endpoint_fields
    projected = rate_endpoint_fields(runner._aggregate_full_repetitions([result], requested=1))
    if task == "classification":
        # R9C: this fixture executes the current prepared loop through Top-k.
        # Historical logits-only evidence stays covered by the R9B regressions.
        assert projected["completed_task_fps"] == result["fps_makespan"]
        assert projected["host_output_fps"] is None
        assert result["postprocess_completed_frames"] == 3
        assert result["completed_task_stage"] == "classification_top1_top5"
    else:
        assert projected["completed_task_fps"] == result["fps_makespan"]
        assert projected["completed_task_measurement_times_s"] == [result["measured_makespan_s"]]
    if task == "direct_bn6":
        assert result["normalization_frozen"] is True
        negative = copy.deepcopy(semantic)
        negative["normalization_frozen"] = False
        result = attach(case, copy.deepcopy(row), negative)
        assert result["ok"] is False
        assert result["normalization_frozen"] is True
        assert owned(result) == before
