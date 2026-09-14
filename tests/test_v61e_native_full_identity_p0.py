from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


def _sealed_input_fixture(
    monkeypatch, module, tmp_path: Path, image_path: Path,
):
    artifact_root = (
        tmp_path / "native_full_outputs"
        / "model=resnet50" / "backend=native_full_deepx"
        / "setup=unspecified" / "comparison=unspecified"
    )
    artifact_root.mkdir(parents=True)
    tensor = artifact_root / "runtime_input.bin"
    tensor.write_bytes(np.zeros((224, 224, 3), dtype=np.uint8).tobytes())
    digest = hashlib.sha256(tensor.read_bytes()).hexdigest()
    semantic_identity = {"schema": "test-preprocessing", "version": 3}
    numeric_identity = {"schema": "test-numeric-input", "version": 3}
    manifest = artifact_root / "native_full_input_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    validated = {
        "runtime_input_file": str(tensor),
        "runtime_input_name": "input",
        "runtime_input_shape": [224, 224, 3],
        "runtime_input_dtype": "uint8",
        "runtime_input_bytes": tensor.stat().st_size,
        "runtime_input_sha256": digest,
        "runtime_input_layout": "HWC",
        "runtime_preprocessing_identity": semantic_identity,
        "runtime_preprocessing_sha256": "a" * 64,
        "runtime_numeric_input_identity": numeric_identity,
        "runtime_numeric_input_sha256": "b" * 64,
        "input_image": str(image_path),
        "input_image_sha256": hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest(),
    }
    monkeypatch.setattr(
        module,
        "_validated_runtime_input_manifest",
        lambda _path, **_kwargs: (dict(validated), "verified"),
    )
    return manifest, {
        "prepared_input_file": str(tensor),
        "prepared_input_sha256": digest,
        "prepared_input_file_sha256": digest,
        "prepared_input_bytes": tensor.stat().st_size,
        "prepared_input_name": "input",
        "prepared_input_shape": [224, 224, 3],
        "prepared_input_dtype": "uint8",
        "prepared_input_layout": "HWC",
        "runtime_preprocessing_identity": semantic_identity,
        "runtime_preprocessing_sha256": "a" * 64,
        "runtime_numeric_input_identity": numeric_identity,
        "runtime_numeric_input_sha256": "b" * 64,
        "prepared_input_source_image_id": image_path.name,
        "prepared_input_source_image_sha256": hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest(),
        "prepared_input_binding_verified": True,
        "prepared_input_source": "sealed_semantic_dump_runtime_tensor",
        "prepared_feed_contract_version": (
            "deepx-sealed-runtime-input-v3"
        ),
    }


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _full_manifest(
    root: Path, *, model: str = "resnet50", backend: str = "native_full_hailo8",
    setup_id: str = "h8", comparison_backend: str = "hailo8",
    contract_family: str = "classification_logits",
) -> Path:
    path = (
        root / "native_full_outputs" / f"model={model}" / f"backend={backend}"
        / f"setup={setup_id}" / f"comparison={comparison_backend}"
        / "native_full_outputs_manifest.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "onnx-splitpoint/runner-output-dump",
        "schema_version": 3,
        "model": model,
        "backend": backend,
        "setup_id": setup_id,
        "comparison_backend": comparison_backend,
        "case": "full",
        "execution_mode": "native_full_baseline",
        "task": "detection" if contract_family in {"raw_head", "decoded_nms"} else "classification",
        "contract_family": contract_family,
        "outputs": [],
    }), encoding="utf-8")
    return path


def test_infer_vstreams_receives_writable_c_contiguous_inputs() -> None:
    from onnx_splitpoint_tool.runners.backends.hailo_backend import _HailoSession

    class Pipe:
        def infer(self, inputs):
            arr = inputs["hef_input"]
            assert arr.flags.c_contiguous is True
            assert arr.flags.writeable is True
            return {"hef_output": np.asarray([[1.0]], dtype=np.float32)}

    session = object.__new__(_HailoSession)
    session._pipe = Pipe()
    session._network_group = object()
    session._network_group_params = object()
    session.persistent_activation = True
    session._active_handle = object()
    session.input_names = ["input"]
    session.runtime_input_shapes = {"input": (2, 2, 3)}
    session._input_name_canonical_to_hef = {"input": "hef_input"}
    session._input_contig_cache = {}
    session._hef_output_names = ["hef_output"]
    session._output_name_hef_to_canonical = {"hef_output": "output"}
    session.output_shapes = {"output": (1,)}

    readonly = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    readonly.setflags(write=False)
    assert readonly.flags.c_contiguous and not readonly.flags.writeable
    result = session.infer({"input": readonly})
    assert result["output"].tolist() == [1.0]


def test_generated_runner_hailo_staging_also_makes_readonly_inputs_writable() -> None:
    text = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    start = text.index("def _ensure_c_contiguous_cached")
    end = text.index("\ndef _adapt_tensor", start)
    namespace = {"np": np, "Dict": dict}
    exec("from __future__ import annotations\n" + text[start:end], namespace)

    readonly = np.arange(12, dtype=np.uint8).reshape(3, 4)
    readonly.setflags(write=False)
    staged = namespace["_ensure_c_contiguous_cached"]({}, "input", readonly)
    mapped = namespace["_make_tensor_map_contiguous"]({"input": readonly})["input"]
    for arr in (staged, mapped):
        assert arr.flags.c_contiguous is True
        assert arr.flags.writeable is True
        np.testing.assert_array_equal(arr, readonly)


def test_native_full_manifest_cross_setup_binding_hard_fails(tmp_path: Path) -> None:
    mod = _load_script("v61e_identity_validator_mismatch", Path("scripts/native_producer_validate_visualize.py"))
    manifest = _full_manifest(tmp_path, setup_id="h8")
    row = {
        "model": "resnet50", "backend": "native_full_hailo8", "case": "full",
        "setup_id": "h10", "comparison_backend": "hailo8",
        "execution_mode": "native_full_baseline",
        "output_dump_manifest": str(manifest),
    }
    with pytest.raises(mod.NativeFullBindingError, match="setup_id_mismatch"):
        mod._find_dump(None, [tmp_path], row)


def test_native_full_rebased_manifest_ambiguity_hard_fails(tmp_path: Path) -> None:
    mod = _load_script("v61e_identity_validator_ambiguous", Path("scripts/native_producer_validate_visualize.py"))
    manifests = [_full_manifest(tmp_path / name) for name in ("copy_a", "copy_b")]
    suffix = Path("native_full_outputs") / Path(*manifests[0].parts[manifests[0].parts.index("native_full_outputs") + 1:])
    remote = Path("/remote/eval") / suffix
    row = {
        "model": "resnet50", "backend": "native_full_hailo8", "case": "full",
        "setup_id": "h8", "comparison_backend": "hailo8",
        "execution_mode": "native_full_baseline",
        "output_dump_manifest": str(remote),
    }
    with pytest.raises(mod.NativeFullBindingError, match="ambiguous_native_full_binding"):
        mod._find_dump(None, [tmp_path / "copy_a", tmp_path / "copy_b"], row)


def test_native_full_raw_head_is_accelerator_only_and_not_claimable(tmp_path: Path) -> None:
    mod = _load_script("v61e_identity_validator_raw", Path("scripts/native_producer_validate_visualize.py"))
    manifest = _full_manifest(tmp_path, model="yolo26s", contract_family="raw_head")
    row = {
        "model": "yolo26s", "backend": "native_full_hailo8", "case": "full",
        "setup_id": "h8", "comparison_backend": "hailo8",
        "execution_mode": "native_full_baseline",
    }
    gate = mod._native_full_e2e_contract_gate(manifest, row, "detection")
    assert gate["e2e_claim_eligible"] is False
    assert gate["e2e_scope"] == "accelerator_only"
    assert gate["contract_consistent"] is False
    assert gate["e2e_contract_reason"] == "raw_head_without_frozen_timed_host_decode_nms"


def test_deepx_full_requires_explicit_performance_input_contract(tmp_path: Path) -> None:
    mod = _load_script("v61e_identity_validator_deepx", Path("scripts/native_producer_validate_visualize.py"))
    manifest = _full_manifest(
        tmp_path, backend="native_full_deepx", comparison_backend="deepx",
    )
    row = {
        "model": "resnet50", "backend": "native_full_deepx", "case": "full",
        "setup_id": "h8", "comparison_backend": "deepx",
        "execution_mode": "native_full_baseline",
    }
    blocked = mod._native_full_e2e_contract_gate(manifest, row, "classification")
    allowed = mod._native_full_e2e_contract_gate(
        manifest, {**row, "performance_input_contract_mode": "explicit"}, "classification",
    )
    assert blocked["e2e_claim_eligible"] is False
    assert blocked["e2e_contract_reason"] == "deepx_performance_input_contract_not_explicit"
    assert allowed["e2e_claim_eligible"] is True


def test_deepx_explicit_contract_is_default_and_probes_are_opt_in(monkeypatch, tmp_path: Path) -> None:
    mod = _load_script("v61e_deepx_contract", Path("scripts/native_full_semantic_dump.py"))
    tensor = np.zeros((1, 3, 2, 2), dtype=np.float32)
    hwc = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(
        mod, "_prepare_image_tensor",
        lambda *args, **kwargs: (tensor, hwc, {"layout": "NCHW"}),
    )
    from PIL import Image
    image = tmp_path / "sample.png"
    Image.fromarray(hwc, mode="RGB").save(image)
    contract = {"input": {
        "shape": [1, 3, 2, 2], "dtype": "float32", "layout": "NCHW",
        "normalization": "imagenet_mean_std", "color_space": "RGB",
        "preprocess_mode": "resize",
    }}
    strict = mod._deepx_candidate_tensors(image, contract, "classification")
    diagnostic = mod._deepx_candidate_tensors(
        image, contract, "classification", allow_diagnostic_probes=True,
    )
    assert [row[3] for row in strict] == ["contract"]
    assert len(diagnostic) > 1
    with pytest.raises(RuntimeError, match="explicit input contract is incomplete"):
        mod._deepx_candidate_tensors(
            image, {"input": {"shape": [1, 3, 2, 2], "dtype": "float32"}},
            "classification",
        )


def test_native_full_remote_resource_copies_match_sources() -> None:
    for name in (
        "native_full_baseline_eval_runner.py",
        "native_full_semantic_dump.py",
        "smoke_hailo10_hef_runner.py",
        "smoke_hailo10_full_from_benchmarkset.py",
        "native_producer_validate_visualize.py",
        "native_producer_final_report.py",
    ):
        assert (Path("scripts") / name).read_bytes() == (
            Path("onnx_splitpoint_tool/resources/remote_scripts") / name
        ).read_bytes()


def test_generated_deepx_full_path_is_explicit_contract_first() -> None:
    suite = Path("onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt").read_text(encoding="utf-8")
    binding = Path("onnx_splitpoint_tool/workflow/deepx_build_binding.py").read_text(encoding="utf-8")
    assert 'ONNX_SPLITPOINT_DEEPX_ALLOW_DIAGNOSTIC_FALLBACK", "0"' in suite
    assert '"status": "explicit_input_contract_missing"' in suite
    assert 'performance_source = "explicit_deepx_contract_required"' in suite
    for token in (
        '"dtype": "uint8"', '"layout": "HWC"',
        'preprocess_mode = "resize" if task_name == "classification" else "letterbox"',
    ):
        assert token in binding


def test_deepx_full_exact_counter_requires_requested_prepared_feed_count(monkeypatch, tmp_path: Path) -> None:
    mod = _load_script("v61e_deepx_exact_work_units", Path("scripts/native_full_baseline_eval_runner.py"))
    exact_image = tmp_path / "exact_pair.jpg"
    exact_image.write_bytes(b"exact-pair-image")
    prepared_manifest, prepared_binding = _sealed_input_fixture(
        monkeypatch, mod, tmp_path, exact_image,
    )
    monkeypatch.setattr(mod, "_suite_python_env", lambda _ns, _backend: (sys.executable, {}, []))
    monkeypatch.setattr(mod, "_first_case", lambda _root: ("b001", tmp_path, tmp_path / "runner.py"))
    monkeypatch.setattr(mod, "_resolve_image", lambda *_args, **_kwargs: (exact_image, "exact_map"))
    monkeypatch.setattr(mod, "_run", lambda *_args, **_kwargs: {"rc": 0, "timed_out": False})
    result = {
        "fps_makespan": 25.0,
        "latency_mean_ms": 40.0,
        "result_source": str(tmp_path / "benchmark_results_deepx.json"),
            "result_row": {
                "run_id": "deepx_m1_full",
                "runtime_ok": True,
                "performance_benchmark_source": "dx_engine_prepared_feed",
                "prepared_feed_contract_version": "deepx-sealed-runtime-input-v3",
                "completed_frames": 137,
                    "deepx_prepared_feed_benchmark": {
                        "image": str(exact_image),
                        "completed_frames": 137,
                        "completed_work_units": 137,
                        "completed_work_units_source":
                            "dx_engine_prepared_feed_timed_loop",
                        "completed_work_units_status":
                            "exact_runtime_counter",
                        "makespan_s": 5.48,
                        "fps_makespan": 25.0,
                        **prepared_binding,
                        "input_contract": {"input": {"dtype": "uint8", "shape": [224, 224, 3]}},
                    },
            },
    }
    monkeypatch.setattr(mod, "_metrics_from_rows", lambda _paths: dict(result))
    ns = SimpleNamespace(
        frames=137,
        warmup=0,
        timeout=30,
        duration_s=60.0,
        engine_build_python="auto",
        diagnostic_deepx_input_probes=False,
    )
    row = mod._generic_full_via_suite(
        tmp_path, "resnet50", "native_full_deepx", "deepx_m1_full", ns,
        prepared_input_manifest=prepared_manifest,
    )
    assert row["frames"] == 137
    assert row["completed_frames"] == 137
    assert row["completed_work_units"] == 137
    assert row["completed_work_units_status"] == "exact_runtime_counter"

    result["result_row"] = {
        "performance_benchmark_source": "dx_engine_prepared_feed",
        "prepared_feed_contract_version": "deepx-sealed-runtime-input-v3",
        "completed_frames": 136,
    }
    row = mod._generic_full_via_suite(
        tmp_path, "resnet50", "native_full_deepx", "deepx_m1_full", ns,
        prepared_input_manifest=prepared_manifest,
    )
    assert row["completed_frames"] is None
    assert row["completed_work_units"] is None
    assert row["completed_work_units_status"] == "unavailable_or_count_mismatch"


def test_generated_deepx_energy_loop_attests_exact_requested_frames() -> None:
    suite = Path("onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt").read_text(encoding="utf-8")
    plan = Path("scripts/native_producer_energy_plan.py").read_text(encoding="utf-8")
    assert 'if bool(getattr(args, "energy_measurement_only", False)):' in suite
    assert 'getattr(args, "throughput_frames", 0) or runs' in suite
    assert '"completed_frames": len(times)' in suite
    assert '"dx_engine_prepared_feed_timed_loop"' in suite
    assert '"dx_engine_prepared_feed_frozen_completion_timed_loop"' in suite
    assert "child = _shell_join_with_fresh_output" in plan
    assert "child = _wrap_runtime" not in plan
