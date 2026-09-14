from __future__ import annotations

import copy
import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.native_command_contract import (
    seal_native_command_contract,
    verify_native_command_contract,
)
from scripts import native_producer_final_report as final_report


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "native_hailo_trt_fifo_from_benchmarkset.py"
SPEC = importlib.util.spec_from_file_location(
    "native_hailo8_v272_completed_detection", RUNNER
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sealed_hailo8_command_contract(producer_impl: str) -> dict:
    python_sha = "1" * 64
    python_path = "/verified/python"
    resolved_python = "/verified/python-real"
    hailo_site = (
        "/home/nx/hailo_py/lib/python3.10/site-packages"
    )

    def artifact(name: str, digest: str) -> dict:
        return {
            "path": f"/verified/{name}",
            "sha256": digest,
        }

    payload = {
        "complete": True,
        "backend": "hailo8_to_trt",
        "model": "yolov7",
        "case": "b001",
        "precision": "uint8_cast_fp16",
        "setup_id": "orin_nx_hailo8_01",
        "comparison_backend": "hailo8",
        "runner": "scripts/native_hailo_trt_fifo_from_benchmarkset.py",
        "runner_sha256": "2" * 64,
        "python_executable": python_path,
        "interpreter_identity": {
            "executable": python_path,
            "resolved_executable": (
                resolved_python
                if producer_impl == "hailo8_python_vstreams_fifo"
                else python_path
            ),
            "executable_sha256": python_sha,
        },
        "benchmark_set": "/verified/benchmark_set",
        "input_image": "/verified/image.jpg",
        "input_image_sha256": "3" * 64,
        "artifacts": {
            "python_executable": {
                "path": python_path,
                "sha256": python_sha,
            },
            "hef": artifact("part1.hef", "4" * 64),
            "engine": artifact("part2.engine", "5" * 64),
            "native_executable": artifact("native", "6" * 64),
            "generated_cpp": artifact("main.cpp", "7" * 64),
            "cmake": artifact("CMakeLists.txt", "8" * 64),
        },
        "runtime_options": {
            "warmup": 1,
            "queue_depth": 2,
            "producer_impl": producer_impl,
        },
        "boundary_contract": {
            "boundary_layout_effective": "NCHW",
        },
    }
    if producer_impl == "hailo8_python_vstreams_fifo":
        payload["interpreter_identity"].update({
            "runtime_mode": (
                "system_tensorrt_with_process_local_hailo_sites"
            ),
            "process_local_extra_sites": [hailo_site],
        })
        payload["runtime_options"].update({
            "task": "detection",
            "mixed_runtime_site_policy": (
                "site.addsitedir_after_system_defaults"
            ),
            "process_local_extra_sites": [hailo_site],
        })
        payload["artifacts"]["native_executable"] = {
            "path": resolved_python,
            "sha256": python_sha,
        }
        payload["artifacts"]["native_trt_consumer_source"] = {
            "path": "/verified/native_trt_consumer.py",
            "sha256": "9" * 64,
        }
        payload["mixed_runtime_contract"] = {
            "status": "ready",
            "runtime_mode": (
                "system_tensorrt_with_process_local_hailo_sites"
            ),
            "site_policy": "site.addsitedir_after_system_defaults",
            "python_executable": python_path,
            "resolved_python_executable": resolved_python,
            "process_local_extra_sites": [hailo_site],
            "modules": {
                "tensorrt": (
                    "/usr/lib/python3.10/dist-packages/"
                    "tensorrt/__init__.py"
                ),
                "hailo_platform": (
                    f"{hailo_site}/hailo_platform/__init__.py"
                ),
                "numpy": (
                    "/home/nx/.local/lib/python3.10/"
                    "site-packages/numpy/__init__.py"
                ),
                "PIL": (
                    "/usr/local/lib/python3.10/dist-packages/"
                    "PIL/__init__.py"
                ),
            },
            "cudart": "libcudart.so",
            "native_trt_consumer_source": (
                "/verified/native_trt_consumer.py"
            ),
            "native_trt_consumer_source_sha256": "9" * 64,
            "source_closure_ok": True,
        }
    return seal_native_command_contract(payload)


def test_hailo8_detection_command_contract_verifies_mixed_runtime_closure():
    contract = _sealed_hailo8_command_contract(
        "hailo8_python_vstreams_fifo"
    )
    verified, status = verify_native_command_contract(contract)

    assert verified == contract
    assert status == "hash_schema_identity_and_artifacts_verified"


@pytest.mark.parametrize(
    "mutation",
    [
        "source_artifact_missing",
        "source_closure_false",
        "site_policy_pypath",
        "runtime_sites_drift",
        "identity_sites_drift",
        "mixed_sites_drift",
        "runtime_mode_drift",
        "module_missing",
        "source_path_drift",
        "source_sha_drift",
        "cpp_producer",
    ],
)
def test_hailo8_detection_command_contract_rejects_resealed_mixed_runtime_drift(
    mutation: str,
):
    contract = copy.deepcopy(
        _sealed_hailo8_command_contract(
            "hailo8_python_vstreams_fifo"
        )
    )
    contract.pop("contract_sha256")
    if mutation == "source_artifact_missing":
        contract["artifacts"].pop("native_trt_consumer_source")
    elif mutation == "source_closure_false":
        contract["mixed_runtime_contract"]["source_closure_ok"] = False
    elif mutation == "site_policy_pypath":
        contract["runtime_options"][
            "mixed_runtime_site_policy"
        ] = "PYTHONPATH"
    elif mutation == "runtime_sites_drift":
        contract["runtime_options"][
            "process_local_extra_sites"
        ] = ["/tmp/runtime-site"]
    elif mutation == "identity_sites_drift":
        contract["interpreter_identity"][
            "process_local_extra_sites"
        ] = ["/tmp/identity-site"]
    elif mutation == "mixed_sites_drift":
        contract["mixed_runtime_contract"][
            "process_local_extra_sites"
        ] = ["/tmp/mixed-site"]
    elif mutation == "runtime_mode_drift":
        contract["interpreter_identity"][
            "runtime_mode"
        ] = "hailo_venv"
    elif mutation == "module_missing":
        contract["mixed_runtime_contract"]["modules"].pop(
            "tensorrt"
        )
    elif mutation == "source_path_drift":
        contract["mixed_runtime_contract"][
            "native_trt_consumer_source"
        ] = "/verified/other_consumer.py"
    elif mutation == "source_sha_drift":
        contract["mixed_runtime_contract"][
            "native_trt_consumer_source_sha256"
        ] = "a" * 64
    elif mutation == "cpp_producer":
        contract["runtime_options"][
            "producer_impl"
        ] = "hailo8_cpp_vstreams_fifo"
    else:  # pragma: no cover - parametrization exhaustiveness
        raise AssertionError(mutation)

    verified, status = verify_native_command_contract(
        seal_native_command_contract(contract)
    )

    assert verified is None
    assert status.startswith("native_command_contract_")


class _Prepared:
    input_names = ["images"]


class _Backend:
    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def run(self, _prepared, _inputs):
        with self._lock:
            self.calls += 1
            value = self.calls
        return SimpleNamespace(
            outputs={
                "split_boundary": np.full(
                    (1, 4), value, dtype=np.float32
                )
            }
        )


class _TRT:
    inputs = ["trt_input"]
    shapes = {"trt_input": (1, 4)}
    dtypes = {"trt_input": np.dtype(np.float32)}

    def __init__(self) -> None:
        self.calls = 0

    def run(self, feeds):
        assert list(feeds) == ["trt_input"]
        self.calls += 1
        return {
            "head": np.asarray(feeds["trt_input"], dtype=np.float32)
        }


class _Completion:
    def __init__(self, *, fail_on: int | None = None) -> None:
        self.completed_count = 0
        self.fail_on = fail_on

    def process(self, _outputs):
        next_count = self.completed_count + 1
        if self.fail_on == next_count:
            raise RuntimeError("completion failed")
        self.completed_count = next_count
        return {"detections": []}

    def attestation(self, *, completed_work_units: int):
        endpoint = {
            "endpoint_contract_hash": "a" * 64,
            "output_endpoint_id": "decoded_nms",
        }
        comparison = {
            "endpoint_contract_hash": "b" * 64,
            "output_endpoint_id": "decoded_nms_portable",
        }
        return {
            "attested": True,
            "status": "passed",
            "observation_relation": "same_hotloop_sentinel",
            "exact_result_claim_bound": True,
            "completed_work_units": completed_work_units,
            "completion_count": self.completed_count,
            "execution_contract_sha256": "c" * 64,
            "completed_endpoint_contract": endpoint,
            "comparison_endpoint_contract": comparison,
            "artifact_sha256": "d" * 64,
            "schema_sha256": "e" * 64,
            "content_sha256": "f" * 64,
            "invocation_sha256": "1" * 64,
            "relation_sha256": "2" * 64,
        }


def _run(*, completion, warmup_completion=None, frames=4, warmup=0):
    backend = _Backend()
    trt = _TRT()
    result = MODULE._hailo8_python_fifo_run(
        backend,
        _Prepared(),
        {"images": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
        trt,
        frames=frames,
        warmup=warmup,
        queue_depth=2,
        duration_s=0.0,
        completion_runtime=completion,
        warmup_completion_runtime=warmup_completion,
        expected_boundary_name="split_boundary",
        expected_boundary_shape=[1, 4],
        expected_boundary_dtype="float32",
    )
    return backend, trt, result


def test_hailo8_fifo_counts_only_completed_detection_frames():
    measured = _Completion()
    warmup = _Completion()
    backend, trt, result = _run(
        completion=measured,
        warmup_completion=warmup,
        frames=5,
        warmup=2,
    )

    assert backend.calls == 7
    assert trt.calls == 7
    assert warmup.completed_count == 2
    assert measured.completed_count == 5
    assert result["completed_work_units"] == 5
    assert result["completed_frames"] == 5
    assert result["produced_frames"] == 5
    assert result["postprocess_completed_frames"] == 5
    assert result["postprocess_completion_verified"] is True
    assert result["postprocess_included"] is True
    assert result["measurement_boundary"] == (
        "workers_ready_to_last_completed_task_frame"
    )
    assert result["last_completion_source"] == (
        "same_hotloop_completed_task_sentinel"
    )
    assert result["completion_execution_attestation"]["status"] == "passed"
    assert result["boundary_copy_count"] == 1
    assert result["boundary_copy_total"] == 5


def test_hailo8_fifo_fails_closed_when_completion_fails():
    with pytest.raises(RuntimeError, match="completion failed"):
        _run(
            completion=_Completion(fail_on=2),
            frames=5,
        )


def test_hailo8_fifo_requires_separate_warmup_completion_runtime():
    with pytest.raises(
        RuntimeError,
        match="completion runtime missing from warmup",
    ):
        _run(completion=_Completion(), frames=1, warmup=1)


def test_hailo8_boundary_mapping_rejects_ambiguous_outputs():
    trt = _TRT()
    with pytest.raises(RuntimeError, match="exactly one Part1 output"):
        MODULE._hailo8_python_boundary(
            {
                "a": np.zeros((1, 4), dtype=np.float32),
                "b": np.zeros((1, 4), dtype=np.float32),
            },
            trt,
        )


def test_hailo8_energy_boundary_name_attestation_rejects_equal_size_alias():
    trt = _TRT()
    with pytest.raises(
        RuntimeError,
        match="runtime_boundary_name_mismatch",
    ):
        MODULE._hailo8_python_boundary(
            {
                "wrong_equal_size_output": np.zeros(
                    (1, 4), dtype=np.float32
                ),
            },
            trt,
            expected_name="attested_boundary",
            expected_shape=[1, 4],
            expected_dtype="float32",
        )


def test_energy_hotloop_receives_exact_attested_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prepared_file = tmp_path / "prepared.bin"
    prepared_file.write_bytes(
        np.zeros((1, 4), dtype=np.float32).tobytes()
    )
    hef = tmp_path / "part1.hef"
    engine = tmp_path / "part2.engine"
    executable = tmp_path / "python"
    image = tmp_path / "image.png"
    for path in (hef, engine, executable, image):
        path.write_bytes(b"x")
    options = {
        "warmup": 0,
        "dump_outputs": False,
        "dump_boundary": False,
        "task": "detection",
        "preprocess_mode_requested": "auto",
        "preprocess_mode_effective": "letterbox",
        "letterbox_pad_value_requested": 0,
        "letterbox_pad_value_effective": 0,
        "letterbox_pad_value": 0,
        "queue_depth": 2,
        "hailo_format": "float32",
        "copy_outputs": True,
        "producer_impl": "hailo8_python_vstreams_fifo",
        "completion_execution_contract": {
            "contract_sha256": "c" * 64,
        },
    }
    prepared_contract = {
        "name": "images",
        "shape": [1, 4],
        "dtype": "float32",
        "task": "detection",
        "preprocess_mode_requested": "auto",
        "preprocess_mode_effective": "letterbox",
        "letterbox_pad_value_requested": 0,
        "letterbox_pad_value_effective": 0,
        "letterbox_pad_value": 0,
        "pad_value_effective": 0,
    }
    binding = {
        "benchmark_set": str(tmp_path),
        "case": "b001",
        "precision": "fp16",
        "hw_arch": "hailo8",
        "input_image": str(image),
        "runtime_options": options,
        "prepared_input_contract": prepared_contract,
        "runtime_boundary_evidence": {
            "status": "exact_runtime_boundary_verified",
            "output_count": 1,
            "output_name": "attested_boundary",
            "output_shape": [1, 4],
            "output_dtype": "float32",
        },
        "artifacts": {
            "native_executable": {"path": str(executable)},
            "hef": {"path": str(hef)},
            "engine": {"path": str(engine)},
            "prepared_input": {"path": str(prepared_file)},
        },
    }
    args = SimpleNamespace(
        energy_preflight_attestation="attestation.json",
        energy_preflight_nonce="nonce",
        source_contract_sha256="s" * 64,
        energy_preflight_max_age_s=300.0,
        warmup=0,
        build=False,
        dump_outputs=False,
        dump_boundary=False,
        duration_s=1.0,
        benchmark_set=str(tmp_path),
        case="b001",
        precision="fp16",
        image=str(image),
        hw_arch="hailo8",
        queue_depth=2,
        hailo_format="float32",
        task="detection",
        preprocess_mode="auto",
        copy_outputs=True,
        letterbox_pad_value=0,
        result_json=str(tmp_path / "result.json"),
        frames=1,
        device_id="",
    )
    monkeypatch.setattr(
        MODULE,
        "load_split_energy_workload_binding",
        lambda *_args, **_kwargs: (binding, "verified"),
    )
    monkeypatch.setattr(
        MODULE,
        "verify_detection_completion_execution_contract",
        lambda contract: dict(contract),
    )
    monkeypatch.setattr(
        MODULE,
        "DetectionCompletionRuntime",
        lambda contract: SimpleNamespace(contract=contract),
    )
    fake_prepared = SimpleNamespace(input_names=["images"])
    fake_trt = object()
    monkeypatch.setattr(
        MODULE,
        "_open_hailo8_python_runtime",
        lambda **_kwargs: (object(), fake_prepared, fake_trt),
    )
    monkeypatch.setattr(
        MODULE,
        "_close_hailo8_python_runtime",
        lambda *_args: None,
    )
    captured = {}

    def fake_fifo(*_args, **kwargs):
        captured.update(kwargs)
        return {"completed_work_units": 3}

    monkeypatch.setattr(
        MODULE, "_hailo8_python_fifo_run", fake_fifo
    )

    assert MODULE._energy_workload_only(args) == 0
    assert captured["expected_boundary_name"] == "attested_boundary"
    assert captured["expected_boundary_shape"] == [1, 4]
    assert captured["expected_boundary_dtype"] == "float32"


def test_repetition_aggregate_preserves_fresh_runtime_scope():
    scope = (
        "fresh_hailo_vstreams_trt_completion_runtime_per_repetition"
    )
    rows = []
    for index in range(3):
        rows.append({
            "fps_makespan": 10.0 + index,
            "completed_work_units": 4,
            "runtime_instance_id": f"runtime-{index}",
            "repetition_runtime_scope": scope,
        })
    result = MODULE._aggregate_repetition_payloads(rows)
    assert result["repetition_runtime_scope"] == scope
    assert result["repetition_independence_verified"] is True
    assert result["fps_makespan"] == 11.0


def test_detection_runner_no_longer_contains_cpp_tail_blocker():
    source = RUNNER.read_text(encoding="utf-8")
    assert "detection_split_completed_tail_not_measured_in_cpp_hotloop" not in source
    assert '"runtime_api": "vstreams"' in source
    assert "same_hotloop_completed_task_sentinel" in source


def test_final_report_uses_verified_hailo8_python_producer_impl(
    tmp_path: Path,
):
    implementation = "hailo8_python_vstreams_fifo"
    contract = _sealed_hailo8_command_contract(implementation)
    result_path = tmp_path / "native_fifo_results.json"
    result_path.write_text(
        json.dumps({
            "ok": True,
            "producer_impl": implementation,
            "native_command_contract": contract,
            "native_command_contract_sha256": contract["contract_sha256"],
        }),
        encoding="utf-8",
    )
    analysis_dir = tmp_path / "analysis_tables"
    analysis_dir.mkdir()
    (analysis_dir / "native_fifo_eval_runner.json").write_text(
        json.dumps({
            "rows": [{
                "model": "yolov7",
                "case": "b001",
                "precision": "uint8_cast_fp16",
                "ok": True,
                "report": str(result_path),
            }],
        }),
        encoding="utf-8",
    )

    rows = final_report._rows_from_native_fifo_runner(tmp_path)

    assert len(rows) == 1
    assert rows[0]["producer_impl"] == implementation
    assert "python" in rows[0]["note"]


def test_final_report_rejects_mutated_or_conflicting_producer_impl():
    implementation = "hailo8_python_vstreams_fifo"
    contract = _sealed_hailo8_command_contract(implementation)
    source = {
        "producer_impl": implementation,
        "native_command_contract": contract,
        "native_command_contract_sha256": contract["contract_sha256"],
    }
    assert (
        final_report._verified_native_producer_impl(source)
        == implementation
    )

    mutated = copy.deepcopy(source)
    mutated["native_command_contract"]["runtime_options"][
        "producer_impl"
    ] = "hailo8_cpp_vstreams_fifo"
    assert final_report._verified_native_producer_impl(mutated) == ""

    conflicting = dict(source)
    conflicting["producer_impl"] = "hailo8_cpp_vstreams_fifo"
    assert final_report._verified_native_producer_impl(conflicting) == ""
