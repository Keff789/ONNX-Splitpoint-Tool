from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/native_hailo_trt_fifo_from_benchmarkset.py"
SPEC = importlib.util.spec_from_file_location("v2783_dual_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dual_endpoint_child_commands_are_isolated(tmp_path: Path) -> None:
    args = SimpleNamespace(
        benchmark_set=str(tmp_path / "benchmark_set"), case="b066",
        hw_arch="hailo8", precision="uint8_dequant_fp16",
        image=str(tmp_path / "image.jpg"), frames=1000, warmup=100,
        repetitions=3, duration_s=0.0, queue_depth=3,
        hailo_format="uint8", preprocess_mode="letterbox",
        letterbox_pad_value=114, raw_preprocess_scope="paper_image",
        build=True, run=True, device_id="", setup_id="orin_nx_hailo8_01",
        eval_run_id="run", source_run_id="hailo8_to_trt",
        model_id="yolov7_paper", native_split_quality_binding="",
        expected_runner_sha256="", expected_image_sha256="",
        expected_hef_sha256="", expected_engine_sha256="",
        expected_boundary_layout="", source_contract_sha256="",
        expected_executable_sha256="", copy_outputs=True,
    )
    raw = MODULE._dual_child_command(
        args=args, endpoint="raw_model_outputs", work=tmp_path,
        result_json=tmp_path / "raw.json", config_json=tmp_path / "raw-config.json",
    )
    completed = MODULE._dual_child_command(
        args=args, endpoint="completed_task", work=tmp_path,
        result_json=tmp_path / "completed.json", config_json=tmp_path / "completed-config.json",
    )
    assert raw[raw.index("--detection-endpoints") + 1] == "raw_model_outputs"
    assert completed[completed.index("--detection-endpoints") + 1] == "completed_task"
    assert raw[raw.index("--work-dir") + 1] != completed[completed.index("--work-dir") + 1]
    assert "--dump-outputs" in raw and "--dump-boundary" in raw
    assert "--dump-outputs" in completed and "--dump-boundary" in completed


def test_dual_endpoint_relation_requires_exact_boundary_and_raw_heads(tmp_path: Path) -> None:
    def manifest(root: Path, value: bytes) -> tuple[Path, Path]:
        root.mkdir(parents=True)
        output = root / "head.bin"; output.write_bytes(value)
        out_manifest = root / "out.json"
        out_manifest.write_text(
            '{"outputs":[{"index":0,"name":"head","dtype":"float32","shape":[1],"file":"head.bin"}]}'
        )
        boundary = root / "boundary.bin"; boundary.write_bytes(b"boundary")
        input_file = root / "input.bin"; input_file.write_bytes(b"input")
        boundary_manifest = root / "boundary.json"
        boundary_manifest.write_text(
            '{"dtype":"uint8","shape":[1,8],"nbytes":8,"file":"boundary.bin","input_dump":"input.bin"}'
        )
        return out_manifest, boundary_manifest

    raw_out, raw_boundary = manifest(tmp_path / "raw", b"same")
    completed_out, completed_boundary = manifest(tmp_path / "completed", b"same")
    hef = tmp_path / "part1.hef"; hef.write_bytes(b"hef")
    engine = tmp_path / "part2.engine"; engine.write_bytes(b"engine")
    base = {
        "hef": str(hef), "engine": str(engine),
        "input_image_sha256": "a" * 64,
    }
    raw = dict(base, native_fifo_output_manifest=str(raw_out), native_fifo_boundary_manifest=str(raw_boundary))
    completed = dict(base, native_fifo_output_manifest=str(completed_out), native_fifo_boundary_manifest=str(completed_boundary))
    relation = MODULE._dual_endpoint_relation(raw, completed)
    assert relation["verified"] is True
    assert relation["raw_outputs_exact"] is True
    assert relation["boundary_exact"] is True

    (tmp_path / "completed" / "head.bin").write_bytes(b"different")
    relation = MODULE._dual_endpoint_relation(raw, completed)
    assert relation["verified"] is False
    assert relation["raw_outputs_exact"] is False


def test_raw_detection_contract_does_not_require_completed_runtime(monkeypatch, tmp_path: Path) -> None:
    # Source-level regression: a raw C++ Detection endpoint remains a valid
    # command contract without the Python mixed-runtime completion closure.
    source = RUNNER.read_text(encoding="utf-8")
    assert 'or producer_impl != "hailo8_python_vstreams_fifo"' in source
    assert 'args.detection_endpoints == "dual"' in source
    assert 'args.detection_endpoints == "completed_task"' in source


def test_dual_parent_publishes_one_combined_result_with_two_endpoints(
    monkeypatch, tmp_path: Path,
) -> None:
    bs = tmp_path / "benchmark_set"
    bs.mkdir()
    (bs / "benchmark_set.json").write_text('{"model_name":"yolov7_paper"}')
    image = tmp_path / "image.jpg"; image.write_bytes(b"image")
    hef = tmp_path / "part1.hef"; hef.write_bytes(b"hef")
    engine = tmp_path / "part2.engine"; engine.write_bytes(b"engine")
    work = tmp_path / "work"; work.mkdir()
    args = SimpleNamespace(
        benchmark_set=str(bs), case="b066", hw_arch="hailo8",
        precision="uint8_dequant_fp16", image=str(image), frames=100,
        warmup=10, repetitions=1, duration_s=0.0, queue_depth=3,
        hailo_format="uint8", preprocess_mode="letterbox",
        letterbox_pad_value=114, raw_preprocess_scope="paper_image",
        build=True, run=True, device_id="", setup_id="orin_nx_hailo8_01",
        eval_run_id="run", source_run_id="hailo8_to_trt",
        model_id="yolov7_paper", native_split_quality_binding="",
        expected_runner_sha256="", expected_image_sha256="",
        expected_hef_sha256="", expected_engine_sha256="",
        expected_boundary_layout="", source_contract_sha256="",
        expected_executable_sha256="", copy_outputs=True,
        result_json="", config_json="",
    )

    def fake_run(command, check=False):  # noqa: ANN001, ARG001
        endpoint = command[command.index("--detection-endpoints") + 1]
        result_path = Path(command[command.index("--result-json") + 1])
        output_dir = Path(command[command.index("--output-dir") + 1])
        boundary_dir = Path(command[command.index("--boundary-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        boundary_dir.mkdir(parents=True, exist_ok=True)
        head = output_dir / "head.bin"; head.write_bytes(b"same-head")
        output_manifest = output_dir / "manifest.json"
        output_manifest.write_text(
            '{"outputs":[{"index":0,"name":"head","dtype":"float32",'
            '"shape":[1],"file":"head.bin"}]}'
        )
        boundary = boundary_dir / "boundary.bin"; boundary.write_bytes(b"boundary")
        prepared = boundary_dir / "input.bin"; prepared.write_bytes(bytes(range(24)))
        boundary_manifest = boundary_dir / "manifest.json"
        boundary_manifest.write_text(
            '{"dtype":"uint8","shape":[1,8],"nbytes":8,'
            '"file":"boundary.bin","input_dump":"input.bin"}'
        )
        payload = {
            "ok": True,
            "hef": str(hef),
            "engine": str(engine),
            "input_image_sha256": MODULE._sha256_file(image),
            "native_fifo_output_manifest": str(output_manifest),
            "native_fifo_boundary_manifest": str(boundary_manifest),
            "fps_makespan": 97.0 if endpoint == "raw_model_outputs" else 16.0,
            "handoff_ms": 0.3,
            "completion_tail_ms": 47.0 if endpoint == "completed_task" else None,
            "measurement_endpoint": endpoint,
        }
        if endpoint == "raw_model_outputs":
            options = {"task":"detection", "preprocess_mode_requested":"letterbox",
                       "preprocess_mode_effective":"letterbox", "letterbox_pad_value_requested":114,
                       "letterbox_pad_value_effective":114, "letterbox_pad_value":114}
            contract = {"complete":True, "input_image_sha256":MODULE._sha256_file(image),
                        "runtime_options": options,
                        "prepared_input_contract": {**options, "pad_value_effective":114,
                            "dtype":"uint8", "layout":"HWC", "shape":[2,4,3],
                            "source_image_sha256":MODULE._sha256_file(image)},
                        "artifacts":{"prepared_input":{"path":str(prepared),"sha256":MODULE._sha256_file(prepared)}}}
            contract["contract_sha256"] = MODULE._stable_json_sha256(contract)
            payload["native_command_contract"] = contract
        else:
            # The parent transfers the raw endpoint's actual payload, not the JPEG.
            source = Path(command[command.index("--prepared-input-rgb") + 1])
            assert source.read_bytes() == prepared.read_bytes()
            assert command[command.index("--expected-prepared-input-sha256") + 1] == MODULE._sha256_file(source)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(__import__("json").dumps(payload))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    rc = MODULE._run_detection_dual_endpoint(
        bs=bs, case="b066", work=work, args=args,
    )
    assert rc == 0
    combined = __import__("json").loads(
        (work / "native_fifo_results.json").read_text()
    )
    assert combined["ok"] is True
    assert combined["primary_performance_endpoint"] == "p2_output"
    assert combined["legacy_performance_endpoint"] == "raw_model_outputs"
    assert combined["application_performance_endpoint"] == "completed_detection"
    assert combined["legacy_application_performance_endpoint"] == "completed_task"
    assert combined["throughput_primary_fps"] == 97.0
    assert combined["application_throughput_fps"] == 16.0
    # Legacy top-level fps_makespan remains the Completed-Task value for the
    # existing Energy/Quality consumers; primary performance is explicit.
    assert combined["fps_makespan"] == 16.0
    assert combined["raw_model_outputs_fps_makespan"] == 97.0
    assert combined["completed_task_fps_makespan"] == 16.0
    assert combined["endpoint_relation_verified"] is True
    assert set(combined["endpoint_results"]) == {
        "raw_model_outputs", "completed_task",
    }
    rows = __import__("json").loads(
        (bs / "benchmark_results_native_fifo_b066.json").read_text()
    )["results"]
    assert len(rows) == 1
    assert rows[0]["throughput_primary_fps"] == 97.0
    assert rows[0]["completed_task_fps_makespan"] == 16.0
    assert rows[0]["p2_output_fps"] == 97.0
    assert rows[0]["completed_detection_fps"] == 16.0
