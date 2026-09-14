from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/native_trt_from_benchmarkset.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("v27920_trt_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_onnx(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"], name=marker)],
        f"trt-cache-{marker}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    onnx.save(helper.make_model(graph), str(path))


def _write_fake_trtexec(path: Path, log: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#!{sys.executable}
import hashlib
import sys
from pathlib import Path

log = Path({str(log)!r})
with log.open("a", encoding="utf-8") as handle:
    handle.write(" ".join(sys.argv[1:]) + "\\n")
if "--help" in sys.argv[1:]:
    print("--memPoolSize")
    raise SystemExit(0)
source_arg = next((v for v in sys.argv[1:] if v.startswith("--onnx=")), "")
engine_arg = next((v for v in sys.argv[1:] if v.startswith("--saveEngine=")), "")
if not source_arg or not engine_arg:
    raise SystemExit(64)
source = Path(source_arg.split("=", 1)[1])
engine = Path(engine_arg.split("=", 1)[1])
engine.parent.mkdir(parents=True, exist_ok=True)
engine.write_bytes(b"fake-trt-engine\\0" + hashlib.sha256(source.read_bytes()).digest())
raise SystemExit(0)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    benchmark_set = tmp_path / "benchmark_set"
    full = benchmark_set / "models/regnet_x_1_6gf.onnx"
    part1 = benchmark_set / "b132/regnet_x_1_6gf_part1_native.onnx"
    part2 = benchmark_set / "b132/regnet_x_1_6gf_part2_native.onnx"
    _write_onnx(full, "full-v1")
    _write_onnx(part1, "part1-v1")
    _write_onnx(part2, "part2-v1")
    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps({"model_id": "regnet_x_1_6gf"}), encoding="utf-8",
    )
    return benchmark_set, full, part1, part2


def _command(
    *, benchmark_set: Path, cache: Path, trtexec: Path, report: Path,
    allow_build: bool = True,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--benchmark-set", str(benchmark_set),
        "--case", "b132",
        "--variants", "full,part1,part2",
        "--precision", "fp16",
        "--out-dir", str(cache),
        "--trtexec", str(trtexec),
        "--workspace-mode", "mempool",
        "--workspace-mb", "64",
        "--no-run-smoke",
        "--json-out", str(report),
    ]
    if not allow_build:
        command.append("--no-build")
    return command


def _invoke(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _calls(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


def test_same_contract_second_run_is_three_hits_without_any_trtexec(
    tmp_path: Path,
) -> None:
    benchmark_set, _full, _part1, _part2 = _fixture(tmp_path)
    cache = tmp_path / "cache"
    fake_log = tmp_path / "trtexec.log"
    trtexec = tmp_path / "bin/trtexec"
    _write_fake_trtexec(trtexec, fake_log)

    cold_report = tmp_path / "cold.json"
    cold = _invoke(_command(
        benchmark_set=benchmark_set,
        cache=cache,
        trtexec=trtexec,
        report=cold_report,
    ))
    assert cold.returncode == 0, cold.stdout
    assert len(_calls(fake_log)) == 3
    cold_rows = json.loads(cold_report.read_text(encoding="utf-8"))["artifacts"]
    assert [row["trt_cache"]["status"] for row in cold_rows] == [
        "MISS", "MISS", "MISS",
    ]
    assert all(row["trt_cache"]["outcome"] == "built" for row in cold_rows)

    engine_identity = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in cache.rglob("*.engine")
    }
    calls_before = fake_log.read_bytes()
    warm_report = tmp_path / "warm.json"
    warm = _invoke(_command(
        benchmark_set=benchmark_set,
        cache=cache,
        trtexec=trtexec,
        report=warm_report,
    ))
    assert warm.returncode == 0, warm.stdout
    assert fake_log.read_bytes() == calls_before
    assert "[trt-cache] HIT role=full model=regnet_x_1_6gf" in warm.stdout
    assert "[trt-cache] HIT role=part1 model=regnet_x_1_6gf" in warm.stdout
    assert "[trt-cache] HIT role=part2 model=regnet_x_1_6gf" in warm.stdout
    warm_rows = json.loads(warm_report.read_text(encoding="utf-8"))["artifacts"]
    assert all(row["trt_cache"]["status"] == "HIT" for row in warm_rows)
    assert all(row["build"]["compiler_dispatched"] is False for row in warm_rows)
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in cache.rglob("*.engine")
    } == engine_identity


def test_source_change_rebuilds_only_affected_part_and_logs_reason(
    tmp_path: Path,
) -> None:
    benchmark_set, _full, _part1, part2 = _fixture(tmp_path)
    cache = tmp_path / "cache"
    fake_log = tmp_path / "trtexec.log"
    trtexec = tmp_path / "bin/trtexec"
    _write_fake_trtexec(trtexec, fake_log)
    assert _invoke(_command(
        benchmark_set=benchmark_set,
        cache=cache,
        trtexec=trtexec,
        report=tmp_path / "cold.json",
    )).returncode == 0

    _write_onnx(part2, "part2-v2")
    changed_report = tmp_path / "changed.json"
    changed = _invoke(_command(
        benchmark_set=benchmark_set,
        cache=cache,
        trtexec=trtexec,
        report=changed_report,
    ))
    assert changed.returncode == 0, changed.stdout
    assert len(_calls(fake_log)) == 4
    assert "[trt-cache] MISS role=part2" in changed.stdout
    assert "reason=source_onnx_mismatch" in changed.stdout
    rows = json.loads(changed_report.read_text(encoding="utf-8"))["artifacts"]
    assert [row["trt_cache"]["status"] for row in rows] == [
        "HIT", "HIT", "MISS",
    ]


def test_no_build_rejects_bare_engine_without_dispatch(tmp_path: Path) -> None:
    benchmark_set, _full, _part1, _part2 = _fixture(tmp_path)
    cache = tmp_path / "cache"
    fake_log = tmp_path / "trtexec.log"
    trtexec = tmp_path / "bin/trtexec"
    _write_fake_trtexec(trtexec, fake_log)
    report = tmp_path / "no-build.json"

    full_engine = cache / "full/fp16/full_fp16.engine"
    full_engine.parent.mkdir(parents=True)
    full_engine.write_bytes(b"unsealed-engine")
    result = _invoke(_command(
        benchmark_set=benchmark_set,
        cache=cache,
        trtexec=trtexec,
        report=report,
        allow_build=False,
    ))
    assert result.returncode == 2
    assert not fake_log.exists()
    rows = json.loads(report.read_text(encoding="utf-8"))["artifacts"]
    assert rows[0]["trt_cache"]["reason"] == "receipt_missing"
    assert all(row["build"]["compiler_dispatched"] is False for row in rows)


def test_read_only_probe_reuses_existing_receipt_and_detects_builder_abi_change(
    tmp_path: Path,
) -> None:
    module = _load_builder()
    benchmark_set, full, _part1, _part2 = _fixture(tmp_path)
    cache = tmp_path / "cache"
    fake_log = tmp_path / "trtexec.log"
    trtexec = tmp_path / "bin/trtexec"
    _write_fake_trtexec(trtexec, fake_log)
    assert _invoke(_command(
        benchmark_set=benchmark_set,
        cache=cache,
        trtexec=trtexec,
        report=tmp_path / "cold.json",
    )).returncode == 0

    hit = module.probe_trt_cache(
        role="trt_full",
        model_id="regnet_x_1_6gf",
        source_onnx=full,
        cache_roots=[cache],
        precision="fp16",
        expected_builder_abi={
            "trtexec_path": str(trtexec),
            "trtexec_sha256": hashlib.sha256(trtexec.read_bytes()).hexdigest(),
        },
    )
    assert hit["status"] == "HIT"
    assert hit["reason"] == "compatible_receipt"
    assert hit["artifact_path"].endswith("full_fp16.engine")
    assert hit["identity"]
    calls_before = fake_log.read_bytes()

    full_engine = cache / "full/fp16/full_fp16.engine"
    exact, mismatch_reason = module._verify_engine_cache_candidate(
        source_onnx=full,
        engine=full_engine,
        receipt_path=full_engine.parent / "engine_build_receipt.json",
        trtexec=str(trtexec),
        precision="fp16",
        shapes="",
        workspace_mb=128,
        workspace_mode="mempool",
        retry_without_shapes=True,
        retry_workspace_alt=True,
        extra_build_args=[],
    )
    assert exact is None
    assert mismatch_reason == "build_contract_mismatch"
    assert fake_log.read_bytes() == calls_before

    trtexec.write_text(
        trtexec.read_text(encoding="utf-8") + "\n# changed builder ABI\n",
        encoding="utf-8",
    )
    miss = module.probe_trt_cache(
        role="trt_full",
        model_id="regnet_x_1_6gf",
        source_onnx=full,
        cache_roots=[cache],
        precision="fp16",
        expected_builder_abi={"trtexec_path": str(trtexec)},
    )
    assert miss["status"] == "MISS"
    assert miss["reason"] == "builder_abi_mismatch"
    assert fake_log.read_bytes() == calls_before
