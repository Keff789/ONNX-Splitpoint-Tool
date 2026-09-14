from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.native_split_quality import (
    seal_native_split_quality_binding,
)
from onnx_splitpoint_tool.benchmark import remote_run
from tests.test_v269f_native_split_receipt_validation import _fixture_payload


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _builder_abi(trtexec: Path) -> dict[str, object]:
    return {
        "trtexec_path": str(trtexec),
        "trtexec_sha256": _sha(trtexec),
        "trtexec_size_bytes": trtexec.stat().st_size,
        "selected_gpu_target": {
            "visible_device_index": "0",
            "name": "Test GPU",
            "compute_capability": "8.7",
            "driver_api_version": "12020",
        },
        "linked_runtime_libraries": [{
            "name": "libnvinfer.so",
            "size_bytes": 3,
            "sha256": "1" * 64,
        }],
    }


def _builder_abi_sha256(abi: dict[str, object]) -> str:
    contract = remote_run._trt_engine_builder_abi_contract(abi)
    return hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def _owner(
    namespace: Path, *, builder_abi_sha256: str,
    stable_key: str = "",
) -> None:
    payload = {
        "schema": "onnx-splitpoint/managed-trt-cache-owner",
        "schema_version": 1,
        "owner": "onnx-splitpoint-tool",
        "cache_key": namespace.name,
        "trt_builder_abi_sha256": builder_abi_sha256,
    }
    if stable_key:
        payload["trt_engine_cache_key"] = stable_key
    (namespace / ".splitpoint_trt_cache_owner.json").write_text(
        json.dumps(payload)
    )


def _suite(tmp_path: Path, runs: list[dict[str, object]]) -> Path:
    suite = tmp_path / "suite"
    (suite / "models").mkdir(parents=True)
    (suite / "models/model.onnx").write_bytes(b"full-model-v1")
    (suite / "b024").mkdir()
    (suite / "b024/model_part2_b24.onnx").write_bytes(b"part2-model-v1")
    (suite / "b024/split_manifest.json").write_text("{}")
    (suite / "benchmark_set.json").write_text(json.dumps({
        "model_id": "model", "model": "models/model.onnx",
        "cases": [{"id": "b024"}],
    }))
    (suite / "benchmark_plan.json").write_text(json.dumps({"runs": runs}))
    return suite


def _vendor_native_quality_validator(suite: Path) -> None:
    vendored = suite / "splitpoint_runners"
    vendored.mkdir(exist_ok=True)
    project_root = Path(remote_run.__file__).resolve().parents[2]
    for name in ("native_command_contract.py", "native_split_quality.py"):
        (vendored / name).write_bytes(
            (project_root / "onnx_splitpoint_tool" / name).read_bytes()
        )


def _receipt(
    *, leaf: Path, source_bytes: bytes, builder: Path, role: str,
    precision_flags: tuple[str, ...] = ("--fp16",),
    workspace_mb: int = 4096,
    shapes: str = "",
    engine_precision: str = "fp16",
    extra_flags: tuple[str, ...] = (),
) -> None:
    leaf.mkdir(parents=True, exist_ok=True)
    source = leaf / "source.onnx"
    engine = leaf / f"{role}_{engine_precision}.engine"
    source.write_bytes(source_bytes)
    engine.write_bytes((role + "-engine").encode())
    payload = {
        "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
        "schema_version": 1,
        "build_returncode": 0,
        "dry_run": False,
        "command": [
            str(builder.resolve()), f"--onnx={source.resolve()}",
            f"--saveEngine={engine.resolve()}", *precision_flags,
            *([f"--shapes={shapes}"] if shapes else []),
            *([f"--workspace={workspace_mb}"] if workspace_mb > 0 else []),
            *extra_flags,
        ],
        "source_onnx": str(source.resolve()),
        "source_onnx_sha256": _sha(source),
        "engine": str(engine.resolve()),
        "engine_sha256": _sha(engine),
        "trtexec": str(builder.resolve()),
        "trtexec_sha256": _sha(builder),
    }
    payload["receipt_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    (leaf / "engine_build_receipt.json").write_text(json.dumps(payload))


class _LocalReadOnlyTransport:
    def __init__(self, remote_base: Path):
        self.host = SimpleNamespace(remote_base_dir=str(remote_base))
        self.commands: list[str] = []

    def resolve_path_read_only(self, value: str, timeout_s: int = 0) -> str:
        del value, timeout_s
        return str(Path(self.host.remote_base_dir).resolve())

    def run_read_only(self, command: str, timeout: int = 0):
        self.commands.append(command)
        result = subprocess.run(
            ["bash", "-c", command], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout or 30,
        )
        return result.returncode, result.stdout


def test_full_only_and_stage2_roles_are_exact_and_setup_scoped(tmp_path: Path) -> None:
    builder = tmp_path / "trtexec"
    builder.write_bytes(b"trtexec-v1")
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    base = tmp_path / "remote"
    base.mkdir()

    full_suite = _suite(tmp_path / "full", [{
        "id": "ort_tensorrt", "type": "onnxruntime", "variants": ["full"],
        "provider": "tensorrt",
        "stage1": {"provider": "tensorrt"},
        "stage2": {"provider": "tensorrt"},
    }])
    key = remote_run._stable_trt_engine_cache_key(
        full_suite, builder_abi=abi,
    )
    full_source = full_suite / "models/model.onnx"
    leaf = (
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key
        / remote_run._trt_persistent_engine_relative_dir(
            role="full", source_onnx_sha256=_sha(full_source),
            precision="fp16",
        )
    )
    _receipt(
        leaf=leaf, source_bytes=full_source.read_bytes(), builder=builder,
        role="full",
    )
    _owner(
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key,
        builder_abi_sha256=_builder_abi_sha256(abi),
    )
    result = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=full_suite,
        setup_id="orin_nx_hailo8_01", active_run_ids=["ort_tensorrt"],
        resolved_remote_base=str(base), builder_abi=abi,
    )
    assert [(row["role"], row["status"], row["item_id"]) for row in result["observations"]] == [
        ("trt_full", "HIT", "orin_nx_hailo8_01/full"),
    ]

    split_suite = _suite(tmp_path / "split", [{
        "id": "hailo8_to_trt", "type": "matrix",
        "stage1": {"hw_arch": "hailo8"},
        "stage2": {"provider": "tensorrt"},
        "variants": ["part1", "part2", "composed"],
        "case_id": "b024",
    }, {
        "id": "trt_to_hailo8", "type": "matrix",
        "stage1": {"provider": "tensorrt"},
        "stage2": {"hw_arch": "hailo8"},
        "variants": ["part1", "part2", "composed"],
    }])
    requirement = remote_run._trt_preflight_run_requirements(
        split_suite, active_run_ids=["hailo8_to_trt", "trt_to_hailo8"],
    )
    assert requirement["full_required"] is False
    assert requirement["p2_cases"] == ["b024"]

    # Scheduler rows sometimes retain the numeric boundary representation;
    # the probe/item contract must still address the canonical suite case.
    plan_path = split_suite / "benchmark_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["runs"][0]["case_id"] = 24
    plan_path.write_text(json.dumps(plan))
    numeric = remote_run._trt_preflight_run_requirements(
        split_suite, active_run_ids=["hailo8_to_trt"],
    )
    assert numeric["p2_cases"] == ["b024"]


def test_remote_probe_hits_current_and_compatible_legacy_part2(tmp_path: Path) -> None:
    suite = _suite(tmp_path / "input", [{
        "id": "hailo8_to_trt", "type": "matrix",
        "stage1": {"hw_arch": "hailo8"},
        "stage2": {"provider": "tensorrt"},
        "variants": ["composed"], "case_id": "b024",
    }])
    builder = tmp_path / "trtexec"
    builder.write_bytes(b"trtexec-v1")
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    base = tmp_path / "remote"
    base.mkdir()
    key = remote_run._stable_trt_engine_cache_key(suite, builder_abi=abi)
    source = suite / "b024/model_part2_b24.onnx"
    # This is the exact pre-v2.79.20 suite namespace consumed by runtime
    # migration, not merely an arbitrary same-prefix directory.
    legacy_key = remote_run._stable_suite_cache_key(suite)
    leaf = (
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516"
        / legacy_key / "b024/part2/fp16"
    )
    _receipt(
        leaf=leaf, source_bytes=source.read_bytes(), builder=builder,
        role="part2",
    )
    _owner(
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516"
        / legacy_key,
        builder_abi_sha256=_builder_abi_sha256(abi),
        stable_key=key,
    )
    transport = _LocalReadOnlyTransport(base)
    result = remote_run.probe_remote_trt_artifact_cache(
        transport=transport, suite_dir=suite,
        setup_id="orin_nx_hailo8_01", setup_accelerator="hailo8",
        active_run_ids=["hailo8_to_trt"],
        resolved_remote_base=str(base), builder_abi=abi,
    )
    assert result["status"] == "ok"
    assert result["hardware_action_performed"] is False
    assert len(transport.commands) == 1
    assert result["observations"][0]["status"] == "HIT"
    assert result["observations"][0]["role"] == "trt_p2"
    assert result["observations"][0]["item_id"] == "orin_nx_hailo8_01/b024"
    assert result["observations"][0]["source_namespace"] == "managed_legacy"


def test_tamper_is_miss_and_unreachable_probe_is_unknown(tmp_path: Path) -> None:
    suite = _suite(tmp_path / "input", [{
        "id": "ort_tensorrt", "type": "onnxruntime", "variants": ["full"],
        "provider": "tensorrt",
    }])
    builder = tmp_path / "trtexec"
    builder.write_bytes(b"trtexec-v1")
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    base = tmp_path / "remote"
    base.mkdir()
    key = remote_run._stable_trt_engine_cache_key(suite, builder_abi=abi)
    source = suite / "models/model.onnx"
    leaf = (
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key
        / remote_run._trt_persistent_engine_relative_dir(
            role="full", source_onnx_sha256=_sha(source), precision="fp16",
        )
    )
    _receipt(leaf=leaf, source_bytes=source.read_bytes(), builder=builder, role="full")
    _owner(
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key,
        builder_abi_sha256=_builder_abi_sha256(abi),
    )
    (leaf / "full_fp16.engine").write_bytes(b"tampered")
    miss = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="setup_a", active_run_ids=["ort_tensorrt"],
        resolved_remote_base=str(base), builder_abi=abi,
    )
    assert miss["observations"][0]["status"] == "MISS"
    assert miss["observations"][0]["reason"] == "engine_sha256_mismatch"

    class Unreachable(_LocalReadOnlyTransport):
        def resolve_path_read_only(self, value: str, timeout_s: int = 0) -> str:
            raise RuntimeError("network unreachable")

    unknown = remote_run.probe_remote_trt_artifact_cache(
        transport=Unreachable(base), suite_dir=suite,
        setup_id="setup_b", active_run_ids=["ort_tensorrt"],
        builder_abi=abi,
    )
    assert unknown["observations"][0]["status"] == "UNKNOWN"
    assert unknown["observations"][0]["item_id"] == "setup_b/full"
    assert unknown["observations"][0]["reason"] == "remote_trt_cache_probe_unavailable"


def test_receipt_precision_workspace_and_shape_contract_are_enforced(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path / "input", [{
        "id": "ort_tensorrt", "type": "onnxruntime", "variants": ["full"],
        "provider": "tensorrt", "native_trt_workspace_mb": 2048,
    }])
    builder = tmp_path / "trtexec"
    builder.write_bytes(b"trtexec-v1")
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    base = tmp_path / "remote"
    base.mkdir()
    key = remote_run._stable_trt_engine_cache_key(suite, builder_abi=abi)
    source = suite / "models/model.onnx"
    leaf = (
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key
        / remote_run._trt_persistent_engine_relative_dir(
            role="full", source_onnx_sha256=_sha(source), precision="fp16",
        )
    )

    # A fp32 engine must not accept the fp16 flag merely because the receipt
    # bytes and ABI are otherwise sound.
    _receipt(
        leaf=leaf, source_bytes=source.read_bytes(), builder=builder,
        role="full", workspace_mb=2048,
    )
    _owner(
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key,
        builder_abi_sha256=_builder_abi_sha256(abi),
    )
    fp32 = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="setup", active_run_ids=["ort_tensorrt"],
        args=SimpleNamespace(add_args="--native-trt-precision fp32"),
        resolved_remote_base=str(base), builder_abi=abi,
    )
    assert fp32["observations"][0]["status"] == "MISS"
    assert fp32["observations"][0]["reason"] == "build_contract_mismatch"

    int8 = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="setup", active_run_ids=["ort_tensorrt"],
        args=SimpleNamespace(add_args="--native-trt-precision int8"),
        resolved_remote_base=str(base), builder_abi=abi,
    )
    assert int8["observations"][0]["status"] == "MISS"
    assert int8["observations"][0]["reason"] == "build_contract_mismatch"

    # Restore fp16 but bind the wrong workspace.  This is also an incompatible
    # engine request, not a cache HIT.
    _receipt(
        leaf=leaf, source_bytes=source.read_bytes(), builder=builder,
        role="full", workspace_mb=1024,
    )
    workspace = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="setup", active_run_ids=["ort_tensorrt"],
        resolved_remote_base=str(base), builder_abi=abi,
    )
    assert workspace["observations"][0]["status"] == "MISS"
    assert workspace["observations"][0]["reason"] == "receipt_workspace_mismatch"

    # The intentionally tiny fake ONNX cannot establish a dynamic input
    # contract.  A non-empty --shapes must therefore remain UNKNOWN rather
    # than being guessed compatible or declared absent.
    _receipt(
        leaf=leaf, source_bytes=source.read_bytes(), builder=builder,
        role="full", workspace_mb=2048, shapes="input:1x3x640x640",
    )
    shape = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="setup", active_run_ids=["ort_tensorrt"],
        resolved_remote_base=str(base), builder_abi=abi,
    )
    assert shape["observations"][0]["status"] == "UNKNOWN"
    assert shape["observations"][0]["reason"] == "shape_contract_unavailable"


def test_legacy_namespace_must_bind_the_current_full_builder_abi(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path / "input", [{
        "id": "ort_tensorrt", "type": "onnxruntime", "variants": ["full"],
        "provider": "tensorrt",
    }])
    builder = tmp_path / "trtexec"
    builder.write_bytes(b"same-trtexec-on-two-gpu-abis")
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    base = tmp_path / "remote"
    base.mkdir()
    current_key = remote_run._stable_trt_engine_cache_key(
        suite, builder_abi=abi,
    )
    legacy_key = current_key.rsplit("-", 1)[0] + "-1111111111111111"
    namespace = (
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516"
        / legacy_key
    )
    source = suite / "models/model.onnx"
    leaf = namespace / "full/fp16"
    _receipt(
        leaf=leaf, source_bytes=source.read_bytes(), builder=builder,
        role="full",
    )
    _owner(namespace, builder_abi_sha256="7" * 64)

    result = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="gpu_87", active_run_ids=["ort_tensorrt"],
        resolved_remote_base=str(base), builder_abi=abi,
    )
    row = result["observations"][0]
    assert row["status"] == "MISS"
    assert row["reason"] == "legacy_owner_builder_abi_mismatch"

    (namespace / ".splitpoint_trt_cache_owner.json").unlink()
    missing_owner = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="gpu_87", active_run_ids=["ort_tensorrt"],
        resolved_remote_base=str(base), builder_abi=abi,
    )
    assert missing_owner["observations"][0]["status"] == "MISS"
    assert missing_owner["observations"][0]["reason"] == "legacy_owner_missing"


def test_active_run_ids_select_the_exact_runtime_contract_and_namespace(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path / "input", [{
        "id": "ort_tensorrt_fp32", "type": "onnxruntime",
        "provider": "tensorrt", "precision": "fp32",
        "workspace_mb": 1024,
    }, {
        "id": "hailo8_to_trt", "type": "matrix",
        "stage1": {"hw_arch": "hailo8"},
        "stage2": {"provider": "tensorrt"},
        "variants": ["composed"], "case_id": "b024",
        "native_trt_precision": "uint8_dequant_fp16",
        "native_trt_workspace_mb": 2048,
    }])
    builder = tmp_path / "trtexec"
    builder.write_bytes(b"trtexec")
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    selected = remote_run._trt_engine_runtime_contract(
        suite, active_run_ids=["hailo8_to_trt"],
    )
    assert selected["native_trt_precision"] == "uint8_dequant_fp16"
    assert selected["native_trt_workspace_mb"] == 2048
    assert selected["plan_precision_overrides"] == ["uint8_dequant_fp16"]
    assert selected["plan_workspace_overrides"] == [2048]

    selected_key = remote_run._stable_trt_engine_cache_key(
        suite, builder_abi=abi, active_run_ids=["hailo8_to_trt"],
    )
    unfiltered_key = remote_run._stable_trt_engine_cache_key(
        suite, builder_abi=abi,
    )
    assert selected_key != unfiltered_key

    base = tmp_path / "remote"
    base.mkdir()
    source = suite / "b024/model_part2_b24.onnx"
    leaf = (
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516"
        / selected_key
        / remote_run._trt_persistent_engine_relative_dir(
            role="part2", case_id="b024",
            source_onnx_sha256=_sha(source),
            precision="uint8_dequant_fp16",
        )
    )
    _receipt(
        leaf=leaf, source_bytes=source.read_bytes(), builder=builder,
        role="part2", workspace_mb=2048,
    )
    _owner(
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516"
        / selected_key,
        builder_abi_sha256=_builder_abi_sha256(abi),
    )
    result = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="setup", active_run_ids=["hailo8_to_trt"],
        resolved_remote_base=str(base), builder_abi=abi,
    )
    assert result["stable_key"] == selected_key
    assert result["observations"][0]["status"] == "MISS"
    assert result["observations"][0]["reason"] == (
        "special_precision_direct_source_not_allowed"
    )


def test_generic_uint8_cast_bridge_is_a_warm_part2_hit(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path / "input", [{
        "id": "hailo8_to_trt", "type": "matrix",
        "stage1": {"hw_arch": "hailo8"},
        "stage2": {"provider": "tensorrt"},
        "variants": ["composed"], "case_id": "b024",
        "native_trt_precision": "uint8_cast_fp16",
    }])
    builder = tmp_path / "trtexec"
    builder.write_bytes(b"trtexec-cast")
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    base = tmp_path / "remote"
    base.mkdir()
    key = remote_run._stable_trt_engine_cache_key(
        suite, builder_abi=abi, active_run_ids=["hailo8_to_trt"],
    )
    original = suite / "b024/model_part2_b24.onnx"
    leaf = (
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key
        / remote_run._trt_persistent_engine_relative_dir(
            role="part2", case_id="b024",
            source_onnx_sha256=_sha(original),
            precision="uint8_cast_fp16",
        )
    )
    _receipt(
        leaf=leaf, source_bytes=b"generated-uint8-cast-bridge",
        builder=builder, role="part2",
        engine_precision="uint8_cast_fp16",
    )
    bridge = leaf / "source.onnx"
    (leaf / "uint8_cast_bridge_meta.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/uint8-cast-bridge",
        "schema_version": 1,
        "source": str(original.resolve()),
        "source_sha256": _sha(original),
        "source_size_bytes": original.stat().st_size,
        "bridge": str(bridge.resolve()),
        "bridge_sha256": _sha(bridge),
        "bridge_size_bytes": bridge.stat().st_size,
        "input_name": "boundary",
        "input_dtype": "UINT8",
        "cast_output": "boundary__uint8_cast_to_float",
        "cast_to": "FLOAT",
        "replaced_uses": 1,
        "precision_tag": "uint8_cast_fp16",
    }))
    _owner(
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key,
        builder_abi_sha256=_builder_abi_sha256(abi),
    )

    result = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="h8", active_run_ids=["hailo8_to_trt"],
        resolved_remote_base=str(base), builder_abi=abi,
    )
    row = result["observations"][0]
    assert row["status"] == "HIT", row
    assert row["evidence"]["source_binding"] == "generic_uint8_cast_bridge"
    assert row["evidence"]["generic_bridge_source_verification"] == (
        "source_bytes_verified"
    )


def test_ort_runtime_never_claims_a_native_engine_hit_or_opens_ssh(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path / "input", [{
        "id": "ort_tensorrt", "type": "onnxruntime", "variants": ["full"],
        "provider": "tensorrt",
    }])

    class NoRemoteProbe:
        host = SimpleNamespace(remote_base_dir="/must/not/be/read")

        def resolve_path_read_only(self, *_args, **_kwargs):
            raise AssertionError("ORT UNKNOWN must precede remote resolution")

        def run_read_only(self, *_args, **_kwargs):
            raise AssertionError("ORT UNKNOWN must not open SSH")

    args = SimpleNamespace(add_args="--trt-runtime ort")
    contract = remote_run._trt_engine_runtime_contract(
        suite, args=args, active_run_ids=["ort_tensorrt"],
    )
    assert contract["trt_runtime"] == "ort"
    result = remote_run.probe_remote_trt_artifact_cache(
        transport=NoRemoteProbe(), suite_dir=suite,
        setup_id="trt", active_run_ids=["ort_tensorrt"], args=args,
    )
    assert result["observations"][0]["status"] == "UNKNOWN"
    assert result["observations"][0]["reason"] == (
        "ort_tensorrt_cache_probe_not_implemented"
    )
    assert remote_run._extract_trt_runtime_mode_from_add_args(
        "--trt-runtime-mode=native"
    ) == "native"


def test_runtime_environment_activation_selects_builder_abi(
    tmp_path: Path,
) -> None:
    builder = tmp_path / "env/bin/trtexec"
    builder.parent.mkdir(parents=True)
    builder.write_bytes(b"environment-specific-trtexec")
    builder.chmod(0o755)
    payload = _builder_abi(builder)

    class AbiTransport:
        command = ""

        def run_read_only(self, command: str, timeout: int = 0):
            del timeout
            self.command = command
            return 0, "SPLITPOINT_TRT_BUILDER_ABI=" + json.dumps(payload)

    transport = AbiTransport()
    observed = remote_run._remote_trt_builder_abi(
        transport,  # type: ignore[arg-type]
        remote_venv="source ~/venvs/trt/bin/activate",
    )
    assert observed["trtexec_sha256"] == _sha(builder)
    assert 'source "$HOME"/venvs/trt/bin/activate' in transport.command
    assert transport.command.index("source ") < transport.command.index(
        "SPLITPOINT_TRT_ABI_PY"
    )


def test_unknown_args_wrong_leaf_and_malformed_candidate_cannot_false_hit(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path / "input", [{
        "id": "ort_tensorrt", "type": "onnxruntime", "variants": ["full"],
        "provider": "tensorrt",
    }])
    builder = tmp_path / "trtexec"
    builder.write_bytes(b"trtexec")
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    base = tmp_path / "remote"
    base.mkdir()
    key = remote_run._stable_trt_engine_cache_key(suite, builder_abi=abi)
    namespace = base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key
    source = suite / "models/model.onnx"
    good_leaf = namespace / remote_run._trt_persistent_engine_relative_dir(
        role="full", source_onnx_sha256=_sha(source), precision="fp16",
    )
    _receipt(
        leaf=good_leaf, source_bytes=source.read_bytes(), builder=builder,
        role="full",
    )
    _owner(namespace, builder_abi_sha256=_builder_abi_sha256(abi))

    # An earlier malformed candidate is isolated; it cannot poison discovery
    # of the later exact canonical receipt.
    bad_leaf = namespace / "aaa-malformed"
    _receipt(
        leaf=bad_leaf, source_bytes=source.read_bytes(), builder=builder,
        role="full",
    )
    bad_receipt_path = bad_leaf / "engine_build_receipt.json"
    bad = json.loads(bad_receipt_path.read_text())
    bad["schema_version"] = "oops"
    bad.pop("receipt_sha256")
    bad["receipt_sha256"] = hashlib.sha256(json.dumps(
        bad, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    bad_receipt_path.write_text(json.dumps(bad))
    hit = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="setup", active_run_ids=["ort_tensorrt"],
        resolved_remote_base=str(base), builder_abi=abi,
    )
    assert hit["observations"][0]["status"] == "HIT"

    # A self-consistent receipt with an unbound build-affecting argument is
    # not compatible with the requested engine contract.
    _receipt(
        leaf=good_leaf, source_bytes=source.read_bytes(), builder=builder,
        role="full", extra_flags=("--builderOptimizationLevel=0",),
    )
    unknown_arg = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="setup", active_run_ids=["ort_tensorrt"],
        resolved_remote_base=str(base), builder_abi=abi,
    )
    assert unknown_arg["observations"][0]["status"] == "MISS"
    assert unknown_arg["observations"][0]["reason"] == (
        "receipt_unknown_build_args"
    )

    wrong_base = tmp_path / "wrong-remote"
    wrong_namespace = (
        wrong_base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key
    )
    wrong_leaf = wrong_namespace / "full" / _sha(source) / "fp32"
    _receipt(
        leaf=wrong_leaf, source_bytes=source.read_bytes(), builder=builder,
        role="full",
    )
    _owner(wrong_namespace, builder_abi_sha256=_builder_abi_sha256(abi))
    wrong = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(wrong_base), suite_dir=suite,
        setup_id="setup", active_run_ids=["ort_tensorrt"],
        resolved_remote_base=str(wrong_base), builder_abi=abi,
    )
    assert wrong["observations"][0]["status"] == "MISS"
    assert wrong["observations"][0]["reason"] == "canonical_leaf_mismatch"


def test_missing_current_owner_and_unreadable_inventory_are_unknown(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path / "input", [{
        "id": "ort_tensorrt", "type": "onnxruntime", "variants": ["full"],
        "provider": "tensorrt",
    }])
    builder = tmp_path / "trtexec"
    builder.write_bytes(b"trtexec")
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    base = tmp_path / "remote"
    base.mkdir()
    key = remote_run._stable_trt_engine_cache_key(suite, builder_abi=abi)
    source = suite / "models/model.onnx"
    leaf = (
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key
        / remote_run._trt_persistent_engine_relative_dir(
            role="full", source_onnx_sha256=_sha(source), precision="fp16",
        )
    )
    _receipt(
        leaf=leaf, source_bytes=source.read_bytes(), builder=builder,
        role="full",
    )
    missing = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="setup", active_run_ids=["ort_tensorrt"],
        resolved_remote_base=str(base), builder_abi=abi,
    )
    assert missing["observations"][0]["status"] == "UNKNOWN"
    assert missing["observations"][0]["reason"] == "current_owner_missing"

    malformed_base = tmp_path / "malformed-remote"
    cache_parent = malformed_base / "_onnx_splitpoint_cache"
    cache_parent.mkdir(parents=True)
    (cache_parent / "tensorrt_managed_v27516").write_text("not a directory")
    unreadable = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(malformed_base), suite_dir=suite,
        setup_id="setup", active_run_ids=["ort_tensorrt"],
        resolved_remote_base=str(malformed_base), builder_abi=abi,
    )
    assert unreadable["observations"][0]["status"] == "UNKNOWN"
    assert unreadable["observations"][0]["reason"] == "remote_cache_unreadable"


def test_incomplete_native_quality_binding_cannot_create_a_false_hit(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path / "input", [{
        "id": "hailo8_to_trt", "type": "matrix",
        "stage1": {"hw_arch": "hailo8"},
        "stage2": {"provider": "tensorrt"},
        "variants": ["composed"], "case_id": "b024",
        "native_trt_precision": "uint8_dequant_fp16",
    }])
    benchmark_set = json.loads((suite / "benchmark_set.json").read_text())
    benchmark_set["benchmark_task"] = "detection"
    (suite / "benchmark_set.json").write_text(json.dumps(benchmark_set))
    _vendor_native_quality_validator(suite)
    builder = tmp_path / "trtexec"
    builder.write_bytes(b"trtexec")
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    base = tmp_path / "remote"
    base.mkdir()
    key = remote_run._stable_trt_engine_cache_key(
        suite, builder_abi=abi, active_run_ids=["hailo8_to_trt"],
    )
    original = suite / "b024/model_part2_b24.onnx"
    namespace = base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key
    leaf = namespace / remote_run._trt_persistent_engine_relative_dir(
        role="part2", case_id="b024",
        source_onnx_sha256=_sha(original),
        precision="uint8_dequant_fp16",
    )
    _receipt(
        leaf=leaf, source_bytes=b"generated-dequant-bridge",
        builder=builder, role="part2",
        engine_precision="uint8_dequant_fp16",
    )
    receipt_path = leaf / "engine_build_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    build_source = leaf / "source.onnx"
    engine = leaf / "part2_uint8_dequant_fp16.engine"
    # This intentionally satisfies the old shallow structural check while
    # omitting the strict runtime's Part1/policy/quantization cross-links.
    binding = {
        "preselection": {
            "setup_id": "setup", "model_id": "model",
            "case_id": "b024", "backend": "hailo8_to_trt",
        },
        "engine_build_receipt_sha256": receipt["receipt_sha256"],
        "artifacts": {
            "source_part2_onnx": {
                "path": str(original.resolve()), "sha256": _sha(original),
            },
            "build_part2_onnx": {
                "path": str(build_source.resolve()),
                "sha256": _sha(build_source),
            },
            "engine": {
                "path": str(engine.resolve()), "sha256": _sha(engine),
            },
            "engine_build_receipt": {
                "path": str(receipt_path.resolve()),
                "sha256": _sha(receipt_path),
            },
        },
    }
    binding["binding_sha256"] = hashlib.sha256(json.dumps(
        binding, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    (leaf / "native_split_quality_binding.json").write_text(
        json.dumps(binding)
    )
    _owner(namespace, builder_abi_sha256=_builder_abi_sha256(abi))

    result = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="setup", active_run_ids=["hailo8_to_trt"],
        resolved_remote_base=str(base), builder_abi=abi,
    )
    row = result["observations"][0]
    assert row["status"] == "UNKNOWN"
    assert row["reason"] == "native_binding_strict_validation_failed"
    assert row["evidence"]["candidate_failures"][0]["validator_status"] == (
        "native_split_quality_binding_schema_or_role_invalid"
    )


def test_complete_quality_first_binding_is_a_binding_addressed_hit(
    tmp_path: Path,
) -> None:
    """The runtime's exact local validator authorizes its intentional path."""

    suite = tmp_path / "suite"
    (suite / "models").mkdir(parents=True)
    (suite / "models/model.onnx").write_bytes(b"full-yolo26s")
    (suite / "b038").mkdir()
    (suite / "b038/model_part2_b38.onnx").write_bytes(
        b"ONNX-source-part2-v1"
    )
    (suite / "b038/split_manifest.json").write_text("{}")
    (suite / "benchmark_set.json").write_text(json.dumps({
        "model_id": "yolo26s", "model": "models/model.onnx",
        "benchmark_task": "detection", "cases": [{"id": "b038"}],
    }))
    (suite / "benchmark_plan.json").write_text(json.dumps({"runs": [{
        "id": "hailo8_to_trt", "type": "matrix",
        "stage1": {"hw_arch": "hailo8"},
        "stage2": {"provider": "tensorrt"},
        "variants": ["composed"], "case_id": "b038",
        "native_trt_precision": "uint8_dequant_fp16",
    }]}))
    _vendor_native_quality_validator(suite)

    # The stable key omits the builder's path; use its known bytes to locate
    # the namespace before creating the production-style Quality-FIRST tree.
    provisional_builder = tmp_path / "provisional-trtexec"
    provisional_builder.write_bytes(b"#!/bin/sh\nexit 0\n")
    provisional_builder.chmod(0o755)
    provisional_abi = _builder_abi(provisional_builder)
    stable_key = remote_run._stable_trt_engine_cache_key(
        suite, builder_abi=provisional_abi,
        active_run_ids=["hailo8_to_trt"],
    )
    base = tmp_path / "remote"
    namespace = (
        base / "_onnx_splitpoint_cache/tensorrt_managed_v27516"
        / stable_key
    )
    quality_root = (
        namespace / "native_split_quality/hailo8_setup/yolo26s/b038"
        / "hailo8_to_trt/quality-selection-key"
        / "engine_cache/b038/part2/uint8_dequant_fp16"
    )
    payload, paths = _fixture_payload(quality_root)
    selected_part1 = suite / "b038/hailo/hailo8/part1/compiled.hef"
    selected_part1.parent.mkdir(parents=True)
    selected_part1.write_bytes(paths["part1_runtime"].read_bytes())

    # Match the preflight's exact workspace build contract while retaining the
    # complete producer receipt/meta/binding cross-link chain.
    receipt = dict(payload["engine_build_receipt"])
    receipt.pop("receipt_sha256", None)
    receipt["command"] = [*receipt["command"], "--workspace=4096"]
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    paths["engine_build_receipt"].write_text(json.dumps(receipt))
    native_meta = dict(payload["native_trt_meta"])
    native_meta["build"] = {
        **dict(native_meta["build"]), "cmd": list(receipt["command"]),
    }
    native_meta["engine_build_receipt"] = dict(receipt)
    paths["native_trt_meta"].write_text(json.dumps(native_meta))

    def artifact(path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()), "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }

    payload["engine_build_receipt"] = receipt
    payload["engine_build_receipt_sha256"] = receipt["receipt_sha256"]
    payload["native_trt_meta"] = native_meta
    payload["native_trt_meta_sha256"] = canonical_json_sha256(native_meta)
    payload["artifacts"]["engine_build_receipt"] = artifact(
        paths["engine_build_receipt"]
    )
    payload["artifacts"]["native_trt_meta"] = artifact(
        paths["native_trt_meta"]
    )
    binding = seal_native_split_quality_binding(payload)
    binding_path = paths["engine_build_receipt"].parent / (
        "native_split_quality_binding.json"
    )
    binding_path.write_text(json.dumps(binding))

    actual_abi = _builder_abi(paths["trtexec"])
    assert remote_run._stable_trt_engine_cache_key(
        suite, builder_abi=actual_abi,
        active_run_ids=["hailo8_to_trt"],
    ) == stable_key
    _owner(
        namespace, builder_abi_sha256=_builder_abi_sha256(actual_abi),
    )

    result = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="hailo8_setup", setup_accelerator="hailo8",
        active_run_ids=["hailo8_to_trt"],
        resolved_remote_base=str(base), builder_abi=actual_abi,
    )
    row = result["observations"][0]
    assert row["status"] == "HIT", row
    assert row["evidence"]["source_binding"] == (
        "strict_native_split_quality_binding"
    )
    assert row["evidence"]["quality_binding_addressed"] is True
    assert row["evidence"]["validator_status"] == (
        "local_files_rehashed_and_exact_cross_links_verified"
    )
    assert row["evidence"]["validator_source_identity"].startswith(
        "splitpoint_runners.native_split_quality@sha256:"
    )
    assert row["receipt_path"] == str(paths["engine_build_receipt"].resolve())

    # The read-only probe must never execute arbitrary Python smuggled into a
    # generated suite.  Only byte-identical installed/vendored validators are
    # trusted; drift downgrades the quality-specific candidate to UNKNOWN.
    validator_path = suite / "splitpoint_runners/native_split_quality.py"
    validator_path.write_text(
        validator_path.read_text() + "\n# tampered generated suite\n"
    )
    untrusted = remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="hailo8_setup", setup_accelerator="hailo8",
        active_run_ids=["hailo8_to_trt"],
        resolved_remote_base=str(base), builder_abi=actual_abi,
    )
    untrusted_row = untrusted["observations"][0]
    assert untrusted_row["status"] == "UNKNOWN"
    assert untrusted_row["reason"] == (
        "native_binding_strict_validation_unavailable"
    )
    assert "differs from trusted installed source" in (
        untrusted_row["evidence"]["candidate_failures"][0]["validator_error"]
    )
