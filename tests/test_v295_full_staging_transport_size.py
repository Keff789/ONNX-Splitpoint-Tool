"""Large Full metadata survives ordinary process-lease and SSH argv quoting."""
import ast
import base64
import json
import os
import random
import shlex
import string
import subprocess
import zlib

from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseOperation, RemoteProcessLeaseScope
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig, SSHTransport
from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from test_v2952_full_staging_metadata import LocalTransport, inventory, put, selective_full


def test_yolo26m_sized_full_manifest_preserves_staging_and_fits_leased_ssh_argument(selective_full):
    point = selective_full
    manifest = json.loads((point.local / point.case / "split_manifest.json").read_text())
    generator = random.Random(20261015)
    # Match the actual YOLO26m metadata's ~79-KB transmitted program, including
    # a compressed manifest with distinct saved tensor descriptions.
    alphabet = string.ascii_letters + string.digits + "_'\""
    manifest["original_tensor_descriptions"] = [
        "".join(generator.choices(alphabet, k=48)) for _ in range(650)]
    point.contract["retained_compiler_metadata"] = [
        {"name": "tensor'" + str(index), "shape": [1, 3, 640, 640]} for index in range(650)]
    put(point.local / point.case / "split_manifest.json", manifest)
    point.item["full_source_case_metadata"] = native._full_case_metadata(
        point.primary, point.contract, deepx_runtime_mode="A4")
    before_local, before_remote = inventory(point.local), inventory(point.remote)
    scope = RemoteProcessLeaseScope("test-full-large-manifest", "test-session")
    operation = RemoteProcessLeaseOperation(scope=scope, operation_id="ssh-capture-" + "a" * 32,
        token="b" * 64, control_argv_prefix=["ssh", "example.invalid"])
    ssh = SSHTransport(HostConfig(id="test", label="test", host="example.invalid", user="nx"),
        remote_lease_scope=scope)
    captured = {}

    class Transport(LocalTransport):
        def run(self, command, **kwargs):
            if not captured:
                captured["command"] = command
                captured["argv"] = ssh._ssh_cmd(operation.wrap_remote_command(command))
            return super().run(command, **kwargs)

    native._stage_full_benchmark_set(Transport(point.tmp), point.item, point.tmp / "receipts")
    command = shlex.split(captured["command"])
    assert command[1:3] == ["-B", "-c"]
    bootstrap = ast.parse(command[3])
    decode = next(node for node in ast.walk(bootstrap) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "b64decode")
    program = zlib.decompress(base64.b64decode(ast.literal_eval(decode.args[0]))).decode("utf-8")
    staging_call = next(node for node in ast.walk(ast.parse(program)) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "_copy_full_benchmark")
    arguments = [ast.literal_eval(node) for node in staging_call.args]
    assert arguments[:4] == [point.item["source_benchmark_set"], point.item["full_benchmark_set"],
        point.item["full_source_contract"], point.item.get("full_source_output_endpoint_attestation")]
    assert arguments[5] == "A4"
    assert zlib.decompress(base64.b64decode(arguments[4]["case_manifest_zlib_base64"])) == (
        point.local / point.case / "split_manifest.json").read_bytes()
    original_command = shlex.quote(command[0]) + " -B - <<'NATIVE_FULL_ACCEPTANCE_METADATA'\n" + program + "NATIVE_FULL_ACCEPTANCE_METADATA"
    original_argv = ssh._ssh_cmd(operation.wrap_remote_command(original_command))
    limit = 32 * os.sysconf("SC_PAGESIZE")
    assert 65_000 < len(original_command.encode()) < 95_000
    assert max(len(value.encode()) for value in original_argv) >= limit
    largest = max(captured["argv"], key=lambda value: len(value.encode()))
    assert len(largest.encode()) < limit * 0.8
    # Only a harmless local executable is spawned with the complete argument;
    # the SSH command and remote lease are constructed, never executed.
    assert subprocess.run(["/bin/true", largest], check=False).returncode == 0
    assert inventory(point.local) == before_local and inventory(point.remote) == before_remote
    assert (point.fresh / point.case / "split_manifest.json").read_bytes() == (
        point.local / point.case / "split_manifest.json").read_bytes()
    assert (point.fresh / "output_contracts.json").read_bytes() == (point.local / "output_contracts.json").read_bytes()
