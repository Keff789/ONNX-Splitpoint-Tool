"""Exercise the actual Native byte/capability verifier before hardware dispatch.

Only SSH/rsync transport is substituted. The source inventory, source bytes,
verification program, and error formatting are the production implementations.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from onnx_splitpoint_tool.remote_runtime_closure import native_remote_package_closure
from onnx_splitpoint_tool.runners import native_split_quality_runtime as runtime
from onnx_splitpoint_tool.workflow import runner


SOURCE = Path(runner.__file__).resolve().parents[2]
CLOSURE = native_remote_package_closure()


class _LocalTransport:
    """Run the exact generated verification code in a fresh isolated Python."""

    def __init__(self, remote_root: Path, *, tamper=False):
        self.remote_root = remote_root
        self.tamper = tamper
        self.events = []
        self.proofs = []

    def __call__(self, command, *, label, timeout_s):
        if command[0] == "rsync":
            source = Path(command[-2])
            destination = Path(command[-1].split(":", 1)[1])
            assert destination.is_relative_to(self.remote_root)
            shutil.copy2(source, destination)
            if self.tamper:
                destination.write_bytes(destination.read_bytes() + b"\n# transport corruption\n")
            self.events.append("transfer")
            return subprocess.CompletedProcess(command, 0, "", "")

        assert command[0] == "ssh", command
        payload = shlex.split(command[-1])
        if payload[:2] == ["mkdir", "-p"]:
            assert len(payload) == 3
            destination = Path(payload[2])
            assert destination.is_relative_to(self.remote_root)
            destination.mkdir(parents=True, exist_ok=True)
            self.events.append("mkdir")
            return subprocess.CompletedProcess(command, 0, "", "")

        # Fail if the production helper unexpectedly attempts an import,
        # hardware command, shell fallback, or a different verifier protocol.
        assert payload[:2] == ["python", "-c"] and len(payload) == 3, payload
        assert "missing_tokens" in payload[2]
        self.events.append("verify")
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-c", payload[2]],
            cwd=self.remote_root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        self.proofs.append(json.loads(completed.stdout))
        return completed


def _sync(transport, relative, tokens):
    return runner._sync_remote_package_asset_v263(
        ssh="nx@isolated-native-test",
        remote_tool_dir=str(transport.remote_root),
        relative_path=relative,
        required_tokens=tokens,
        timeout=20,
        process_runner=transport,
    )


@pytest.mark.parametrize(
    "relative,module,tokens", CLOSURE,
    ids=[relative for relative, _module, _tokens in CLOSURE],
)
def test_actual_shipped_native_assets_pass_production_verifier(tmp_path, relative, module, tokens):
    transport = _LocalTransport(tmp_path / "remote tool")
    steps = _sync(transport, relative, tokens)
    assert transport.events == ["mkdir", "transfer", "verify"]
    assert transport.proofs[0]["missing_tokens"] == []
    assert transport.proofs[0]["sha256"] == transport.proofs[0]["expected"]
    assert steps[-1]["rc"] == 0
    assert steps[-1]["sha256_ok"] is True
    assert (transport.remote_root / relative).read_bytes() == (SOURCE / relative).read_bytes()


def test_matching_hash_cannot_hide_missing_required_capability(tmp_path, monkeypatch):
    relative = "onnx_splitpoint_tool/release_identity.py"
    tokens = next(tokens for path, _module, tokens in CLOSURE if path == relative)
    source = tmp_path / "source without required capability"
    mutated = source / relative
    mutated.parent.mkdir(parents=True)
    # The sender and receiver agree exactly; the transferred program still
    # lacks a required capability. Reproduce hash_ok=True / rc=7 directly.
    original = (SOURCE / relative).read_text()
    assert "BUILD_ID" in original
    mutated.write_text(original.replace("BUILD_ID", "UNRELATED_ID"))
    monkeypatch.setattr(runner, "__file__", str(source / "onnx_splitpoint_tool/workflow/runner.py"))
    transport = _LocalTransport(tmp_path / "remote")
    with pytest.raises(RuntimeError, match=r"rc=7 hash_ok=True missing=.*BUILD_ID"):
        _sync(transport, relative, tokens)
    proof = transport.proofs[0]
    assert proof["sha256"] == proof["expected"]
    assert "BUILD_ID" in proof["missing_tokens"]
    assert transport.events == ["mkdir", "transfer", "verify"]


def test_remote_hash_drift_rejected_even_with_all_capabilities(tmp_path):
    relative = "onnx_splitpoint_tool/quality_result_contract.py"
    tokens = next(tokens for path, _module, tokens in CLOSURE if path == relative)
    transport = _LocalTransport(tmp_path / "remote", tamper=True)
    with pytest.raises(RuntimeError, match=r"rc=7 hash_ok=False missing=\[\]"):
        _sync(transport, relative, tokens)
    proof = transport.proofs[0]
    assert proof["sha256"] != proof["expected"]
    assert proof["missing_tokens"] == []
    assert transport.events == ["mkdir", "transfer", "verify"]


@pytest.mark.parametrize("policy", ["cache_verify_only", "strict_warm_cache"])
def test_executed_cache_block_preserves_correct_policy_and_details(monkeypatch, policy):
    if policy == "cache_verify_only":
        monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", policy)
    else:
        monkeypatch.delenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", raising=False)
    error = runtime._cache_verify_block(artifact="existing-part2.engine", reason="receipt_invalid")
    assert isinstance(error, RuntimeError)
    assert str(error) == (
        "cache_miss_blocked:artifact_policy=" + policy
        + ":compiler=trtexec:artifact=existing-part2.engine:reason=receipt_invalid"
    )
