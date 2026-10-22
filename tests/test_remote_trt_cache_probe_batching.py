from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import shlex
import zlib
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark import remote_run
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig, SSHTransport, shell_quote
from tests.test_v27920_remote_trt_cache_preflight import (
    _LocalReadOnlyTransport, _builder_abi, _builder_abi_sha256, _owner,
    _receipt, _sha, _suite, _vendor_native_quality_validator,
)


def decode_payload(command):
    # Inspect the normal builder's literals; never execute its decoded program.
    bootstrap = ast.parse(shlex.split(command)[-1])
    literal = next(node.value for node in ast.walk(bootstrap)
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)
                   and len(node.value) > 100)
    program = zlib.decompress(base64.b64decode(literal)).decode()
    tree = ast.parse(program)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "P" for t in node.targets))
    encoded = next(node.value for node in ast.walk(assignment)
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)
                   and len(node.value) > 100)
    return json.loads(base64.b64decode(encoded))


def argument_bytes(command):
    return len(shell_quote(command).encode("utf-8")) + 1


def entropy(n):
    return "".join(hashlib.sha256(str(i).encode()).hexdigest()
                   for i in range((n + 63) // 64))[:n]


def payload():
    return {"model_id": "model", "setup_id": "h10", "precision": "fp16",
            "native_quality_validator": {"modules": {"actual": {"source": entropy(4000)}}},
            "requirements": [{"role": "trt_p2", "item_id": f"h10/b{i:03d}",
                              "case_id": f"b{i:03d}", "run_ids": [entropy(40000)[i:]]}
                             for i in range(3)]}


def test_exact_final_ssh_argument_and_small_command_is_unchanged():
    original = payload()
    original["requirements"] = original["requirements"][:1]
    command = remote_run._remote_trt_cache_probe_command(original)
    transport = SSHTransport(HostConfig(id="h10", label="test", host="invalid"))
    assert argument_bytes(command) == len(transport._ssh_cmd(command)[-1].encode()) + 1
    before = copy.deepcopy(original)
    batches, oversized = remote_run._remote_trt_cache_probe_batches(original)
    assert oversized == []
    assert len(batches) == 1
    assert batches[0]["command"] == command
    assert batches[0]["payload"] == original == before
    assert batches[0]["ssh_argument_bytes"] == argument_bytes(command)


def test_batches_retain_every_shared_field_and_requirement_once(monkeypatch):
    original = payload()
    sizes = [argument_bytes(remote_run._remote_trt_cache_probe_command(
        dict(original, requirements=[row]))) for row in original["requirements"]]
    monkeypatch.setattr(remote_run, "_TRT_CACHE_PROBE_SSH_ARGUMENT_LIMIT_BYTES", max(sizes))
    before = copy.deepcopy(original)
    batches, oversized = remote_run._remote_trt_cache_probe_batches(original)
    assert oversized == []
    assert len(batches) > 1
    assert [r for batch in batches for r in batch["payload"]["requirements"]] == original["requirements"]
    for batch in batches:
        assert decode_payload(batch["command"]) == batch["payload"]
        assert {k: v for k, v in batch["payload"].items() if k != "requirements"} == {
            k: v for k, v in original.items() if k != "requirements"}
        assert batch["ssh_argument_bytes"] <= max(sizes)
    assert original == before
    assert batches == remote_run._remote_trt_cache_probe_batches(original)[0]


def test_oversized_singleton_does_not_omit_other_requirements(monkeypatch):
    original = payload()
    original["requirements"][1]["run_ids"] = [entropy(250000)]
    cap = max(argument_bytes(remote_run._remote_trt_cache_probe_command(
        dict(original, requirements=[original["requirements"][i]]))) for i in (0, 2))
    monkeypatch.setattr(remote_run, "_TRT_CACHE_PROBE_SSH_ARGUMENT_LIMIT_BYTES", cap)
    batches, oversized = remote_run._remote_trt_cache_probe_batches(original)
    assert [row["requirement"] for row in oversized] == [original["requirements"][1]]
    assert oversized[0]["ssh_argument_bytes"] > cap
    assert [r for b in batches for r in b["payload"]["requirements"]] == [
        original["requirements"][0], original["requirements"][2]]


def suite_fixture(tmp_path):
    cases = ["b024", "b025", "b026"]
    suite = _suite(tmp_path / "input", [{"id": "trt", "type": "onnxruntime",
                                      "provider": "tensorrt", "variants": ["part2"],
                                      "case_ids": cases}])
    for case in cases[1:]:
        (suite / case).mkdir()
        (suite / case / f"model_part2_{case}.onnx").write_bytes(case.encode())
        (suite / case / "split_manifest.json").write_text("{}")
    _vendor_native_quality_validator(suite)
    builder = tmp_path / "trtexec"
    builder.write_bytes(b"trtexec-v1")
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    base = tmp_path / "remote"
    base.mkdir()
    key = remote_run._stable_trt_engine_cache_key(suite, builder_abi=abi)
    namespace = base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key
    for case in cases:
        source = remote_run._trt_preflight_part2_source(suite, case)
        leaf = namespace / remote_run._trt_persistent_engine_relative_dir(
            role="part2", case_id=case, source_onnx_sha256=_sha(source), precision="fp16")
        _receipt(leaf=leaf, source_bytes=source.read_bytes(), builder=builder, role="part2")
    _owner(namespace, builder_abi_sha256=_builder_abi_sha256(abi))
    return suite, abi, base, namespace


def probe(suite, abi, base, transport, **kwargs):
    return remote_run.probe_remote_trt_artifact_cache(
        transport=transport, suite_dir=suite, setup_id="h10", active_run_ids=["trt"],
        resolved_remote_base=str(base), builder_abi=abi, **kwargs)


def force_singletons(monkeypatch, payload):
    cap = max(argument_bytes(remote_run._remote_trt_cache_probe_command(
        dict(payload, requirements=[row]))) for row in payload["requirements"])
    monkeypatch.setattr(remote_run, "_TRT_CACHE_PROBE_SSH_ARGUMENT_LIMIT_BYTES", cap)
    return cap


def test_normal_verifier_batched_hit_miss_owner_tamper_equivalence(tmp_path, monkeypatch):
    suite, abi, base, namespace = suite_fixture(tmp_path)
    local = _LocalReadOnlyTransport(base)
    baseline = probe(suite, abi, base, local)
    original = decode_payload(local.commands[0])
    assert len(original["requirements"]) == 3
    assert [r["status"] for r in baseline["observations"]] == ["HIT"] * 3
    cap = force_singletons(monkeypatch, original)
    local.commands.clear()
    batched = probe(suite, abi, base, local)
    assert len(local.commands) > 1
    assert batched["observations"] == baseline["observations"]
    assert all(argument_bytes(command) <= cap for command in local.commands)
    # The unchanged owning-host verifier still rejects bytes and owners.
    next(namespace.rglob("part2_fp16.engine")).write_bytes(b"changed")
    tampered = probe(suite, abi, base, local)
    assert sum(r["status"] == "MISS" for r in tampered["observations"]) == 1
    (namespace / ".splitpoint_trt_cache_owner.json").unlink()
    unknown = probe(suite, abi, base, local)
    assert [r["status"] for r in unknown["observations"]] == ["UNKNOWN"] * 3


def test_responses_must_match_exact_model_setup_role_item_once(tmp_path, monkeypatch):
    suite, abi, base, _ = suite_fixture(tmp_path)
    local = _LocalReadOnlyTransport(base)
    probe(suite, abi, base, local)
    original = decode_payload(local.commands[0])
    force_singletons(monkeypatch, original)
    class Wrong(_LocalReadOnlyTransport):
        def run_read_only(self, command, timeout=0):
            rc, out = super().run_read_only(command, timeout)
            if len(self.commands) == 1:
                prefix, raw = out.split("=", 1)
                value = json.loads(raw)
                value["observations"][0]["item_id"] = "h10/wrong"
                out = prefix + "=" + json.dumps(value)
            return rc, out
    wrong = Wrong(base)
    result = probe(suite, abi, base, wrong)
    assert result["status"] == "unknown"
    assert len(wrong.commands) == 1
    assert all(r["status"] == "UNKNOWN" for r in result["observations"])
    assert all(r["reason"] == "remote_trt_cache_probe_prior_failure_not_dispatched"
               for r in result["observations"][1:])
    assert len({r["item_id"] for r in result["observations"]}) == 3


def test_batches_share_one_elapsed_timeout_budget(tmp_path, monkeypatch):
    suite, abi, base, _ = suite_fixture(tmp_path)
    local = _LocalReadOnlyTransport(base)
    probe(suite, abi, base, local)
    force_singletons(monkeypatch, decode_payload(local.commands[0]))
    clock = [100.0]
    monkeypatch.setattr(remote_run.time, "monotonic", lambda: clock[0])
    class Slow(_LocalReadOnlyTransport):
        timeouts = []
        def run_read_only(self, command, timeout=0):
            self.timeouts.append(timeout)
            result = super().run_read_only(command, timeout)
            clock[0] += 20.0
            return result
    slow = Slow(base)
    result = probe(suite, abi, base, slow, timeout_s=30)
    assert slow.timeouts == [30.0, 10.0]
    assert [r["status"] for r in result["observations"]] == ["HIT", "HIT", "UNKNOWN"]
    assert result["observations"][-1]["reason"] == "remote_trt_cache_probe_budget_exhausted"


@pytest.mark.parametrize("command", ["echo 'é雪'", "printf '%s' \"a'b\"", "", "line1\nline2"])
def test_argument_measurement_uses_actual_quoting_utf8_and_nul(command):
    transport = SSHTransport(HostConfig(id="h8", label="test", host="invalid"))
    assert argument_bytes(command) == len(transport._ssh_cmd(command)[-1].encode("utf-8")) + 1


@pytest.mark.parametrize("mutation", ["top_model", "top_setup", "stable_key", "schema", "row_model",
    "row_setup", "role", "item", "duplicate", "missing", "status"])
def test_response_rejects_detached_or_duplicate_identities(mutation):
    request = dict(model_id="model", setup_id="h8", stable_key="key", remote_base="/base",
                   requirements=[dict(role="trt_p2", item_id="h8/b001"),
                                 dict(role="trt_p2", item_id="h8/b002")])
    response = dict(schema="onnx-splitpoint/remote-trt-cache-preflight", schema_version=1,
                    status="ok", **{k: request[k] for k in ("model_id", "setup_id", "stable_key", "remote_base")},
                    observations=[dict(model_id="model", setup_id="h8", status="HIT", **row)
                                  for row in request["requirements"]])
    if mutation.startswith("top_"):
        response[mutation[4:] + "_id"] = "wrong"
    elif mutation in {"stable_key", "schema"}:
        response[mutation] = "wrong"
    elif mutation.startswith("row_"):
        response["observations"][0][mutation[4:] + "_id"] = "wrong"
    elif mutation in {"role", "item", "status"}:
        response["observations"][0]["item_id" if mutation == "item" else mutation] = "wrong"
    elif mutation == "duplicate":
        response["observations"][1] = dict(response["observations"][0])
    elif mutation == "missing":
        response["observations"].pop()
    with pytest.raises(RuntimeError):
        remote_run._remote_trt_cache_probe_response(
            "SPLITPOINT_REMOTE_TRT_CACHE_PREFLIGHT=" + json.dumps(response), request)


def test_expensive_local_preparation_does_not_consume_old_verifier_budget(tmp_path, monkeypatch):
    suite, abi, base, _ = suite_fixture(tmp_path)
    clock = [100.0]
    monkeypatch.setattr(remote_run.time, "monotonic", lambda: clock[0])
    prepare = remote_run._remote_trt_cache_probe_batches
    def slow_prepare(payload):
        result = prepare(payload)
        clock[0] += 1200.0
        return result
    monkeypatch.setattr(remote_run, "_remote_trt_cache_probe_batches", slow_prepare)
    class Capturing(_LocalReadOnlyTransport):
        def run_read_only(self, command, timeout=0):
            assert timeout == 180.0
            return super().run_read_only(command, timeout)
    result = probe(suite, abi, base, Capturing(base))
    assert [r["status"] for r in result["observations"]] == ["HIT"] * 3


def test_oversized_item_is_unknown_while_other_normal_verifiers_continue(tmp_path, monkeypatch):
    suite, abi, base, _ = suite_fixture(tmp_path)
    plan = remote_run._trt_preflight_run_requirements
    def large_plan(*args, **kwargs):
        result = plan(*args, **kwargs)
        result["generic_p2_run_ids_by_case"]["b025"] = [entropy(250000)]
        return result
    monkeypatch.setattr(remote_run, "_trt_preflight_run_requirements", large_plan)
    local = _LocalReadOnlyTransport(base)
    result = probe(suite, abi, base, local)
    assert [r["status"] for r in result["observations"]] == ["HIT", "UNKNOWN", "HIT"]
    assert result["observations"][1]["reason"] == "remote_trt_cache_probe_payload_too_large"
    dispatched = [row for cmd in local.commands for row in decode_payload(cmd)["requirements"]]
    assert [r["case_id"] for r in dispatched] == ["b024", "b026"]
    assert result["status"] == "partial"


@pytest.mark.parametrize("failure", [124, 70, 1, "exception"])
def test_transport_failure_never_launches_another_unleased_batch(tmp_path, monkeypatch, failure):
    suite, abi, base, _ = suite_fixture(tmp_path)
    local = _LocalReadOnlyTransport(base)
    probe(suite, abi, base, local)
    force_singletons(monkeypatch, decode_payload(local.commands[0]))
    class Failed(_LocalReadOnlyTransport):
        def run_read_only(self, command, timeout=0):
            self.commands.append(command)
            if failure == "exception":
                raise OSError("local read-only transport failure")
            return failure, "remote completion is not proven"
    failed = Failed(base)
    result = probe(suite, abi, base, failed)
    assert len(failed.commands) == 1
    assert [r["status"] for r in result["observations"]] == ["UNKNOWN"] * 3
    assert [r["reason"] for r in result["observations"]][1:] == [
        "remote_trt_cache_probe_prior_failure_not_dispatched"] * 2
    for row in result["observations"][1:]:
        assert row["evidence"]["expected_source_sha256"]
        assert row["evidence"]["prior_probe_failure"]["item_ids"] == ["h10/b024:generic"]
    assert all(b["status"] == "not_dispatched" for b in result["probe_batches"][1:])


def test_later_failure_preserves_prior_verified_hit_without_further_dispatch(tmp_path, monkeypatch):
    suite, abi, base, _ = suite_fixture(tmp_path)
    local = _LocalReadOnlyTransport(base)
    probe(suite, abi, base, local)
    force_singletons(monkeypatch, decode_payload(local.commands[0]))
    class FailsSecond(_LocalReadOnlyTransport):
        def run_read_only(self, command, timeout=0):
            if len(self.commands) == 1:
                self.commands.append(command)
                return 124, "remote completion is not proven"
            return super().run_read_only(command, timeout)
    failed = FailsSecond(base)
    result = probe(suite, abi, base, failed)
    assert len(failed.commands) == 2
    assert [r["status"] for r in result["observations"]] == ["HIT", "UNKNOWN", "UNKNOWN"]
    assert result["status"] == "partial"
    assert result["observations"][-1]["reason"] == "remote_trt_cache_probe_prior_failure_not_dispatched"
