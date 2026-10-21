"""Mixed controls observe the historical prepared input without restaging it."""
import hashlib
import json
import sys

import pytest

from onnx_splitpoint_tool.workflow.generic_completed_task import bind_runtime_source_closure, download_case
from tests.test_generic_completion_dispatch import LocalTransport


@pytest.mark.parametrize("conflict", [False, True])
def test_mixed_source_closure_preserves_bound_historical_input(tmp_path, conflict):
    bundle = tmp_path / "bundle"
    (bundle / "runner").mkdir(parents=True)
    (bundle / "splitpoint_runners").mkdir()
    (bundle / "runner/run_split_onnxruntime.py").write_text("# fixture runner\n")
    (bundle / "splitpoint_runners/generic_energy.py").write_text("# fixture library\n")
    historical = tmp_path / "closed_archive/input.npy"
    historical.parent.mkdir()
    historical.write_bytes(b"immutable fixture prepared input")
    before = historical.read_bytes()
    descriptor = {"path": str(historical), "sha256": hashlib.sha256(before).hexdigest()}
    row = {"remote_root": "/new/runtime", "remote_python_executable": sys.executable,
           "runtime_config": {"mixed_input_control": {"path": "/new/control.json", "sha256": "a" * 64},
                              "input_binding": {"prepared_input": descriptor}}, "staging_files": []}
    if conflict:
        row["staging_files"] = [{"remote": str(historical), "sha256": "b" * 64}]
        with pytest.raises(ValueError, match="prepared input source binding conflicts"):
            bind_runtime_source_closure(row, bundle)
        assert "runtime_source_files" not in row["runtime_config"]
    else:
        bind_runtime_source_closure(row, bundle)
        assert row["runtime_config"]["runtime_source_files"].count(descriptor) == 1
        assert row["staging_files"] == []
        bind_runtime_source_closure(row, bundle)  # Stable revalidation without new writes.
    assert historical.read_bytes() == before


@pytest.mark.parametrize("change", [None, "download", "previous"])
def test_mixed_control_evidence_survives_normal_collection_without_rewrite(tmp_path, change):
    remote = tmp_path / "remote/result"
    report = remote / "mixed_input_control/repetition_1.json"
    report.parent.mkdir(parents=True)
    original = b'{"status": "passed", "purpose": "transport fixture only"}\n'
    report.write_bytes(original)
    descriptor = {"path": "mixed_input_control/repetition_1.json",
                  "sha256": hashlib.sha256(original).hexdigest()}
    (remote / "generic_completion_results.json").write_text(json.dumps({
        "repetitions": [{"mixed_input_control": {"report": descriptor}}]}))
    local = tmp_path / "collected"
    if change == "download":
        report.write_bytes(b"changed remote diagnostic")
    elif change == "previous":
        target = local / descriptor["path"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"previous different evidence")
    if change:
        with pytest.raises(ValueError, match="sample (changed|differs)"):
            download_case(LocalTransport(), str(remote), local, 1)
        if change == "previous":
            assert target.read_bytes() == b"previous different evidence"
    else:
        assert download_case(LocalTransport(), str(remote), local, 1)["result_collected"] is True
        assert (local / descriptor["path"]).read_bytes() == original
        assert (local / "downloads/attempt_001/result" / descriptor["path"]).read_bytes() == original
