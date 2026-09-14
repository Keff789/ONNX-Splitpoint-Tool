"""The real compact exporter retains both rejected and selected attempt logs."""
import json
import zipfile
from pathlib import Path

import pytest
from onnx_splitpoint_tool.workflow.debug_pack import create_evaluation_debug_pack

MEASUREMENT = "reports/native_energy_measurements/measurements/case/plan/attempt"


@pytest.mark.parametrize("missing", [False, True])
def test_retry_logs_and_selection_history_survive_export(tmp_path, missing):
    root = tmp_path / "run"
    base = root / MEASUREMENT
    first = base / "run_002/workload_stdout.log"
    retry = base / "repeat_retry_attempts/repeat_002/attempt_01/run_000/workload_stdout.log"
    for path, body in ((first, b"rejected marker error\n"), (retry, b"selected completion\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    history = [{"attempt_index": i, "selected": bool(i), "run_directory": str(p.parent)}
               for i, p in enumerate((first, retry))]
    aggregate = base / "energy_aggregate.json"
    original = json.dumps({"runs": [{"run_index": 2, "logical_repeat_index": 2,
                                    "selected_repeat_attempt_index": 1,
                                    "repeat_attempt_history": history}]}).encode()
    aggregate.write_bytes(original)
    (root / "evaluation_workflow.log").write_text("completed\n")
    (retry.parent / "tensor.npy").write_bytes(b"must not be exported")
    if missing:
        retry.unlink()
    archive = tmp_path / "debug.zip"
    create_evaluation_debug_pack(root, archive)
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("debug_pack_manifest.json"))
        evidence = manifest["native_energy_attempt_diagnostics"]
        assert z.read(aggregate.relative_to(root).as_posix()) == original
        assert z.read(first.relative_to(root).as_posix()) == first.read_bytes()
        assert evidence["complete"] is (not missing)
        assert evidence["selected_stdout_members"] == [retry.relative_to(root).as_posix()]
        assert not any(name.endswith(".npy") for name in z.namelist())
        if missing:
            assert evidence["missing_source_members"] == [retry.relative_to(root).as_posix()]
            assert manifest["complete"] is False
        else:
            assert z.read(retry.relative_to(root).as_posix()) == retry.read_bytes()
    assert aggregate.read_bytes() == original
