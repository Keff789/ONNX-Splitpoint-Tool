import json
from pathlib import Path

from onnx_splitpoint_tool.benchmark.schema import dumps_json_safe, write_json_atomic


class _FakeOpsetImport:
    def __init__(self, domain: str = "", version: int = 13) -> None:
        self.domain = domain
        self.version = version


class _FakeGraph:
    def __init__(self) -> None:
        self.name = "fake_graph"
        self.node = [object(), object(), object()]
        self.input = [object()]
        self.output = [object(), object()]
        self.initializer = [object()]


class _FakeModelProto:
    def __init__(self) -> None:
        self.graph = _FakeGraph()
        self.opset_import = [_FakeOpsetImport("", 13)]
        self.ir_version = 9
        self.producer_name = "unit-test"
        self.producer_version = "1.0"
        self.model_version = 1
        self.domain = ""

    def SerializeToString(self) -> bytes:
        return b"fake-model-bytes"

    def ListFields(self) -> list[object]:
        return []


def test_write_json_atomic_sanitizes_modelproto_like_objects(tmp_path: Path) -> None:
    payload = {
        "manifest": {
            "raw_model": _FakeModelProto(),
            "model_path": Path("cases/b216/model.onnx"),
            "tags": {"beta", "alpha"},
        }
    }

    out_path = tmp_path / "manifest.json"
    write_json_atomic(out_path, payload)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert written["manifest"]["model_path"] == "cases/b216/model.onnx"
    assert written["manifest"]["tags"] == ["alpha", "beta"]
    assert written["manifest"]["raw_model"]["graph_name"] == "fake_graph"
    assert written["manifest"]["raw_model"]["node_count"] == 3
    assert written["manifest"]["raw_model"]["input_count"] == 1
    assert written["manifest"]["raw_model"]["output_count"] == 2
    assert written["manifest"]["raw_model"]["initializer_count"] == 1
    assert written["manifest"]["raw_model"]["opset_imports"] == [{"domain": "", "version": 13}]


def test_dumps_json_safe_breaks_circular_references() -> None:
    payload: dict[str, object] = {}
    payload["self"] = payload

    written = json.loads(dumps_json_safe(payload, indent=2))
    assert written["self"] == {"__circular_ref__": "dict"}
