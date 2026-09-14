from __future__ import annotations

import ast
import json
from pathlib import Path
import tarfile
from typing import Any, Optional

from onnx_splitpoint_tool.gui.controller import _copy_runner_lib
from onnx_splitpoint_tool.remote.bundle import (
    build_suite_bundle,
    remote_minimal_bundle_patterns,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_TEMPLATE = (
    ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"
)
CANONICAL_ENDPOINT = ROOT / "onnx_splitpoint_tool/native_output_endpoint.py"


def _template_function(name: str) -> Any:
    source = RUNNER_TEMPLATE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNNER_TEMPLATE))
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    namespace = {
        "Any": Any,
        "Optional": Optional,
        "Path": Path,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(RUNNER_TEMPLATE), "exec"),
        namespace,
    )
    return namespace[name]


def test_validation_source_is_resolved_once_before_both_task_branches(
    tmp_path: Path,
) -> None:
    resolve = _template_function("_resolve_validation_source_path")
    case_dir = tmp_path / "suite" / "b001"
    case_dir.mkdir(parents=True)

    assert resolve("", case_dir) is None
    assert resolve(None, case_dir) is None
    assert resolve("../validation", case_dir) == (case_dir / "../validation").resolve()
    absolute = tmp_path / "absolute-validation"
    assert resolve(str(absolute), case_dir) == absolute.resolve()

    tree = ast.parse(RUNNER_TEMPLATE.read_text(encoding="utf-8"))
    stores = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "_validation_source_path"
        and isinstance(node.ctx, ast.Store)
    ]
    loads = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "_validation_source_path"
        and isinstance(node.ctx, ast.Load)
    ]
    assert len(stores) == 1
    assert loads
    assert stores[0] < min(loads)


def test_runner_vendors_canonical_endpoint_attestor_and_imports_after_suite_path(
    tmp_path: Path,
) -> None:
    _copy_runner_lib(tmp_path)
    vendored = tmp_path / "splitpoint_runners/native_output_endpoint.py"
    assert vendored.read_bytes() == CANONICAL_ENDPOINT.read_bytes()

    source = RUNNER_TEMPLATE.read_text(encoding="utf-8")
    suite_path_call = source.index("\n_maybe_add_suite_runtime_to_syspath()\n")
    vendored_leaf_lookup = source.index(
        'candidate / "splitpoint_runners" / "native_output_endpoint.py"'
    )
    vendored_leaf_load = source.index(
        'spec_from_file_location(\n        "_onnx_splitpoint_suite_native_output_endpoint"'
    )
    canonical_import = source.index(
        "from onnx_splitpoint_tool import native_output_endpoint as "
        "_canonical_endpoint_module"
    )
    assert suite_path_call < vendored_leaf_lookup
    assert vendored_leaf_lookup < vendored_leaf_load < canonical_import
    assert "from splitpoint_runners.native_output_endpoint import (" not in source


def test_remote_minimal_bundle_keeps_authoritative_output_contracts(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    contract = {
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": "yolo26s",
        "contracts": [{
            "schema": "onnx-splitpoint/output-contract",
            "schema_version": 1,
            "model_id": "yolo26s",
            "backend": "deepx_m1",
            "variant": "full",
            "contract_status": "recorded",
            "endpoint_mode": "decoded",
            "host_tail_required": False,
            "postprocessing_required": False,
        }],
    }
    (suite / "output_contracts.json").write_text(
        json.dumps(contract, sort_keys=True), encoding="utf-8",
    )
    artifact_contract = suite / "deepx/deepx_m1/full/output_contract.json"
    artifact_contract.parent.mkdir(parents=True)
    artifact_contract.write_text(
        json.dumps({
            "endpoint_mode": "decoded",
            "host_tail_required": False,
            "postprocessing_required": False,
        }, sort_keys=True),
        encoding="utf-8",
    )

    includes, excludes = remote_minimal_bundle_patterns()
    assert "output_contracts.json" in includes
    archive = tmp_path / "suite.tar.gz"
    stats = build_suite_bundle(
        suite, archive, includes=includes, excludes=excludes,
    )
    manifest = json.loads(Path(stats.manifest_path).read_text(encoding="utf-8"))
    bundled = {str(row["rel"]) for row in manifest["files"]}
    assert "output_contracts.json" in bundled
    assert "deepx/deepx_m1/full/output_contract.json" in bundled
    with tarfile.open(archive, "r:gz") as handle:
        members = set(handle.getnames())
    assert "output_contracts.json" in members
    assert "deepx/deepx_m1/full/output_contract.json" in members
