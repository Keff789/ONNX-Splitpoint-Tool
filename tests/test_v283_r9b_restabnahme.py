"""R9B actual DeepX no-op metadata replay; all model/engine payloads are synthetic.

Exercises the actual remote probe script locally without compiler, hardware,
CUDA or network. Historic H8/H10 geometry and receipt fields are preserved.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
from onnx_splitpoint_tool.benchmark import remote_run
from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.native_split_quality import (
    known_native_split_policy, materialize_native_split_preselection,
    seal_native_split_quality_binding, validate_native_split_quality_binding,
)
from tests.test_v27920_remote_trt_cache_preflight import (
    _LocalReadOnlyTransport, _builder_abi, _builder_abi_sha256, _owner,
    _sha, _vendor_native_quality_validator,
)
PAIRS = json.loads((Path(__file__).parent / "fixtures" /
    "v283_r9b_deepx_noop.json").read_text())["pairs"]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _artifact(path):
    return {"path": str(path.resolve()), "sha256": _sha(path),
            "size_bytes": path.stat().st_size}


def _replace(value, replacements):
    if isinstance(value, dict):
        return {k: _replace(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace(v, replacements) for v in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value


def _pair_fixture(tmp_path, index, *, pair_name="recorded-pair", parent_tag=""):
    original = copy.deepcopy(PAIRS[index]["binding"])
    model = original["preselection"]["model_id"]
    case = original["preselection"]["case_id"]
    task = original["preselection"]["task"]
    selection = original["preselection"]
    backend, setup = selection["backend"], selection["setup_id"]
    precision = selection["precision"]
    suite = tmp_path / "suite"
    (suite / "models").mkdir(parents=True, exist_ok=True)
    (suite / "models/model.onnx").write_bytes(b"synthetic-full-mobilenet")
    (suite / case).mkdir(exist_ok=True)
    raw_source = suite / case / "model_part2.onnx"
    raw_source.write_bytes(b"synthetic-canonical-part2-identical-across-backends")
    _write(suite / case / "split_manifest.json", {})
    _write(suite / "benchmark_set.json", {
        "model_id": model, "model": "models/model.onnx",
        "benchmark_task": task, "cases": [{"id": case}],
    })
    run = {"id": backend, "type": "matrix", "case_id": case,
           "stage1": {"hw_arch": backend.removesuffix("_to_trt")},
           "stage2": {"provider": "tensorrt"}, "variants": ["composed"]}
    _write(suite / "benchmark_plan.json", {"runs": [run],
        "native_split_quality_selection": {
            "schema": "onnx-splitpoint/native-split-quality-selection",
            "schema_version": 1, "applicable": True,
            "split_backends": [backend.removesuffix("_to_trt")],
            "split_selection_source": "native_backends",
        },
    })
    _vendor_native_quality_validator(suite)
    builder = tmp_path / "bin/trtexec"
    builder.parent.mkdir(exist_ok=True)
    builder.write_bytes(b"#!/bin/sh\nexit 99\n")  # must never execute
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    base = tmp_path / "remote"
    stable_key = remote_run._stable_trt_engine_cache_key(
        suite, builder_abi=abi, active_run_ids=[backend])
    namespace = base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / stable_key
    root = namespace / "native_split_quality" / setup / model / case / backend / pair_name
    leaf = root / "engine_cache" / case / "part2" / precision
    paths, replacements = {}, {}
    for role, old in original["artifacts"].items():
        if role == "trtexec":
            paths[role] = builder
        elif role in {"part1_runtime", "boundary_metadata", "native_trt_meta"}:
            paths[role] = root / Path(old["path"]).name
        elif role == "source_part2_onnx":
            paths[role] = root / "benchmark_set" / case / "source_part2.onnx"
        else:
            paths[role] = leaf / Path(old["path"]).name
        replacements[old["path"]] = str(paths[role].resolve())
        if role not in {"boundary_metadata", "native_trt_meta", "engine_build_receipt", "trtexec"}:
            paths[role].parent.mkdir(parents=True, exist_ok=True)
            paths[role].write_bytes(raw_source.read_bytes() if role in {"source_part2_onnx", "build_part2_onnx"}
                                   else ("synthetic-" + role + "-" + backend + (parent_tag if role == "part1_runtime" else "")).encode())
        if paths[role].is_file():
            replacements[old["sha256"]] = _sha(paths[role])
    metadata = _replace(original["boundary_metadata_payload"], replacements)
    metadata.pop("metadata_sha256")
    metadata["part1_artifact_size_bytes"] = paths["part1_runtime"].stat().st_size
    metadata["metadata_sha256"] = canonical_json_sha256(metadata)
    _write(paths["boundary_metadata"], metadata)
    policy = known_native_split_policy(model_id=model, case_id=case,
                                      setup_id=setup, backend=backend)
    selected = materialize_native_split_preselection(
        policy=policy, part1_artifact=_artifact(paths["part1_runtime"]),
        boundary_metadata=metadata, boundary_metadata_artifact=_artifact(paths["boundary_metadata"]))
    receipt = _replace(original["engine_build_receipt"], replacements)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _write(paths["engine_build_receipt"], receipt)
    meta = _replace(original["native_trt_meta_payload"], replacements)
    meta["engine_build_receipt"] = receipt
    meta["build"]["cmd"] = receipt["command"]
    _write(paths["native_trt_meta"], meta)
    boundary = {key: selected[key] for key in original["boundary_contract"]}
    payload = {
        "eval_run_id": "synthetic-metadata-replay", "source_run_id": backend,
        "quality_completed": True, "performance_claims_emitted": False,
        "preselection": selected, "preselection_sha256": selected["selection_sha256"],
        "artifacts": {role: _artifact(path) for role, path in paths.items()},
        "boundary_contract": boundary, "boundary_contract_sha256": canonical_json_sha256(boundary),
        "engine_build_receipt": receipt, "engine_build_receipt_sha256": receipt["receipt_sha256"],
        "native_trt_meta": meta, "native_trt_meta_sha256": canonical_json_sha256(meta),
        "producer_command": ["python", "native_trt_from_benchmarkset.py"],
    }
    selected_part1 = suite / case / "deepx/deepx_m1/part1/model.dxnn"
    selected_part1.parent.mkdir(parents=True, exist_ok=True)
    selected_part1.write_bytes(paths["part1_runtime"].read_bytes())
    binding = seal_native_split_quality_binding(payload)
    verified, status = validate_native_split_quality_binding(binding, verification_mode="local")
    assert verified, status
    binding_path = root / "native_split_quality_binding.json"
    _write(binding_path, binding)
    _owner(namespace, builder_abi_sha256=_builder_abi_sha256(abi))
    return dict(suite=suite, base=base, abi=abi, setup=setup, backend=backend,
                precision=precision, paths=paths, binding_path=binding_path,
                binding=binding, source=raw_source, selected_part1=selected_part1, payload=payload, namespace=namespace)


def _probe(fixture):
    return remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(fixture["base"]), suite_dir=fixture["suite"],
        setup_id=fixture["setup"], setup_accelerator=fixture["backend"].removesuffix("_to_trt"),
        active_run_ids=[fixture["backend"]], resolved_remote_base=str(fixture["base"]),
        builder_abi=fixture["abi"],
    )



@pytest.mark.parametrize('index', [0, 1, 2], ids=['mobilenet_current', 'mobilenet_old_parent', 'yolo11l'])
def test_real_noop_metadata_enters_complete_generated_validator(tmp_path, index):
    f = _pair_fixture(tmp_path, index)
    before = {p: p.read_bytes() for p in f['base'].rglob('*') if p.is_file()}
    r = _probe(f); row = r['observations'][0]
    assert row['status'] == 'HIT', r
    assert row['evidence']['source_binding'] == 'strict_native_split_quality_binding'
    assert row['evidence']['validator_status'] == 'local_files_rehashed_and_exact_cross_links_verified'
    assert row['evidence']['quality_binding_addressed'] is True
    assert before == {p: p.read_bytes() for p in f['base'].rglob('*') if p.is_file()}


def _reseal_fixture(f, change):
    """Only synthetic files: reject semantic contradictions beyond hash damage."""
    payload = copy.deepcopy(f['payload'])
    change(payload)
    meta = payload['native_trt_meta']
    _write(f['paths']['native_trt_meta'], meta)
    payload['native_trt_meta_sha256'] = canonical_json_sha256(meta)
    payload['artifacts']['native_trt_meta'] = _artifact(f['paths']['native_trt_meta'])
    binding = seal_native_split_quality_binding(payload)
    _write(f['binding_path'], binding)
    return binding


@pytest.mark.parametrize('damage', ['applied', 'requested', 'replaced_uses', 'bool_uses', 'bridge_output', 'dtype', 'layout', 'bridge_hash', 'shape'])
def test_noop_semantic_mutations_cannot_become_direct_hash_hits(tmp_path, damage):
    f = _pair_fixture(tmp_path, 0)
    def change(p):
        b = p['native_trt_meta']['uint8_cast_bridge']
        if damage == 'applied': b['boundary_layout']['applied'] = True
        elif damage == 'requested': b['boundary_layout']['requested'] = 'memory_nhwc_to_nchw'
        elif damage == 'layout': b['boundary_layout']['effective'] = 'memory_nhwc_to_nchw'
        elif damage == 'replaced_uses': b['replaced_uses'] = 1
        elif damage == 'bool_uses': b['replaced_uses'] = False
        elif damage == 'bridge_output': b['bridge_output'] = 'cast_output'
        elif damage == 'dtype': b['input_dtype'] = 'UINT8'
        elif damage == 'bridge_hash': b['bridge_sha256'] = 'a'*64
        elif damage == 'shape':
            b['input_shape'] = [1, 1, 1, 960]
            p['native_trt_meta']['inputs'][0]['shape'] = b['input_shape']
    try:
        _reseal_fixture(f, change)
    except ValueError:
        # Sealing rejects some contradictions first; require actual generated
        # validator to reject the equivalent self-hashed embedded mutation too.
        b=copy.deepcopy(f['binding']);p={'native_trt_meta':b['native_trt_meta_payload']};change(p)
        b['native_trt_meta_payload_sha256']=canonical_json_sha256(b['native_trt_meta_payload'])
        b.pop('binding_sha256');b['binding_sha256']=canonical_json_sha256(b);_write(f['binding_path'],b)
    result = _probe(f)
    assert result['observations'][0]['status'] != 'HIT', result
    if damage in {'applied','requested','replaced_uses','bool_uses','bridge_output','shape'}:
        assert result['observations'][0]['reason'] == 'native_binding_noop_float_contract_mismatch', result


@pytest.mark.parametrize('damage', ['parent', 'missing_binding', 'engine_bytes', 'receipt_crosslink', 'source_crosslink', 'owner_abi', 'builder_bytes', 'namespace'])
def test_noop_complete_artifact_gates_stay_mandatory(tmp_path, damage):
    f = _pair_fixture(tmp_path, 0)
    if damage == 'parent': f['selected_part1'].write_bytes(b'other-selected-parent')
    elif damage == 'missing_binding': f['binding_path'].unlink()
    elif damage == 'engine_bytes': f['paths']['engine'].write_bytes(b'changed-engine')
    elif damage == 'builder_bytes': f['paths']['trtexec'].write_bytes(b'changed-builder')
    elif damage == 'owner_abi': _owner(f['namespace'], builder_abi_sha256='a'*64)
    elif damage == 'namespace':
        f['namespace'].rename(f['namespace'].with_name('unrelated-model-namespace'))
    else:
        b=copy.deepcopy(f['binding']);role='engine_build_receipt' if damage=='receipt_crosslink' else 'build_part2_onnx'
        original=f['paths'][role];other=original.with_name('crosslink-'+original.name);other.write_bytes(original.read_bytes())
        b['artifacts'][role]['path']=str(other)
        b.pop('binding_sha256');b['binding_sha256']=canonical_json_sha256(b);_write(f['binding_path'],b)
    r=_probe(f);assert r['observations'][0]['status']!='HIT',r


@pytest.mark.parametrize('field,value', [
    ('part1_artifact_sha256','a'*64), ('part1_artifact_size_bytes',1),
    ('expected_backend','hailo8_to_trt'), ('expected_task','detection'),
    ('case_id','b062'), ('engine_precision','uint8_cast_fp16'),
    ('engine_precision','uint8_dequant_fp16'), ('native_policy_sha256','b'*64),
    ('model_id','wrong-model'), ('setup_id','other-setup'),
])
def test_actual_generated_probe_rejects_other_current_contract(tmp_path, field, value):
    import base64,re
    f=_pair_fixture(tmp_path,0)
    class MutatedCurrentRequest(_LocalReadOnlyTransport):
        def run_read_only(self, command, timeout=0):
            encoded=re.search(r'P = json.loads\(base64.b64decode\("([^"]+)"\)',command).group(1)
            p=json.loads(base64.b64decode(encoded))
            if field in {'model_id','setup_id'}:p[field]=value
            else:p['requirements'][0][field]=value
            return super().run_read_only(remote_run._remote_trt_cache_probe_command(p),timeout=timeout)
    r=remote_run.probe_remote_trt_artifact_cache(transport=MutatedCurrentRequest(f['base']),suite_dir=f['suite'],setup_id=f['setup'],setup_accelerator='deepx',active_run_ids=[f['backend']],resolved_remote_base=str(f['base']),builder_abi=f['abi'])
    assert r['observations'][0]['status']!='HIT',r


def test_wrong_duplicate_does_not_hide_later_matching_parent(tmp_path):
    old = _pair_fixture(tmp_path, 0, pair_name="aaa-old-parent", parent_tag="-old")
    current = _pair_fixture(tmp_path, 0, pair_name="zzz-current-parent", parent_tag="-current")
    assert old['paths']['source_part2_onnx'] != old['paths']['build_part2_onnx']
    assert old['paths']['source_part2_onnx'].read_bytes() == old['paths']['build_part2_onnx'].read_bytes()
    assert old['binding_path'].as_posix() < current['binding_path'].as_posix()
    for fixture in (old, current):
        valid, reason = validate_native_split_quality_binding(
            json.loads(fixture['binding_path'].read_text()), verification_mode="local")
        assert valid, reason
    r = _probe(current)
    assert r['observations'][0]['status'] == 'HIT', r
    assert r['observations'][0]['receipt_path'] == str(current['paths']['engine_build_receipt'])
    current['binding_path'].unlink()
    r = _probe(current)
    assert r['observations'][0]['status'] != 'HIT', r
    assert 'native_binding_part1_mismatch' in str(r), r


def test_public_version_alias_matches_canonical_release_in_fresh_process():
    import subprocess,sys
    r=subprocess.run([sys.executable,'-B','-c','import onnx_splitpoint_tool as p; from onnx_splitpoint_tool import release_identity as i; assert p.__version__ == i.VERSION == i.RELEASE == "2.83"; print(p.__file__)'],capture_output=True,text=True,check=True)
    assert str(Path(__file__).resolve().parents[1]/'onnx_splitpoint_tool/__init__.py') in r.stdout
