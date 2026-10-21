"""Normal read-only receipt preflight, synthetic files and no accelerator SDK."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark import remote_run
from tests.test_generic_quality_completion import descriptor, make_config
from tests.test_v27920_remote_trt_cache_preflight import (
    _LocalReadOnlyTransport, _builder_abi, _builder_abi_sha256, _owner, _receipt,
    _sha, _vendor_native_quality_validator,
)


def _h8_representative(config, suite):
    hw = config["completion_contract"]["processor_contract"]["input_hw"]
    path = suite / "test-only-prepared-input.bin"
    path.write_bytes(bytes(hw[0] * hw[1] * 3))
    config["representative_input"] = {
        "image_id": config["expected_image_ids"][0], "source_image_sha256": "a" * 64,
        "original_wh": [80, 60], "runtime_input_name": "images",
        "prepared_input": {**descriptor(path), "shape": [*hw, 3], "dtype": "uint8", "format": "raw"},
        "runtime_input_adapter": {"backend": "hailo8", "task": "detection",
            "function": "_completion_hailo_inputs", "runtime_input_format": "float32"},
    }


def _fixture(tmp_path, backend="hailo10h"):
    suite = tmp_path / "suite"
    case = suite / "b039"
    case.mkdir(parents=True)
    f = make_config(case)
    config = f["config"]
    config["identity"].update(comparison_backend=backend, setup_id="test_setup")
    config["path_root"] = "suite_root"
    if backend == "hailo8":
        _h8_representative(config, suite)
    artifacts = config["expected_runtime_artifacts"]
    for role in ("source_part1_onnx", "source_part2_onnx"):
        old = Path(artifacts[role]["path"])
        path = case / ("model_part1_b39.onnx" if role == "source_part1_onnx" else "model_part2_b39.onnx")
        path.write_bytes(old.read_bytes())
        artifacts[role] = descriptor(path)
        artifacts[role]["path"] = str(path.relative_to(suite))
    (case / "split_manifest.json").write_text("{}")
    (suite / "models").mkdir()
    (suite / "models/model.onnx").write_bytes(b"test-only-full-model")
    (suite / "benchmark_set.json").write_text(json.dumps({
        "model_id": "yolo26s", "model_name": "yolo26s", "model": "models/model.onnx",
        "benchmark_task": "detection", "cases": [{"id": "b039"}],
    }))
    _vendor_native_quality_validator(suite)
    # Preserve the actual normal-loader fixture source files under this suite.
    sources = suite / "fixture_sources"
    sources.mkdir()
    for row in config["runtime_source_files"]:
        old = Path(row["path"])
        path = sources / old.name
        path.write_bytes(old.read_bytes())
        row["path"] = str(path.relative_to(suite))
    run = {"id": "vendor_to_trt", "type": "matrix", "variants": ["composed"],
           "case_ids": ["b039"], "quality_source_only": True,
           "hardware_setup_id": "test_setup", "native_trt_build": False,
           "native_trt_precision": "fp16", "trt_runtime": "native",
           "stage1": {"type": "deepx" if backend == "deepx_m1" else "hailo",
                      "hw_arch": backend},
           "stage2": {"type": "onnxruntime", "provider": "tensorrt"}}
    builder = tmp_path / "trtexec"
    builder.write_bytes(b"test-only-trtexec")
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    base = tmp_path / "remote"
    # A previous owner namespace is deliberately pinned, as in the real b039.
    root = base / "_onnx_splitpoint_cache/tensorrt_managed_v27516/yolo26s-1111111111111111"
    source = case / "model_part2_b39.onnx"
    leaf = root / remote_run._trt_persistent_engine_relative_dir(
        role="part2", case_id="b039", source_onnx_sha256=_sha(source), precision="fp16")
    _receipt(leaf=leaf, source_bytes=source.read_bytes(), builder=builder, role="part2")
    _receipt(leaf=root / "full/test-only", source_bytes=(suite / "models/model.onnx").read_bytes(),
             builder=builder, role="full")
    _owner(root, builder_abi_sha256=_builder_abi_sha256(abi), stable_key="old-engine-key")
    for role, name in (("engine", "part2_fp16.engine"),
                       ("engine_build_receipt", "engine_build_receipt.json"),
                       ("build_part2_onnx", "source.onnx")):
        artifacts[role] = descriptor(leaf / name)
    config["generic_artifact_binding"]["artifacts"] = deepcopy(artifacts)
    origin = tmp_path / "source_config.json"
    plan = {"runs": [run]}

    def publish():
        origin.write_text(json.dumps(config))
        run["generic_quality_completion_configs"] = {"b039": "quality_runtime/b039.json"}
        run["generic_quality_completion_config_sha256"] = {"b039": _sha(origin)}
        run["generic_quality_completion_sources"] = {"b039": descriptor(origin)}
        (suite / "benchmark_plan.json").write_text(json.dumps(plan))

    publish()
    transport = _LocalReadOnlyTransport(base)

    def probe():
        return remote_run.probe_remote_trt_artifact_cache(
            transport=transport, suite_dir=suite, setup_id="test_setup",
            builder_abi=abi, resolved_remote_base=str(base))

    return locals()


@pytest.mark.parametrize("backend", ["hailo10h", "deepx_m1", "hailo8"])
def test_explicit_generic_recipe_pins_existing_fp16_without_native_policy(tmp_path, backend):
    f = _fixture(tmp_path, backend)
    result = f["probe"]()
    assert result["local_errors"] == []
    req = result["requirements"][0]
    assert req["recipe_source"] == "generic_quality_completion_config"
    assert req["engine_precision"] == "fp16"
    assert req["runtime_precision_identity"] == "float32_layout_fp16"
    assert "native_policy_sha256" not in req
    row = result["observations"][0]
    assert row["status"] == "HIT", row
    assert row["artifact_path"] == f["artifacts"]["engine"]["path"]
    assert row["evidence"]["source_binding"] == "generic_quality_completion_config"
    assert row["evidence"]["namespace_owner_status"] == "legacy_owner_engine_key_changed"
    assert not (f["suite"] / "quality_runtime").exists()  # preflight performs no staging
    assert len(f["transport"].commands) == 1


@pytest.mark.parametrize("fault", ["engine", "receipt", "source", "builder"])
def test_actual_pinned_remote_bytes_and_receipt_still_required(tmp_path, fault):
    f = _fixture(tmp_path)
    path = {"engine": f["leaf"] / "part2_fp16.engine",
            "receipt": f["leaf"] / "engine_build_receipt.json",
            "source": f["leaf"] / "source.onnx", "builder": f["builder"]}[fault]
    path.write_bytes(path.read_bytes() + b"drift")
    result = f["probe"]()
    assert result["observations"][0]["status"] != "HIT"


@pytest.mark.parametrize("fault", ["config", "source", "source_closure", "setup", "backend",
                                   "precision", "build_source", "build_enabled", "mixed"])
def test_local_explicit_contract_drift_never_falls_back_to_native(tmp_path, fault):
    f = _fixture(tmp_path)
    if fault == "config":
        f["origin"].write_bytes(f["origin"].read_bytes() + b" ")
    elif fault == "source":
        (f["case"] / "model_part1_b39.onnx").write_bytes(b"drift")
    elif fault == "source_closure":
        next(f["sources"].iterdir()).write_bytes(b"drift")
    else:
        if fault in {"setup", "backend", "precision"}:
            key = {"setup": "setup_id", "backend": "comparison_backend", "precision": "precision"}[fault]
            f["config"]["identity"][key] = "wrong"
        elif fault == "build_source":
            f["artifacts"]["build_part2_onnx"]["sha256"] = "a" * 64
            f["config"]["generic_artifact_binding"]["artifacts"] = deepcopy(f["artifacts"])
        elif fault == "build_enabled":
            f["run"]["native_trt_build"] = True
        elif fault == "mixed":
            extra = deepcopy(f["run"]); extra["id"] = "another_vendor_to_trt"
            for key in list(extra):
                if key.startswith("generic_quality_completion_"):
                    extra.pop(key)
            f["plan"]["runs"].append(extra)
        f["publish"]()
    result = f["probe"]()
    assert result["local_errors"]
    assert result["observations"][0]["status"] == "UNKNOWN"
    assert f["transport"].commands == []
    assert "native_policy_sha256" not in result["requirements"][0]


@pytest.mark.parametrize("malformed", [None, {}, "bad"])
def test_malformed_explicit_optin_cannot_become_native_default(tmp_path, malformed):
    f = _fixture(tmp_path)
    f["run"]["generic_quality_completion_configs"] = malformed
    (f["suite"] / "benchmark_plan.json").write_text(json.dumps(f["plan"]))
    result = f["probe"]()
    assert result["local_errors"]
    assert f["transport"].commands == []


@pytest.mark.parametrize("replacement", ["engine_and_receipt", "receipt_whitespace"])
def test_valid_but_different_receipt_or_engine_is_not_the_selected_artifact(tmp_path, replacement):
    f = _fixture(tmp_path)
    receipt = f["leaf"] / "engine_build_receipt.json"
    if replacement == "engine_and_receipt":
        engine = f["leaf"] / "part2_fp16.engine"
        engine.write_bytes(b"another-valid-build")
        payload = json.loads(receipt.read_text())
        payload["engine_sha256"] = _sha(engine)
        payload.pop("receipt_sha256")
        from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
        payload["receipt_sha256"] = canonical_json_sha256(payload)
        receipt.write_text(json.dumps(payload))
    else:
        receipt.write_bytes(receipt.read_bytes() + b" ")
    row = f["probe"]()["observations"][0]
    assert row["status"] == "MISS"
    assert row["reason"] == "generic_quality_artifact_mismatch"


def test_missing_owner_or_changed_abi_never_authorizes_explicit_generic(tmp_path):
    f = _fixture(tmp_path)
    (f["root"] / ".splitpoint_trt_cache_owner.json").unlink()
    assert f["probe"]()["observations"][0]["status"] != "HIT"
    _owner(f["root"], builder_abi_sha256="a" * 64, stable_key="old-engine-key")
    assert f["probe"]()["observations"][0]["status"] != "HIT"


def _bridge_fixture(tmp_path, index):
    from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
    from onnx_splitpoint_tool.native_split_quality import (
        known_native_split_policy, materialize_native_split_preselection,
        seal_native_split_quality_binding, validate_native_split_quality_binding,
    )
    from tests.test_v2803_fix2_trt_preflight import _replace, _write
    pairs = json.loads((Path(__file__).parent / 'fixtures/generic_quality_existing_bridge_metadata.json').read_text())['pairs']
    original = deepcopy(pairs[index]['binding']); pre = original['preselection']
    backend, setup, precision = pre['backend'], pre['setup_id'], pre['precision']
    suite = tmp_path / 'suite'; case = suite / 'b003'; case.mkdir(parents=True)
    fixture = make_config(case); config = fixture['config']; config['path_root'] = 'suite_root'
    config['identity'].update(case_id='b003', setup_id=setup, precision=precision,
        comparison_backend={'hailo8_to_trt':'hailo8', 'hailo10h_to_trt':'hailo10h','deepx_to_trt':'deepx_m1'}[backend])
    if backend == 'hailo8_to_trt':
        _h8_representative(config, suite)
    (suite/'models').mkdir(); (suite/'models/model.onnx').write_bytes(b'synthetic-full-yolo26s')
    source = case/'model_part2_b3.onnx';source.write_bytes(b'synthetic-canonical-P2')
    part1 = case/'model_part1_b3.onnx';part1.write_bytes(b'synthetic-canonical-P1')
    (case/'split_manifest.json').write_text('{}')
    _write(suite/'benchmark_set.json', {'model_id':'yolo26s','model':'models/model.onnx',
        'model_name':'yolo26s','benchmark_task':'detection','cases':[{'id':'b003'}]})
    run={'id':backend,'type':'matrix','variants':['composed'],'case_ids':['b003'],
        'quality_source_only':True,'native_trt_build':False,'native_trt_precision':'fp16','trt_runtime':'native',
        'hardware_setup_id':setup,'stage1':{'hw_arch':backend.removesuffix('_to_trt')},
        'stage2':{'provider':'tensorrt'}}
    _write(suite/'benchmark_plan.json',{'runs':[run]});_vendor_native_quality_validator(suite)
    builder=tmp_path/'trtexec';builder.write_bytes(b'#!/bin/sh\nexit 99\n');builder.chmod(0o755)
    abi=_builder_abi(builder);base=tmp_path/'remote'
    namespace=base/'_onnx_splitpoint_cache/tensorrt_managed_v27516'/remote_run._stable_trt_engine_cache_key(suite,builder_abi=abi)
    root=namespace/'native_split_quality'/setup/'yolo26s/b003'/backend/'original-compiled-bridge'
    leaf=root/'engine_cache/b003/part2'/precision
    paths,replacements={},{}
    noop=original['artifacts']['source_part2_onnx']['sha256']==original['artifacts']['build_part2_onnx']['sha256']
    for role,old in original['artifacts'].items():
        if role=='trtexec':path=builder
        elif role in {'part1_runtime','boundary_metadata','native_trt_meta'}:path=root/Path(old['path']).name
        elif role=='source_part2_onnx':path=root/'benchmark_set/b003/source_part2.onnx'
        else:path=leaf/Path(old['path']).name
        paths[role]=path;replacements[old['path']]=str(path.resolve())
        if role not in {'boundary_metadata','native_trt_meta','engine_build_receipt','trtexec'}:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(source.read_bytes() if role=='source_part2_onnx' or role=='build_part2_onnx' and noop else ('synthetic-'+role).encode())
        if path.is_file():replacements[old['sha256']]=_sha(path)
    metadata=_replace(original['boundary_metadata_payload'],replacements);metadata.pop('metadata_sha256')
    metadata['part1_artifact_size_bytes']=paths['part1_runtime'].stat().st_size
    metadata['metadata_sha256']=canonical_json_sha256(metadata);_write(paths['boundary_metadata'],metadata)
    policy=known_native_split_policy(model_id='yolo26s',case_id='b003',setup_id=setup,backend=backend)
    selected=materialize_native_split_preselection(policy=policy,part1_artifact=descriptor(paths['part1_runtime']),
        boundary_metadata=metadata,boundary_metadata_artifact=descriptor(paths['boundary_metadata']))
    receipt=_replace(original['engine_build_receipt'],replacements);receipt.pop('receipt_sha256')
    receipt['receipt_sha256']=canonical_json_sha256(receipt);_write(paths['engine_build_receipt'],receipt)
    meta=_replace(original['native_trt_meta_payload'],replacements);meta['engine_build_receipt']=receipt
    meta['build']['cmd']=receipt['command'];_write(paths['native_trt_meta'],meta)
    boundary={key:selected[key] for key in original['boundary_contract']}
    binding=seal_native_split_quality_binding({'eval_run_id':'synthetic-original-Generic','source_run_id':backend,
        'quality_completed':True,'performance_claims_emitted':False,'preselection':selected,
        'preselection_sha256':selected['selection_sha256'],'artifacts':{role:descriptor(path) for role,path in paths.items()},
        'boundary_contract':boundary,'boundary_contract_sha256':canonical_json_sha256(boundary),
        'engine_build_receipt':receipt,'engine_build_receipt_sha256':receipt['receipt_sha256'],
        'native_trt_meta':meta,'native_trt_meta_sha256':canonical_json_sha256(meta),
        'producer_command':['python','native_trt_from_benchmarkset.py']})
    verified,status=validate_native_split_quality_binding(binding,verification_mode='local');assert verified,status
    artifacts=config['expected_runtime_artifacts']
    for role in ('part1_runtime','boundary_metadata','build_part2_onnx','engine','engine_build_receipt'):
        artifacts[role]=descriptor(paths[role])
    for role,path in [('source_part1_onnx',part1),('source_part2_onnx',source)]:
        artifacts[role]={**descriptor(path),'path':str(path.relative_to(suite))}
    config['generic_artifact_binding']['artifacts']=deepcopy(artifacts)
    sources=suite/'fixture_sources';sources.mkdir()
    for record in config['runtime_source_files']:
        old=Path(record['path']);path=sources/old.name;path.write_bytes(old.read_bytes());record['path']=str(path.relative_to(suite))
    container=tmp_path/'original_generic_composed_request.json'
    source_request={'model_id':'yolo26s','case_id':'b003','setup_id':setup,
                    'runtime_precision_identity':precision,'native_split_quality_binding':binding}
    _write(container,source_request)
    config['existing_part2_bridge']={'schema':'onnx-splitpoint/existing-part2-bridge-reference','schema_version':1,
        'binding_sha256':binding['binding_sha256'],'policy_sha256':selected['policy_sha256'],'precision':precision,
        'binding_source':{**descriptor(container),'path':'quality_runtime/assets/b003/existing_part2_bridge/composed_request.json',
                          'local_path':str(container),'pointer':'/native_split_quality_binding'}}
    config['original_run_cfg']={'native_trt_precision':precision,'native_trt_workspace_mb':4096,
                               'boundary_mode':'raw_uint8_hailo' if precision=='uint8_dequant_fp16' else 'canonical_float32'}
    origin=tmp_path/'source_config.json'
    def publish():
        _write(origin,config)
        run.update(generic_quality_completion_configs={'b003':'quality_runtime/b003.json'},
            generic_quality_completion_config_sha256={'b003':_sha(origin)},
            generic_quality_completion_sources={'b003':descriptor(origin)})
        _write(suite/'benchmark_plan.json',{'runs':[run]})
    publish();_owner(namespace,builder_abi_sha256=_builder_abi_sha256(abi))
    transport=_LocalReadOnlyTransport(base)
    def probe():return remote_run.probe_remote_trt_artifact_cache(transport=transport,suite_dir=suite,
        setup_id=setup,builder_abi=abi,resolved_remote_base=str(base))
    return locals()


@pytest.mark.parametrize('index',[0,1,2],ids=['H8_uint8','H10_uint8','DeepX_float32_noop'])
def test_explicit_original_generic_bridge_uses_normal_full_compiled_validator(tmp_path,index):
    f=_bridge_fixture(tmp_path,index);r=f['probe']();assert not r['local_errors'],r
    req=r['requirements'][0];assert req['engine_precision']==f['precision']
    assert req['recipe_source']=='generic_quality_completion_config' and 'native_policy_sha256' not in req
    row=r['observations'][0];assert row['status']=='HIT',row
    assert row['evidence']['source_binding']=='generic_quality_existing_part2_bridge'
    assert row['evidence']['binding_sha256']==f['binding']['binding_sha256']
    assert row['evidence']['existing_bridge_does_not_admit_current_quality'] is True
    assert row['evidence']['validator_status']=='local_files_rehashed_and_exact_cross_links_verified'
    assert not list(f['base'].rglob('native_split_quality_binding.json'))  # exact bound original request is the container


@pytest.mark.parametrize('fault',['container_drift','swapped_comparison_seal','wrong_policy','boundary','part1','bridge','missing_reference'])
def test_explicit_bridge_drift_never_uses_another_binding_or_direct_recipe(tmp_path,fault):
    f=_bridge_fixture(tmp_path,1)
    if fault=='container_drift':f['container'].write_bytes(f['container'].read_bytes()+b' ')
    elif fault=='swapped_comparison_seal':
        f['config']['existing_part2_bridge']['binding_sha256']='a'*64;f['publish']()
    elif fault=='wrong_policy':f['config']['existing_part2_bridge']['policy_sha256']='a'*64;f['publish']()
    elif fault=='missing_reference':f['config'].pop('existing_part2_bridge');f['publish']()
    else:
        path=f['paths'][{'boundary':'boundary_metadata','part1':'part1_runtime','bridge':'build_part2_onnx'}[fault]]
        path.write_bytes(path.read_bytes()+b'drift')
    result=f['probe']();assert result['observations'][0]['status']!='HIT'


def test_normal_stager_copies_only_bound_request_bytes_and_never_follows_its_fields(tmp_path):
    from onnx_splitpoint_tool.workflow.generic_quality_sidecars import materialize_generic_quality_sidecars
    f=_bridge_fixture(tmp_path,1)
    unrelated=tmp_path/'not-a-stage-instruction';unrelated.write_bytes(b'preserve me')
    source_request=json.loads(f['container'].read_text())
    source_request['arbitrary_referenced_asset']={'path':str(unrelated),'destination':'unexpected-copy'}
    f['container'].write_text(json.dumps(source_request))
    f['config']['existing_part2_bridge']['binding_source'].update(
        sha256=_sha(f['container']),size_bytes=f['container'].stat().st_size)
    from onnx_splitpoint_tool.gui.controller import _copy_runner_lib
    _copy_runner_lib(f['suite'])
    f['config']['runtime_source_files'].extend(
        {**descriptor(path), 'path':str(path.relative_to(f['suite']))}
        for path in sorted((f['suite']/'splitpoint_runners').rglob('*.py')))
    f['publish']()
    before={p for p in f['suite'].rglob('*') if p.is_file()}
    plan=json.loads((f['suite']/'benchmark_plan.json').read_text())
    result=materialize_generic_quality_sidecars(suite_dir=f['suite'],model_id='yolo26s',benchmark_plan=plan)
    copied=f['suite']/'quality_runtime/assets/b003/existing_part2_bridge/composed_request.json'
    assert copied.read_bytes()==f['container'].read_bytes()
    assert {p for p in f['suite'].rglob('*') if p.is_file()}-before=={
        copied,f['suite']/'quality_runtime/b003.json'}
    assert unrelated.read_bytes()==b'preserve me'
    assert [r['role'] for r in result[0]['assets']]==['existing_part2_bridge_binding_source']
    # The existing child loader reads the actual staged container and strict
    # compiled artifacts independently after staging, with no SDK/inference.
    import os, subprocess, sys
    command=[sys.executable, '-B', '-c',
        'from splitpoint_runners.generic_quality_completion import load_quality_completion_config; '
        'import sys; load_quality_completion_config(sys.argv[1],check_files=True)',
        str(f['suite']/'quality_runtime/b003.json')]
    environment={**os.environ, 'PYTHONPATH':str(f['suite']), 'PYTHONDONTWRITEBYTECODE':'1'}
    checked=subprocess.run(command,cwd=f['suite'],env=environment,capture_output=True,text=True)
    assert checked.returncode==0,checked.stderr
    copied.write_bytes(copied.read_bytes()+b'drift')
    rejected=subprocess.run(command,cwd=f['suite'],env=environment,capture_output=True,text=True)
    assert rejected.returncode!=0 and 'source/artifact bytes changed' in rejected.stderr


def test_valid_separate_comparison_binding_cannot_replace_original_generic_seal(tmp_path):
    from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
    from onnx_splitpoint_tool.native_split_quality import validate_native_split_quality_binding
    f=_bridge_fixture(tmp_path,1)
    different=deepcopy(f['binding']);different['eval_run_id']='separate-native-comparison'
    different.pop('binding_sha256');different['binding_sha256']=canonical_json_sha256(different)
    verified,status=validate_native_split_quality_binding(different,verification_mode='local')
    assert verified,status
    request=json.loads(f['container'].read_text());request['native_split_quality_binding']=different
    f['container'].write_text(json.dumps(request))
    f['config']['existing_part2_bridge']['binding_source'].update(
        sha256=_sha(f['container']),size_bytes=f['container'].stat().st_size)
    f['publish']()
    report=f['probe']()
    assert report['observations'][0]['status']=='UNKNOWN'
    assert 'original_generic_bridge_mismatch' in report['local_errors'][0]
    assert not f['transport'].commands
