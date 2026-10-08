"""Energy must replay the measured completion mode and publish fresh evidence."""
import ast
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from onnx_splitpoint_tool import native_detection_postprocess as ndp
from onnx_splitpoint_tool.native_command_contract import split_energy_detection_completion_mode
from scripts import native_deepx_trt_e2e_from_benchmarkset as deepx
from scripts import native_hailo10_trt_e2e_from_benchmarkset as hailo10
from scripts import run_native_producer_energy_from_summary as consumer
from test_v272_detection_hotloop_integration import _FakeDeepX, _FakeHailo10Session, _FakeTRT
from test_v27931_hailo_manifest_energy import _fresh
from test_v279_native_three_stage import _decoded_completion_contract


def _main(module):
    return next(n for n in ast.parse(Path(module.__file__).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')


def _energy_exit(module):
    return next(n for n in ast.walk(_main(module)) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'energy_binding is not None'
                and any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'exact_count' for t in x.targets) for x in n.body))


@pytest.mark.parametrize('mode', ['', 'auto', 'legacy_attested_hotloop', 'garbage'])
def test_energy_mode_requires_explicit_successful_choice(mode):
    with pytest.raises(ValueError, match='completion_runtime_mode_unbound'):
        split_energy_detection_completion_mode({'runtime_options': {'completion_runtime_mode': mode}})


@pytest.mark.parametrize('module,expected_sites', [(deepx, 2), (hailo10, 4)])
@pytest.mark.parametrize('mode', ['strict', 'bound_dense', 'fast_oracle_outside_timing'])
def test_all_actual_factory_sites_use_sealed_mode_and_preserve_performance_selection(module, expected_sites, mode):
    calls = [n for n in ast.walk(_main(module)) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'create_detection_completion_runtime']
    assert len(calls) == expected_sites
    for call in calls:
        node = next(k.value for k in call.keywords if k.arg == 'runtime_mode')
        code = compile(ast.Expression(body=node), module.__file__, 'eval')
        ns = SimpleNamespace(completion_runtime_mode='auto')
        scope = {**module.__dict__, 'ns': ns, 'energy_binding': {'runtime_options': {'completion_runtime_mode': mode}}}
        assert eval(code, scope) == mode
        scope['energy_binding'] = None
        assert eval(code, scope) == 'auto'


@pytest.mark.parametrize('module', [deepx, hailo10])
def test_actual_fifo_fresh_marker_crosses_unchanged_energy_importer(module, tmp_path, capsys):
    planned, _, run, directory = _fresh(tmp_path)
    outputs, contract = _decoded_completion_contract()
    command = planned['native_command_contract']
    binding = copy.deepcopy(command)
    binding['runtime_options']['completion_runtime_mode'] = 'fast_oracle_outside_timing'
    class TRT(_FakeTRT):
        def run_prepared(self): return outputs
    runtime = ndp.create_detection_completion_runtime(contract, runtime_mode='fast_oracle_outside_timing',
        source_paths=(module.__file__,), sentinel_context={'runtime_instance_id':'energy-fixture','repetition_id':'energy-fixture','input_image_sha256':command['input_image_sha256']})
    kwargs = dict(frames=3, warmup=0, queue_depth=2, task='detection', completion_runtime=runtime, warmup_completion_runtime=None)
    if module is deepx:
        result = module._deepx_fifo_run(_FakeDeepX(), np.zeros(1,np.float32), TRT(), **kwargs)
    else:
        result = module._hailo10_async_fifo_run(_FakeHailo10Session(), {'images':np.zeros(1,np.float32)}, TRT(), inflight=2, **kwargs)
    result['ok'] = True
    ns = SimpleNamespace(source_contract_sha256=command['contract_sha256'], energy_preflight_nonce='fresh-nonce', energy_workload_only=True)
    scope = {**module.__dict__, 'report':result, 'energy_binding':binding, 'ns':ns, 'task':'detection', 'task_effective':'detection'}
    exit_node = _energy_exit(module)
    exec(compile(ast.Module(body=exit_node.body, type_ignores=[]), module.__file__, 'exec'), scope)
    marker_node = next(n for n in ast.walk(_main(module)) if isinstance(n, ast.If)
                       and ast.unparse(n.test) == "ns.energy_workload_only and report.get('ok')")
    exec(compile(ast.Module(body=marker_node.body, type_ignores=[]), module.__file__, 'exec'), scope)
    stdout = capsys.readouterr().out
    path = directory/'workload_stdout.log';path.write_text(stdout)
    run['collector_and_workload_log_diagnostics']={'workload_stdout':{'available':True,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
    assert stdout.count('__SPLITPOINT_ENERGY_COMPLETION__=') == 1
    verified, reason = consumer._verify_fresh_fast_energy_completion(run,planned,run_dir=directory)
    assert verified is not None, reason
    assert verified['completed_work_units'] == 3
    for key,bad in [('completion_runtime_mode','strict'),('completion_observation_relation','same_hotloop_sentinel'),('postprocess_completed_frames',2)]:
        changed=copy.deepcopy(result);changed[key]=bad;scope['report']=changed
        with pytest.raises(RuntimeError,match='completion'):
            exec(compile(ast.Module(body=exit_node.body,type_ignores=[]),module.__file__,'exec'),scope)


def _energy_binding_fixture(module, mode):
    options=dict(task='detection',warmup=0,queue_depth=2,inflight=2,producer_impl='auto',part1_onnx_used=False,
        quantized_inputs=True,quantized_outputs=True,copy_outputs=True,canonical_input_slot_names=['images'],canonical_output_slot_names=['boundary'],
        prepared_input_bound=True,preprocess_mode_requested='auto',preprocess_mode_effective='letterbox',
        letterbox_pad_value_requested=114,letterbox_pad_value_effective=114,letterbox_pad_value=114,
        completion_runtime_mode=mode)
    prepared=dict(format='numpy_npy_v1',shape=[1,3,8,8],dtype='uint8',c_contiguous=True,task='detection',
        preprocess='exact_performance_prepared_tensor_persisted',normalization='hef_quant_info_from_unit_float32',
        preprocess_mode_requested='auto',preprocess_mode='letterbox',preprocess_mode_effective='letterbox',
        letterbox_pad_value_requested=114,letterbox_pad_value_effective=114,letterbox_pad_value=114,pad_value_effective=114,
        slot_order=['images'],entries=[dict(name='images',artifact_name='prepared_input',shape=[1,3,8,8],dtype='uint8',c_contiguous=True)])
    return dict(backend='deepx_to_trt' if module is deepx else 'hailo10h_to_trt',runtime_options=options,
        artifacts={'prepared_input':{'path':'/bound.npy'}},prepared_input_contract=prepared,
        benchmark_set='/bound/benchmark_set',case='b009',precision='uint8_dequant_fp16',input_image='/bound/image.jpg',
        hw_arch='hailo10h',boundary_contract={'boundary_layout_effective':'as_input'},
        python_executable='/bound/python',runner='scripts/runner.py',contract_sha256='c'*64)


@pytest.mark.parametrize('module',[deepx,hailo10])
@pytest.mark.parametrize('mode',['strict','bound_dense','fast_oracle_outside_timing'])
def test_energy_argv_and_preflight_bind_resolved_mode(module,mode,monkeypatch):
    from onnx_splitpoint_tool.native_command_contract import split_energy_runtime_argv
    binding=_energy_binding_fixture(module,mode)
    argv=split_energy_runtime_argv(binding,duration_s=60,fresh_output_root='/new',remote_tool_dir='/tool',preflight_attestation_path='/preflight')
    assert argv[argv.index('--completion-runtime-mode')+1]==mode
    ns=SimpleNamespace(energy_preflight_attestation='/preflight',energy_preflight_nonce='nonce',source_contract_sha256='c'*64,
        energy_preflight_max_age_s=300,warmup=0,build_missing_engine=False,force_rebuild_engine=False,dump_outputs=False,dump_boundary=False,
        repetitions=1,duration_s=60,benchmark_set=binding['benchmark_set'],case='b009',precision=binding['precision'],image=binding['input_image'],
        queue_depth=2,boundary_layout='as_input',task='detection',preprocess_mode='auto',letterbox_pad_value=114,
        hw_arch='hailo10h',inflight=2,producer_impl='auto',quantized_inputs=True,quantized_outputs=True,copy_outputs=True,completion_runtime_mode=mode)
    monkeypatch.setattr(module,'load_split_energy_workload_binding',lambda *a,**k:(binding,'verified-fixture'))
    monkeypatch.setattr(module,'_bound_detection_completion_contract',lambda value:{})
    check=module._deepx_energy_binding if module is deepx else module._hailo10_energy_binding
    verified,reason=check(ns);assert verified is binding,reason
    ns.completion_runtime_mode='auto';verified,reason=check(ns)
    assert verified is None and 'cli_completion_runtime_mode_mismatch'in reason
    ns.completion_runtime_mode=mode;binding['runtime_options'].pop('completion_runtime_mode')
    verified,reason=check(ns);assert verified is None and 'runtime_mode_unbound'in reason
    with pytest.raises(ValueError,match='runtime_mode_unbound'):
        split_energy_runtime_argv(binding,duration_s=60,fresh_output_root='/new',remote_tool_dir='/tool',preflight_attestation_path='/preflight')


def test_normal_sync_cold_target_imports_both_exact_energy_runners(tmp_path):
    """Use production sync/verifiers and isolate imports from this checkout."""
    from onnx_splitpoint_tool.remote_runtime_closure import native_remote_package_closure
    from onnx_splitpoint_tool.workflow import runner
    from test_v2803_fix5_native_asset_sync import _LocalTransport, _sync

    target = tmp_path / 'cold native runtime'
    transport = _LocalTransport(target)
    for relative, _module, tokens in native_remote_package_closure():
        _sync(transport, relative, tokens)
    for module in (hailo10, deepx):
        name = Path(module.__file__).name
        runner._sync_remote_script_v60i(
            ssh='nx@isolated-native-test', remote_tool_dir=str(target),
            script_name=name,
            required_tokens=('--completion-runtime-mode', '--energy-preflight-nonce',
                             '__SPLITPOINT_ENERGY_COMPLETION__='),
            timeout=20, process_runner=transport,
        )
        source = Path(module.__file__)
        assert (target / 'scripts' / name).read_bytes() == source.read_bytes()
        mirror = source.parent.parent / 'onnx_splitpoint_tool/resources/remote_scripts' / name
        assert mirror.read_bytes() == source.read_bytes()

    program = f"""
import sys, pathlib, json
root = pathlib.Path({str(target)!r})
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'scripts'))
import native_hailo10_trt_e2e_from_benchmarkset as hailo10
import native_deepx_trt_e2e_from_benchmarkset as deepx
from onnx_splitpoint_tool.native_command_contract import split_energy_detection_completion_mode
assert deepx.NativeTRT is hailo10.NativeTRT
assert deepx._NATIVE_TRT_IMPORT_ERROR is None
for module in (deepx, hailo10):
    assert pathlib.Path(module.__file__).is_relative_to(root)
    assert pathlib.Path(module.split_energy_detection_completion_mode.__code__.co_filename).is_relative_to(root)
    assert module.split_energy_detection_completion_mode is split_energy_detection_completion_mode
    assert module.split_energy_detection_completion_mode({{'runtime_options': {{'completion_runtime_mode': 'fast_oracle_outside_timing'}}}}) == 'fast_oracle_outside_timing'
print(json.dumps({{'imported': True, 'inference_started': False}}))
"""
    completed = subprocess.run(
        [sys.executable, '-I', '-B', '-c', program], cwd=target,
        text=True, capture_output=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {'imported': True, 'inference_started': False}


@pytest.mark.parametrize('shape,layout', [
    ((8, 10, 3), 'HWC'), ((1, 8, 10, 3), 'NHWC'),
    ((3, 8, 10), 'CHW'), ((1, 3, 8, 10), 'NCHW'),
])
@pytest.mark.parametrize('quantized', [False, True])
def test_hailo10_declares_executed_layout_before_timing_without_tensor_change(
    tmp_path, shape, layout, quantized,
):
    from PIL import Image

    image = tmp_path / 'asymmetric.png'
    pixels = np.arange(5 * 9 * 3, dtype=np.uint8).reshape(5, 9, 3)
    Image.fromarray(pixels).save(image)

    class Session:
        def quantize_input(self, name, value):
            assert name == 'images' and value.dtype == np.float32
            return np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)

    prepared = SimpleNamespace(
        input_names=['images'], handle=SimpleNamespace(
            runtime_input_shapes={'images': shape}, input_shapes={}, session=Session(),
        ),
    )
    options = dict(image=str(image), task='detection', preprocess_mode='resize')
    original = hailo10._make_input(prepared, quantized, **options)
    metadata = {}
    inputs = hailo10._make_input(
        prepared, quantized, **options, prepared_input_metadata=metadata,
    )
    assert metadata == {'images': {'layout': layout}}
    assert inputs['images'].shape == shape
    assert inputs['images'].dtype == original['images'].dtype
    assert inputs['images'].tobytes() == original['images'].tobytes()

    # Execute the normal pre-timer persistence block as shipped. Only the SDK
    # setup preceding it is substituted; arrays and contract serialization are real.
    branch = next(node for node in ast.walk(_main(hailo10))
                  if isinstance(node, ast.If) and ast.unparse(node.test) == 'energy_binding is not None'
                  and any(isinstance(item, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'inputs'
                                  for target in item.targets) for item in node.orelse))
    start = next(index for index, item in enumerate(branch.orelse)
                 if isinstance(item, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'prepared_dir'
                         for target in item.targets))
    scope = {**hailo10.__dict__, 'prepared': prepared, 'inputs': inputs,
             'prepared_input_metadata': metadata, 'prepared_input_artifacts': {},
             'out_dir': tmp_path, 'report': {}, 'task_effective': 'detection',
             'preprocess_mode_effective': 'resize',
             'ns': SimpleNamespace(image=str(image), quantized_inputs=quantized,
                                   preprocess_mode='resize', letterbox_pad_value=114)}
    exec(compile(ast.Module(body=branch.orelse[start:], type_ignores=[]),
                 hailo10.__file__, 'exec'), scope)
    contract = scope['prepared_input_contract']
    assert contract['layout'] == layout
    assert contract['entries'][0]['layout'] == layout
    assert contract['normalization'] == (
        'hef_quant_info_from_unit_float32' if quantized else 'divide_255_float32'
    )
    stored = np.load(scope['prepared_input_artifacts']['prepared_input_00']['path'],
                     allow_pickle=False)
    assert stored.tobytes() == original['images'].tobytes()
