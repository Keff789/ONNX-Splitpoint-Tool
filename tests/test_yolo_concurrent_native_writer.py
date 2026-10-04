"""Prepared concurrent Native writer checks; all accelerator leaves are software.

The fixture executes real completion callbacks and existing Quality/Native
writers. It does not claim accelerator performance or scientific Quality PASS.
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import subprocess
import time
import hashlib
import os

import numpy as np
import pytest

from scripts import native_hailo_trt_fifo_from_benchmarkset as parent
from onnx_splitpoint_tool.native_command_contract import verify_native_command_contract
from onnx_splitpoint_tool.native_split_quality import seal_native_split_quality_binding
from onnx_splitpoint_tool.trt_quality_chain import split_binding_set_from_central_quality_summary
from tests import test_yolo_augmentation_quality_release as quality

START = ('yolov7_paper', 'b066', 'orin_nx_hailo8_01', 'hailo8_to_trt')


def test_concurrent_helper_child_bootstraps_toolroot_without_installed_package(tmp_path):
    script = Path(parent.__file__).with_name('native_hailo_trt_concurrent_three_stage_from_benchmarkset.py')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, '-S', str(script), '--help'], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert '--campaign-config' in result.stdout


def _read(path):
    return json.loads(Path(path).read_text())


def _writer_fixture(tmp_path, monkeypatch):
    """Real central request/consumer; synthetic existing SDK artifact bytes."""
    runner = quality.make_runner(tmp_path, [START])
    from tests.test_yolo_quality_connected_chain import _authority_inputs
    _authority_inputs(runner)
    original_binding = quality.binding
    outputs = {name: np.full((1, 3, grid, grid, 85), -20., np.float32)
               for name, grid in (('output', 80), ('clone_1', 40), ('clone_2', 20))}

    def binding(*args, **kwargs):
        bound = original_binding(*args, **kwargs)
        meta = copy.deepcopy(bound['native_trt_meta'])
        meta['outputs'] = [dict(name=name, shape=list(value.shape), elem_type='FLOAT', has_dynamic=False)
                           for name, value in outputs.items()]
        meta_path = Path(bound['artifacts']['native_trt_meta']['path'])
        quality.write(meta_path, meta)
        bound['native_trt_meta'] = meta
        bound['native_trt_meta_sha256'] = quality.canonical_json_sha256(meta)
        bound['artifacts']['native_trt_meta'] = quality.write(meta_path, meta)
        return seal_native_split_quality_binding(bound)

    monkeypatch.setattr(quality, 'binding', binding)
    target = runner.augmentation_plan['rows'][0]
    request, _ = quality.emit_request(runner, target, runner.run_dir / 'producer')
    runner._queue_central_quality_requests(target['model_id'])
    result = runner._central_quality_futures[str(request.resolve())].result(timeout=120)
    assert result['technical_status'] == 'completed', result
    summary = _read(runner._publish_case_quality([result]))
    sets = split_binding_set_from_central_quality_summary(summary, eval_run_id=runner.run_id,
        setup_id=target['setup_id'], selections=[dict(target, backend=target['direction'])])
    selected = next(iter(sets['bindings_by_model_case_backend'].values()))
    suite = runner.run_dir / 'native_producers/software_concurrent/hailo8/yolov7_paper/benchmark_set'
    quality.write(suite / 'benchmark_set.json', {'model_id':START[0], 'benchmark_task':'detection'})
    quality.write(suite / 'output_contracts.json', {
        'schema':'onnx-splitpoint/output-contracts', 'schema_version':1, 'model_id':START[0],
        'task':'detection', 'contracts':[{
            'schema':'onnx-splitpoint/output-contract', 'schema_version':1, 'model_id':START[0],
            'task':'detection', 'backend':'tensorrt', 'variant':'full', 'contract_status':'recorded',
            'endpoint_mode':'raw_head', 'host_tail_required':True, 'postprocessing_required':True}]})
    from PIL import Image
    image = suite / 'input.png'
    Image.new('RGB', (640,480), (42,96,128)).save(image)
    bound_path = suite / 'binding.json'
    quality.write(bound_path, selected)
    monkeypatch.setitem(sys.modules, 'hailo_platform', SimpleNamespace(HEF=lambda path: SimpleNamespace(
        get_input_vstream_infos=lambda:[SimpleNamespace(name='images', shape=(640,640,3))])))
    def open_runtime(**kwargs):
        tensor = selected['boundary_metadata_payload']['boundary_tensor']
        trt_input = selected['native_trt_meta']['inputs'][0]
        backend = SimpleNamespace(run=lambda prepared, feeds: SimpleNamespace(outputs={
            tensor['runtime_name']: np.zeros(tensor['shape'], np.uint8)}), cleanup=lambda prepared:None)
        prepared = SimpleNamespace(input_names=['images'],
            handle=SimpleNamespace(runtime_input_shapes={'images':(640,640,3)}))
        trt = SimpleNamespace(inputs=[trt_input['name']], shapes={trt_input['name']:trt_input['shape']},
            dtypes={trt_input['name']:np.uint8}, run=lambda feeds:outputs, close=lambda:None)
        return backend, prepared, trt
    monkeypatch.setattr(parent, '_open_hailo8_python_runtime', open_runtime)
    args = SimpleNamespace(benchmark_set=str(suite), case='b066', hw_arch='hailo8',
        precision='uint8_dequant_fp16', image=str(image), frames=1000, warmup=100, repetitions=3,
        queue_depth=3, post_queue_depth=4, duration_s=0., setup_id=START[2],
        eval_run_id=runner.run_id, source_run_id='hailo8_to_trt', model_id=START[0],
        native_split_quality_binding=str(bound_path), hailo_format='uint8', task='detection',
        preprocess_mode='letterbox', preprocess_mode_effective='letterbox', letterbox_pad_value=114,
        prepared_input_rgb='', expected_prepared_input_sha256='', device_id='',
        copy_outputs=True, dump_outputs=True, dump_boundary=True, output_dir='', boundary_dir='',
        build=False, completion_runtime_mode='fast_oracle_outside_timing',
        expected_runner_sha256='', expected_image_sha256='', expected_hef_sha256='',
        expected_engine_sha256='', expected_boundary_layout='', expected_executable_sha256='',
        source_contract_sha256='', result_json='')
    work = suite / 'native_pipeline/b066/hailo_to_trt/uint8_dequant_fp16'
    work.mkdir(parents=True)
    return runner, selected, suite, work, args, outputs


def test_real_quality_and_prepared_config_use_current_artifacts(tmp_path, monkeypatch):
    runner, selected, suite, work, args, outputs = _writer_fixture(tmp_path, monkeypatch)
    try:
        config, inputs, prepared = parent._concurrent_campaign_config(
            bs=suite, case='b066', hef=Path(selected['artifacts']['part1_runtime']['path']),
            engine=Path(selected['artifacts']['engine']['path']), image=Path(args.image),
            work=work, args=args, quality_binding=selected)
        assert config['engine'] == selected['artifacts']['engine']
        assert config['quality_binding_sha256'] == selected['binding_sha256']
        assert np.array_equal(np.fromfile(prepared, np.uint8).reshape(640,640,3), inputs['images'])
        assert config['outputs'] == [dict(name=name, shape=list(value.shape), dtype='float32')
                                     for name, value in outputs.items()]
        assert config['completion_contract']['processor_contract'] == config['processor_contract']
    finally:
        runner._shutdown_management_services()


@pytest.mark.parametrize('damage', ['engine', 'image', 'input_size'])
def test_prepared_config_rejects_wrong_bound_input_before_runtime(tmp_path, monkeypatch, damage):
    # The ordinary command expectations and prepared-feed checks precede the
    # child invocation. Never replace them with a successful runtime fixture.
    runner, selected, suite, work, args, _ = _writer_fixture(tmp_path, monkeypatch)
    try:
        if damage == 'engine': args.expected_engine_sha256 = '0' * 64
        elif damage == 'image': args.expected_image_sha256 = '0' * 64
        else:
            path = work / 'wrong-input.bin'; path.write_bytes(b'bad')
            args.prepared_input_rgb = str(path)
        monkeypatch.setattr(parent, '_invoke_concurrent_campaign', lambda *a, **k: pytest.fail('runtime must not start'))
        with pytest.raises(RuntimeError):
            parent._run_concurrent_campaign(bs=suite, case='b066',
                hef=Path(selected['artifacts']['part1_runtime']['path']),
                engine=Path(selected['artifacts']['engine']['path']), image=Path(args.image),
                work=work, args=args, quality_binding=selected)
    finally:
        runner._shutdown_management_services()


def test_parent_helper_real_cpp_callbacks_normal_native_and_energy_writers(tmp_path, monkeypatch, capsys):
    """Full writer entry: normal requests, C++ queues, 3300 real Fast callbacks.

    Only the process envelope, compiler and accelerator SDK implementations are
    controlled. No reader, binding verifier, timer or success flag is replaced.
    """
    from scripts import native_hailo_trt_concurrent_three_stage_from_benchmarkset as helper
    from scripts import native_producer_final_report as final
    from scripts import native_producer_energy_plan as energy
    from tests.test_yolo_concurrent_campaign_runtime import compile_software_runtime
    runner, selected, suite, work, args, _ = _writer_fixture(tmp_path, monkeypatch)
    try:
        config, _, _ = parent._concurrent_campaign_config(
            bs=suite, case='b066', hef=Path(selected['artifacts']['part1_runtime']['path']),
            engine=Path(selected['artifacts']['engine']['path']), image=Path(args.image),
            work=work, args=args, quality_binding=selected)
        library = compile_software_runtime(runner.run_dir / 'sdk-runtime',
                                           boundary_contract=config['boundary_contract'])
        original_spec = importlib.util.spec_from_file_location

        def spec_factory(name, location, *a, **kw):
            spec = original_spec(name, location, *a, **kw)
            if str(location).endswith('/three_stage_canary.py'):
                real_exec = spec.loader.exec_module

                def exec_module(module):
                    real_exec(module)
                    module.build_cpp = lambda _work: library
                spec.loader.exec_module = exec_module
            return spec

        monkeypatch.setattr(importlib.util, 'spec_from_file_location', spec_factory)
        calls = []

        def child(command, **kwargs):
            assert Path(command[1]).name == 'native_hailo_trt_concurrent_three_stage_from_benchmarkset.py'
            calls.append(list(command))
            with monkeypatch.context() as child_scope:
                child_scope.setattr(sys, 'argv', command[1:])
                code = helper.main()
            return subprocess.CompletedProcess(command, code)

        monkeypatch.setattr(subprocess, 'run', child)
        argv = ['native_hailo_trt_fifo_from_benchmarkset.py', '--benchmark-set', str(suite),
                '--case', 'b066', '--hw-arch', 'hailo8', '--precision', args.precision,
                '--image', args.image, '--model-id', START[0], '--setup-id', START[2],
                '--eval-run-id', runner.run_id, '--source-run-id', START[3],
                '--native-split-quality-binding', args.native_split_quality_binding,
                '--frames','1000','--warmup','100','--repetitions','3','--queue-depth','3',
                '--no-build','--dump-outputs','--dump-boundary','--copy-outputs']
        monkeypatch.setattr(sys, 'argv', argv)
        assert parent.main() == 0
        assert len(calls) == 1
        result_path = work / 'native_fifo_results.json'
        raw = _read(result_path)
        command, reason = verify_native_command_contract(raw['native_command_contract'])
        assert command, reason
        assert raw['producer_impl'] == 'hailo8_cpp_concurrent_three_stage'
        assert raw['repetition_count_valid'] == 3
        assert [r['completed_work_units'] for r in raw['repetition_records']] == [1000]*3
        assert raw['native_split_quality_consumer_status'] == 'exact_quality_native_engine_command_and_boundary_match'
        # Negative admission uses the already saved observations; no extra
        # completion loops and no mutation of the successful runtime evidence.
        runtime_report = _read(work / 'concurrent_campaign_runtime.json')
        actual_config = _read(work / 'concurrent_campaign_config.json')
        for damage in ('missing_repetition', 'wrong_time', 'wrong_count', 'included_preprocess', 'undrained', 'wrong_engine'):
            damaged = copy.deepcopy(runtime_report)
            if damage == 'missing_repetition': damaged['repetitions'].pop()
            elif damage == 'wrong_time': damaged['repetitions'][0]['measurement_start_ns'] = 0
            elif damage == 'wrong_count': damaged['repetitions'][0]['completed_work_units'] -= 1
            elif damage == 'included_preprocess': damaged['repetitions'][0]['preprocessing_included'] = True
            elif damage == 'undrained': damaged['repetitions'][0]['pipeline_drained'] = False
            else: damaged['config']['engine']['sha256'] = 'f'*64
            with pytest.raises(RuntimeError):
                parent._concurrent_campaign_payload(damaged, actual_config, work=work, args=args, inputs={})
        # The normal outer process envelope records only its actual return/file.
        source = suite.parent.parent
        envelope = {**raw, 'model':START[0], 'report':str(result_path), 'result_ok':raw['ok'],
                    'child_result_fresh':True, 'returncode':0,
                    'native_fifo_result_sha256':parent._sha256_file(result_path),
                    'native_fifo_result_size_bytes':result_path.stat().st_size}
        quality.write(source / 'analysis_tables/native_fifo_eval_runner.json', {'rows':[envelope]})
        report_dir = runner.run_dir / 'reports/native'
        monkeypatch.setattr(sys, 'argv', ['native_producer_final_report.py', '--root',str(source),
            '--out-dir',str(report_dir),'--eval-root',str(runner.run_dir)])
        assert final.main() == 0
        summary = _read(report_dir / 'native_producer_combined_summary.json')
        assert len(summary['rows']) == 1
        row = summary['rows'][0]
        assert row['ok'] is True and row['repetition_count_valid'] == 3, row
        assert row['producer_impl'] == 'hailo8_cpp_concurrent_three_stage'
        assert row['native_split_final_portable_binding_valid'] is True, row
        monkeypatch.setattr(sys, 'argv', ['native_producer_energy_plan.py', '--summary',
            str(report_dir / 'native_producer_combined_summary.json'), '--out-dir', str(tmp_path/'energy-plan'),
            '--duration-s','60','--runs','3','--remote-root','/software/unexecuted',
            '--remote-tool-dir','/software/tool','--measure-all-runtime-successful','--screening-energy',
            '--hailo8-ssh','software@h8'])
        assert energy.main() == 0
        plan = _read(tmp_path/'energy-plan/native_producer_energy_plan.json')
        assert len(plan['rows']) == 1 and not plan.get('excluded_rows'), plan
        from scripts import native_split_energy_preflight as preflight
        from scripts import run_native_producer_energy_from_summary as energy_consumer
        from onnx_splitpoint_tool.native_command_contract import split_energy_runtime_argv
        nonce = 'software-concurrent-energy-unique-invocation'
        attestation, code = preflight.build_attestation(contract_payload=command, nonce=nonce,
            expected_contract_sha256=command['contract_sha256'],
            expected_preflight_script_sha256=parent._sha256_file(Path(preflight.__file__)),
            tool_root=Path(parent.__file__).resolve().parents[1], valid_for_s=180)
        assert code == 0, attestation
        energy_dir = tmp_path/'energy-run'; energy_dir.mkdir()
        attest_path = energy_dir/'preflight.json'; quality.write(attest_path, attestation)
        energy_argv = split_energy_runtime_argv(command, duration_s=60,
            fresh_output_root=str(energy_dir/'fresh'),
            remote_tool_dir=str(Path(parent.__file__).resolve().parents[1]),
            preflight_attestation_path=str(attest_path), preflight_nonce=nonce)
        monkeypatch.setattr(sys, 'argv', energy_argv[2:])
        capsys.readouterr()
        started = time.time_ns()
        energy_rc = parent.main()
        ended = time.time_ns()
        stdout = capsys.readouterr().out
        (energy_dir/'workload_stdout.log').write_text(stdout)
        assert energy_rc == 0
        fresh_path = Path(energy_argv[energy_argv.index('--result-json')+1])
        fresh = _read(fresh_path)
        assert fresh['completed_work_units'] > 1000 and fresh['warmup'] == 0
        assert fresh['makespan_ms'] >= 60000
        assert len(fresh['request_latency']['pairs']) == fresh['completed_work_units']
        marker = json.loads(next(line.split('=',1)[1] for line in stdout.splitlines()
                                 if line.startswith('__SPLITPOINT_ENERGY_COMPLETION__=')))
        assert marker == {key:value for key,value in fresh.items()
                          if key not in {'request_latency','repetition_records','repetition_evidence'}}
        assert len(stdout.encode()) < 8*1024*1024
        energy_run = {'preflight_evidence':{'ok':True,'nonce':nonce},
            'runtime_work_unit_evidence':{'exact':True,'count':fresh['completed_work_units']},
            'workload_timing':{'status':'ok','rc':energy_rc,'start_ns':started,'end_ns':ended},
            'collector_and_workload_log_diagnostics':{'workload_stdout':{
                'available':True,'bytes':len(stdout.encode()),'sha256':hashlib.sha256(stdout.encode()).hexdigest()}}}
        verified, reason = energy_consumer._verify_fresh_fast_energy_completion(
            energy_run, plan['rows'][0], run_dir=energy_dir)
        assert verified is not None, reason
        quality.write(tmp_path / 'native-writer-acceptance.json', {
            'scope':'software SDK leaves; no accelerator or collector execution',
            'normal_parent_returncode':0, 'native_repetitions':3, 'completed_tasks_per_repetition':1000,
            'quality_binding_source':str(runner.run_dir/'quality_management/central_quality_summary.json'),
            'normal_summary':str(report_dir/'native_producer_combined_summary.json'),
            'normal_energy_rows':len(plan['rows']), 'collectors_started':0,
            'software_energy_duration_s':60, 'software_energy_completed_tasks':fresh['completed_work_units'],
            'energy_stdout_bytes':len(stdout.encode()), 'fresh_energy_consumer_status':verified['status']})
    finally:
        runner._shutdown_management_services()
