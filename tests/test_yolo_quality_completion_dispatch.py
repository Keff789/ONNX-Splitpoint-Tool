"""Normal completion dispatch/download/parser with actual software timer.

The only replaced execution boundary is the accelerator process: its two SDK
calls return deterministic fixture tensors. Completion, functional comparison,
atomic result/sample writes, download and reentry admission stay real.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path

import numpy as np
import pytest

from tests.test_generic_completion_dispatch import LocalTransport, prepared_case
from tests.test_v2912_generic_split_completion import _module, _fixture
from onnx_splitpoint_tool.generic_completion_reporting import load_valid_repetitions, write_completion_reports
from onnx_splitpoint_tool.workflow import generic_completed_task as entry


class SoftwareCompletionTransport(LocalTransport):
    def __init__(self, module, *, fail_repetition=None):
        super().__init__()
        self.module = module
        self.inference_counts = []
        self.fail_repetition = fail_repetition

    def run_streaming(self, command, *, on_line, **kwargs):
        argv = shlex.split(command)
        config = json.loads(Path(argv[argv.index('--generic-completion-config')+1]).read_text())
        index = int(argv[argv.index('--generic-completion-repetition')+1])
        output = Path(argv[argv.index('--out-dir')+1])
        reference = config['functional_reference']['outputs']
        outputs = {name:self.module._completion_bound_array(spec) for name,spec in reference.items()}
        counts = {'p1':0, 'p2':0}
        def sdk_part1():
            counts['p1'] += 1
            if index == self.fail_repetition and counts['p1'] == 101:
                raise RuntimeError('controlled SDK failure after completed warmup')
            return outputs
        def sdk_part2(inputs):
            counts['p2'] += 1
            return inputs
        self.calls.append((config['identity'],index))
        try:
            rc = self.module._run_generic_split_completion(config, output, sdk_part1, lambda value:value,
                sdk_part2, {key:Path(spec['path']) for key,spec in config['expected_runtime_artifacts'].items()},
                clone_payload=(lambda values:{key:np.array(value,copy=True) for key,value in values.items()})
                    if config.get('original_run_cfg', {}).get('streaming_impl') == 'generic' else None,
                repetition_index=index)
        except RuntimeError as error:
            if 'controlled SDK failure' not in str(error):
                raise
            on_line(str(error))
            rc = 1
        self.inference_counts.append(counts)
        on_line(f'controlled SDK process completed {index}/3 with real completion timer')
        return rc


@pytest.mark.parametrize('streaming_impl', ['generic', 'auto'])
def test_real_software_runtime_through_dispatch_download_parser_and_selective_reentry(tmp_path, monkeypatch, streaming_impl):
    module = _module(monkeypatch)
    config, _ = _fixture(tmp_path)
    row = prepared_case(tmp_path)
    # This local seam is explicitly classification, not invented detector data.
    # The three-model detector integration can use the same transport with its
    # normal bound Native/Generic configuration.
    config['identity'] = row['identity']
    config['original_run_cfg'].update(row['runtime_config']['original_run_cfg'])
    config['original_run_cfg']['streaming_impl'] = streaming_impl
    row['runtime_config'] = config
    for key in ('input_binding','completion_contract','measurement_boundary','expected_runtime_artifacts',
                'functional_reference','original_run','task'):
        row[key] = config[key]
    transport = SoftwareCompletionTransport(module)
    entry.run_case(row, transport, tmp_path / 'output', 1)
    assert [index for _,index in transport.calls] == [1,2,3]
    assert transport.inference_counts == [{'p1':1100,'p2':1100}]*3
    valid = load_valid_repetitions(row['result_path'], row)
    assert len(valid) == 3
    assert all(value['functional_control']['checked_outside_timer'] for value in valid)
    assert all(value['handoff_payload_copy'] == (streaming_impl == 'generic') for value in valid)
    output = tmp_path / 'reports'
    summary = write_completion_reports({'rows':[row]}, tmp_path / 'output/results', output)
    assert summary['status_counts']['completed'] == 1 and summary['valid_repetitions'] == 3
    before = Path(row['result_path']).read_bytes()
    entry.run_case(row, transport, tmp_path / 'output', 1)
    assert len(transport.calls) == 3 and Path(row['result_path']).read_bytes() == before


def test_actual_runtime_failure_preserves_valid_repeat_and_resume_only_missing(tmp_path, monkeypatch):
    config, _ = _fixture(tmp_path)
    row = prepared_case(tmp_path)
    config['identity'] = row['identity']
    config['original_run_cfg'].update(row['runtime_config']['original_run_cfg'])
    row['runtime_config'] = config
    for key in ('input_binding','completion_contract','measurement_boundary','expected_runtime_artifacts',
                'functional_reference','original_run','task'):
        row[key] = config[key]
    transport = SoftwareCompletionTransport(_module(monkeypatch), fail_repetition=2)
    with pytest.raises(RuntimeError, match='Generic case stopped'):
        entry.run_case(row, transport, tmp_path/'output', 1)
    valid = load_valid_repetitions(row['result_path'],row)
    assert len(valid) == 1 and valid[0]['repetition_index'] == 1
    saved = json.loads(json.dumps(valid[0]))
    summary = write_completion_reports({'rows':[row]},tmp_path/'output/results',tmp_path/'reports')
    assert summary['status_counts']['partial'] == 1 and summary['valid_repetitions'] == 1
    transport.fail_repetition = None
    entry.run_case(row,transport,tmp_path/'output',1)
    valid = load_valid_repetitions(row['result_path'],row)
    assert [index for _,index in transport.calls] == [1,2,2,3]
    assert len(valid) == 3 and valid[0] == saved


@pytest.mark.parametrize('family,model,case_id,setup,backend', [
    ('yolo11','yolo11l','b120','orin_nx_hailo8_01','hailo8'),
    ('yolo26','yolo26s','b038','orin_nx_hailo10_01','hailo10h'),
    ('yolov7','yolov7_paper','b044','orin_nx_deepx_m1_01','deepx'),
])
def test_three_new_detector_startcases_real_completion_download_and_import(
        tmp_path, monkeypatch, family, model, case_id, setup, backend):
    from tests.test_v2912_generic_split_completion import _descriptor
    from onnx_splitpoint_tool.native_split_quality import known_native_split_policy
    if family == 'yolo11':
        from tests.test_v27923_decoded_completion import _outputs, _runtime
        outputs = _outputs(); processor = _runtime(tmp_path, outputs)
    elif family == 'yolo26':
        from tests.test_v272_detection_completion_runtime import _decoded_outputs, _runtime
        outputs = _decoded_outputs(); processor = _runtime(outputs)
    else:
        from tests.test_v272_yolov7_head_mapping import _raw_source
        from onnx_splitpoint_tool.native_detection_postprocess import build_detection_completion_runtime
        outputs = {f'head{stride}':np.full((1,3,640//stride,640//stride,85), -20., dtype=np.float32)
                   for stride in (8,16,32)}
        outputs['head8'][0,0,40,40,:4] = 0.
        outputs['head8'][0,0,40,40,4:6] = 10.
        processor = build_detection_completion_runtime(model_id=model, outputs=outputs,
            input_hw=[640,640], original_wh=[640,480], source_endpoint_contract=_raw_source(outputs,endpoint_hash='a'*64))
    config, _ = _fixture(tmp_path)
    row = prepared_case(tmp_path, setup=setup, task='detection')
    row['identity'].update(model_id=model, case_id=case_id, setup_id=setup,
        direction=backend+'_to_trt', comparison_backend=backend,
        precision=known_native_split_policy(model_id=model,case_id=case_id,
                                            setup_id=setup,backend=backend+'_to_trt')['precision'],
        output_endpoint_id=processor.execution_contract['comparison_endpoint_contract']['endpoint_contract_hash'])
    row.update(row['identity'])
    row['case_key'] = '|'.join(row['identity'].values())
    row['case_output_relative'] = f'{setup}/{model}/{case_id}'
    row['result_path'] = str(tmp_path / 'output/results' / row['case_output_relative'] / 'generic_completion_results.json')
    # These are synthetic accelerator leaves, not a measured Native pair.
    row['historical_pair'] = {}
    row.pop('native_measurement_boundary', None)
    row.pop('measurement_boundary_comparison', None)
    config.update(identity=row['identity'], task='detection', completion_contract=processor.execution_contract)
    config['original_run_cfg'].update(row['runtime_config']['original_run_cfg'])
    reference = {}
    for index,(name,value) in enumerate(outputs.items()):
        path = tmp_path / f'reference_{index}.npy'; np.save(path,value,allow_pickle=False)
        reference[name] = _descriptor(path,value)
    config['functional_reference']['outputs'] = reference
    config['functional_reference']['tolerances'] = dict(match_iou_threshold=.5,min_match_ratio=.95,min_mean_iou=.98)
    row['runtime_config'] = config
    for key in ('input_binding','completion_contract','measurement_boundary','expected_runtime_artifacts',
                'functional_reference','original_run','task'):
        row[key] = config[key]
    transport = SoftwareCompletionTransport(_module(monkeypatch))
    entry.run_case(row,transport,tmp_path/'output',1,total=12)
    assert [index for _,index in transport.calls] == [1,2,3]
    assert transport.inference_counts == [{'p1':1100,'p2':1100}]*3
    valid = load_valid_repetitions(row['result_path'],row)
    assert len(valid) == 3 and all(value['completed_task_count']==1000 for value in valid)
    assert all(value['last_completed_result']['detection_count']>0 for value in valid)
    summary = write_completion_reports({'rows':[row]},tmp_path/'output/results',tmp_path/'reports')
    assert summary['status_counts']['completed']==1 and summary['valid_repetitions']==3
    assert summary['technical_pairs']==0 and summary['claim_pairs']==0
    before = Path(row['result_path']).read_bytes()
    entry.run_case(row,transport,tmp_path/'output',1,total=12)
    assert len(transport.calls)==3 and Path(row['result_path']).read_bytes()==before
