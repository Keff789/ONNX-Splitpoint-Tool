"""Selective Quality uses the normal writer, with no Generic timing leaves."""
import argparse
import json
from types import SimpleNamespace

import pytest

from tests.test_v2912_generic_split_completion import _module


def arguments(**changes):
    args = dict(quality_source_only=True, variants='composed',
        native_split_quality_binding='binding.json',
        existing_management_cpu_reference_json='{}',
        validation_images='frozen-dataset', validation_max_images=5000,
        quality_gate_json=json.dumps({'statistics':{'execution_location':'central_management'}}),
        generic_completion_config='', quality_evidence_only=False, build_only=False,
        energy_measurement_only=False, output_dump_only=False, external_output_dump='',
        full_only_quality_identity_json='', warmup=100, runs=1000, phase_runs=5,
        throughput_frames=24, throughput_warmup_frames=6,
        native_trt_build=True, native_trt_no_fallback=False, viz='auto', no_report_plots=False)
    args.update(changes)
    return SimpleNamespace(**args)


@pytest.mark.parametrize('change', [
    {'variants':'full'}, {'variants':'part1,part2,composed'}, {'variants':'composed,composed'},
    {'native_split_quality_binding':''}, {'existing_management_cpu_reference_json':''},
    {'validation_images':''}, {'validation_max_images':0},
    {'quality_gate_json':json.dumps({'statistics':{'execution_location':'local'}})},
    *[{key: True} for key in ('quality_evidence_only','build_only','energy_measurement_only','output_dump_only')],
    {'generic_completion_config':'completion.json'}, {'external_output_dump':'outputs.json'},
    {'full_only_quality_identity_json':'{}'},
])
def test_quality_source_rejects_other_scopes_before_runtime(monkeypatch, change):
    module = _module(monkeypatch)
    with pytest.raises(SystemExit):
        module._configure_quality_source_only(arguments(**change), argparse.ArgumentParser())


def test_quality_source_disables_all_timing_budgets_and_builds(monkeypatch):
    module = _module(monkeypatch)
    args = arguments()
    module._configure_quality_source_only(args, argparse.ArgumentParser())
    assert [args.warmup,args.runs,args.phase_runs,args.throughput_frames,args.throughput_warmup_frames] == [0]*5
    assert args.native_trt_build is False and args.native_trt_no_fallback is True
    ordinary = arguments(quality_source_only=False)
    before = dict(vars(ordinary))
    module._configure_quality_source_only(ordinary, argparse.ArgumentParser())
    assert vars(ordinary) == before


def test_completion_from_quality_source_keeps_real_100_1000_3_path(tmp_path, monkeypatch):
    from tests import test_v2912_generic_split_completion as completion_tests
    old_fixture = completion_tests._fixture
    def source_fixture(path):
        config, logits = old_fixture(path)
        config['original_run_cfg']['quality_source_only'] = True
        return config, logits
    monkeypatch.setattr(completion_tests, '_fixture', source_fixture)
    # This existing test enters actual generated main three times, verifies
    # 3300 tasks in each real pipeline leaf and a fourth no-work reentry.
    completion_tests.test_real_generated_main_uses_generic_pipeline_and_never_cpu_reference(tmp_path, monkeypatch)


def test_generated_composed_quality_writer_has_no_performance_dispatch(tmp_path, monkeypatch):
    from tests.yolo_quality_generic_fixture import make_generic_fixture
    from onnx_splitpoint_tool.native_split_quality import known_native_split_policy
    timing_calls = []
    class TimingForbidden(BaseException):
        pass
    def forbidden(*args, **kwargs):
        timing_calls.append(args)
        raise TimingForbidden('Generic performance dispatch forbidden in Quality source')
    class ObserveRuntime:
        def __getattr__(self, key):
            return getattr(monkeypatch, key)
        def setattr(self, target, name, value, *args, **kwargs):
            monkeypatch.setattr(target, name, value, *args, **kwargs)
            if name == '_create_session':
                monkeypatch.setattr(target, '_bench', forbidden)
                monkeypatch.setattr(target, '_bench_streaming_pipeline', forbidden)
    row = dict(model_id='yolo11l', case_id='b120', setup_id='orin_nx_hailo8_01',
               direction='hailo8_to_trt')
    row['precision'] = known_native_split_policy(model_id=row['model_id'], case_id=row['case_id'],
        setup_id=row['setup_id'], backend=row['direction'])['precision']
    result = make_generic_fixture(tmp_path, ObserveRuntime(), row,
        {'eval_run_id':'quality-source-only', 'quality_source_only':True})
    report = result['raw_report']
    assert timing_calls == []
    assert result['request']['record_count'] == 5000
    assert report['run_cfg']['quality_source_only'] is True
    assert report['quality_source_only'] is True
    assert report['performance_measurement_executed'] is False
    assert report['variant_status']['composed'] == 'ok'
    assert report['throughput'] is None
    for timing in report['timings'].values():
        assert timing['mean_ms'] is None and timing['std_ms'] is None and timing['runs_ms'] is None
    assert report['timings']['composed']['phase_profile'] is None
    assert report['timing_breakdown']['phase_runs_requested'] == 0
    assert result['calls']['full'] == 0
    # The fixture itself loads the actual exported request through the normal
    # strict central consumer and checks unchanged original reference bytes.
