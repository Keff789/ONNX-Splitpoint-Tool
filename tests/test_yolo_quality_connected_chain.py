"""Three new identities through normal local writers, binders and consumers.

Only accelerator SDK tensors and the external completion process are software
fixtures. No accelerator, compiler, Full model inference or collector starts.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from onnx_splitpoint_tool.generic_completion_plan import bind_completion_rows, IDENTITY_FIELDS
from onnx_splitpoint_tool.generic_completion_reporting import write_completion_reports, load_valid_repetitions
from onnx_splitpoint_tool.workflow import generic_completed_task as completion
from onnx_splitpoint_tool.workflow.yolo_augmentation import _normal_pairs
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_files, build_normalized_results_payload
from onnx_splitpoint_tool.workflow.start_snapshot import build_profile_start_snapshot, public_start_snapshot_metadata
from tests.test_yolo_augmentation_quality_release import make_runner, STARTS, write
from tests.test_yolo_quality_native_completion import native_observation, stored_original_reference
from tests.test_yolo_quality_completion_dispatch import SoftwareCompletionTransport
from tests.test_v2912_generic_split_completion import _module
from tests.yolo_quality_generic_fixture import make_generic_fixture


def _read(path):
    return json.loads(Path(path).read_text())


def _cli(module, argv, monkeypatch):
    with monkeypatch.context() as local:
        local.setattr(sys, 'argv', [module.__file__, *map(str, argv)])
        rc = module.main()
    assert rc == 0, (module.__name__, rc)


def _outputs(model):
    if model == 'yolo11l':
        from tests.test_v27923_decoded_completion import _outputs
        return _outputs(), 'decoded_pre_nms'
    if model == 'yolo26s':
        from tests.test_v272_detection_completion_runtime import _decoded_outputs
        outputs = _decoded_outputs()
        outputs['detections'][..., :4] *= 10
        return outputs, 'decoded_nms'
    outputs = {name: np.full((1, 3, 640//stride, 640//stride, 85), -20., np.float32)
               for name, stride in (('output',8), ('clone_1',16), ('clone_2',32))}
    outputs['output'][0, 0, 40, 40, :4] = 0.
    outputs['output'][0, 0, 40, 40, 4:6] = 10.
    return outputs, 'raw_detection_head'


def _authority_inputs(runner):
    """Create input policy metadata through the existing snapshot builder."""
    profile = copy.deepcopy(runner.profile_payload)
    profile['name'] = 'three_new_sdk_fixture'
    snapshot = build_profile_start_snapshot(profile_request='software-fixture', source_profile=profile,
        resolved_profile=profile, profile_id='software-fixture', profile_path='', profile_source='test')
    metadata = public_start_snapshot_metadata(snapshot)
    version = 'v2.91.2-quality-release-repair'
    write(runner.run_dir/'run_manifest.json', {'schema':'onnx-splitpoint/evaluation-run-manifest',
        'run_id':runner.run_id, 'workflow_version':version, 'tool_version':'2.91.2',
        'profile_start_snapshot':metadata})
    write(runner.run_dir/'profile.yaml', profile)
    write(runner.run_dir/'reports/native_producer_stage.json', {
        'schema':'onnx-splitpoint/native-producer-stage', 'run_id':runner.run_id,
        'workflow_version':version, 'profile_start_snapshot_sha256':metadata['snapshot_sha256'],
        'profile_selection_snapshot_sha256':metadata['requested_selection']['selection_snapshot_sha256'],
        'profile_selection_fingerprint':metadata['selection_fingerprint'],
        'native_split_quality_first':{'required':True, 'applicable':True}})


def _normal_index(observation, source_root, backend):
    """The outer SDK process supplies its actual exit/fresh-file envelope.

    The observations, counts, tensor seals and command identity come unchanged
    from the real Native hotloop and existing artifact writers.
    """
    row = copy.deepcopy(observation['row'])
    path = observation['result_path']
    # Keep the immutable SDK output in place and expose the same file through
    # the normal case/backend/precision discovery layout used by the validator.
    backend_dir = 'hailo_to_trt' if backend == 'hailo8_to_trt' else backend
    alias = path.parent/'native_pipeline'/row['case']/backend_dir/row['precision']
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink():
        assert alias.resolve() == path.parent.resolve()
    else:
        alias.symlink_to(path.parent, target_is_directory=True)
    report_path = alias/path.name
    row.update(report=str(report_path), native_fifo_result=str(report_path), result_ok=row['ok'],
        child_result_fresh=True, returncode=0, timed_out=False,
        native_fifo_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        native_fifo_result_size_bytes=path.stat().st_size)
    names = {'hailo8_to_trt':'native_fifo_eval_runner.json',
             'hailo10h_to_trt':'native_hailo10h_producer_e2e_eval.json',
             'deepx_to_trt':'native_deepx_producer_e2e_eval.json'}
    write(source_root/'analysis_tables'/names[backend], {'rows':[row]})


def test_three_new_quality_native_completion_import_and_energy_plan(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path/'cache'))
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path/'matplotlib'))
    from scripts import native_producer_final_report as final
    from scripts import native_producer_validate_visualize as validator
    from scripts import native_producer_energy_plan as energy
    from scripts import run_evalrun_native_producer_variants as coordinator
    from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy

    runner = make_runner(tmp_path)
    run = runner.run_dir
    observations, generic, results, old_references = [], [], [], []
    try:
        _authority_inputs(runner)
        for selected_row in runner.augmentation_plan['rows']:
            model = selected_row['model_id']
            outputs, endpoint = _outputs(model)
            suite_root = run/'models'/model/'benchmark_set/generated'
            suite_root.mkdir(parents=True)
            # Generated Suite subprocess.run is an outer SDK child leaf;
            # restore the global subprocess module immediately afterwards.
            with monkeypatch.context() as child:
                produced = make_generic_fixture(suite_root, child, selected_row, {
                    'workflow_runner':runner, 'outputs':outputs, 'endpoint_mode':endpoint,
                    'input_hw':(640,640), 'original_wh':(640,480)})
            generic.append(produced)
            assert runner._queue_central_quality_requests(model) == 1
            request = produced['composed_request_path'].resolve()
            result = runner._central_quality_futures[str(request)].result(timeout=180)
            assert result['technical_status'] == 'completed'
            results.append(result)
            print('CONNECTED_QUALITY_COMPLETED', model, flush=True)
        central_path = runner._publish_case_quality(results, complete=True)
        central = _read(central_path)
        # The canonical consumer publication is also used by Finalreport and
        # the normal validator; no positive binding/result map is injected.
        write(run/'quality_management/central_quality_summary.json', central)
        for selected_row, produced in zip(runner.augmentation_plan['rows'], generic):
            model, backend, setup = (selected_row[k] for k in ('model_id','direction','setup_id'))
            # Use the normal generated-suite discovery name. The target is
            # the exact Suite which produced these immutable request bytes.
            (run/'models'/model/'benchmark_set/legacy_suite').symlink_to(
                produced['case'].parent, target_is_directory=True)
            physical = backend.removesuffix('_to_trt')
            cfg = {'native_release_mode':'per_case', 'full_baselines':{'enabled':False},
                'remotes':{physical:{'setup_id':setup,'ssh':'fixture@'+setup}},
                'central_quality_summary':str(central_path)}
            leaf = {'id':model+'_'+setup, 'models':[model], 'backends':[physical],
                'split_backends':[physical], 'case_policy':'case_map_only',
                'case_map':{model:[selected_row['case_id']]}, 'release_kind':'split',
                'precision':selected_row['precision'], 'artifact_namespace':model+'_'+setup,
                'full_baselines':{'enabled':False}}
            prepared, paths, _, _, _ = coordinator._prepare_case_release(run, cfg, leaf, central)
            binding_path = next(iter(next(iter(paths.values())).values()))
            bindings = _read(binding_path)['bindings_by_model_case_backend']
            selected = next(iter(bindings.values()))
            assert selected['source_request_sha256'] == hashlib.sha256(produced['request_bytes']).hexdigest()
            assert selected['producer_binding_sha256'] == produced['binding']['binding_sha256']
            source_root = run/'native_producers'/physical
            if model == 'yolo11l':
                from tests import test_v27923_decoded_completion as h8_fixture
                monkeypatch.setattr(h8_fixture, 'DECLARATION', produced['case'].parent/'output_contracts.json')
            observation = native_observation(selected, source_root/model/'benchmark_set', model_path=produced['full_path'])
            assert all(np.array_equal(outputs, produced['runner_module'].np.asarray(observation['outputs'][name]))
                       for name, outputs in _outputs(model)[0].items())
            _normal_index(observation, source_root, backend)
            observations.append(observation)
            old, policy = stored_original_reference(observation, tmp_path/'stored_original'/model)
            old_references.append(old)
            normalized, sources = normalize_benchmark_files(model_id=model, source_paths=[produced['aggregate_path']])
            write(run/'models'/model/'benchmark_results/normalized_results.json',
                build_normalized_results_payload(model_id=model, results=normalized, sources=sources, run_root=run))
            print('CONNECTED_NATIVE_COMPLETED', model, flush=True)
    finally:
        runner._shutdown_management_services()

    reports = run/'reports'
    summary = reports/'native_producer_combined_summary.json'
    roots = run/'native_producers'
    base_args = ['--root',roots,'--recursive','--out-dir',reports,'--eval-root',run]
    _cli(final, base_args, monkeypatch)
    native = _read(summary)
    assert len(native['rows']) == 3
    assert native['native_split_quality_authority']['valid'], native['native_split_quality_authority']
    assert all(row['ok'] for row in native['rows']), [
        (r['model'],r.get('status'),r.get('failure_reason')) for r in native['rows']]
    original_validation = tmp_path/'stored_original/validation.json'
    write(original_validation, {'rows':old_references})
    validation = reports/'native_validation/native_producer_validation_summary.json'
    policies = [AccuracyGatePolicy.from_mapping(
        json.loads(_read(item['aggregate_path'])[0]['run_cfg']['quality_gate_json']))
        for item in generic]
    assert len({policy.sha256() for policy in policies}) == 1
    assert all(result['task_quality_policy_sha256'] == policies[0].sha256() for result in results)
    # The validator is explicitly forbidden from creating a model session.
    import onnxruntime
    with monkeypatch.context() as readonly:
        readonly.setattr(onnxruntime, 'InferenceSession', lambda *a, **k: (_ for _ in ()).throw(
            AssertionError('Full inference forbidden in stored-output replay')))
        _cli(validator, ['--summary',summary,'--out-dir',validation.parent,'--eval-root',run,
            '--existing-outputs-only','--existing-reference-validation-summary',original_validation,
            '--central-quality-summary',run/'quality_management/central_quality_summary.json',
            '--quality-gate-json',json.dumps(policies[0].as_dict()),'--root',roots], readonly)
    verified = _read(validation)
    failures = [{k:r.get(k) for k in ('model','status','semantic_ok','self_reference_reason',
        'semantic_validation_status','structural_contract_reason','numerical_similarity_reason')}
        for r in verified['rows']]
    assert len(verified['rows']) == 3 and verified['technical_error_count'] == 0, failures
    assert all(row['semantic_ok'] is True for row in verified['rows']), failures
    _cli(final, [*base_args,'--quality-summary',validation], monkeypatch)
    (reports/'native_producer_summary.json').write_bytes(summary.read_bytes())
    pairs = _normal_pairs(run)
    assert len(pairs) == 3, pairs
    assert all(row['generic_source_path'] == row['original_source'] for row in pairs)
    scope = [{**{key:row[key] for key in IDENTITY_FIELDS},'original_source':row['original_source']} for row in pairs]
    bound = bind_completion_rows(scope,pairs,summary,validation,run,
        scope_source=str(tmp_path/'software-fixture-selection'),start_selection=lambda rows:[r['case_key'] for r in rows])
    output = tmp_path/'completion'
    module = _module(monkeypatch)
    transport = SoftwareCompletionTransport(module)
    for index, row in enumerate(bound['rows'],1):
        remote = str(tmp_path/'sdk_process_workspace'/row['setup_id'])
        config, files = completion.runtime_configuration(row,remote,'connected-sdk-fixture')
        row.update(runtime_config=config,staging_files=files,remote_root=remote,
            result_path=str(output/'results'/row['case_output_relative']/'generic_completion_results.json'))
        completion.run_case(row,transport,output,index,total=3)
        assert len(load_valid_repetitions(row['result_path'],row)) == 3
    summary_out = write_completion_reports(bound,output/'results',output/'reports')
    assert summary_out['status_counts']['completed'] == 3 and summary_out['valid_repetitions'] == 9
    assert summary_out['technical_pairs'] == 3, summary_out
    previous = {r['case_key']:Path(r['result_path']).read_bytes() for r in bound['rows']}
    for index,row in enumerate(bound['rows'],1):
        completion.run_case(row,transport,output,index,total=3)
        assert Path(row['result_path']).read_bytes() == previous[row['case_key']]
    assert len(transport.calls) == 9
    _cli(energy, ['--summary',summary,'--validation-summary',validation,'--out-dir',tmp_path/'energy-plan',
        '--duration-s','60','--runs','3','--remote-root','/fixture/unexecuted','--remote-tool-dir','/fixture/tool',
        '--measure-all-runtime-successful','--screening-energy',
        '--hailo8-ssh','fixture@h8','--hailo10-ssh','fixture@h10','--deepx-ssh','fixture@dx'],monkeypatch)
    energy_rows = _read(tmp_path/'energy-plan/native_producer_energy_plan.json')['rows']
    identity = lambda row:(row.get('model_id') or row.get('model'),row.get('case_id') or row.get('case'),row['setup_id'])
    assert {identity(row) for row in energy_rows} == {row[:3] for row in STARTS}
    write(tmp_path/'connected-acceptance.json', {'status':'passed','normal_quality_results':len(results),
        'normal_native_results':len(native['rows']),'normal_validation_results':len(verified['rows']),
        'normal_completion_bound_rows':len(bound['rows']),'completion_summary':summary_out,
        'normal_energy_plan_rows':len(energy_rows),'energy_collectors_started':0,
        'sdk_process_dispatches':len(transport.calls),'reentry_extra_dispatches':0,
        'scope':'three exact new starter identities; software SDK leaves, not hardware measurements'})
