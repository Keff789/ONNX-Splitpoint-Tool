"""Existing setup scheduler -> real N5000 queue/publication -> Native release.

Only the outer remote batch is replaced. No physical inference or Native
performance is invoked, and no positive producer result is fabricated.
"""
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import threading

import pytest

from onnx_splitpoint_tool.workflow.execution_binding import PhysicalSetupDispatchManager
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.workflow.yolo_augmentation import YOLOAugmentationRunner
from tests.test_yolo_augmentation_quality_release import make_runner, emit_request, STARTS, write
from tests.test_v269f_variant_native_split_quality_first import _evalrun


@pytest.fixture
def first_runner(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    runner.__class__ = YOLOAugmentationRunner
    runner.augmentation_resume = False
    runner.augmentation_execution_output = tmp_path / 'attempt'
    runner._augmentation_first_quality_ready = Future()
    runner.augmentation_plan['start_case_keys'] = ['|'.join(row[:3]) for row in STARTS]
    template = _evalrun(tmp_path / 'authority')
    manifest = json.loads((template / 'run_manifest.json').read_text())
    manifest['run_id'] = runner.run_id
    write(runner.run_dir / 'run_manifest.json', manifest)
    (runner.run_dir / 'profile.yaml').write_bytes((template / 'profile.yaml').read_bytes())
    stage = json.loads((template / 'reports/native_producer_stage.json').read_text())
    write(runner.run_dir / 'reports/native_producer_stage.json', dict(stage, run_id=runner.run_id))
    backends = dict(hailo8_to_trt='hailo8', hailo10h_to_trt='hailo10h', deepx_to_trt='deepx')
    cfg = dict(native_release_mode='per_case', backends=list(backends.values()),
        split_backends=list(backends.values()), full_baselines={'enabled':False},
        remotes={backends[b]:{'setup_id':s,'ssh':'fixture@'+s} for _,_,s,b in STARTS},
        variants=[])
    for row in runner.augmentation_plan['rows']:
        m,c,s,b = [row[k] for k in ('model_id','case_id','setup_id','direction')]
        write(runner.run_dir/'models'/m/'benchmark_set'/c/'split_manifest.json', {})
        write(runner.run_dir/'models'/m/'benchmark_set/benchmark_set.json', {'benchmark_task':'detection'})
        cfg['variants'].append(dict(id=m, models=[m], case_map={m:[c]}, case_policy='case_map_only',
            backends=[backends[b]], split_backends=[backends[b]], precision=row['precision']))
    # Supply the ordinary resolved configuration fixture; all release planners,
    # request readers, validators and producer-binding writers remain real.
    variants=cfg.pop('variants')
    monkeypatch.setattr(runner, '_native_producer_config', lambda: dict(cfg))
    monkeypatch.setattr(runner, '_normal_native_release_variants', lambda actual: variants)
    monkeypatch.setattr(runner, '_materialize_variant_native_remotes', lambda actual: cfg['remotes'])
    yield runner
    runner._shutdown_management_services()


def test_real_request_queue_and_native_release_open_all_other_setup_heads(first_runner, tmp_path, monkeypatch):
    runner = first_runner
    starts, requests = [], []
    lock = threading.Lock()
    models = [row[0] for row in STARTS]

    def outer_steps(self, model, row, *, dispatch_manager, on_setup_complete, **kwargs):
        def dispatch():
            def remote(group):
                setup = group['target_id']
                first = model == STARTS[0][0] and setup == STARTS[0][2]
                with lock:
                    starts.append((model, setup, self._augmentation_first_quality_ready.done()))
                if first:
                    requests.append(emit_request(self, self.augmentation_plan['rows'][0], tmp_path/'producer')[0])
                else:
                    assert self._augmentation_first_quality_ready.result()['prepared_native_leaf']
                    assert (self.augmentation_execution_output/'first-quality-native-binding.json').is_file()
                on_setup_complete(model, setup, {})
                return {"target_id":setup,"statuses":["ok"],"records":[{"status":"ok"}]}
            futures = [dispatch_manager.submit_group(model, s, remote, {'target_id':s}) for _,_,s,_ in STARTS]
            dispatch_manager.finish_registration(model)
            return [f.result(timeout=120) for f in futures]
        value = yield dispatch
        return value

    monkeypatch.setattr(EvaluationWorkflowRunner, '_stage_run_benchmarks_steps', outer_steps)
    manager = PhysicalSetupDispatchManager(max_workers=3, model_ids=models)
    try:
        generators = [runner._stage_run_benchmarks_steps(model,{},dispatch_manager=manager) for model in models]
        dispatches = [next(g) for g in generators]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(d) for d in dispatches]
            outputs = [future.result(timeout=120) for future in futures]
        for generator, output in zip(generators, outputs):
            with pytest.raises(StopIteration): generator.send(output)
    finally:
        manager.shutdown(wait=True, cancel_futures=True)
    assert starts[0] == (STARTS[0][0], STARTS[0][2], False)
    assert len(starts) == 9 and all(done for *_,done in starts[1:])
    proof = json.loads((runner.augmentation_execution_output/'first-quality-native-binding.json').read_text())
    bound_path = next(iter(next(iter(proof['native_split_quality_bindings'].values())).values()))
    binding_set = json.loads(__import__('pathlib').Path(bound_path).read_text())
    selected = next(iter(binding_set['bindings_by_model_case_backend'].values()))
    assert selected['source_request_sha256'] == hashlib.sha256(requests[0].read_bytes()).hexdigest()
    assert not list((runner.run_dir/'reports/native_producer_variants').glob('*/performance_stage_result.json'))
    assert not (runner.run_dir/'native_producers').exists()


@pytest.mark.parametrize('failure', ['missing_request','bad_candidate','missing_group','partial_with_valid_request'])
def test_first_failure_keeps_every_other_group_unstarted(first_runner, tmp_path, monkeypatch, failure):
    runner = first_runner
    called = []
    def outer_steps(self, model, row, *, dispatch_manager, on_setup_complete, **kwargs):
        def dispatch():
            def remote(group):
                setup=group['target_id']; called.append(setup)
                if failure in {'bad_candidate','partial_with_valid_request'}:
                    request,_=emit_request(self,self.augmentation_plan['rows'][0],tmp_path/'producer')
                    if failure=='bad_candidate':
                        payload=json.loads(request.read_text()); (request.parent/payload['candidate']['path']).write_text('changed')
                on_setup_complete(model,setup,{})
                return {"target_id":setup,"statuses":["partial" if failure=="partial_with_valid_request" else "ok"],"records":[]}
            setups=[s for _,_,s,_ in STARTS if failure!='missing_group' or s!=STARTS[0][2]]
            futures=[dispatch_manager.submit_group(model,s,remote,{'target_id':s}) for s in setups]
            dispatch_manager.finish_registration(model)
            errors=[]
            for future in futures:
                try: future.result(timeout=120)
                except Exception as exc: errors.append(exc)
            raise errors[0]
        yield dispatch
    monkeypatch.setattr(EvaluationWorkflowRunner,'_stage_run_benchmarks_steps',outer_steps)
    generator=runner._stage_run_benchmarks_steps('yolo11l',{})
    try:
        dispatch=next(generator)
        with pytest.raises(Exception): dispatch()
    finally:
        generator.close()
    assert called == ([] if failure=='missing_group' else [STARTS[0][2]])
    assert runner._augmentation_first_quality_ready.exception() is not None
    assert not (runner.augmentation_execution_output/'first-quality-native-binding.json').exists()


def test_preserved_quality_uses_normal_request_and_release_without_remote(first_runner,tmp_path):
    runner=first_runner
    request,_=emit_request(runner,runner.augmentation_plan['rows'][0],tmp_path/'producer')
    assert runner._queue_central_quality_requests('yolo11l')==1
    result=runner._central_quality_futures[str(request.resolve())].result(timeout=120)
    assert result['technical_status']=='completed'
    runner._preserved_central_quality_results=[result]
    runner._preserved_central_quality_requests={str(request.resolve()):{
        'source_request_sha256':hashlib.sha256(request.read_bytes()).hexdigest()}}
    runner._central_quality_futures.clear()
    runner.options.resume_missing=True
    assert runner._accept_first_quality()
    assert not runner._central_quality_futures
    assert runner._augmentation_first_quality_ready.done()
    assert not (runner.run_dir/'native_producers').exists()
