"""Actual same-feed binding across raw C++ and completed Python endpoints."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = Path(os.environ.get('V281_IMAGE_RUNNER', ROOT / 'scripts/native_hailo_trt_fifo_from_benchmarkset.py'))
SPEC = importlib.util.spec_from_file_location('v281_shared_input_runner', PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def prepared(shape):
    return SimpleNamespace(input_names=['images'], handle=SimpleNamespace(runtime_input_shapes={'images':shape}))


@pytest.mark.parametrize('layout,shape', [('HWC',(8,11,3)), ('NHWC',(1,8,11,3)), ('CHW',(3,8,11)), ('NCHW',(1,3,8,11))])
@pytest.mark.parametrize('quantized', [True,False])
def test_actual_shared_rgb_bytes_preserve_padding_channels_layout_and_scaling(tmp_path, monkeypatch, layout, shape, quantized):
    rgb = np.full((8,11,3),114,dtype=np.uint8)
    rgb[2:7,1:10] = np.arange(5*9*3,dtype=np.uint8).reshape(5,9,3)
    source = tmp_path/'cpp-prepared.bin'; source.write_bytes(rgb.tobytes())
    # A second image decode/resize would reproduce the original defect.
    monkeypatch.setattr(runner, 'Image', None)
    result = runner._hailo8_python_input(prepared(shape), tmp_path/'unavailable.jpg', quantized=quantized,
        preprocess_mode='letterbox', letterbox_pad_value=114, prepared_input_rgb=source,
        expected_prepared_input_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())['images']
    expected = rgb if quantized else rgb.astype(np.float32)/np.float32(255)
    if layout in {'CHW','NCHW'}: expected=expected.transpose(2,0,1)
    if layout in {'NHWC','NCHW'}: expected=expected[None]
    assert result.shape == shape
    assert result.dtype == ('uint8' if quantized else 'float32')
    assert result.flags.c_contiguous
    assert result.tobytes() == np.ascontiguousarray(expected).tobytes()
    for _ in range(3):
        reused=runner._hailo8_reuse_python_input(prepared(shape), {'images':result})
        assert reused['images'] is result


@pytest.mark.parametrize('mutation', ['hash','short','large','missing','channels'])
def test_shared_rgb_corruption_and_unsupported_input_fail_before_runtime(tmp_path, mutation):
    source=tmp_path/'feed.bin'; source.write_bytes(bytes(range(8*10*3)))
    sha=hashlib.sha256(source.read_bytes()).hexdigest(); shape=(8,10,3)
    if mutation=='hash': source.write_bytes(bytes(reversed(source.read_bytes())))
    elif mutation=='short': source.write_bytes(source.read_bytes()[:-1])
    elif mutation=='large': source.write_bytes(source.read_bytes()+b'!')
    elif mutation=='missing': source.unlink()
    elif mutation=='channels': shape=(8,10,1)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        runner._hailo8_python_input(prepared(shape), tmp_path/'no-image.jpg', quantized=True,
            preprocess_mode='resize',letterbox_pad_value=114,prepared_input_rgb=source,expected_prepared_input_sha256=sha)


def raw_contract(tmp_path):
    image=tmp_path/'image.jpg'; image.write_bytes(b'original compressed bytes')
    source=tmp_path/'input.bin'; source.write_bytes(bytes(range(24)))
    boundary=tmp_path/'boundary.bin'; boundary.write_bytes(b'hef output')
    manifest=tmp_path/'boundary.json'; manifest.write_text(json.dumps({'file':'boundary.bin','input_dump':'input.bin'}))
    options={'task':'detection','preprocess_mode_requested':'auto','preprocess_mode_effective':'letterbox',
        'letterbox_pad_value_requested':114,'letterbox_pad_value_effective':114,'letterbox_pad_value':114}
    contract={'complete':True,'input_image_sha256':runner._sha256_file(image),'runtime_options':options,
        'prepared_input_contract':{**options,'pad_value_effective':114,'dtype':'uint8','layout':'HWC','shape':[2,4,3],
            'source_image_sha256':runner._sha256_file(image)},
        'artifacts':{'prepared_input':{'path':str(source),'sha256':runner._sha256_file(source)}}}
    contract['contract_sha256']=runner._stable_json_sha256(contract)
    return {'native_command_contract':contract,'native_fifo_boundary_manifest':str(manifest)}, SimpleNamespace(image=str(image),task='detection',preprocess_mode='auto',letterbox_pad_value=114)


def test_parent_binds_sealed_actual_consumed_feed(tmp_path):
    payload,args=raw_contract(tmp_path)
    result=runner._shared_rgb_from_raw_endpoint(payload,args=args)
    assert result['path']==str(tmp_path/'input.bin')
    assert result['sha256']==hashlib.sha256(bytes(range(24))).hexdigest()


@pytest.mark.parametrize('mutation', ['seal','artifact','image','pad','consumed','layout','size'])
def test_parent_does_not_accept_different_preparation_or_stale_feed(tmp_path,mutation):
    payload,args=raw_contract(tmp_path)
    contract=payload['native_command_contract']
    if mutation=='seal': contract['complete']=False
    elif mutation=='artifact': (tmp_path/'input.bin').write_bytes(bytes(reversed(range(24))))
    elif mutation=='image': (tmp_path/'image.jpg').write_bytes(b'changed image')
    elif mutation=='pad': args.letterbox_pad_value=0
    elif mutation=='consumed':
        (tmp_path/'different.bin').write_bytes(b'different actual feed')
        (tmp_path/'boundary.json').write_text(json.dumps({'file':'boundary.bin','input_dump':'different.bin'}))
    elif mutation in {'layout','size'}:
        contract.pop('contract_sha256')
        if mutation=='layout':contract['prepared_input_contract']['layout']='CHW'
        else:contract['prepared_input_contract']['shape']=[3,4,3]
        contract['contract_sha256']=runner._stable_json_sha256(contract)
    with pytest.raises(RuntimeError,match='shared_prepared_'):
        runner._shared_rgb_from_raw_endpoint(payload,args=args)


def test_runtime_repetition_rejects_changed_input_metadata():
    inputs={'images':np.zeros((8,11,3),dtype=np.uint8)}
    with pytest.raises(RuntimeError,match='runtime_input_drift'):
        runner._hailo8_reuse_python_input(prepared((8,12,3)),inputs)


def test_parent_keeps_raw_evidence_when_shared_binding_fails(tmp_path,monkeypatch):
    payload, partial_args=raw_contract(tmp_path)
    args=SimpleNamespace(**vars(partial_args),result_json='')
    calls=[]
    def command(**kw):
        calls.append(kw['endpoint'])
        return [str(kw['result_json'])]
    def raw_run(command,check=False):
        result=Path(command[0]);result.parent.mkdir(parents=True,exist_ok=True)
        result.write_text(json.dumps(payload))
        (tmp_path/'input.bin').write_bytes(b'corrupt after raw completion')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner,'_dual_child_command',command)
    monkeypatch.setattr(runner.subprocess,'run',raw_run)
    target=tmp_path/'native_fifo_results.json'
    target.write_text('{"ok":true,"stale":"prior run"}')
    assert runner._run_detection_dual_endpoint(bs=tmp_path,case='b067',work=tmp_path,args=args)==8
    result=json.loads(target.read_text())
    assert result['ok'] is False
    assert result['failure_reason']=='shared_prepared_input_binding_failed'
    assert result['completed_task_execution']=='not_started_invalid_shared_input'
    assert result['phases'][0]['returncode']==0
    assert Path(result['endpoint_result_paths']['raw_model_outputs']).is_file()
    assert calls==['raw_model_outputs']
    assert 'stale' not in result


def test_raw_repetitions_pin_first_consumed_bytes(tmp_path):
    source=tmp_path/'prepared.bin';source.write_bytes(b'first actual raw feed')
    pinned=runner._pin_raw_prepared_feed(source)
    assert runner._pin_raw_prepared_feed(source,pinned)==pinned
    source.write_bytes(b'other equally sized!')
    with pytest.raises(RuntimeError,match='repetition_feed_changed'):
        runner._pin_raw_prepared_feed(source,pinned)


def test_actual_raw_launcher_returns_failure_when_later_process_changes_feed(tmp_path,monkeypatch):
    """Exercise the real repetition launcher, including its exit code and JSON."""
    bs=tmp_path/'benchmark_set';bs.mkdir()
    (bs/'benchmark_set.json').write_text('{"task":"detection","model_id":"yolo11l"}')
    image=tmp_path/'image.jpg';image.write_bytes(b'fixed image')
    hef=tmp_path/'part1.hef';hef.write_bytes(b'existing HEF')
    engine=tmp_path/'part2.engine';engine.write_bytes(b'existing engine')
    work=tmp_path/'work';(work/'build').mkdir(parents=True)
    (work/'build/split_native_hailo_trt_fifo').write_bytes(b'existing native executable')
    target=work/'native_fifo_results.json'
    target.write_text('{"ok":true,"stale":"earlier run"}')
    monkeypatch.setattr(runner,'_find_hef',lambda *a:hef)
    monkeypatch.setattr(runner,'_find_engine',lambda *a:engine)
    monkeypatch.setattr(runner,'_engine_boundary_contract',lambda *a:{})
    monkeypatch.setattr(runner,'_verify_replay_expectations',lambda *a,**kw:{})
    monkeypatch.setattr(runner,'_native_command_contract',lambda **kw:{'complete':True})
    commands=[]
    def child(command,cwd=None):
        commands.append(command)
        feed=Path(command[command.index('--prepared-input-out')+1]);feed.parent.mkdir(parents=True,exist_ok=True)
        if len(commands)==1:
            assert '--prepared-input-rgb' not in command
            feed.write_bytes(b'first prepared RGB')
        else:
            assert Path(command[command.index('--prepared-input-rgb')+1])==feed
            assert feed.read_bytes()==b'first prepared RGB'
            feed.write_bytes(b'changed prepared RGB')
        result=Path(command[command.index('--out')+1]);result.parent.mkdir(parents=True,exist_ok=True)
        result.write_text('{"ok":true,"frames":2}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess,'run',child)
    monkeypatch.setattr(runner.sys,'argv',[str(PATH),'--benchmark-set',str(bs),'--case','b067','--image',str(image),
        '--work-dir',str(work),'--no-build','--detection-endpoints','raw_model_outputs','--repetitions','3'])
    assert runner.main()==8
    result=json.loads(target.read_text())
    assert len(commands)==2
    assert result['ok'] is False and result['returncode']==8
    assert result['repetitions_completed']==1
    assert result['error']=='shared_prepared_raw_repetition_feed_changed'
    assert 'stale' not in result
