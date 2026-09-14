"""Normal artifact publication/reuse across fresh controllers; SDK is synthetic.

These tests do not represent CUDA, HEF readability or quality acceptance.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool import hailo_negative_evidence as evidence
from onnx_splitpoint_tool.preprocessing_contract import canonical_image_preprocessing_contract

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def publication_case(tmp_path, monkeypatch):
    """Real selected interpreter metadata; only the DFC SDK is a local fixture."""
    previous = ROOT / 'tests' / 'test_v27934_hailo_backend.py'
    spec = importlib.util.spec_from_file_location('v280_sdk_fixture_source', previous)
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    # The long-standing SDK boundary fixture provides input/calibration files
    # and environment limits. Undo its resolver stubs for this real selection.
    original_resolver = backend._resolve_managed_venv_python
    original_metadata = backend._hailo_sdk_version_token_from_managed_venv
    original_token = backend._hailo_sdk_version_token
    case = fixture_module.managed.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(backend, '_resolve_managed_venv_python', original_resolver)
    monkeypatch.setattr(backend, '_hailo_sdk_version_token_from_managed_venv', original_metadata)
    monkeypatch.setattr(backend, '_hailo_sdk_version_token', original_token)
    venv = tmp_path / 'selected_dfc'
    subprocess.run([sys.executable, '-B', '-m', 'venv', '--without-pip', '--system-site-packages', str(venv)],
                   check=True, capture_output=True, timeout=30)
    sites = list((venv / 'lib').glob('python*/site-packages'))
    assert len(sites) == 1
    site = sites[0]
    # The tool test interpreter dependencies remain available to this temporary
    # child. No installed application/SDK/driver file is changed.
    # Include effective dependency search roots as well as site-packages:
    # pip --target / .pth-projected test dependencies otherwise disappear in
    # the nested venv (Python does not recursively process another .pth root).
    (site / 'test_dependencies.pth').write_text('\n'.join(dict.fromkeys(
        str(Path(p).resolve()) for p in sys.path if p and Path(p).is_dir())) + '\n')
    (site / 'hailo_sdk_client.py').write_bytes((tmp_path / 'sdk_boundary' / 'hailo_sdk_client.py').read_bytes())
    meta = site / 'hailo_sdk_client-BOUNDARY_TEST.dist-info'
    meta.mkdir()
    (meta / 'METADATA').write_text('Metadata-Version: 2.1\nName: hailo_sdk_client\nVersion: BOUNDARY_TEST\n')
    monkeypatch.setenv('PYTHONPATH', str(ROOT))
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    # Add Conv defaults and unused initializer cleanup so the actual normal
    # fixup changes graph bytes and source/compiler bindings are distinct.
    inp = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 8, 8])
    out = helper.make_tensor_value_info('outputs', TensorProto.FLOAT, [1, 3, 8, 8])
    weights = helper.make_tensor('weights', TensorProto.FLOAT, [3, 3, 1, 1], np.eye(3, dtype=np.float32).ravel())
    graph = helper.make_graph([helper.make_node('Conv', ['images', 'weights'], ['outputs'])], 'fixture', [inp], [out], [weights])
    onnx.save(helper.make_model(graph), case['onnx_path'])
    case.update(fixup=True, wsl_venv_activate=str(venv / 'bin' / 'activate'))
    return case


def _run_controller(case, out, *, compute='cpu', guard=False, cache_only=False):
    """Fresh real Python process using normal auto dispatch and real metadata."""
    kwargs = {k: str(v) if isinstance(v, Path) else v for k, v in case.items()}
    kwargs.update(outdir=str(out), backend='venv', compute_device=compute, cache_only=cache_only)
    request = out.parent / (out.name + '_request.json')
    result_path = out.parent / (out.name + '_result.json')
    request.write_text(json.dumps(kwargs))
    program = '''
import dataclasses, importlib.abc, json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sdk_import_attempts = []
if sys.argv[4] == 'guard':
    class BlockSDK(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'hailo_sdk_client','tensorflow'}:
                sdk_import_attempts.append(fullname)
                raise AssertionError('SDK import on cache reuse: ' + fullname)
    sys.meta_path.insert(0, BlockSDK())
from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool import hailo_compiler_context as context
if sys.argv[4] == 'guard':
    def forbidden(*a, **kw):
        raise AssertionError('Compiler or GPU probe on cache reuse')
    backend._run_streamed_subprocess = forbidden
    context.resolve_hailo_compiler_context = forbidden
result = backend.hailo_build_hef_auto(**json.loads(Path(sys.argv[2]).read_text()), on_log=lambda stream, line: print(stream + ': ' + line))
payload = dataclasses.asdict(result)
payload['controller_pid'] = os.getpid()
payload['sdk_import_attempts'] = sdk_import_attempts
Path(sys.argv[3]).write_text(json.dumps(payload))
'''
    p = subprocess.run([sys.executable, '-I', '-B', '-c', program, str(ROOT), str(request), str(result_path),
                        'guard' if guard else 'build'], capture_output=True, text=True, timeout=40)
    assert p.returncode == 0, p.stdout + '\n' + p.stderr
    result = json.loads(result_path.read_text())
    result['_console'] = p.stdout + p.stderr
    return result


def _payload_files(root):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file() and p.name in {'compiled.hef', 'hailo_hef_build_receipt.json', 'cache_meta.json'}}


def test_normal_publication_then_two_fresh_gpu_preference_hits_keep_recipe(publication_case, tmp_path):
    case = publication_case
    original = Path(case['onnx_path']).read_bytes()
    first = _run_controller(case, tmp_path / 'first')
    assert first['ok'], first.get('error')
    assert first['calib_info']['cache_hit'] is False
    assert first['details']['compiler_dispatch_count'] == 1
    assert first['last_stage'] == 'publication'
    receipt = backend._load_valid_hailo_receipt(Path(first['hef_path']))
    assert receipt['source_onnx_sha256'] != receipt['compiler_onnx_sha256']
    assert receipt['source_onnx_sha256'] == hashlib.sha256(original).hexdigest()
    assert receipt['compiler_onnx_sha256'] == receipt['cache_payload']['model_sha256']
    assert receipt['publish_artifacts'] is True and receipt['diagnostic_only'] is False
    cache_before = _payload_files(tmp_path / 'production_cache')
    store_before = _payload_files(tmp_path / 'production_store')
    assert cache_before and store_before
    second = _run_controller(case, tmp_path / 'second', compute='gpu', guard=True)
    third = _run_controller(case, tmp_path / 'second', compute='gpu', guard=True)
    assert len({first['controller_pid'], second['controller_pid'], third['controller_pid']}) == 3
    for result in (second, third):
        assert result['ok'], result.get('error')
        assert result['calib_info']['cache_hit'] is True
        assert result['calib_info']['compiler_dispatch_count'] == 0
        hit_receipt = result['calib_info']['build_receipt']
        assert hit_receipt == receipt
        assert Path(result['hef_path']).read_bytes() == Path(first['hef_path']).read_bytes()
    assert _payload_files(tmp_path / 'production_cache') == cache_before
    assert _payload_files(tmp_path / 'production_store') == store_before
    assert Path(case['onnx_path']).read_bytes() == original


def test_successful_structured_result_cannot_become_activation_calibration():
    payload = {'ok': True, 'last_stage': None, 'calib_info': {'source': 'image_preprocess',
               'cache_payload': {'activation_part1_sha256': '', 'calibration_count': 500}},
               'details': {'phase_events': [{'phase': 'compile', 'state': 'completed'},
                                            {'phase': 'publication', 'state': 'completed'}]}}
    line = '__SPLITPOINT_HAILO_RESULT__' + json.dumps(payload)
    assert backend._hailo_stage_from_line(line) is None
    result = backend._hef_result_from_payload(payload, elapsed_default=0, hw_arch='hailo10h',
             net_name='model_full', backend_default='venv', returncode=0, last_stage='activation_calibration')
    assert result.last_stage == 'publication'


@pytest.mark.parametrize('line', [
    '[hailo][calib] activation_from_part1 calibration started',
    'Part1 calibration batch 1 of 500',
    '[hailo][activation] generating calibration inputs',
])
def test_real_activation_progress_still_classified(line):
    assert backend._hailo_stage_from_line(line) == 'activation_calibration'


def test_private_negative_evidence_reason_is_nonpublishing_not_missing_sdk(publication_case, tmp_path):
    case = publication_case
    case.update(force=True, publish_artifacts=False)
    result = _run_controller(case, tmp_path / 'diagnostic')
    assert result['ok'], result.get('error')
    assert 'diagnostic_nonpublishing' in result['_console']
    assert 'managed_compiler_identity_unavailable' not in result['_console']
    assert not (tmp_path / 'production_cache').exists()
    assert not (tmp_path / 'production_store').exists()
    assert not (tmp_path / 'production_evidence').exists()


@pytest.mark.parametrize('metadata_fault', ['missing', 'ambiguous'])
def test_cache_only_missing_managed_identity_never_imports_controller_sdk(publication_case, tmp_path, metadata_fault):
    case = publication_case
    first = _run_controller(case, tmp_path / 'first')
    assert first['ok'], first.get('error')
    for path in (tmp_path / 'selected_dfc').rglob('hailo_sdk_client-BOUNDARY_TEST.dist-info/METADATA'):
        if metadata_fault == 'missing':
            path.unlink()
        else:
            extra = path.parent.parent / 'hailo_sdk_client-OTHER.dist-info'
            extra.mkdir()
            (extra / 'METADATA').write_text('Metadata-Version: 2.1\nName: hailo_sdk_client\nVersion: OTHER\n')
    cache_before = _payload_files(tmp_path / 'production_cache')
    result = _run_controller(case, tmp_path / 'identity_unavailable', compute='gpu', guard=True, cache_only=True)
    assert result['ok'] is False
    assert result['sdk_import_attempts'] == []
    assert result['failure_kind'] == 'cache_miss_blocked'
    assert result['unsupported_reason'] == 'compiler_identity_unavailable'
    assert result['details']['compiler_dispatch_allowed'] is False
    assert _payload_files(tmp_path / 'production_cache') == cache_before


@pytest.fixture
def inventory_case(tmp_path, monkeypatch):
    """Synthetic receipt inventory, not runnable image or accelerator data."""
    from onnx_splitpoint_tool import hailo_model_diagnostics_v27934 as diagnostic
    spec = importlib.util.spec_from_file_location('v280_inventory_fixture', ROOT / 'tests' / 'test_v27934_model_diagnostic.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    args, _ = previous.baseline.__wrapped__(tmp_path, monkeypatch)
    compiler = tmp_path / 'mobilenet_v3_large_hailo_fixed.onnx'
    fixed, _ = backend.fix_onnx_for_hailo(onnx.load(args['source_onnx']), add_conv_defaults=True)
    onnx.save(fixed, compiler)
    contract = canonical_image_preprocessing_contract('classification', (224, 224))
    key, payload = backend._hailo_cache_key(model_path=compiler, activation_part1=None,
        hw_arch='hailo10h', opt_level=1, calib_dir=args['calibration_dir'], calib_count=500,
        calib_batch_size=8, extra_model_script='', start_nodes=None, end_nodes=None,
        preprocessing_contract=contract, effective_calib_count=500, calibration_storage='memmap',
        calibration_memory_cap_bytes=256 * 1024 * 1024, net_name='mobilenet_v3_large_full',
        net_input_shapes={'images': [1, 3, 224, 224]}, disable_rt_metadata_extraction=True)
    receipt = backend._write_hailo_receipt(hef_path=args['cpu_hef'], source_onnx=args['source_onnx'],
        compiler_onnx=compiler, hw_arch='hailo10h', net_name='mobilenet_v3_large_full',
        preprocessing_contract=contract, preprocessing_sha256=payload['preprocessing_contract_sha256'],
        cache_key=key, cache_payload=payload, calibration_identity=payload['calibration_identity'], calibration_count=500)
    cache = tmp_path / 'cache'
    backend._publish_hailo_bundle(source_hef=args['cpu_hef'], destination=cache / key / 'compiled.hef', receipt=receipt)
    venv = tmp_path / 'inventory_dfc'
    (venv / 'bin').mkdir(parents=True)
    (venv / 'bin' / 'python').symlink_to(sys.executable)
    (venv / 'bin' / 'activate').write_text('# Synthetic fixture, never executed\n')
    metadata = venv / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages' / 'hailo_dataflow_compiler-5.3.0.dist-info'
    metadata.mkdir(parents=True)
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: hailo-dataflow-compiler\nVersion: 5.3.0\n')
    request = diagnostic.prepare_request(**{**args, 'compiler_onnx': compiler, 'venv': venv / 'bin' / 'python'})
    request_file = tmp_path / 'inventory_request.json'
    request_file.write_text(json.dumps(request))
    test_home = tmp_path / 'test_home'
    test_home.mkdir()
    env = {**os.environ, 'HOME': str(test_home), 'PYTHONDONTWRITEBYTECODE': '1',
           'ONNX_SPLITPOINT_HAILO_CACHE_ROOT': str(cache), 'ONNX_SPLITPOINT_HAILO_CACHE_ENABLED': '1',
           'ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT': str(tmp_path / 'untouched_store'),
           'ONNX_SPLITPOINT_BUILD_EVIDENCE_ROOT': str(tmp_path / 'untouched_evidence'),
           'ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED': '0'}
    return request_file, request, env, cache


def _run_inventory_launcher(inventory_case, output):
    request_file, _, env, _ = inventory_case
    p = subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'scripts' / 'hailo_reuse_probe_v280.py'),
                        '--request', str(request_file), '--output-parent', str(output)],
                       capture_output=True, text=True, timeout=35, env=env)
    dirs = list(output.glob('v280_hailo_reuse_*'))
    work = next(d for d in dirs if d.is_dir())
    summary = json.loads((work / 'summary.json').read_text())
    return p, summary, work


def test_bounded_target_launcher_two_fresh_normal_cache_hits_and_compact_zip(inventory_case, tmp_path):
    import zipfile
    _, request, _, cache = inventory_case
    before = _payload_files(cache)
    p, summary, work = _run_inventory_launcher(inventory_case, tmp_path / 'reports')
    assert p.returncode == 0, p.stdout + p.stderr + json.dumps(summary)
    assert summary['status'] == 'two_fresh_process_cache_hits_pass'
    assert summary['gpu_publication'] == 'NOT_RUN'
    assert len({r['controller_pid'] for r in summary['runs']}) == 2
    for row in summary['runs']:
        assert row['exact_cpu_receipt_match'] is True
        assert row['source_onnx_sha256'] == request['source_onnx']['sha256']
        assert row['compiler_onnx_sha256'] == request['compiler_onnx']['sha256']
        assert row['compiler_dispatch_count'] == 0
        assert row['sdk_import_attempts'] == [] and row['child_process_attempts'] == []
    assert _payload_files(cache) == before
    assert not (tmp_path / 'untouched_store').exists()
    assert not (tmp_path / 'untouched_evidence').exists()
    with zipfile.ZipFile(work.with_suffix('.zip')) as z:
        assert len(z.namelist()) == 5
        assert all(Path(n).suffix in {'.json', '.log'} for n in z.namelist())
        assert json.loads(z.read(f'{work.name}/summary.json')) == summary


def test_bounded_target_launcher_missing_cache_is_incomplete_without_compiler(inventory_case, tmp_path):
    import shutil
    shutil.rmtree(inventory_case[3])
    p, summary, _ = _run_inventory_launcher(inventory_case, tmp_path / 'reports')
    assert p.returncode == 2, p.stdout + p.stderr
    assert summary['status'] == 'incomplete'
    assert len(summary['runs']) == 1
    row = summary['runs'][0]
    assert row['cache_hit'] is False
    assert row['sdk_import_attempts'] == [] and row['child_process_attempts'] == []
    assert row['model_build'] == 'NOT_RUN'
    assert not inventory_case[3].exists()


def test_bounded_target_launcher_respects_existing_platform_interlock(inventory_case, tmp_path):
    import fcntl
    lock = Path(inventory_case[2]['HOME']) / '.onnx_splitpoint_tool' / 'locks' / 'workflow_platform_interlock.lock'
    lock.parent.mkdir(parents=True)
    with lock.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        p, summary, _ = _run_inventory_launcher(inventory_case, tmp_path / 'reports')
    assert p.returncode == 2
    assert summary['runs'] == []
    assert 'diagnostic_workflow_or_platform_operation_active' in summary['error']
    assert lock.exists()


@pytest.mark.parametrize('field,value', [('family', 'hailo8'), ('model', 'resnet50'), ('publication', 'productive')])
def test_target_reuse_inventory_cannot_expand_diagnostic_scope(inventory_case, tmp_path, field, value):
    path, request, _, cache = inventory_case
    request[field] = value
    path.write_text(json.dumps(request))
    before = _payload_files(cache)
    p, summary, _ = _run_inventory_launcher(inventory_case, tmp_path / 'reports')
    assert p.returncode == 2
    assert len(summary['runs']) == 1
    row = summary['runs'][0]
    assert row['status'] == 'incomplete'
    assert row['sdk_import_attempts'] == [] and row['child_process_attempts'] == []
    assert 'diagnostic_' in row['error']
    assert _payload_files(cache) == before
