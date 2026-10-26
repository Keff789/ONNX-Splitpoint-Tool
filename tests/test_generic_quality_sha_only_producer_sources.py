"""Historical SHA-only source sizes are derived from real complete sources only."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool import generic_producer_binding as producer
from onnx_splitpoint_tool.workflow import generic_quality_sidecars as sidecars
from test_generic_quality_completion import descriptor
from test_generic_quality_producer_sources import admitted
from test_generic_producer_creation_binding import _normal_creation


@pytest.fixture
def historical(admitted, tmp_path):
    arguments, evidence, config, original, copied = admitted
    primary = tmp_path / 'actual_producer'
    suite = primary / 'models/yolo26s/benchmark_set/legacy_suite'
    suite.parent.mkdir(parents=True)
    old_suite = arguments['suite_dir']
    shutil.move(str(old_suite), suite)
    arguments['suite_dir'] = suite
    copied = suite / 'quality_runtime/b039.json'
    for item in config['runtime_source_files']:
        item.pop('size_bytes')
    for item in evidence['runtime_source_files']:
        item['path'] = str(suite / Path(item['path']).relative_to(old_suite))
        item.pop('size_bytes')

    def publish():
        encoded = json.dumps(config)
        original.write_text(encoded);copied.write_text(encoded)
        for row in (arguments['profile']['run_profiles'][0], arguments['benchmark_plan']['runs'][0]):
            row['generic_quality_completion_sources']['b039'] = descriptor(original)
            row['generic_quality_completion_config_sha256']['b039'] = descriptor(original)['sha256']
        creation = _normal_creation(primary, setup=config['identity']['setup_id'],
                                    profile_extra=arguments['profile'])
        arguments['profile'] = creation.profile
        (suite / 'benchmark_plan.json').write_text(json.dumps(arguments['benchmark_plan']))
        arguments['producer_run_binding'] = producer.build_producer_run_binding(
            primary, session_id=creation.session)
        evidence['config_file'] = descriptor(copied)
    publish()
    return SimpleNamespace(arguments=arguments, evidence=evidence, config=config,
                           original=original, copied=copied, publish=publish)


def test_37_actual_byte_sizes_are_separate_and_original_config_evidence_unchanged(historical):
    state=historical
    before=(state.original.read_bytes(),state.copied.read_bytes(),deepcopy(state.evidence))
    authority=sidecars.admit_generic_quality_producer_sources(**state.arguments)
    assert authority.validate_evidence_sources(state.evidence) is None
    proof=authority.runtime_source_size_derivation()
    assert proof['schema']=='onnx-splitpoint/historical-sha-only-runtime-source-size-derivation-v1'
    assert proof['original_config_request_and_evidence_rewritten'] is False
    assert proof['original_runtime_source_files']==state.config['runtime_source_files']
    assert len(proof['actual_runtime_source_files'])==37
    for item in proof['actual_runtime_source_files']:
        actual=descriptor(state.arguments['suite_dir']/item['path'])
        assert item['sha256']==actual['sha256'] and item['size_bytes']==actual['size_bytes']
    assert all('size_bytes' not in item for item in proof['original_runtime_source_files'])
    assert before==(state.original.read_bytes(),state.copied.read_bytes(),state.evidence)
    proof['producer_run_binding']['session_id']='f'*32
    assert authority.runtime_source_size_derivation()['producer_run_binding']==state.arguments['producer_run_binding']


@pytest.mark.parametrize('fault',['missing_creation','wrong_session','wrong_run','wrong_profile','wrong_plan'])
def test_missing_or_wrong_actual_creation_and_scope_reject(historical,fault):
    args=deepcopy(historical.arguments)
    if fault=='missing_creation':args.pop('producer_run_binding')
    elif fault=='wrong_session':args['producer_run_binding']['session_id']='b'*32
    elif fault=='wrong_run':args['producer_run_binding']['run_dir']=str(Path(args['producer_run_binding']['run_dir']).parent)
    elif fault=='wrong_profile':args['profile']['name']='different'
    else:args['benchmark_plan']['model_id']='other'
    with pytest.raises((ValueError,OSError,KeyError)):
        sidecars.admit_generic_quality_producer_sources(**args)


@pytest.mark.parametrize('fault',['mixed_size','wrong_sha','missing_member','extra_actual_member','symlink'])
def test_derived_sizes_never_admit_mixed_or_wrong_bytes_or_namespace(historical,fault):
    state=historical
    item=state.config['runtime_source_files'][0]
    path=state.arguments['suite_dir']/item['path']
    if fault=='mixed_size':
        item['size_bytes']=path.stat().st_size;state.publish()
    elif fault=='wrong_sha':
        item['sha256']='0'*64;state.publish()
    elif fault=='missing_member':path.unlink()
    elif fault=='extra_actual_member':
        (state.arguments['suite_dir']/'splitpoint_runners/foreign.py').write_text('# foreign')
    else:
        other=path.with_suffix('.retained');path.rename(other);path.symlink_to(other)
    with pytest.raises((ValueError,OSError)):
        sidecars.admit_generic_quality_producer_sources(**state.arguments)


@pytest.mark.parametrize('fault',['actual_bytes','actual_creation','observed_sha','observed_size'])
def test_derived_authority_rechecks_all_bytes_creation_and_observed_evidence(historical,fault):
    state=historical
    authority=sidecars.admit_generic_quality_producer_sources(**state.arguments)
    evidence=deepcopy(state.evidence)
    if fault=='actual_bytes':
        path=state.arguments['suite_dir']/state.config['runtime_source_files'][0]['path']
        path.write_bytes(path.read_bytes()+b'# changed')
    elif fault=='actual_creation':
        path=Path(state.arguments['producer_run_binding']['run_dir'])/'run_manifest.json'
        value=json.loads(path.read_text());value['current_session_id']='b'*32;path.write_text(json.dumps(value))
    elif fault=='observed_sha':evidence['runtime_source_files'][0]['sha256']='0'*64
    else:evidence['runtime_source_files'][0]['size_bytes']=1
    with pytest.raises((ValueError,OSError)):
        authority.validate_evidence_sources(evidence)


def test_optional_observed_size_is_checked_against_real_derived_size(historical):
    state=historical
    authority=sidecars.admit_generic_quality_producer_sources(**state.arguments)
    evidence=deepcopy(state.evidence)
    for item in evidence['runtime_source_files']:
        item['size_bytes']=Path(item['path']).stat().st_size
    assert authority.validate_evidence_sources(evidence) is None
