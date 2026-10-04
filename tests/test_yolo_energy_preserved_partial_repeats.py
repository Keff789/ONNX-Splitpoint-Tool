"""Read-only partial energy projection from the normal stored-repeat verifier."""
import copy
import json
from pathlib import Path
import shlex

import pytest

from onnx_splitpoint_tool.native_energy_reporting import (
    collect_native_energy, _preserved_campaign_repeat_records, _energy_repeat_records,
)
from test_yolo_authorized_energy_budget import case, write
from test_yolo_h8_energy_budget_limits import h8, install
from test_yolo_h8_energy_retry_delivery import rig_for
from test_v283_campaign_energy_budget import measure
from test_yolo_energy_partial_reporting import publish


@pytest.fixture
def partial(h8, tmp_path, monkeypatch):
    rig = rig_for(tmp_path, monkeypatch, ['success'])
    original = measure(rig, 'original-three', row='seed', campaign_max_retries=2,
        campaign_max_transport_failures=20, run_count=3, exact_run_count=True,
        invalid_repeat_max_retries=0)
    assert original['ok']
    chain = copy.deepcopy(original['task_budget']['chains'][0])
    h8.root['tasks'][h8.row]['chains'][0] = chain
    write(h8.path, h8.root)
    install(h8)
    command = tmp_path / 'workload.sh'; command.write_text(rig.command + '\n')
    row = {'backend':'hailo8_to_trt','model':'yolo26s','case':'b038','setup_id':'orin_nx_hailo8_01',
        'precision':'uint8_dequant_fp16','contract':'uint8_dequant',
        'command_file':str(command),'successful_command_contract_sha256':'a'*64,'duration_s':1,
        'measurement_effective_repeats':3,'measurement_requested_repeats':3,'window_method_ab':{},
        'measure_command':shlex.join(['energy_measurement_cli.py','measure',
            '--campaign-budget-file',str(h8.path),'--campaign-row-id',h8.row,
            '--command-file',str(command),'--preflight-expected-command-contract-sha256','a'*64,
            '--duration','1']), 'screening_only':True, 'claim_eligible':False}
    item = {'row':row,'ok':False,'run':{'rc':1,'energy_aggregate_complete':False,
        'energy_aggregate_verified':False,'energy_task_budget':{'checkpoint':str(h8.path),
            'row_id':h8.row,'source_id':h8.source},'energy_collector_attempt_count':2,
        'energy_failed_collector_attempt_count':1,'execution_status':'INCOMPLETE',
        'energy_not_started_reason':'task_execution_interrupted',
        'measurement_start_observation':{'collector_started_repeat_count':2}}}
    return h8, rig, chain, item


def test_verified_original_partial_remains_visible_without_aggregate_pass(partial, tmp_path):
    h8, rig, chain, item = partial
    source = Path(chain['run_directory'])/'energy_summary.json'
    original = source.read_bytes(); budget = h8.path.read_bytes(); item_before=copy.deepcopy(item)
    assert len(_preserved_campaign_repeat_records(item)) == 1
    publish(tmp_path/'projection',[item])
    observed = collect_native_energy(tmp_path/'projection',include_partial=True)[0]
    assert observed['energy_repeat_valid_n'] == 1 and observed['energy_repeat_requested_n'] == 3
    assert observed['energy_repeat_status'] == 'incomplete_valid_repeats'
    assert observed['ok'] is False and observed['claim_eligible'] is False
    assert observed['energy_physical_collector_attempt_count'] == 2
    assert observed['energy_not_started_reason'] == 'task_execution_interrupted'
    assert source.read_bytes() == original and h8.path.read_bytes() == budget and item == item_before


@pytest.mark.parametrize('mutation',['foreign_row','changed_command','changed_contract','changed_duration',
    'source_unverified','duplicate_valid'])
def test_invalid_or_ambiguous_partial_is_not_promoted(partial, mutation):
    h8, rig, chain, item = partial
    if mutation == 'foreign_row': item['row']['case']='b999'
    elif mutation == 'changed_command': Path(item['row']['command_file']).write_text('different workload')
    elif mutation == 'changed_contract': item['row']['successful_command_contract_sha256']='b'*64
    elif mutation == 'changed_duration': item['row']['duration_s']=60
    elif mutation == 'source_unverified':
        p=Path(chain['run_directory'])/'collector_stdout.log'; p.write_bytes(p.read_bytes().replace(b'verified=true',b'verified=false'))
    else:
        value=json.loads(h8.path.read_text());value['tasks'][h8.row]['chains'].append(copy.deepcopy(chain));write(h8.path,value)
    assert _preserved_campaign_repeat_records(item) == []


def test_complete_aggregate_and_visible_repeat_never_double_count(partial):
    h8, rig, chain, item = partial
    entry=json.loads((Path(chain['run_directory'])/'energy_summary.json').read_text())
    item['run']['stdout']=json.dumps(entry)
    assert len(_energy_repeat_records(item)) == 1
    item['run']['energy_aggregate']={'repeat_contract_complete':True,'run_count':3,
        'valid_postprocessed_runs':3}
    assert _preserved_campaign_repeat_records(item) == []
