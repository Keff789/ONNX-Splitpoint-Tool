"""THESIS20 local validation preserves technical source provenance."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from scripts import native_producer_energy_plan as energy
from scripts import native_producer_validate_visualize as validator
from scripts import run_native_producer_energy_from_summary as runner


def _fixture():
    path = Path(__file__).with_name("test_v269f_native_split_final_energy_integrity.py")
    spec = importlib.util.spec_from_file_location("thesis20_energy_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_aggregate_keeps_verified_split_source_without_granting_claim(tmp_path):
    fixture = _fixture()
    row, manifest, _ = fixture._quality_first_row(tmp_path)
    final = fixture._load_script("native_producer_final_report.py")
    row.update(final._verify_split_semantic_artifacts(
        result_path=None, manifest_path=manifest,
        manifest_sha256=fixture._sha256(manifest), sources=(row,),
    ))
    evidence, status = energy._split_quality_energy_evidence(row, row["native_command_contract"])
    assert status == "portable_join_and_semantic_payload_bytes_rehashed"
    row["_split_quality_energy_evidence"] = evidence
    provenance, conflicts = energy._planned_quality_provenance(row, None)
    assert not conflicts
    assert provenance["source_request_sha256"] == row["source_request_sha256"]
    assert provenance["validation_dataset_sha256"] == ""
    plan = {**evidence, **provenance,
            "successful_command_contract_sha256": evidence["native_command_contract_sha256"],
            "native_split_energy_quality_binding": evidence,
            "native_split_energy_quality_binding_sha256": evidence["evidence_sha256"],
            "native_split_quality_consumer_attestation": row["native_split_quality_consumer_attestation"]}
    assert runner._verify_split_energy_quality_binding(plan)[1] == "sealed_quality_binding_verified"
    old_plan = {**plan, "source_request_sha256": ""}
    with pytest.raises(ValueError, match="generic_source_request_sha256_drift"):
        runner._verify_split_energy_quality_binding(old_plan)
    for field, value in (
        ("source_request_sha256", "f" * 64),
        ("successful_command_contract_sha256", "f" * 64),
        ("native_split_quality_consumer_attestation", {}),
    ):
        with pytest.raises(ValueError):
            runner._verify_split_energy_quality_binding({**plan, field: value})
    for validation in (
        {"source_request_sha256": "f" * 64},
        {"source_request_sha256": "invalid"},
        {"source_request_sha256": row["source_request_sha256"], "quality_source_sha256": "f" * 64},
    ):
        with pytest.raises(ValueError, match="validation_source_request_sha256_drift"):
            energy._planned_quality_provenance(row, validation)
    assert energy._planned_quality_provenance(row, {"source_request_sha256": row["source_request_sha256"]}) == (provenance, [])
    invalid_row = copy.deepcopy(row)
    invalid_row["native_split_quality_source_request_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="generic_source_request_sha256_drift"):
        energy._planned_quality_provenance(invalid_row, None)


def test_unverified_runtime_observation_cannot_supply_quality_source():
    row = {"source_request_sha256": "a" * 64,
           "_split_quality_energy_evidence": {"native_split_quality_required": False}}
    provenance, conflicts = energy._planned_quality_provenance(row, None)
    assert not conflicts and not any(provenance.values())


def test_not_supported_is_terminal_without_claim_or_technical_error(tmp_path, monkeypatch, capsys):
    row = {"backend": "hailo8_to_trt", "model": "yolo11l", "case": "b001", "ok": False,
           "status": "not_supported", "quality_applicability": "not_applicable_unsupported_part2_inputs",
           "source_manifest": "/original/split_manifest.json"}
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"rows": [row]}))
    monkeypatch.setattr(validator, "_eval_root_from_summary", lambda _: tmp_path)
    monkeypatch.setattr(validator, "_native_split_authority", lambda _: {})
    monkeypatch.setattr(validator.sys, "argv", ["validator", "--summary", str(summary), "--out-dir", str(tmp_path / "out")])
    assert validator.main() == 0
    report = json.loads((tmp_path / "out/native_producer_validation_summary.json").read_text())
    result = report["rows"][0]
    assert result["status"] == "not_supported"
    assert result["technical_quality_error"] is False
    assert result["claim_ok"] is False
    assert result["source_manifest"] == row["source_manifest"]
    assert "row=1/1" in capsys.readouterr().out


def test_autoprobe_receives_validation_output_dir(tmp_path, monkeypatch):
    output = tmp_path / "validation"
    manifest = tmp_path / "original" / "output.json"
    observed = []
    monkeypatch.setattr(validator, "_find_self_reference_input_manifest", lambda *a, **k: (None, "boundary", "missing"))
    monkeypatch.setattr(validator, "_expected_detection_contract", lambda *a, **k: ("raw_head", "fixture"))
    monkeypatch.setattr(validator, "_is_unbound_yolov7_raw_self_reference_target", lambda *a: False)
    monkeypatch.setattr(validator, "_precomputed_full_self_reference_detection", lambda *a, **k: None)
    monkeypatch.setattr(validator, "_auto_run_yolo_full_self_reference_probe", lambda *a, **k: observed.append(k["out_dir"]))
    validator._full_onnx_self_reference_detection(manifest, tmp_path, out_dir=output)
    assert observed == [output]


def test_campaign_partial_repeats_reuse_without_new_process_starts(tmp_path, monkeypatch):
    from test_v283_campaign_energy_budget import campaign_rig, measure
    import shlex
    rig = campaign_rig(tmp_path, monkeypatch, ['success'])
    first = measure(rig, 'first', run_count=1, exact_run_count=True, invalid_repeat_max_retries=0)
    assert first['ok']
    assert rig.counts() == {'preflight': 1, 'collector': 1, 'workload': 1}
    second = measure(rig, 'second', reuse_campaign_repeats=True)
    assert second['ok'] and second['repeat_contract_complete']
    assert rig.counts() == {'preflight': 3, 'collector': 3, 'workload': 3}

    assert second['runs'][0]['campaign_reused_repeat']['run_directory'] == str(tmp_path / 'first/run_000')
    command_file = tmp_path / 'workload-command.sh'
    command_file.write_text(rig.command)
    command = shlex.join(['python', 'energy_measurement_cli.py', 'measure', '--campaign-budget-file', str(tmp_path / 'campaign.json'),
                          '--campaign-row-id', 'row-a', '--command-file', str(command_file), '--duration', '1',
                          '--preflight-expected-command-contract-sha256', 'a' * 64])
    directory, evidence = runner._selected_energy_repeat_run_dir(
        second['runs'][0], measurement_root=tmp_path / 'second', logical_index=0, campaign_command=command,
    )
    assert directory == tmp_path / 'first/run_000'
    assert evidence['repeat_selection_provenance'] == 'verified_existing_campaign_repeat'
    third = measure(rig, 'third', reuse_campaign_repeats=True)
    assert third['ok'] and third['repeat_contract_complete']
    assert rig.counts() == {'preflight': 3, 'collector': 3, 'workload': 3}
    changed = copy.deepcopy(second['runs'][0]); changed['energy_total_j'] += 1
    with pytest.raises(ValueError, match='source_payload_drift'):
        runner._selected_energy_repeat_run_dir(changed, measurement_root=tmp_path / 'second', logical_index=0, campaign_command=command)
    with pytest.raises(ValueError, match='command contract drift'):
        measure(rig, 'wrong-contract', reuse_campaign_repeats=True, preflight_expected_command_contract_sha256='b' * 64)
    assert rig.counts() == {'preflight': 3, 'collector': 3, 'workload': 3}


def test_normal_energy_supplement_extends_cohort_reuses_rows_and_second_start(tmp_path, monkeypatch):
    import hashlib
    import sys
    import test_v2712_energy_resume as fixture
    original_plan = fixture._plan
    original_row = fixture._plan_row
    monkeypatch.setattr(fixture, '_plan', lambda *a, **kw: {
        **original_plan(*a, **kw), 'energy_profile_requested_runs_per_row': 3,
        'energy_effective_runs_per_row': 3, 'energy_repeat_expansion_applied': False,
        'energy_repeat_expansion_reason': 'none'})
    monkeypatch.setattr(fixture, '_plan_row', lambda *a, **kw: {
        **original_row(*a, **kw), 'measurement_profile_requested_repeats': 3,
        'measurement_effective_repeats': 3, 'measurement_repeat_expansion_applied': False,
        'measurement_repeat_expansion_reason': 'none'})
    namespace = fixture._namespace()
    old_out, summary, previous, old_plan, _ = fixture._resume_fixture(tmp_path)
    (tmp_path / 'campaign.json').write_text(json.dumps({'campaign': True, 'tasks': {'fixture': {'stop_reason': '', 'source_id': 'fixture'}}, 'sources': {'fixture': {'stop_reason': ''}}}))
    campaign_args = ' --campaign-budget-file ' + str(tmp_path / 'campaign.json') + ' --campaign-row-id fixture --campaign-repeats 3 --campaign-max-retries 2 --campaign-max-transport-failures 20'
    for row in old_plan['rows']:
        row['measure_command'] += campaign_args
    for item in previous['rows']:
        item['row']['measure_command'] += campaign_args
    aggregate = previous['rows'][0]['run']['energy_aggregate']
    aggregate_path = Path(previous['rows'][0]['row']['measurement_output_dir']) / 'energy_aggregate.json'
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate))
    previous['rows'][0]['run'].update(energy_aggregate_path=str(aggregate_path), energy_aggregate_sha256=hashlib.sha256(aggregate_path.read_bytes()).hexdigest())
    prior = old_out / 'native_producer_energy_results.json'
    prior.write_text(json.dumps(previous)); prior_bytes = prior.read_bytes()
    extended = copy.deepcopy(old_plan)
    extended['rows'].append(fixture._plan_row(tmp_path/'new-row', backend='hailo8_to_trt', model='resnet50', case='b040', setup_id='setup-a', attempt='old-plan'))
    first_out = tmp_path / 'supplement-1'
    def with_campaign(row, index, plan_dir):
        row['measure_command'] += campaign_args
        return row
    calls = fixture._install_fake_runner(namespace, out=first_out, old_plan=extended, row_transform=with_campaign)
    prepared = []
    namespace['main'].__globals__['_prepare_resume_measurement_cohort'] = lambda rows, **kw: (prepared.append(rows) or {'ok': True, 'measurement_wrapper_allowed': True})

    def argv(out, prior):
        result = ['runner', '--summary', str(summary), '--out-dir', str(out), '--previous-results', str(prior)]
        for field, value in fixture._EXECUTION_CONTEXT.items():
            result += ['--' + field.replace('_', '-'), value]
        return result

    monkeypatch.setattr(sys, 'argv', argv(first_out, prior))
    assert namespace['main']() == 0
    assert len(calls) == 3 and len(prepared[0]) == 3
    first_report = json.loads((first_out / 'native_producer_energy_results.json').read_text())
    assert first_report['supplement_reused_row_count'] == 1
    assert len(first_report['rows']) == 4
    assert first_report['rows'][0]['row'] == previous['rows'][0]['row']
    assert first_report['rows'][0]['run'] == previous['rows'][0]['run']
    assert prior.read_bytes() == prior_bytes
    second_out = tmp_path / 'supplement-2'
    calls = fixture._install_fake_runner(namespace, out=second_out, old_plan=extended, row_transform=with_campaign)
    monkeypatch.setattr(sys, 'argv', argv(second_out, first_out / 'native_producer_energy_results.json'))
    assert namespace['main']() == 0
    assert calls == []
    second = json.loads((second_out / 'native_producer_energy_results.json').read_text())
    assert second['supplement_reused_row_count'] == 4
    assert second['started_measurement_count'] == 0 and second['ok'] is True
    changed = copy.deepcopy(first_report['plan_payload']); changed['rows'][0]['energy_scope'] = 'FS'
    with pytest.raises(ValueError, match='contract drift'):
        namespace['_supplement_previous_results'](previous, changed)
