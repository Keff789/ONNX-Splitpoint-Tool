import copy
import csv
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_rate_endpoints import report_rate_fields, format_rate_endpoints
from onnx_splitpoint_tool.native_performance_reporting import collect_native_performance_matrix
from onnx_splitpoint_tool.workflow.runner import _native_concise_summary_v60w
from onnx_splitpoint_tool.workflow.scientific_reporting import _write_reports

FIXTURES = json.loads((Path(__file__).parent/'fixtures/v283_r9a_nachabnahme/historical_full_rows.json').read_text())


@pytest.mark.parametrize('fixture', FIXTURES, ids=lambda f:f['row']['backend'])
def test_original_full_collection_retains_diagnostic_across_reporting(fixture, tmp_path):
    row = copy.deepcopy(fixture['row'])
    root = tmp_path/'native_producers/family'
    table = root/'analysis_tables/native_full_baseline_eval.json'
    table.parent.mkdir(parents=True)
    table.write_text(json.dumps({'rows':[row]}))
    row['source_root'] = str(root)
    row['report'] = '/remote/last-repetition.json'
    result = report_rate_fields(row)
    assert result['rate_endpoint_source'] == str(table)
    assert result['completed_task_fps'] is None
    assert result['historical_fps'] == row['fps_median']
    assert result['historical_fps_ci95_low'] == row['fps_ci95_low']
    assert result['historical_fps_ci95_high'] == row['fps_ci95_high']
    assert result['historical_performance_endpoint'] == row['comparison_endpoint_stratum']
    row.update(result)
    reports = tmp_path/'reports'; reports.mkdir()
    (reports/'native_producer_combined_summary.json').write_text(json.dumps({'rows':[row]}))
    _, concise = _native_concise_summary_v60w(reports)
    matrix = collect_native_performance_matrix(tmp_path)
    normalized, = matrix['observations']
    for projected in [row, concise[0], normalized]:
        assert projected['historical_fps'] == result['historical_fps']
        assert projected['historical_fps_ci95_low'] == result['historical_fps_ci95_low']
        assert projected['historical_performance_endpoint'] == result['historical_performance_endpoint']
        assert projected['completed_task_fps'] is None
        assert 'Historische Diagnose' in format_rate_endpoints(projected)
    _write_reports(reports/'scientific', {'created_at':'2026-09-16','profile_id':'offline',
        'rows':[],'summary':{},'native_performance_matrix':matrix})
    table = reports/'scientific/native_performance_observations.csv'
    assert str(result['historical_fps']) in table.read_text()


@pytest.mark.parametrize('mutation', ['identity', 'ambiguous', 'runtime_id'])
def test_canonical_full_never_selects_by_age_or_last_report(tmp_path, mutation):
    row=copy.deepcopy(FIXTURES[0]['row']); raw=copy.deepcopy(row)
    root=tmp_path/'collected';table=root/'analysis_tables/native_full_baseline_eval.json';table.parent.mkdir(parents=True)
    rows=[raw]
    if mutation=='identity':raw['setup_id']='wrong'
    if mutation=='ambiguous':rows.append(copy.deepcopy(raw))
    if mutation=='runtime_id':raw['repetition_records'][0]['runtime_instance_id']='different'
    table.write_text(json.dumps({'rows':rows}));row['source_root']=str(root)
    result=report_rate_fields(row)
    assert result['completed_task_fps'] is None
    assert result['historical_fps'] is None
    assert result['completed_task_fps_unavailable_reason'] in {
        'canonical_full_identity_missing_or_ambiguous','raw_report_summary_identity_conflict'}
