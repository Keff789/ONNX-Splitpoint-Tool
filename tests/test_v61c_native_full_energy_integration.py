from __future__ import annotations
import json, shlex, subprocess, sys
from pathlib import Path
from tests.test_energy_final_contract_regressions import _with_runtime_contract

def test_remote_native_full_runner_is_standalone():
    root=Path(__file__).resolve().parents[1]
    script=root/'onnx_splitpoint_tool/resources/remote_scripts/native_full_baseline_eval_runner.py'
    cp=subprocess.run([sys.executable,str(script),'--help'],cwd='/tmp',text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
    assert cp.returncode==0,(cp.stdout,cp.stderr)

def test_debug_pack_includes_progress_files():
    root=Path(__file__).resolve().parents[1]
    assert 'reports/native_progress.jsonl' in (root/'onnx_splitpoint_tool/workflow/debug_pack.py').read_text()
    assert 'create_evaluation_debug_pack' in (root/'scripts/create_evaluation_debug_pack.py').read_text()
    assert 'create_evaluation_debug_pack' in (root/'onnx_splitpoint_tool/gui/app.py').read_text()

def test_energy_plan_retains_unpaired_technical_row_and_propagates_runs(tmp_path):
    root=Path(__file__).resolve().parents[1]
    summary=tmp_path/'summary.json'; validation=tmp_path/'validation.json'; out=tmp_path/'out'
    rows=[
      {'ok':True,'backend':'hailo8_to_trt','model':'m','case':'b1','precision':'p','setup_id':'h8','fps_makespan':10},
      {'ok':True,'backend':'native_full_hailo8','model':'m','case':'full','precision':'p','setup_id':'h8','comparison_backend':'hailo8','fps_makespan':8},
      {'ok':True,'backend':'native_full_tensorrt','model':'m','case':'full','precision':'p','setup_id':'h8','comparison_backend':'hailo8','fps_makespan':12},
      {'ok':True,'backend':'deepx_to_trt','model':'m','case':'b2','precision':'p','setup_id':'dx','fps_makespan':5},
    ]
    rows=[_with_runtime_contract(row) for row in rows]
    summary.write_text(json.dumps({'rows':rows}))
    validation.write_text(json.dumps({'rows':[
      {
        **row,
        'task':'classification',
        'top1_match':True,
        'contract_consistent':True,
        'claim_ok':True,
        'semantic_ok':True,
      }
      for row in rows
    ]}))
    cp=subprocess.run([sys.executable,str(root/'scripts/native_producer_energy_plan.py'),'--summary',str(summary),'--validation-summary',str(validation),'--out-dir',str(out),'--hailo8-ssh','host','--deepx-ssh','host','--runs','2'],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
    assert cp.returncode==0,(cp.stdout,cp.stderr)
    payload=json.loads((out/'native_producer_energy_plan.json').read_text())
    assert len(payload['rows'])==4
    assert payload['excluded_rows']==[]
    assert any(
        x.get('reason')=='pair_baseline_missing'
        for x in payload['quality_pairing_posthoc_excluded_rows']
    )
    deepx_row=next(row for row in payload['rows'] if row['backend']=='deepx_to_trt')
    assert deepx_row['diagnostic_only'] is True
    assert deepx_row['energy_claim_eligible'] is False
    assert all('--runs 2' in row['measure_command'] for row in payload['rows'])
    measurement_dirs = []
    for row in payload['rows']:
        parts = shlex.split(row['measure_command'])
        assert '--out' in parts
        command_out = Path(parts[parts.index('--out') + 1]).resolve()
        planned_out = Path(row['measurement_output_dir']).resolve()
        assert command_out == planned_out
        assert planned_out == Path(row['measurement_output_base_dir']).resolve()
        assert out.parent.resolve() in planned_out.parents
        assert planned_out.parent.parent.name == 'measurements'
        assert planned_out.name.startswith('plan_')
        assert not planned_out.exists(), 'planner must not materialize execution output'
        assert row['measurement_output_policy'] == 'runner_materialized_unique_attempt_child'
        assert row['measurement_plan_attempt_id'] == payload['measurement_plan_attempt_id']
        assert row['measurement_setup_id'] == row['setup_id']
        assert row['measurement_requested_repeats'] == payload['energy_runs_per_row'] == 2
        assert row['measurement_run_id'] == parts[parts.index('--run-id') + 1]
        measurement_dirs.append(planned_out)
    assert len(set(measurement_dirs)) == len(payload['rows'])

def test_remote_import_failure_category():
    from onnx_splitpoint_tool.workflow.runner import _native_backend_result_reason_v60y
    row={'error':"ModuleNotFoundError: No module named 'onnx_splitpoint_tool.native_progress'"}
    assert _native_backend_result_reason_v60y(row)=='remote_runner_import_failed'

def test_energy_runner_is_import_safe_from_foreign_cwd(tmp_path):
    root=Path(__file__).resolve().parents[1]
    summary=tmp_path/'summary.json'; validation=tmp_path/'validation.json'; out=tmp_path/'out'
    summary.write_text(json.dumps({'rows':[]}))
    validation.write_text(json.dumps({'rows':[]}))
    env={'PATH':str(Path(sys.executable).parent), 'PYTHONUNBUFFERED':'1'}
    cp=subprocess.run([
        sys.executable, str(root/'scripts/run_native_producer_energy_from_summary.py'),
        '--summary',str(summary),'--validation-summary',str(validation),'--out-dir',str(out),
        '--runs','1',
    ],cwd=tmp_path,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    assert cp.returncode!=0,(cp.stdout,cp.stderr)
    payload=json.loads((out/'native_producer_energy_results.json').read_text())
    assert payload['status']=='blocked_no_runtime_constructible_rows'
    assert payload['ok'] is False and payload['complete'] is False
    assert payload['started_measurement_count']==0


def test_native_energy_repeat_resolution_uses_run_mode_snapshot():
    from onnx_splitpoint_tool.workflow.runner import _native_energy_repeat_count_v61c
    profile={
        'energy': {'repeat_override': 0},
        'execution_preset': {'snapshot': {'energy': {'repeats': 5}}},
    }
    assert _native_energy_repeat_count_v61c(profile, {})==5
    assert _native_energy_repeat_count_v61c(profile, {'repeats': 2})==2


def test_runner_syncs_full_dependencies_and_runs_remote_preflight():
    root=Path(__file__).resolve().parents[1]
    text=(root/'onnx_splitpoint_tool/workflow/runner.py').read_text()
    assert 'native_progress.py' in text
    assert 'smoke_hailo10_full_from_benchmarkset.py' in text
    assert 'native_full_remote_import_preflight' in text
    assert 'full-preflight:' in text


def test_progress_files_are_registered_as_native_artifacts():
    root=Path(__file__).resolve().parents[1]
    text=(root/'onnx_splitpoint_tool/workflow/runner.py').read_text()
    assert 'for progress_name in ("native_progress.json", "native_progress.jsonl")' in text


def test_workflow_syncs_remote_progress_preflights_full_and_persists_progress():
    root=Path(__file__).resolve().parents[1]
    source=(root/'onnx_splitpoint_tool/workflow/runner.py').read_text()
    assert 'script_name="native_progress.py"' in source or '("native_progress.py"' in source
    assert 'native_full_remote_import_preflight' in source
    assert 'remote_python_path = f"PYTHONPATH=' in source
    assert 'native_progress_jsonl' in source and 'native_progress_json' in source

def test_energy_helper_dry_run_retains_unpaired_technical_row(tmp_path):
    root=Path(__file__).resolve().parents[1]
    summary=tmp_path/'summary.json'; validation=tmp_path/'validation.json'; out=tmp_path/'out'
    split_row=_with_runtime_contract(
      {'ok':True,'backend':'hailo8_to_trt','model':'m','case':'b1','precision':'p','setup_id':'h8','fps_makespan':10}
    )
    summary.write_text(json.dumps({'rows':[split_row]}))
    validation.write_text(json.dumps({'rows':[
      {**split_row,'task':'classification','top1_match':True,'contract_consistent':True,'claim_ok':True,'semantic_ok':True},
    ]}))
    cp=subprocess.run([sys.executable,str(root/'scripts/run_native_producer_energy_from_summary.py'),'--summary',str(summary),'--validation-summary',str(validation),'--out-dir',str(out),'--hailo8-ssh','host','--duration-s','1','--timeout','10','--dry-run'],cwd=tmp_path,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
    assert cp.returncode==0,(cp.stdout,cp.stderr)
    payload=json.loads((out/'native_producer_energy_results.json').read_text())
    assert payload['status']=='dry_run' and len(payload['rows'])==1
    assert payload['ok'] is True
    assert payload['rows'][0]['row']['diagnostic_only'] is True
    assert payload['rows'][0]['row']['energy_claim_eligible'] is False
