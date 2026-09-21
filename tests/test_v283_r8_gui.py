"""Real Tk editor/summary matrix, no resolver substitutes and no hardware.

The installed-profile case is explicitly host dependent. The portable case
uses a shipped product profile and the same real widgets and callbacks.
"""
import os
from pathlib import Path
import tkinter as tk

import pytest
import yaml


@pytest.fixture
def root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f'real Tk display unavailable: {exc}')
    root.withdraw()
    yield root
    root.destroy()


@pytest.mark.parametrize('installed', [False, True])
def test_real_editor_mode_save_reload_and_normal_summary(root, tmp_path, monkeypatch, installed):
    from onnx_splitpoint_tool.gui import profile_editor as editor_module
    from onnx_splitpoint_tool.gui.panels import panel_evaluation_workflow as panel
    from onnx_splitpoint_tool.native_execution_contract import resolve_native_execution_contract
    repo = Path(__file__).resolve().parents[1]
    profile = repo/'profiles/CompleteSetDev.yaml' if installed else repo/'onnx_splitpoint_tool/resources/evaluation_profiles/smoke_regression_v1.yaml'
    if not profile.is_file():
        pytest.skip('installed CompleteSetDev profile unavailable')
    for key in ('ONNX_SPLITPOINT_HARDWARE_SETUPS_FILE', 'ONNX_SPLITPOINT_RUN_MODES_FILE', 'URECS_RECEIVE_DIAGNOSTICS'):
        monkeypatch.delenv(key, raising=False)
    errors = []
    # Only interactive dialog responses are controlled. Real editor, variables,
    # loader, registry, modes, serialization and summary remain unchanged.
    monkeypatch.setattr(editor_module.messagebox, 'showerror', lambda *a, **k: errors.append(a))
    editor = editor_module.EvaluationProfileEditor(root, profile_var=tk.StringVar(root, str(profile)))
    editor.withdraw()
    assert editor._load_profile(str(profile)), errors
    saved = tmp_path/'normal.yaml'
    monkeypatch.setattr(editor_module.filedialog, 'asksaveasfilename', lambda **kw: str(saved))
    outcomes = []
    for mode, expected in [('standard', (100, 10, 1)), ('final', (1000, 100, 3)), ('standard', (100, 10, 1))]:
        editor.var_run_mode_id.set(mode)
        editor._on_run_mode_changed()
        root.update()
        result = editor._build_payload()
        contract = resolve_native_execution_contract(result)
        assert tuple(contract[k] for k in ('frames','warmup','repetitions')) == expected
        assert result['execution_preset']['effective']['native_performance_repetitions'] == expected[2]
        editor._save(use_after=False)
        assert saved.is_file(), errors
        assert editor._load_profile(str(saved)), errors
        for energy_on in (False, True):
            editor.var_energy_enabled.set(energy_on)
            editor._on_energy_master_toggle()
            editor._save(use_after=False)
            lines, snapshot = panel._profile_summary_payload(str(saved))
            assert snapshot, lines
            if energy_on and editor.var_native_enabled.get():
                assert any('Native collector:' in line for line in lines)
                assert any('product_default' in line for line in lines)
            outcomes.append({'mode':mode, 'energy':energy_on, 'summary':lines})
    assert not errors
    (tmp_path/'matrix.json').write_text(__import__('json').dumps(outcomes, indent=2))
    editor.destroy()


def test_real_editor_keeps_explicit_budget_and_zero_warmup(root, tmp_path, monkeypatch):
    from onnx_splitpoint_tool.gui import profile_editor as module
    from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
    profile = Path(__file__).resolve().parents[1]/'onnx_splitpoint_tool/resources/evaluation_profiles/smoke_regression_v1.yaml'
    raw = yaml.safe_load(profile.read_text())
    raw['execution_preset'] = {'id':'standard', 'follow_tool_config':True}
    raw['native_producers'] = {'enabled':True, 'frames':217, 'warmup':0, 'repetitions':2,
                               'energy':{'enabled':True, 'mode':'measure', 'task_budget':{'enabled':False}}}
    raw, _ = apply_run_mode(raw, config=default_run_modes_config())
    saved = tmp_path/'custom.yaml'; saved.write_text(yaml.safe_dump(raw))
    errors=[]
    monkeypatch.setattr(module.messagebox,'showerror',lambda *a,**k:errors.append(a))
    editor=module.EvaluationProfileEditor(root,profile_var=tk.StringVar(root,str(saved)))
    editor.withdraw()
    editor.var_run_mode_id.set('final'); editor._on_run_mode_changed(); root.update()
    result=editor._build_payload()
    assert tuple(result['native_producers'][k] for k in ('frames','warmup','repetitions')) == (217,0,2)
    assert result['native_producers']['energy']['task_budget'] == {'enabled':False}
    assert not errors
    editor.destroy()


def test_real_editor_full_split_selection_and_frozen_resume(root, tmp_path, monkeypatch):
    from onnx_splitpoint_tool.gui import profile_editor as module
    from onnx_splitpoint_tool.native_execution_contract import resolve_native_execution_contract
    profile=Path(__file__).resolve().parents[1]/'onnx_splitpoint_tool/resources/evaluation_profiles/smoke_regression_v1.yaml'
    errors=[]
    monkeypatch.setattr(module.messagebox,'showerror',lambda *a,**k:errors.append(a))
    editor=module.EvaluationProfileEditor(root,profile_var=tk.StringVar(root,str(profile)))
    editor.withdraw()
    for key,var in editor.__dict__.items():
        if key.startswith('var_run_') and isinstance(var,tk.BooleanVar): var.set(False)
    editor.var_native_enabled.set(True)
    for full,split in ((True,False),(False,True),(True,True)):
        editor.var_run_hailo.set(full);editor.var_run_trt.set(full)
        editor.var_run_hailo_to_trt.set(split)
        editor._on_run_mode_changed();root.update()
        result=editor._build_payload();native=result['native_producers']
        assert native['full_baselines']['enabled'] is full
        assert ('hailo8' in native['split_backends']) is split
        contract=resolve_native_execution_contract(result)
        assert tuple(contract[k] for k in ('frames','warmup','repetitions'))==(100,10,1)
    # An archived snapshot explicitly detached from today's config retains its
    # native budget even though Standard now has different defaults.
    result['execution_preset']['follow_tool_config']=False
    result['native_producers'].update(frames=1000,warmup=100,repetitions=3)
    result['execution_preset']['snapshot']['runtime']['native'].update(frames=1000,warmup=100,repetitions=3)
    frozen=tmp_path/'frozen.yaml';frozen.write_text(yaml.safe_dump(result))
    assert editor._load_profile(str(frozen)),errors
    loaded=editor._build_payload()
    assert tuple(loaded['native_producers'][k] for k in ('frames','warmup','repetitions'))==(1000,100,3)
    assert loaded['execution_preset']['follow_tool_config'] is False
    editor.destroy()


def test_real_app_stale_collector_and_persistent_source_stop(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.gui import app as gui
    from onnx_splitpoint_tool.gui.panels import panel_evaluation_workflow as panel
    from onnx_splitpoint_tool.energy.config import load_hardware_registry
    from test_v283_campaign_energy_budget import campaign_rig
    import json
    try:
        app=gui.SplitPointAnalyserGUI()
    except tk.TclError as exc:
        pytest.skip(f'real Tk display unavailable: {exc}')
    monkeypatch.setattr(gui.messagebox,'showerror',lambda *a,**k:None)
    monkeypatch.setattr(gui.messagebox,'showwarning',lambda *a,**k:None)
    for key in ('ONNX_SPLITPOINT_HARDWARE_SETUPS_FILE','ONNX_SPLITPOINT_RUN_MODES_FILE'):
        monkeypatch.delenv(key,raising=False)
    profile=Path(__file__).resolve().parents[1]/'onnx_splitpoint_tool/resources/evaluation_profiles/smoke_regression_v1.yaml'
    raw=yaml.safe_load(profile.read_text())
    raw['execution_preset']={'id':'standard','follow_tool_config':True}
    raw['native_producers']={'enabled':True,'energy':{'enabled':True,'mode':'measure'}}
    path=tmp_path/'normal.yaml';path.write_text(yaml.safe_dump(raw))
    app._select_main_tab('evaluation_workflow')
    app.var_eval_workflow_profile.set(str(path));app.var_eval_workflow_out_root.set(str(tmp_path/'runs'))
    app._select_main_tab('evaluation_workflow');app.update()
    text=tk.Text(app)
    def refresh():
        lines,snapshot=panel._profile_summary_payload(str(path))
        panel._commit_profile_summary(app,text,'\n'.join(lines),snapshot)
        app.update()
        return lines,snapshot
    try:
        assert refresh()[1]
        options=app._eval_workflow_snapshot_options(resume_override=False)
        native=options.profile_start_snapshot['resolved_profile']['native_producers']
        assert native['energy']['task_budget']['max_retries']==1
        raw['execution_preset']['id']='final';path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError,match='changed|resolved'):
            app._eval_workflow_snapshot_options(resume_override=False)
        raw['execution_preset']['id']='standard'
        # Only negative cases use a temporary copy of the ordinary registry.
        registry=load_hardware_registry()
        negative=tmp_path/'negative_registry.yaml'
        raw['hardware']={'setups_file':str(negative)}
        for binary in (str(tmp_path/'missing'),'/bin/true'):
            registry['energy_defaults']['collector_binary']=binary
            registry['energy_defaults']['collector_sha256']='0'*64
            negative.write_text(yaml.safe_dump(registry));path.write_text(yaml.safe_dump(raw))
            lines,snapshot=refresh()
            assert not snapshot,lines
            with pytest.raises(ValueError):app._eval_workflow_snapshot_options(resume_override=False)
        # Real process leaves replace the physical meter/workload. The policy
        # comes from the normal real Tk start snapshot above without mocking
        # profile, registry, mode, budget, parser or source-gate resolution.
        rig=campaign_rig(tmp_path,monkeypatch,['success','first_sample','success'])
        policy=native['energy']['task_budget']
        args=dict(task_budget_file=None,task_max_chains=None,task_max_transport_failures=None,
            campaign_budget_file=tmp_path/'campaign.json',campaign_repeats=3,
            campaign_max_retries=policy['max_retries'],campaign_max_transport_failures=policy['max_transport_failures'])
        first=rig.measure('Full',campaign_row_id='Full',**args)
        assert first['error']=='campaign_source_completion_unresolved'
        before=rig.counts()
        for entry in ('Split','WindowProbe','reentry'):
            blocked=rig.measure(entry,campaign_row_id=entry,**args)
            assert blocked['execution_status']=='NOT_RUN'
        assert rig.counts()==before=={'preflight':2,'collector':2,'workload':1}
        (tmp_path/'gui_source_stop.json').write_text(json.dumps({'actual_process_starts':before,'policy':policy,'first':first},indent=2))
    finally:
        app._restore_evaluation_shutdown_signal_broker()
        app.destroy()
