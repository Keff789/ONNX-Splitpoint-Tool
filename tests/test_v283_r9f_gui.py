"""Real Tk profile roundtrip and normal summary/snapshot; no workflow starts."""
import tkinter as tk
from pathlib import Path
import pytest
import yaml


@pytest.mark.parametrize('version',[0,1])
def test_real_editor_retains_output_policy_and_warm_only_admission(tmp_path,monkeypatch,version):
    from onnx_splitpoint_tool.gui import profile_editor as module
    from onnx_splitpoint_tool.gui.panels import panel_evaluation_workflow as panel
    from onnx_splitpoint_tool.gui.app import SplitPointAnalyserGUI
    from onnx_splitpoint_tool.backend_backfill import DEFAULT_BACKFILL
    from onnx_splitpoint_tool.workflow.artifact_cache_preflight import resolve_artifact_cache_preflight_policy
    source=Path(__file__).resolve().parents[1]/'profiles/CompleteSetDev.yaml'
    data=yaml.safe_load(source.read_text())
    data.setdefault('selection_policy',{})['backend_backfill']={**DEFAULT_BACKFILL,'technical_output_contract_version':version,
        'max_candidates_per_backend':4,'max_cold_builds':0,'max_hailo_part1_builds':0,'max_trt_part2_builds':0}
    data.setdefault('native_producers',{})['build_missing_engines']=False
    warm={'enabled':True,'default_expectation':'warm','block_on_unexpected_cold_builds':True}
    data['artifact_cache_preflight']=warm
    data.setdefault('workflow',{})['artifact_cache_preflight']=warm
    initial=tmp_path/'initial.yaml';initial.write_text(yaml.safe_dump(data,sort_keys=False))
    app=SplitPointAnalyserGUI();app.withdraw()
    previous={k:v.get() for k,v in app.__dict__.items() if k.startswith('var_eval_workflow_') and isinstance(v,tk.Variable)}
    errors=[]
    monkeypatch.setattr(module.messagebox,'showerror',lambda *a,**k:errors.append(a))
    monkeypatch.setattr(module.messagebox,'showinfo',lambda *a,**k:None)
    saved=tmp_path/'saved.yaml'
    monkeypatch.setattr(module.filedialog,'asksaveasfilename',lambda **kw:str(saved))
    editor=module.EvaluationProfileEditor(app,profile_var=tk.StringVar(app,str(initial)));editor.withdraw()
    try:
        assert editor._load_profile(str(initial)),errors
        editor._update_run_summary();app.update_idletasks()
        editor._save(use_after=False)
        assert saved.is_file(),errors
        assert editor._load_profile(str(saved)),errors
        result=editor._build_payload()
        assert result['selection_policy']['backend_backfill']['technical_output_contract_version']==version
        assert result['native_producers']['build_missing_engines'] is False
        policy=resolve_artifact_cache_preflight_policy(result)
        assert policy['default_expectation']=='warm' and policy['block_on_unexpected_cold_builds']
        app._select_main_tab('evaluation_workflow');app.var_eval_workflow_profile.set(str(saved));app.var_eval_workflow_out_root.set(str(tmp_path/'unused'))
        lines,snapshot=panel._profile_summary_payload(str(saved));assert snapshot,lines
        texts=[]
        def collect(parent):
            for child in parent.winfo_children():
                if isinstance(child,tk.Text):texts.append(child)
                collect(child)
        collect(app.panel_frames['evaluation_workflow'])
        assert panel._commit_profile_summary(app,texts[0],'\n'.join(lines),snapshot)
        options=app._eval_workflow_snapshot_options(resume_override=False)
        resolved=options.profile_start_snapshot['resolved_profile']
        assert resolved['selection_policy']['backend_backfill']['technical_output_contract_version']==version
        assert resolved['native_producers']['build_missing_engines'] is False
        assert any('technical_output_contract_version' in line and str(version) in line for line in lines)
        assert not errors
        (tmp_path/'summary.widget.txt').write_text(texts[0].get('1.0','end'))
    finally:
        editor.destroy()
        for key,value in previous.items():getattr(app,key).set(value)
        app.destroy()
