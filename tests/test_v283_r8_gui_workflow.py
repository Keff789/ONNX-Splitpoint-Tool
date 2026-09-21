"""Normal Tk app -> queue -> real CPU workflow -> report -> terminal failure.

Run separately with --noconftest: normal GUI configuration and X authority
must remain available. Only dialog responses are controlled. No fake runner,
resolver, queue, compiler result, parser or result projection is used.
The local CPU-only physical scope is currently unsupported. This is an
explicit negative integration test, not a successful Native workflow claim.
"""
import json
import os
from pathlib import Path
import subprocess
import tkinter as tk

import pytest
import yaml


def test_normal_gui_queue_real_cpu_workflow_reports_technical_failure(tmp_path, monkeypatch):
    import onnx
    from onnx import helper, TensorProto
    from PIL import Image
    from onnx_splitpoint_tool.campaign import create_dataset_manifest
    from onnx_splitpoint_tool.gui import app as gui
    from onnx_splitpoint_tool.gui.panels import panel_evaluation_workflow as panel
    for key in ('ONNX_SPLITPOINT_HARDWARE_SETUPS_FILE', 'ONNX_SPLITPOINT_RUN_MODES_FILE'):
        monkeypatch.delenv(key, raising=False)
    try:
        probe = tk.Tk(); probe.destroy()
    except tk.TclError as exc:
        pytest.skip(f'real Tk display unavailable: {exc}')
    labels = tmp_path/'labels.txt'
    labels.write_text('n00000000 class0\nn00000001 class1\nn00000002 class2\n')
    manifests = {}
    for role, delta in [('calibration', 0), ('validation', 1)]:
        directory = tmp_path/role
        for i, color in enumerate([(230,10,10),(10,230,10),(10,10,230)]):
            folder = directory/f'n{i:08d}'; folder.mkdir(parents=True)
            Image.new('RGB',(16,16),tuple(c+delta for c in color)).save(folder/f'class{i}.png')
        manifests[role] = str(create_dataset_manifest(task='classification', role=role,
            dataset_id='r8_'+role, split='train' if role=='calibration' else 'val',
            root=directory, output=tmp_path/(role+'.json'), labels=labels))
    model_path=tmp_path/'tiny_classification.onnx'
    graph=helper.make_graph([
        helper.make_node('GlobalAveragePool',['images'],['pooled']),
        helper.make_node('Flatten',['pooled'],['flat'],axis=1),
        helper.make_node('Identity',['flat'],['output'])], 'r8_cpu',
        [helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,16,16])],
        [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,3])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],ir_version=8)
    onnx.checker.check_model(model);onnx.save(model,model_path)
    registry=tmp_path/'dataset_registry.json';registry.write_text(json.dumps({'manifests':{}}))
    profile={'name':'r8_gui_cpu_fixture','execution_preset':{'id':'standard','follow_tool_config':True},
        'model_suite':{'primary':[{'id':'tiny_classification','onnx':str(model_path),'task':'classification','input_shape':[1,3,16,16]}]},
        'selection_policy':{'max_accepted_cases_per_model':1,'preferred_shortlist':1,'min_gap':1,
            'candidate_search_pool':3,'selection_strategy':'stratified_windows'},
        'run_profiles':[{'id':'ort_cpu','type':'same_backend_reference','full':'cpu','stage1':'cpu','stage2':'cpu','required':True}],
        'hardware':{'selected_setups':[],'selected_groups':[]},'hailo_build':{'targets':[]},
        'native_producers':{'enabled':False,'energy':{'enabled':False}},
        'energy':{'requested_native_energy':False},
        'campaign':{'dataset_registry':str(registry),'dataset_manifests':{'classification':manifests}}}
    path=tmp_path/'cpu.yaml';path.write_text(yaml.safe_dump(profile))
    dialogs=[]
    app_holder=[]
    def dialog(kind):
        def show(title, message='', **kw):
            dialogs.append({'kind':kind,'title':title,'message':str(message)})
            if title=='Evaluation Workflow' and app_holder:
                app_holder[0].after(1000,app_holder[0].quit)
        return show
    commands=[]
    real_popen=subprocess.Popen
    def observe_popen(args,*a,**kw):
        command=[str(x) for x in args] if isinstance(args,(list,tuple)) else [str(args)]
        commands.append(command)
        assert Path(command[0]).name not in {'ssh','scp','rsync','urecs-data-collector'}, command
        return real_popen(args,*a,**kw)
    monkeypatch.setattr(subprocess,'Popen',observe_popen)
    app=gui.SplitPointAnalyserGUI();app_holder.append(app)
    # App initialization installs its diagnostic dialog wrapper. Control only
    # the final interaction after that normal initialization has completed.
    monkeypatch.setattr(gui.messagebox,'showinfo',dialog('info'))
    monkeypatch.setattr(gui.messagebox,'showwarning',dialog('warning'))
    monkeypatch.setattr(gui.messagebox,'showerror',dialog('error'))
    monkeypatch.setattr(gui.messagebox,'askyesno',lambda *a,**k:True)
    app._select_main_tab('evaluation_workflow')
    previous={k:v.get() for k,v in app.__dict__.items() if k.startswith('var_eval_workflow_') and isinstance(v,tk.Variable)}
    try:
        app.var_eval_workflow_profile.set(str(path))
        app.var_eval_workflow_out_root.set(str(tmp_path/'runs'))
        lines,visible=panel._profile_summary_payload(str(path));assert visible,lines
        app._eval_workflow_visible_profile_snapshot=visible
        # Use the real panel's existing commit callback and actual Text widget.
        texts=[]
        def widgets(parent):
            for child in parent.winfo_children():
                if isinstance(child,tk.Text):texts.append(child)
                widgets(child)
        widgets(app.panel_frames['evaluation_workflow'])
        assert texts
        panel._commit_profile_summary(app,texts[0],'\n'.join(lines),visible)
        opts=app._eval_workflow_snapshot_options(resume_override=False)
        assert opts.no_remote and opts.execution_mode=='generate_and_run' and not opts.skip_benchmarks
        state={}
        def start():state['job']=app._queue_evaluation_workflow(resume=False)
        def timeout():
            state['timeout']=True
            record=app._background_jobs.get(state.get('job'))
            if record and record.cancel_callback:record.cancel_callback()
        app.after(50,start);app.after(360000,timeout)
        app.mainloop()
        (tmp_path/'gui_evidence.json').write_text(json.dumps({'dialogs':dialogs,'commands':commands,'state':state},indent=2))
        assert not state.get('timeout'),dialogs
        assert state.get('job'),dialogs
        run_dirs=list((tmp_path/'runs').glob('r8_gui_cpu_fixture_*'))
        assert len(run_dirs)==1,dialogs
        run=run_dirs[0]
        assert (run/'run_manifest.json').is_file()
        assert (run/'reports/run_status_summary.json').is_file(),dialogs
        assert list((run/'jobs').rglob('*.json'))
        assert commands, 'no actual subprocess boundary observed'
        assert any(d['title']=='Evaluation Workflow' for d in dialogs)
        status=json.loads((run/'reports/run_status_summary.json').read_text())
        assert status['technical_status']=='failed'
        assert any('required_scope_setup_missing' in row.get('reason','') for row in status['blocking_reasons'])
        assert app.var_eval_workflow_status.get()=='Lauf fehlgeschlagen'
        assert app._background_jobs[state['job']].status=='error'
        assert any('Lauf fehlgeschlagen' in d['message'] for d in dialogs)
        assert (run/'quality_management/references/tiny_classification/management_cpu_reference_stdout.txt').is_file()
        assert any('benchmark_suite.py' in ' '.join(command) for command in commands)
        assert 'finalization_status=pass' in (tmp_path/'runs/_latest_evaluation_workflow.log').read_text()
    finally:
        for key,value in previous.items():getattr(app,key).set(value)
        app.destroy()
