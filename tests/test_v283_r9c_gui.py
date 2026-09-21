"""Normal Tk result widget and actual reporter/CSV for new and old endpoints."""
import tkinter as tk
import pytest

@pytest.mark.parametrize('endpoint',['topk','host','historical'])
def test_real_classification_result_widget_and_csv(tmp_path,endpoint):
    from test_v283_r9c_classification import build_classification_report_fixture
    from onnx_splitpoint_tool.gui.app import SplitPointAnalyserGUI
    build_classification_report_fixture(tmp_path,endpoint)
    app=SplitPointAnalyserGUI()
    previous={k:v.get() for k,v in app.__dict__.items() if k.startswith('var_eval_workflow_') and isinstance(v,tk.Variable)}
    try:
        app._select_main_tab('evaluation_workflow')
        app._eval_workflow_render_result({'run_dir':str(tmp_path),'status':'local_classification_replay'})
        app.update()
        def texts(parent):
            for child in parent.winfo_children():
                if isinstance(child,tk.Text):yield child.get('1.0','end')
                yield from texts(child)
        rendered='\n'.join(texts(app.panel_frames['evaluation_workflow']))
        assert '250.000 FPS' in rendered
        if endpoint=='topk':assert 'Mean 4.000 / P50 4.000 / P95 4.000 ms' in rendered and 'n=100/100' in rendered
        elif endpoint=='host':assert 'Hostoutputlatenz' in rendered and 'Mean 2.000' in rendered
        else:assert 'Hostoutput (ohne Task-Postprocessing)' in rendered and 'request_timestamps_missing' in rendered
        (tmp_path/'R9C_WIDGET.txt').write_text(rendered)
    finally:
        for key,value in previous.items():getattr(app,key).set(value)
        app.destroy()
