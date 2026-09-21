"""Real Tk rendering of newly paired requests; other callback/queue tests stay in R9A."""
import tkinter as tk
from pathlib import Path


def test_real_widget_request_latency_and_export(tmp_path):
    from test_v283_r9b_request_latency import test_twelve_ms_requests_three_ms_cadence_through_raw_summary_gui_csv
    from onnx_splitpoint_tool.gui.app import SplitPointAnalyserGUI
    test_twelve_ms_requests_three_ms_cadence_through_raw_summary_gui_csv(tmp_path)
    app=SplitPointAnalyserGUI() # Missing DISPLAY must fail this acceptance, never skip.
    previous={k:v.get() for k,v in app.__dict__.items() if k.startswith('var_eval_workflow_') and isinstance(v,tk.Variable)}
    try:
        app._select_main_tab('evaluation_workflow')
        app._eval_workflow_render_result({'run_dir':str(tmp_path),'status':'local_request_pair_replay'})
        app.update()
        def texts(parent):
            for child in parent.winfo_children():
                if isinstance(child,tk.Text):yield child.get('1.0','end')
                yield from texts(child)
        rendered='\n'.join(texts(app.panel_frames['evaluation_workflow']))
        for expected in ('323.625 FPS','Mean 12.000 / P50 12.000 / P95 12.000 ms','n=100/100','ab vorbereitetem Input'):
            assert expected in rendered
        (tmp_path/'R9B_REAL_RESULT_WIDGET.txt').write_text(rendered)
    finally:
        for key,value in previous.items():getattr(app,key).set(value)
        app.destroy()
