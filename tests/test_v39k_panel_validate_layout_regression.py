from pathlib import Path


def test_panel_validate_advanced_widgets_stay_in_advanced_group() -> None:
    src = Path('onnx_splitpoint_tool/gui/panels/panel_validate.py').read_text(encoding='utf-8')
    assert 'cb_transfer = ttk.Combobox(\n            advanced_group,' in src
    assert 'chk_reuse = ttk.Checkbutton(\n            advanced_group,' in src
    assert 'ent_venv = ttk.Entry(advanced_group' in src


def test_panel_validate_split_matrix_row_not_shared_with_diag_tools() -> None:
    src = Path('onnx_splitpoint_tool/gui/panels/panel_validate.py').read_text(encoding='utf-8')
    assert 'diag_tools.grid(row=6' in src
    assert 'Split matrix:").grid(row=7' in src
    assert 'matrix_custom.grid(row=8' in src
    assert 'lbl_matrix.grid(row=9' in src
