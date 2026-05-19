#!/bin/bash
set -e

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"
PYTHON_EXE=""

ensure_venv() {
    if [ -d "$VENV_DIR" ]; then
        echo "Aktiviere Virtual Environment in $VENV_DIR..."
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        PYTHON_EXE="$PWD/$VENV_DIR/bin/python"
        return
    fi

    echo "Virtual Environment '$VENV_DIR' nicht gefunden. Wird erstellt..."
    python3 -m venv "$VENV_DIR"
    echo "Aktiviere neues Virtual Environment..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON_EXE="$PWD/$VENV_DIR/bin/python"
    echo "Aktualisiere pip..."
    "$PYTHON_EXE" -m pip install --upgrade pip
    if [ -f "$REQUIREMENTS" ]; then
        echo "Installiere Abhängigkeiten aus $REQUIREMENTS..."
        "$PYTHON_EXE" -m pip install -r "$REQUIREMENTS"
    else
        echo "Warnung: $REQUIREMENTS wurde nicht gefunden. Überspringe Paketinstallation."
    fi
}

ensure_startup_dependencies() {
    echo "Prüfe/ergänze minimale GUI-Abhängigkeiten ..."
    "$PYTHON_EXE" -m onnx_splitpoint_tool.dependency_bootstrap --groups gui_core
}

ensure_venv
ensure_startup_dependencies

echo "Starte Analyse-GUI..."
"$PYTHON_EXE" analyse_and_split_gui.py

deactivate
