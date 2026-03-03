#!/bin/bash

# Name des venv-Ordners und der Requirements-Datei
VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"

# 1. Prüfen, ob das venv existiert
if [ -d "$VENV_DIR" ]; then
    echo "Aktiviere Virtual Environment in $VENV_DIR..."
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual Environment '$VENV_DIR' nicht gefunden. Wird erstellt..."
    # Verwende python3 (oder python, falls python3 nicht aliasiert ist)
    python3 -m venv "$VENV_DIR"
    
    echo "Aktiviere neues Virtual Environment..."
    source "$VENV_DIR/bin/activate"
    
    echo "Aktualisiere pip..."
    pip install --upgrade pip
    
    # Prüfen, ob die requirements.txt existiert, bevor installiert wird
    if [ -f "$REQUIREMENTS" ]; then
        echo "Installiere Abhängigkeiten aus $REQUIREMENTS..."
        pip install -r "$REQUIREMENTS"
    else
        echo "Warnung: $REQUIREMENTS wurde nicht gefunden. Überspringe Paketinstallation."
    fi
fi

# 2. Python-Skript ausführen
echo "Starte Analyse-GUI..."
python analyse_and_split_gui.py

# 3. venv nach Beenden wieder deaktivieren (optional)
deactivate