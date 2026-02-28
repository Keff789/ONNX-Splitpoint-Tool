#!/bin/bash

# Name des venv-Ordners
VENV_DIR=".venv"

# 1. Prüfen, ob das venv existiert
if [ -d "$VENV_DIR" ]; then
    echo "Aktiviere Virtual Environment in $VENV_DIR..."
    source "$VENV_DIR/bin/activate"
else
    echo "Fehler: Ordner $VENV_DIR wurde nicht gefunden."
    exit 1
fi

# 2. Python-Skript ausführen
echo "Starte Analyse-GUI..."
python analyse_and_split_gui.py

# 3. venv nach Beenden wieder deaktivieren (optional)
deactivate
