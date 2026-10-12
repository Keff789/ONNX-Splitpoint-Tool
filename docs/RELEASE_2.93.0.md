# Release 2.93.0 — reguläre Toolabnahme und Energieintegration

Paket/GUI `2.93.0`, Build/Workflow `v2.93.0`, Entwicklungslinie `v2.79`.
Diese Softwareveröffentlichung integriert den vorhandenen Energieproduktstand
und schließt Konfigurations- und Importlücken. Sie startet keine Messkampagne.

Gespeicherte Null-Warmups bleiben bis zum tatsächlichen Backendrequest erhalten.
GUI-Startcallback und profilgesteuerter CLI-Aufruf erzeugen denselben fachlichen
Backendrequest; nur die Metadatenzeit der Profilauflösung wird beim Vergleich
ausgenommen. Die jeweils gespeicherten Run-mode-Budgets bleiben maßgeblich.
Die GUI zeigt einen ausführbaren, korrekt quotierten CLI-Startbefehl.

Vorbereitete Native-/Generic-Energieparents laufen über die normalen
Produktmodule und behalten ihre vorhandenen Quellen-, Besitzer-, Retry-,
Resume- und Budgetprüfungen. Die Ergebnisimporte einschließlich der gezielten
Generic-Current-/CSV-Projektion liegen nun im Paket. Historische Importbefehle
lösen automatisch auf diesen installierten Leser auf; private Pythonlogik wird
dabei nicht ausgeführt, erhaltene historische Dateien bleiben Provenienzeingaben.
Worktree-Importumleitungen sind dafür nicht nötig. Native-Resume benötigt
kein privates Collector-Modul-Overlay. Historische Quellen und Versuche bleiben
lesbar und erhalten.

Der normale Generic-Workflow behält seine vorhandene Energiepolitik
`native_only`. Der selektive Generic-Completed-Task-Energiepfad für bereits
vorbereitete Parents ist separat erreichbar. Ein vollständiger privater
Fairness-Export bleibt ein eigener Berichtsvorgang; der normale Import verlangt
ihn nicht und verändert keine bereits bestätigten wissenschaftlichen Aussagen.

Die Softwareabnahme nutzt lokale Prozess-, Controller-, Retry-/Resume- und
Quellenfixtures sowie Reprocessing vorhandener Originaldaten. Sechs gespeicherte
Parents mit 23 Punkten liefern exakt die bisherigen Ergebnis- und
Entscheidungsfelder; 236 Originaldateien, Budget, Queue und Vergleichsmatrix
bleiben unverändert. Dies ist keine erneute physische GUI-/Collectorabnahme.
Gültige vorhandene Hardwarebelege bleiben ihre eigenen Originalbelege.

Installation im regulären Quellcheckout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[runtime]'
.venv/bin/onnx-splitpoint-gui
```

Für eine vorhandene Installation verwenden Quellarchiv und
`scripts/update_source_release.sh RELEASE.zip TOOL_DIR` weiterhin den
etablierten Installer, der Dependencies, Profile und lokale Daten erhält.
`scripts/build_source_manifest.py` und `scripts/build_source_release.py`
erzeugen das vollständige verifizierte Quellarchiv.

Normale selektive Einstiege:

```bash
.venv/bin/python -m onnx_splitpoint_tool.workflow.run_evaluation \
  --profile-driven --profile /path/to/profile.yaml --out /path/to/new-runs
.venv/bin/onnx-splitpoint-complete-evaluation \
  --run-dir /path/to/prepared-parent --plan-only --output-dir /path/to/preview
.venv/bin/onnx-splitpoint-complete-evaluation \
  --run-dir /path/to/completed-parent --import-only
```

Ein ausdrücklich beauftragtes Resume verwendet einen eigenen Aufruf:

```bash
.venv/bin/onnx-splitpoint-complete-evaluation \
  --run-dir /path/to/paused-parent --resume
```

Es behält gültige Slots und schließt nur fehlende Wiederholungen. `--import-only`
ist ein separater Modus für bereits vorhandene Ergebnisse.
Der direkte paketierte Leser ist `onnx-splitpoint-energy-import --cohort ...`
für Native bzw. `--parent ...` für Generic. Diese Befehle ersetzen keine
gesonderte Autorisierung einer pausierten Kampagne. Nach dieser Toolphase
bleibt die Restmesskampagne pausiert.
