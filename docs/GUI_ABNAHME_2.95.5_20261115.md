# Lokale GUI-Abnahme 2.95.5 — 15.11.2026

Die korrigierte GUI wurde auf Smartmirror2 tatsächlich bedient. Der abschließende
Softwarelauf besteht mit **149 PASS, 0 FAIL, 0 SKIP** (141,55 s; fünf
Deprecation-Warnungen aus vorhandenen Abhängigkeiten). Die Funktionsmatrix
enthält **251 Prüffälle: 231 PASS, 19 NOT_RUN und 1 BLOCKED**. GUI-, Callback-,
CLI- und ausdrücklich ersetzte externe Aufrufe sind darin getrennt bezeichnet.
Es gibt keine neue Hardware-, Energie- oder Herstellercompilerabnahme.

## Quellenstand und tatsächliche Installation

- Paket/GUI **2.95.5**, BUILD_ID **v2.95.5**, ohne neue Versionskennung oder Tag.
- Ausgangscheckout: `/home/kmika/ONNX-Splitpoint-Tool`, lokales `main` auf
  `5e6606422f1e6484ece5e78c91280e2adef3c84c` mit unveröffentlichter MCP-Arbeit
  und dem vorherigen manuellen Split-Fix. Diese Arbeit wurde gesichert und
  erhalten; MCP-Quellen, SDK-Abhängigkeiten und private Laufdaten gehören
  nicht zu dieser Lieferung.
- Der Remote `https://github.com/Keff789/ONNX-Splitpoint-Tool.git`, Zielbranch
  `main`, enthielt bereits 54 neuere Commits. Die Lieferung wurde daher im
  separaten Branch `codex/gui-acceptance-2955-20261115` auf dem abgerufenen
  `origin/main` **6f0d8f26d6f1426000e78f72afedeae6a722298a** vorbereitet.
  Worktree: `/home/kmika/Worktrees/onnx-gui-acceptance-2955-20261115`.
- Bestehende Pythonumgebung:
  `/home/kmika/ONNX-Splitpoint-Tool/.venv/bin/python`. Keine neuen Pakete
  installiert. `check_acceptance_environment.py` bestand; das ist allein
  eine Abhängigkeitsprüfung.
- Zehn geänderte Produktdateien wurden gezielt in den produktiven Checkout
  übernommen und byteweise mit der geprüften Lieferung verglichen. Der normale
  `./start_gui.sh` lädt nachweislich die reparierten Dateien aus
  `/home/kmika/ONNX-Splitpoint-Tool/onnx_splitpoint_tool/`.
  Der alte Checkout wurde dabei nicht auf einen anderen HEAD umgeschrieben.

## Gefundene Ursachen und Reparaturen

| Produktpfad | Ursache und resultierendes Verhalten |
|---|---|
| Manueller Split | Fehlende Modellquell-/Timeout-Snapshots behoben; externe Gewichte, Full-Kopie, Manifest und ORT verwenden dieselbe gewählte Quelle. Hailo-Timeout wird ausschließlich für angeforderte Builds erfasst. |
| Split-Eingaben und Ziel | Geänderte ungültige Analyseparameter werden vor Dialog/Verzeichniserzeugung verständlich abgewiesen; ein als Datei belegtes Exportziel erzeugt einen Fehlerdialog. |
| Exportabschluss | Fehler ausgewählter Kontextdateien, Runner oder technischer Validierung ergeben `Split failed` vor Manifestpublikation und Hailo-Build. Ein technisch gültiger numerischer ORT-FAIL bleibt als FAIL dokumentiert. |
| TeX | Undefinierten `model_path` beseitigt; Dateiname und Tabellenbeschriftung folgen dem gewählten GuiState-Modell mit Fallback. |
| Advanced Export | Vorschau zeigt Full-Kopie, tatsächliches `validation_core.json`, Validierungsänderungen und wirksame Ordner-/Kontextregeln. Verbindliche Bestandteile erscheinen als deaktivierte Anzeigen. Der bestehende Always-context-Vertrag bleibt erhalten. |
| Kontexttiefe | Gültiges `hops=0` wird im manuellen und im Benchmarkexport erhalten. |
| Frische Moduskonfiguration | Drei Einrückungen von `statistics_engine` in der ausgelieferten YAML repariert; Werte, Seeds und wissenschaftliche Regeln unverändert. |
| Modellvorbereitung | Zwei Legacy-Hide-Schleifen versteckten den einzigen manuellen Einstieg. Dieser ist wieder erreichbar; derselbe laufende Vorbereitungsjob kann nicht doppelt gestartet werden. |
| Verzögerte Fehler | Ausnahmeursachen für Dataset-, Matrix-, Energie- und Ergebnisworker vor dem Ende des `except`-Blocks übernommen; fehlenden Diagnoseimport ergänzt. |
| Jobs | Akzeptierter Abbruch wird idempotent; zweiter Klick ruft den Cancelcallback nicht erneut auf. |
| Tool Config | Ungültiger SSH-Port oder fehlgeschlagenes Speichern stoppt nachfolgende Test-/Energy-Aktionen; ursprüngliche Ursache bleibt sichtbar. |
| Logs | Kopieren erhält den letzten Fehlerrecord samt Traceback; neuer kleiner Exportdialog schreibt die vollständige gewählte Logdatei, Abbruch bleibt neutral. |
| Benchmark-Analyse | Fehlenden `decision_overview`-Import ergänzt; große kanonische JSON-in-CSV-Felder werden mit der vorhandenen Projektgrenze gelesen, globale Grenze danach restauriert. |
| Ergebnisdiagramme | Geschlossene Subplotfenster lösen beim späteren Rerender keinen verwaisten Matplotlib/Tk-Callback mehr aus. |
| Results/Debug Pack | Existierende leere Elternverzeichnisse werden über die vorhandene Run-Discovery abgewiesen; sie gelten nicht länger als Results Bundle. |

Die Änderungen betreffen `gui_app.py`, `gui/app.py`, `gui/benchmark_workflow.py`,
die Panels `panel_export.py`, `panel_validate.py`, `panel_hardware.py`,
`panel_logs.py`, `panel_benchmark_analysis.py`, `benchmark/analysis.py` und die
ausgelieferte Run-Modus-YAML. Hinzu kommen sieben gezielte Testdateien,
Arbeitsstand, dieser Bericht und reguläre Quellenindizes. Keine Profile,
Modelle, Messungen, SDK-Artefakte oder temporären Testausgaben werden mitgeliefert.

## Tatsächliche Bedienwege und Funktionsmatrix

| Bereich | Tatsächlich geprüft | Abgrenzung |
|---|---|---|
| Analyse | Fehlende/ungültige Modelle, MobileNet/YOLOv7-Wechsel, Parameter, Kandidaten/Details, Filter, Rangfolge, Presets, Plots/Toolbar und Exporte | Reale sichtbare GUI und normale Ereignisschleife |
| Advanced Export | b043/b044/b066, Klassifikatorschnitt mit Rename, Full-/Cut-flow/hops=0, Ordner/Abbruch/Leerzeichen, bestehende und blockierte Dateien; ONNX/Manifest/Full/Runner/Grafiken validiert | Hailo ausschließlich ersetzter Compiler im Regressionstest |
| Evaluation Workflow | Profile laden/editieren/validieren/speichern/reload/apply, Pflichtfelder/ungültiges YAML, Modell-/Setup-/Full-/Split-/Generic-/Native-Auswahl, Budgets und resultierende Pläne; kleine Datasetimporte/Registrierung/Verify/Reindex | Positive Start/Resume/Rerun/Finalize-Controllerpfade und Packbuilder ausdrücklich ersetzt; kein echter Remote-Workflow |
| Benchmark | Prepare current ONNX, Generate N=1 CPU-only, Resume vollständig/ungültig/unterbrochen, Artefaktreuse, Cancel/Doppelklick, Harness/Bundle nur eigener Daten | Kein lokaler Benchmarkstart-Button vorhanden: GUI-CPU-Parität real; erzeugter regulärer CPU-Runner zusätzlich ausdrücklich als CLI geprüft |
| Benchmark-Analyse | Historische und neue CPU-Ergebnisse, leer/fehlend/unvollständig/ungültig, A/B/C, Filter, CAL, Tabellen/Diagramme, Toolbar, Report/Publication/ZIP | Fehlende Messwerte bleiben fehlend; keine neuen Hardwaremessungen |
| Jobs | Eigener fortschreitender Worker, Monitor schließen/wieder öffnen, Abbruch und Threadende, Dismiss, UI weiter bedienbar | Desktopöffner separat mit ersetztem Leaf geprüft |
| Tool Config | Acht Kategorien, Edit/Validate/Save/Reload/Apply, YAML-/Portfehler, persistierte Optionen, Artefaktbibliothek an eigenen Testartefakten | SSH-/SDK-/Energie-/Plattformantworten ausdrücklich ersetzt; reale Caches erhalten |
| Logs | Anzeigen, vollständige Ursache kopieren, echter Save-As mit Abbruch/Leerzeichen, bytegleicher Export, Cleanup ausschließlich eigener Logs | Keine Bereinigung produktiver Dateien |
| Normaler Neustart | Originales `./start_gui.sh` → MobileNet → Analyse → b027 → Split/CPU-ORT → TeX/SVG → normales Schließen → gleicher Starter und wiederhergestellte Einstellungen | Zwei Starts: 19,281 s und 20,972 s, Exit 0, keine Tk-Ausnahmen; nur isolierte Testeinstellungen gespeichert |

Die vollständige Tabelle mit **Bereich/Aktion, Eingabe, erwartetem und
tatsächlichem Ergebnis, Testart, Status und Beleg** liegt außerhalb des
Quellbaums unter:

```text
/home/kmika/Reports/gui_gesamttest_2955_20261115/
  FUNKTIONS_TESTTABELLE.md
  funktions_testtabelle.json
  funktions_testtabelle.csv
  inventory_review/                 # Quelleninventar und unabhängiger Abgleich
  analysis_exports/                 # normale Modell-/Export-/Fehlerbedienwege
  workflow/                         # Profile, Daten, Controllerabgrenzung
  benchmark/                        # Suite, Resume, CPU-Runner, Abbruch
  results/                          # historische/neue Ergebnisse und Exporte
  config_jobs_logs/                 # Konfiguration, Jobs, Logs, Neustart
  final_launcher/                   # tatsächlicher produktiver Launcher
```

Alle drei YOLO-Splits und der Klassifikatorschnitt bestanden die echte lokale
ORT-Paritätsprüfung. Beim abschließenden Originalstarter wurden zusätzlich
ONNX-Checker, Manifestschema, Full-Quellhash, Runner, PDF/SVG und TeX geprüft.
Der neu erzeugte CPU-Benchmark lief regulär mit einem Split, einem Durchlauf,
keinem Warmup und ohne Native-/TensorRT-Kaltbuilds. Parität: max_abs=mean_abs=0;
fehlende Full-Timingwerte wurden nicht als Nullmessung ausgegeben. Die Anzeige
und der Export dieser CPU-Ergebnisse wurden anschließend real in der GUI geprüft.

## Softwaretests und genaue Befehle

Entscheidender Sammellauf im Lieferworktree mit bestehendem Python, `-B`,
`PYTHONDONTWRITEBYTECODE=1`, `DISPLAY=:1002`, externen pytest-/Matplotlib-/Tmp-
Verzeichnissen: **149 PASS, 0 FAIL, 0 SKIP**. Produkt-/Testquellen waren vor
und nach dem Lauf bytegleich. Enthalten sind `test_v295_release.py`, alle drei
`test_v2955_full_*.py`, die sieben Callback-Splitfälle, sechs neue
`test_gui_acceptance_*.py`-Module, aktuelle GUI/CLI-/Profilintegration, große
CSV-Felder und gezielt passende bestehende Export-/GUI-/Budgetregressionen.

Exakter ausgeführter argv mit Umgebung und JUnit/stdout:
`inventory_review/final_integration.command.json`, `.junit.xml`, `.stdout.txt`.
Reproduzierbarer äußerer Prüfbefehl:

```bash
/home/kmika/ONNX-Splitpoint-Tool/.venv/bin/python -B \
  /home/kmika/Reports/gui_gesamttest_2955_20261115/inventory_review/run_final_checks.py
```

Die Originalfehler wurden zusätzlich anhand unveränderter Git-Quellen und
realer GUI-Fehlerbelege reproduziert. Alle Vorher-/Nachherlogs bleiben erhalten.
Ein früher breiterer Altregressionslauf ergab 94 PASS/9 FAIL: sechs fehlende
private historische Profilfixtures und drei auf 2.75.47 festgelegte Versions-
assertionen. Ein weiterer historischer Energietest hatte 49 PASS/23 FAIL gegen
eine bereits in der Ausgangsbasis entfernte Policy; ein anderer konnte eine
bereits fehlende alte Hilfsfunktion nicht importieren. Diese Ergebnisse werden
nicht in PASS umgedeutet und keine Energiepolicies dafür zurückgesetzt.
Details und Baselinevergleiche liegen in `inventory_review/integration_review.json`
und `config_jobs_logs/report.json`.

## Grenzen, erhaltene Daten und normaler Start

**BLOCKED:** Der Dataset-Root-Öffnen-Button wurde ausgeführt, seine externe
Dateimanageransicht jedoch nicht visuell verifiziert.
**NOT_RUN:** echte SSH-/SDK-/Vendor-/TensorRT-/Energie-/Plattformaktionen,
Power/Calibration, Kaltbuilds, Cacheindex/Cachelöschung, große Downloads,
Online-Kaggle-/Installationsschritte und COCO-Registrierung. Diese Aktionen
wären außerhalb des genehmigten lokalen Umfangs oder würden produktive
Caches/Daten verändern. Controller-/Fehleranbindungen wurden getrennt mit
ausdrücklichen Ersatzantworten getestet. Ein gültiger vollständiger Debug-Pack-
Build über die GUI wird nicht behauptet; Run-Admission wurde real und die
positive Builder-Anbindung mit Ersatz geprüft. Historische Berichte bleiben
unverändert. Es gab keine neue wissenschaftliche Qualitäts-/Energiekampagne.

Produktive Einstellungen wurden vor/nach den GUI-Prüfungen verglichen und
blieben identisch; HOME wurde nicht umgebogen. Die manuellen Grafikdateien des
vorigen Auftrags wurden erhalten und nicht neu gestaltet. Ungeprüfte MCP-
Arbeit bleibt im ursprünglichen Checkout; der veröffentlichte Quellstand
enthält ausschließlich geprüfte GUI-Reparaturen und notwendige Tests/Dokumente.
Quellenmanifest und Prüfsummen werden mit dem vorhandenen Projektbuilder
aktualisiert und vor Veröffentlichung in Release- und Installed-Scope geprüft.

Normaler produktiver Start:

```bash
cd /home/kmika/ONNX-Splitpoint-Tool
./start_gui.sh
```

Commit-/Pushnachweis und abschließende Status-/Diffdateien stehen im externen
Abschlussbericht desselben Evidence-Verzeichnisses. Es wird weder ein Versions-
tag erstellt noch bestehende Geschichte umgeschrieben; der veröffentlichte
Commit wird nach dem Push direkt am Remote geprüft.
