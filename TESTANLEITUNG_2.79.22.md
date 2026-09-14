# Testanleitung ONNX-Splitpoint-Tool 2.79.22

Build-ID: `v2.79.22-global-negative-build-evidence`

Die vorhandene Negativ-Evidenz wird im gemeinsamen Hailo-Backend verwendet.
Damit gilt sie auch außerhalb der bisherigen Hailo-8-first/Gate-A-Auswahl,
insbesondere für normale EvaluationRuns und direkte Builds. Die Cache-Matrix,
`legacy_unsealed` und `selection_changed` aus 2.79.21 bleiben erhalten.

## Installation und Offline-Prüfung

Das vollständige Lieferpaket entpacken. Das Source-ZIP muss beim Installer
oder in `~/Downloads` liegen. Bei geschlossenem Tool ausführen:

```bash
bash install_v27922_and_collect_acceptance.sh
```

Der Installer aktualisiert `~/ONNX-Splitpoint-Tool`, erhält die vorhandene
`.venv` sowie benutzereigene Profile und prüft die Versionsidentität. Er führt
weder Compiler noch Hardwareläufe aus. Das Ergebnis wird als `EVIDENCE_ZIP`
ausgegeben. Der ursprüngliche Installationsfehler und gegebenenfalls Probleme
bei der Rücksetzung bleiben in den Diagnosen sichtbar.

Die gleiche Offline-Acceptance lässt sich später separat ausführen:

```bash
cd ~/ONNX-Splitpoint-Tool
bash scripts/run_v27922_small_acceptance.sh
```

Erwartete Schlusszeilen:

```text
PASS v2.79.22 smoke
PASS v2.79.22 small acceptance (real cache preflight and hardware EvalRun: NOT_RUN)
```

Die Acceptance prüft dauerhafte Evidenz, exakte Wiederverwendung, den normalen
Backend-Kontrollfluss, die beiden Recovery-/Klassifikationskorrekturen und die
bestehenden Cache-/Publikationsregressionen. Die Tests verwenden temporäre
Daten und simulierte Compiler; ein PASS belegt keine realen Gerätebefunde.

## Zentraler Bestand und Wiederverwendung

Neue Terminalbefunde werden standardmäßig hier gespeichert:

```text
~/.onnx_splitpoint_tool/build_evidence/live_build_evidence.json
```

Weitere kompatible JSON-Indizes im gleichen Ordner werden lesend konsultiert.
Dazu gehört ein vorhandener älterer Index wie `v2783_yolo11_gate_a.json`.
Die Umgebungsvariable `ONNX_SPLITPOINT_BUILD_EVIDENCE_ROOT` erlaubt einen
anderen Evidenzordner, beispielsweise für isolierte Tests. Sie ändert nicht
die Identität des Builds. Vorhandene historische Indizes werden nicht in-place
überschrieben. Diese Version führt keinen Cleanup vorhandener Evidenz aus.

| Gespeicherter Zustand | Wirkung bei exakt gleicher Identität |
|---|---|
| `ARTIFACT_PASS` | Positives Artefakt muss weiterhin die bestehenden Prüfungen bestehen. |
| `PARSER_UNSUPPORTED` | Der identische Parser-/Compilerfehlversuch wird übersprungen. |
| `COMPILE_INFEASIBLE` | Der identische als unmöglich belegte Build wird übersprungen. |
| `TRANSIENT_INFRASTRUCTURE` | Kein dauerhafter Negativtreffer; ein neuer Versuch bleibt möglich. |
| `ABORTED_UNKNOWN` | Kein dauerhafter Negativtreffer; ein neuer Versuch bleibt möglich. |

Modell-/Part-ONNX, Boundary, Endpoint, Hailo-Architektur, Compiler, Recipe,
Kalibrierung und Preprocessing müssen passen. `yolo26s/b364` allein reicht
nicht, um einen Build zu sperren. Geänderte relevante Identität darf einen
neuen Build auslösen. `force=True` übergeht einen exakten Negativbefund nicht,
weil auch interne automatische Wiederholungen dieses Flag verwenden.
Unbekannte Splits bleiben baubar. Ein fehlender Index wird nicht als Beweis
eines gescheiterten Splits behandelt.

Die neue allgemeine Einbindung betrifft die Hailo-Buildpfade. Sie behauptet
keine bereits bestehende globale Negativdatenbank für TensorRT oder DeepX.

## Die beiden Fehlerkorrekturen

Recovery akzeptiert die gültige atomare Hailo-Generation aus 2.79.21 als
zusammengehöriges HEF/Receipt/Cache-Meta-Paket. Die Veröffentlichung verwendet
Symlinks; der Importer darf deshalb nicht mehr allein deren Existenz mit
`unsafe_evidence_symlink` ablehnen. Er muss die vorgesehene Bundle-Struktur
prüfen und eine Generation zusammenhängend lesen. Beliebige externe Symlinks,
unvollständige Pakete und gemischte Generationen bleiben ungültig.

`CUDA memory allocation failed: out of memory` muss
`TRANSIENT_INFRASTRUCTURE` ergeben, auch wenn das Log zusätzlich eine
allgemeine Mapping-Fehlermeldung enthält. Ein echter `Agent infeasible`- oder
Mapping-Befund ohne widersprechenden Ressourcenfehler bleibt
`COMPILE_INFEASIBLE`.

## Prüfung auf dem Gerät nach der laufenden Rettung

Zuerst den vorhandenen Evidenzbestand außerhalb des Tools read-only
inventarisieren und gesondert sichern. Die Installation rekonstruiert keine
gelöschten Logs, Part-ONNXs oder Negativbefunde und startet keine automatische
Suche in alten Runs. Die Wiederherstellung historischer Einträge bleibt eine
separate Aufgabe mit nachvollziehbaren Originalquellen.

Ein späterer begrenzter Kontrolllauf muss für eine vorhandene exakt passende
Negatividentität den Treffer mit Modell, Boundary, Backend und Zustand
anzeigen; für diesen Versuch darf kein Compilerprozess starten. Bei einer
geänderten relevanten Identität muss die Entscheidung einen neuen Versuch
zulassen. Die endgültige Cache-Matrix muss einen bekannten negativen Befund
vom tatsächlich erwarteten Cold Build unterscheiden.

Vor der endgültigen Matrix prüft der Runner gespeicherte Entscheidungen
`known_infeasible` erneut gegen den aktuellen Bestand. Ein inzwischen
wiederhergestelltes gültiges HEF muss als `HIT` erscheinen. Hat sich hingegen
die relevante Compiler- oder Kalibrierungsidentität geändert, ist anhand des
aktuellen Prüfergebnisses `MISS` beziehungsweise `UNKNOWN` auszuweisen; die
alte Negativentscheidung darf nicht unverändert weitergelten. Ändert sich die
Entscheidung erst nach der Matrix, stoppt `cache_preflight_refresh_required`
die verspätete Cold-Build-Ausführung. Zuerst muss eine aktualisierte Matrix
vorliegen.

Windows-Aufrufe geben den tatsächlichen Quellmodellpfad und das Split-Manifest
weiter. Der Linux-Helfer prüft die tatsächlichen Quelldateibytes gegen die
übergebene Source-Identität; er erfindet keine vollständige Modellidentität aus
einer Part-Datei. Passen erhaltene Source-Hashangabe und inzwischen geänderte
Quelldateibytes nicht mehr zusammen, lautet das Ergebnis `CONFLICT`. Daraus
darf weder ein passender Negativtreffer noch ein regulärer Cold Build werden.

Diese Prüfung soll zuerst vorhandene Befunde nutzen. Es ist kein erneuter
stundenlanger Fehlbuild erforderlich, um den Negativ-Lookup zu testen. Ein
regulär benötigter neuer Build kann später zusätzlich prüfen, dass sein
Terminalbefund für den nächsten Run dauerhaft verfügbar ist.

## Statusgrenzen

Die tatsächlich ausgeführten Software- und Installationsergebnisse stehen in
den Protokollen des Lieferpakets. Die Geräte- und Rettungsprüfung ist dort
nicht enthalten:

```text
SMARTMIRROR_NEGATIVE_EVIDENCE_INVENTORY=NOT_RUN
REAL_HAILO_NEGATIVE_REUSE=NOT_RUN
REAL_HAILO_TERMINAL_RECORDING=NOT_RUN
REAL_MULTI_HOST_EVALRUN=NOT_RUN
```
