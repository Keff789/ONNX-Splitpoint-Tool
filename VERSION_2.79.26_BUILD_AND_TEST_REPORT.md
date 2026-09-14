# Build- und Testbericht 2.79.26

Build-ID: `v2.79.26-deepx-full-diagnostics-fix`

Basis: ausgelieferte 2.79.25, DeepX-Abnahme D vom 07.09.2026 und ihr gezielter
Nachexport. Die vorherigen Quellen und Ergebnisdateien bleiben erhalten.

## Begrenzte Korrekturen

- DeepX Full verwendet für `decoded_pre_nms` eine konsistente Output-Auslegung.
  Die bestehende YOLO11-Modellidentität wird bis zur Full-Eingabevorbereitung
  erhalten. Das vorhandene DXNN wird nicht allein wegen dieser Toolkorrektur
  neu kompiliert oder umgedeutet; Vertrags- und Inhaltsprüfungen bleiben erhalten.
- Quality-Aufträge werden bei Workerfehlern genau einmal als beendet verbucht.
  Ein Poolfehler lässt keine unerledigten Phantomaufträge in der Statusanzeige.
- Der Debug-Pack nimmt die erforderlichen vorhandenen kleinen Backend-Ergebnisse,
  Runnerlogs, Validierungsdetails und Verträge auf. Fehlende erforderliche Daten
  werden sichtbar als unvollständige Sammlung ausgewiesen. Der Export startet
  keine Compiler, SSH-Abfragen oder Messungen.
- Der separat korrigierte GPU-Launcher wird mitgeliefert und startet die GUI
  über ihr importierbares Modul. Multiprocessing muss keine `<stdin>`-Datei
  nachladen. Das vorhandene cu126-Overlay bleibt auf DeepX-Compiler beschränkt.
- Releaseidentität, generische Einstiegspunkte und Updater sind gemeinsam auf
  2.79.26 gesetzt. Historische Smoke-Einstiegspunkte und Dokumente bleiben erhalten.

Keine neue Cachehierarchie, Datenbank oder Pipelineebene. Die Quality-Toleranzen
bleiben unverändert. Die in der vorigen Abnahme gemessenen Qualitätswerte werden
weder geändert noch unter einer nachträglich gelockerten Regel als PASS ausgegeben.
`COMPILE_INFEASIBLE` und `PARSER_UNSUPPORTED` bleiben bei exakt passender Identität
wiederverwendbar; `TRANSIENT_INFRASTRUCTURE` bleibt wiederholbar.

## Softwareabnahme

Das finale Softwaregate und die isolierte Installation 2.79.25 → 2.79.26 sind in
`VERIFICATION_V27926.json` und `verification/` des Lieferpakets dokumentiert.
Der Installationstest verwendet eine eigene temporäre Toolinstallation. HEF,
Receipt, DXNN, negative Evidenz und ein benutzereigenes Profil müssen inhaltlich
erhalten bleiben; ebenso das bestehende `.venv`-Verzeichnis. Die exakte
Source-ZIP-Prüfsumme ist im mitgelieferten Installer fest gebunden.

Die GUI-Preflight-Regression läuft als unveränderter Test in einem separaten
frischen Python-Prozess. Dadurch kann ein früherer Headless-Reportimport den
Matplotlib-Backendwechsel nicht blockieren. Der anschließende gemeinsame Lauf
führt alle übrigen ausgewählten Tests aus; der GUI-Test wird dabei genau einmal
gezielt ausgenommen, weil er unmittelbar zuvor ausgeführt wurde. Beide Prozesse
müssen bestehen; ihre Testzahlen werden gemeinsam ausgewiesen.

## Noch ausstehend

Die Geräteabnahme muss bestätigen, dass DeepX Full die gesamte Quality-,
Performance- und Energie-Kette durchläuft. Offline-Reproduktionen mit echten
Diagnosemetadaten und kontrollierten Testdoubles ersetzen diese nicht.

Die zusätzliche Standard-TRT-Part1-Engine ist weiterhin nicht vollständig in
der Preflight-Matrix erfasst. Der Scope dieser Version umfasst diese bereits
dokumentierte Anzeigegrenze nicht. Reale Accuracyverluste werden nicht durch
Statusänderungen behoben; 500-Bilder-Abnahmen bleiben Screening.

```text
REAL_V27926_DEEPX_FULL_ACCEPTANCE=NOT_RUN
REAL_V27926_ACCURACY_RECOVERY=NOT_RUN
REAL_V27926_ENERGY_REFERENCE_VALIDATION=NOT_RUN
REAL_V27926_MULTI_MODEL_NIGHT_RUN=NOT_RUN
```
