# Testanleitung ONNX-Splitpoint-Tool 2.79.23

Build-ID: `v2.79.23-native-reuse-measurement-fixes`

Diese Wartungsversion korrigiert die durch den EvaluationRun mit 2.79.20
belegten Wiederverwendungs-, Laufzeit- und Messprobleme auf Basis von 2.79.22.
Die vorhandenen Artefaktidentitäten und Qualitätsgrenzen gelten weiter.
Es gibt keine YOLOv7-spezifische Änderung an Modell, Splitauswahl oder Export:
Im untersuchten Run wurde das falsche ONNX gewählt.

## Installation und Offline-Prüfung

Das vollständige Lieferpaket entpacken. Das Source-ZIP muss beim Installer
oder in `~/Downloads` liegen. Bei geschlossenem Tool ausführen:

```bash
bash install_v27923_and_collect_acceptance.sh
```

Der Installer aktualisiert `~/ONNX-Splitpoint-Tool`, erhält die vorhandene
`.venv` und benutzereigene Profile. Er startet keine Compiler und keine
Hardwareläufe. Das Ergebnisarchiv wird als `EVIDENCE_ZIP` ausgegeben.
Auch vorhandene lokale Verzeichnisse `artifact_store` und `build_evidence`
direkt unter dem Tool-Ordner bleiben erhalten. Die üblichen zentralen
Verzeichnisse unter `~/.onnx_splitpoint_tool` liegen außerhalb des Update-Ziels.

Die Offline-Prüfung kann später separat ausgeführt werden:

```bash
cd ~/ONNX-Splitpoint-Tool
bash scripts/run_v27923_small_acceptance.sh
```

Erwartete Schlusszeilen:

```text
PASS v2.79.23 smoke
PASS v2.79.23 small acceptance (real cache preflight and hardware EvalRun: NOT_RUN)
```

Die Prüfung umfasst die neuen Fehlerfälle und die vorhandenen Regressionen
für Cache-Publikation, Wiederverwendung und dauerhafte Negativ-Evidenz.
Temporäre Testdaten und simulierte Compiler ersetzen reale Gerätebefunde
nicht. Die konkreten Softwareergebnisse stehen im Lieferpaket.

## Begrenzte Geräteprüfung vor dem nächsten großen Run

1. Den vorhandenen TensorRT-Cache lesend prüfen. Ein passender historischer
   Eintrag muss seine Einzelprüfung erreichen. Ein Treffer benötigt weiterhin
   ein gültiges Artefakt, Receipt und passende Compiler-/Geräteidentität.
   Die Installation selbst startet keine Migration oder Retention.
2. Einen kurzen Native-Split-Kontrolllauf je betroffenem Pfad prüfen: Hailo-8
   mit vollständig übertragenem Runtime-Modul, Hailo-10H MobileNetV3 mit
   korrektem Boundary-Layout und Hailo-10H YOLO11l mit vollständiger Detection.
   Ursprüngliche Laufzeitfehler müssen im Bericht sichtbar bleiben.
3. Vor einem DeepX-Cold-Build die lokale Compilerumgebung prüfen. Ein bekannter
   CUDA-Architekturkonflikt ist ein Infrastrukturproblem und kein negativer
   Befund über den Split. Vorhandene kompatible DXNNs bleiben wiederverwendbar.
4. Eine kurze Full-System-Messung mit vorhandener Kalibrierung durchführen.
   Kalibrierbindung, tatsächlich abgedecktes Messfenster und Sample-Verluste
   getrennt prüfen. Ein korrigiertes Kalibrier-Gate hebt die Qualitätsgrenzen
   und Vollständigkeitsanforderungen der Messkampagne nicht auf.

Ein bekannter exakter Hailo-Buildbefund `COMPILE_INFEASIBLE` muss weiterhin
den erneuten Compilerstart verhindern. `TRANSIENT_INFRASTRUCTURE` ist kein
dauerhafter Ausschluss. Gelöschte Quellen oder Negativbefunde werden durch die
Installation nicht rekonstruiert.

Die gemessenen Qualitätsverluste von YOLO11l sowie ein unvollständiger
Detection-Endpunkt bei TensorRT-Full werden nicht durch gelockerte Grenzwerte
zu gültigen Vergleichsergebnissen erklärt. Für wissenschaftliche Freigaben
gelten die bisherigen Endpunkt-, Qualitäts- und Kampagnenanforderungen.

## Statusgrenzen

```text
REAL_TRT_LEGACY_REUSE=NOT_RUN
REAL_NATIVE_RUNTIME_CONTROLS=NOT_RUN
REAL_DEEPX_COMPILER_PREFLIGHT=NOT_RUN
REAL_FS_ENERGY_MEASUREMENT=NOT_RUN
REAL_MULTI_HOST_EVALRUN=NOT_RUN
```
