# Installation und gezielte Abnahme 2.82

Build: `v2.82-selected-energy-generic-roles-workspace-product-evidence`

Die finale maschinenlesbare Software-/Upgradeprüfung liegt im Lieferbundle unter `VERIFICATION_V282.json`. Hardwareausführung in der Entwicklungsumgebung: **NOT_RUN**. Ein bestandener Installer ist keine Freigabe aller Native-Endpunkte.

## Installation und vorhandene Ergebnisse prüfen

GUI und laufende Workflows geordnet beenden. Das COMPLETE_DELIVERY_BUNDLE entpacken und in dessen Verzeichnis ausführen:

```bash
bash install_and_accept_v282.sh
```

Der Starter verwendet das mitgelieferte, anhand seiner festen Prüfsumme geprüfte SOURCE-ZIP und den bestehenden offiziellen Updater. Danach liest er den originalen Lauf `~/Models/EvaluationRuns/completsetdev_20260913_123411`, erzeugt getrennte Generic-/Energy-Replays und bereitet den gezielten Scope vor. Bei einem anderen Installations- oder Runpfad:

```bash
bash install_and_accept_v282.sh \
  --tool "$HOME/ONNX-Splitpoint-Tool" \
  --source-run "$HOME/Models/EvaluationRuns/completsetdev_20260913_123411" \
  --output-root "$HOME/Downloads"
```

Ein neues `v282_install_and_accept_…zip` enthält den gemeinsamen Abschluss einschließlich Fehlern. Fehlt der Originallauf, kann der Softwareupdate trotzdem abgeschlossen sein; das Gesamtergebnis nennt dann den fehlenden Replayschritt. Archivierte Kopien mit ehemaligen Absolutpfaden benötigen die ausdrücklich angegebenen Root-Abbildungen des separaten Energy-Reimporters. Der Starter rät keine Pfade.

Venvs, Registry, eigene Profile, Kalibrierungen und Caches bleiben erhalten. Es gibt kein implizites Paketupgrade, keine CUDA-Neuinstallation, keine Cachebereinigung und keinen Git-Push. Ein aktiver Workflow oder ein fehlerhafter Paketnachweis blockiert den Update weiterhin.

## Gezielte Zielsystemausführung

Der erste Aufruf startet keine neue Messung. Die vorbereitete `scope.json` und die Profile zeigen jeden konkreten Auftrag. Nach ausreichendem Platz auf dem tatsächlichen DFC-Arbeitsverzeichnis:

```bash
bash install_and_accept_v282.sh --skip-install --run-targeted
```

Diese Ausführung ist bewusst begrenzt:

| Block | Aufträge | Umfang |
|---|---|---|
| H8 Nachholung | YOLO11l b064, YOLO26m b040, YOLO26s b023 | Seriell; nach gescheitertem ersten Cold-Gate keine weiteren abhängigen Kaltbauten |
| H8 Dump-/Decoderprüfung | YOLOv7 (`yolov7_paper`) b009, b011, b044, b063 | Vorhandene HEFs/Engines; korrigierter normaler Native-Pfad |
| Erforderliche Full-Referenzen | Vendor-Full und TensorRT-Full der betroffenen Modelle | Im Scope ausdrücklich sichtbar; reguläre Wiederverwendungsregeln |
| H10 Endpunktklärung | YOLO26m b398 und YOLO26s b364 | Eigener begrenzter Diagnoseauftrag, keine Kompilierung oder Energie |
| Energy | Die beiden vorhandenen H8-Retryfälle | Nur vorhandene Aufnahmen importieren; keine erneute 115-Zeilen-Kampagne |

Die Standardfolgeprüfungen verwenden 500 Validierungsbilder, 500 Bootstrapwiederholungen und Native 1000 Frames / 100 Warmup / 3 Wiederholungen. Hailo bleibt balanced, Opt1, B500, Batch8; Force AUS, generische Energie AUS. Nach jedem erfolgreich nachgeholten H8-Fall prüft ein frischer Prozess denselben gespeicherten Request ausschließlich auf Cache-HIT, null Compilerstarts und identische HEF-/Cacheidentität. Ein inzwischen vorhandener HIT wird nicht künstlich in einen MISS umgewandelt. Die sechs bekannten `COMPILE_INFEASIBLE`-Fälle werden nicht erneut gebaut. RegNet b073 bleibt ein bereits erfolgreicher Build.

Für 640×640-Eingaben beträgt die vorhandene Schätzung `500 × 1 × 3 × 640 × 640 × 4 × 24 + 2 GiB = 61.129.883.648 Bytes` (56,93 GiB). Die frühe Prüfung und die erneute Prüfung vor Dispatch verwenden den tatsächlichen Auftragspfad. Ein anderer freier Datenträger hilft nur, wenn der DFC dort tatsächlich arbeitet. Platzmangel bleibt `TRANSIENT_INFRASTRUCTURE`, nicht Compile-Nichtrealisierbarkeit.

## Einzelwerkzeuge

Softwareprüfung ohne Hardware im installierten Quellverzeichnis. Der Bundle-Installer setzt die mitgelieferte Replayfixture automatisch; beim direkten Aufruf ihren entpackten Pfad angeben:

```bash
V282_REPLAY_NPZ=/pfad/zum/ONNX-Splitpoint-Tool_v2.82_COMPLETE_DELIVERY_BUNDLE/verification/fixtures/deepx_output_value_probe_outputs.npz \
  bash scripts/run_v282_small_acceptance.sh --report /tmp/v282_software.json
```

Kürzere Regression im installierten Tool: `bash scripts/run_v282_short_tests.sh`. Beide Starter verändern keine Hardwarekonfiguration.

Scope ausschließlich vorbereiten:

```bash
bash scripts/run_reference_workflow_v282.sh \
  --source-run "$HOME/Models/EvaluationRuns/completsetdev_20260913_123411"
```

Generic-Replay:

```bash
.venv/bin/python -I -B scripts/replay_generic_exclusions_v282.py \
  --run-root "$HOME/Models/EvaluationRuns/completsetdev_20260913_123411" \
  --out "$HOME/Downloads/v282_generic_replay"
```

Der Energy-Reimporter `scripts/replay_selected_energy_attempts_v282.py --help` erklärt die expliziten Checkpoint- und Root-Argumente. Nur eine vollständig geprüfte, eindeutig ausgewählte Aufnahme zählt. Verworfenes stdout, alte negative Importentscheidung, ursprüngliche Werte und Qualitätsentscheidungen bleiben erhalten.

H10-Diagnose separat:

```bash
.venv/bin/python -I -B scripts/hailo10_yolo26_boundary_probe_v282.py \
  --run-dir "$HOME/Models/EvaluationRuns/completsetdev_20260913_123411" \
  --output-dir "$HOME/Downloads" \
  --ssh nx@192.168.0.145 \
  --remote-python /home/nx/venvs/hailo10/bin/python
```

Falls konkrete lokale Artefaktwurzeln erforderlich sind, `--artifact-root /konkreter/pfad` ergänzen. `--plan-only` prüft zunächst die Bindungen. Standardmäßig ein Originalbild pro Fall; `--images-json` erlaubt ein vorab festgelegtes Fixed4-Set. Ein `capture_pass` beweist nur vollständige Erfassung. Ungültige Endpunkte bleiben technisch ungültig; keine Qualitäts-/Performancefreigabe aus einem Identity-Roundtrip.

## Ergebnisaussagen und Statusachsen

Gültige `quality_decision=fail` und `inconclusive` bleiben fachliche Ergebnisse, erscheinen als Warnung und behalten ihre bisherigen Claim-/Rankingbeschränkungen. Technische Teilprobleme ergeben `partial`; globaler Integritäts-/Exportfehler bleibt `failed`, Benutzerabbruch `cancelled`.

Die Originalprojektion hat weiterhin 126 Native-Zeilen = 115 Erfolge + 6 Ausschlüsse + 5 Blockaden und 42 erfolgreiche Full-Baselines. Qualität bleibt 81 PASS / 46 FAIL / 23 INCONCLUSIVE bei 150 abgeschlossenen Aufträgen (129 primär + 21 Begleitaufträge). Der Reporter erzeugt weder neue Inferenzen noch einen 5000-Bilder-Finalnachweis. Neue YOLOv7-Resultate mit geänderter expliziter Rechenvertragsversion ersetzen keine historischen Qualitätswerte ohne passende Neuberechnung.
