# Testanleitung 2.79.30

Build: `v2.79.30-deepx-full-terminal-closure`. Abnahme gemäß Implementationsplan Revision 2, insgesamt **51 geplante Prüfgruppen**; dies ist keine behauptete Zahl bestandener Pytests.

## Software und Installation – H1

Die bestehende vollständige 29er-Abnahmeauswahl bleibt bis auf ihre feste alte Releaseidentitätsprüfung erhalten. Neue DeepX-/Merge-/Qualitäts-/Abschlussprüfungen und die ausdrücklich geforderten historischen Regressionen sind in `scripts/run_v27930_small_acceptance.sh` eingebunden. Die GUI-Backendprüfung wird in einem eigenen frischen Prozess ausgeführt, damit die Reihenfolge von Matplotlib-Imports keinen falschen Fehler erzeugt. Beide Prozesse müssen bestehen. JUnit-Dateien und JSON nennen ausgeführt, bestanden, Fehler und übersprungen getrennt; fehlende notwendige Abhängigkeiten sind keine Freigabe.

Nur bei geschlossener GUI und beendeten Messprozessen ausführen:

```bash
bash ./install_v27930_and_collect_acceptance.sh
```

Der Installer prüft das exakte Source-ZIP, verwendet den bestehenden Updater, erhält die vorhandene Venv und eigenen Profile und führt die Abnahme aus der aktualisierten Installation aus. Ein laufender oder pausierter Workflow behält seine bestehenden Locks; zusätzlich wird eine zur Installation gehörende laufende/pausierte GUI vor dem Quellenaustausch erkannt. Keine festgeschriebene PID und kein automatischer Prozessabbruch.

Maintainerprüfung vor Auslieferung: endgültiges Archiv frisch entpacken; ausschließlich daraus eine separate 29→30-Installation prüfen. Erhaltene Venv, Profil-/Konfigurations-Testbestände, DXNN/HEF/TensorRT-Testartefakte und negative Evidenz müssen übereinstimmen. Diese Testbestände sind synthetisch. `COMPILE_INFEASIBLE` bleibt negative Evidenz; `TRANSIENT_INFRASTRUCTURE` bleibt wiederholbare Infrastrukturabweichung.

## Lokaler Abschluss – H1b

```bash
bash ./run_terminal_closure_smoke_v27930.sh
```

Der tatsächlich installierte Runner erzeugt eigene synthetische Daten. `fast` und `strict` werden in getrennten Prozessen mit separater Cache-Fixture geprüft. Erwartet: null terminale Cache-Lese-/Schreibaufrufe, kanonische eindeutige Records, echte Pending-/Pass-Verifikation, sichtbare Abschlussphasen und genau ein finales Ereignis. Der Mutationsfall bleibt negativ, Pass wird widerrufen. Der Benutzer-Cache und alte Runs sind keine Testziele. Das Ergebnis zählt nicht als Hardware-/Energiemessung.

## Normaler Full-Pfad – H2

```bash
bash ./run_deepx_full_workflow_smoke_v27930.sh
```

Eine diagnostische Native-Full-Wiederholung von YOLO11l auf `orin_nx_deepx_m1_01`: 100 Frames und 10 Warmup-Frames, ohne Energie/B500-Neuberechnung/Compileraufruf. Grundlage ist der vorhandene lokale D29-Run; erforderliche Dateien werden in eigenes aktuelles Remote-Staging übertragen. Eine entfernte alte Remote-Runstruktur darf den Start nicht verhindern. Fehlt das echte Artefakt oder die gebundene Full-Eingabe, bricht der Launcher konkret ab. Ein anderes Bild oder Split-Tensor wird nicht eingesetzt.

Erwartet: physische Stage `decoded_pre_nms`, vollständige Frozen-NMS, gleiche nachgewiesene Full-Eingabe und gleicher Frozen-Vertrag, `requested=attempted=valid=1`. Der Remote-Normalworkflow hat 300 Sekunden plus 10 Sekunden zum Beenden; der eigene Runtime-Kindprozess 180 Sekunden plus 5 Sekunden zum Beenden. Transfers haben separat 300 Sekunden. Der Launcher beobachtet und beendet eigene Prozesse, bevor das Staging bereinigt werden darf. Details stehen in `--help` und Diagnosebericht. Die frühere 90-Sekunden-Einzelbildgrenze ist keine Zusage für diesen Normalworkflow einschließlich Übertragung.

H2-Hardwarestatus vor Ausführung auf dem Jetson: **NOT_RUN**. Der vollständige Qualitätsloader wird mit echten Export-/Kandidaten-Fixtures offline geprüft; daraus folgt keine neue B500-Hardwareentscheidung.

## Reguläres D-Profil – H3

```bash
bash ./start_deepx_gpu_canary.sh
```

Profil `acceptance_profiles/v27930_acceptance_D_YOLO11l_b003_DeepX_GPU.yaml` laden. Inhalt und gespeicherter Run-Mode-Snapshot erhalten die fachlichen Einstellungen aus 29:

| Bereich | Unveränderte Einstellung |
|---|---|
| Modell/Boundary/Setup | YOLO11l, b003, orin_nx_deepx_m1_01 |
| Native-Pfade | DeepX→TensorRT, DeepX Full, TensorRT Full |
| Native Performance | 3 Wiederholungen, 3.000 Frames, 100 Warmup |
| Native Energie | Full und Split, 3 Wiederholungen, 30 Sekunden, bestehende Zusatzprüfungen |
| Qualität | 500 Bilder, bestehende AP-Margen und Bootstrap-Konfiguration |
| Generic Energy / separate Validation | aus / summary_only |

Erwartete vorhandene Artefakte vor dem Lauf prüfen. Passende Artefakte wiederverwenden; einen unerwarteten Cold Build untersuchen. Das reguläre D-Profil erlaubt weiterhin fehlende Builds, ohne den Cache-Namespace zu ändern.

Ein Lauf ist erst beendet, wenn `finalize_processes` und `finalize_artifacts` abgeschlossen sind, Index und Closure-Report den technischen Abschluss belegen und die GUI nach Runner-Rückgabe den finalen Zustand zeigt. Der versiegelte Run-Log enthält keine nachträglich angehängten Heartbeats; späte Meldungen stehen im Live-Dialog/Parentlog. Der vorhandene Closure-Report ist der persistente Nachweis. `workflow_status=failed` kann mit korrekter `finalization_status=pass` zusammen auftreten.

Während der Versiegelung ist Cancel gemäß bestehender Schreibsperre nicht verfügbar. Ein Abschlussfehler darf nicht als Benutzerabbruch oder erfolgreiche Messung erscheinen. Den Run-Ordner samt Debug-Pack erhalten. H3-Hardwareabnahme vor dem neuen Lauf: **NOT_RUN**.

## Rückkehr zum vorigen Source-Stand

Bei Regression GUI und Messprozesse geordnet beenden, Fehlerbelege sichern und den bestehenden geprüften Installationsweg für 2.79.29 verwenden. Venv und Modellcaches erhalten. 29 bleibt ein Stand mit bekannten Integrations-/Abschlussfehlern; alte Ergebnisse nicht nachträglich auf PASS setzen.
