# ONNX Splitpoint Tool v2.79.32 – gestufte Abnahme

Build-/Workflow-ID: `v2.79.32-native-full-failure-closure`.

Diese Source beschreibt die Softwareänderungen F1–F3. Maßgeblich für den tatsächlich geprüften Lieferstand sind die externen Prüfberichte und JUnitdateien des zugehörigen v32-Bundles. Diese Anleitung enthält keine vorweggenommene Ausführungsfreigabe.

## H32-0: Software und Update

GUI, laufende und pausierte Workflows vor dem Update beenden. Der Bundleinstaller verwendet die vorhandene Tool-Venv und den überprüfenden Source-Updater; Venv, Compiler, eigene Profile, Konfiguration, Kalibrierungen, Ergebnisse und Caches bleiben erhalten. Keine Pakete, CUDA-, Torch- oder Vendorinstallationen aktualisieren. Fehlende Pflichtabhängigkeiten ergeben `environment_blocked`; kein Skip-/Xfail-PASS. Keine Locks löschen.

Reguläres Softwaregate aus dem entpackten Source oder der installierten v32:

```bash
bash scripts/run_v27932_small_acceptance.sh --report /tmp/v27932-acceptance.json
```

`PY` kann ausdrücklich auf die vorhandene Test-/Tool-Venv zeigen. Der Bericht und beide JUnitdateien müssen außerhalb des Sourcebaums liegen. Das Gate umfasst alle bisherigen v31/v30-Verhaltensregressionen sowie die fünf neuen v32-Suiten. Nur die bisherige fest auf v31 prüfende Releaseidentität ist durch die vollständige v32-Releaseprüfung ersetzt. Fehlende Abhängigkeiten blockieren vor den Tests; jeder Pflichtskip oder Xfail blockiert die Freigabe.

Schlägt die Acceptance nach erfolgreichem Sourceupdate fehl, kann bereits v32 installiert sein. `FINAL_STAGE`, tatsächlich installierte Identität und vollständiges Installerlog prüfen. Nach Fehl-Exitcode keinen Hardware-Smoke ausführen. Ein erforderlicher Rückweg verwendet nach Prozessende den überprüften Original-v31-Source und dessen bestehenden Updater; Modelle und historische Ergebnisse nicht zurücksetzen.

## H32-1: Terminal und historischer Replay

Aus dem v32-Lieferordner `run_terminal_closure_smoke_v27932.sh` und `run_complete_set_replay_v27932.sh` aufrufen. Der Terminaltest verwendet die vorhandene produktive Finalisierung für `fast` und `strict`, jeweils normal und mit absichtlicher später Manipulation. Der historische Replay muss 63 vorhandene Fälle, 55 Erfolge, 8 Fehler und 0 zusätzliche Missing-Zeilen sowie 55 Energiezeilen / 165 gültige Replikate erhalten. Das ist neue Softwareprojektion historischer Originalevidence, keine neue Hardwaremessung.

## H32-2 bis H32-6: kurze Hardwaregates

Status dieser Source: `NOT_RUN` auf Zielhardware. Erst nach H32-0/H32-1:

- DeepX YOLO11l Full über `run_deepx_full_workflow_smoke_v27932.sh`: 100 Frames, zehn Warmups, eine Wiederholung, regulärer Fullrunner, vorhandenes passendes DXNN.
- Produktives `imagenet_mean_std` mit den festgelegten 16 R1-Bild-IDs über normale Profile: MobileNetV3, ResNet50, RegNet-X. Passende Caches dürfen wiederverwendet werden; scale-only bleibt unpassend. Anschließend MobileNet zuerst mit frischem gepaartem B500 prüfen.
- Bestehende begrenzte Hailo8- und DeepX-Splits nach den mitgelieferten Integrationsprofilen ausführen. `COMPILE_INFEASIBLE` bleibt ein erklärt negativer Fall; `TRANSIENT_INFRASTRUCTURE` bleibt gesondert und gegebenenfalls wiederholbar.
- Hailo10/YOLO26 AP6b bleibt offen: `run_hailo10_yolo26_boundary_probe_v27932.sh --plan-only` und danach kontrollierte Rohprobe. Keine pauschale Layout-/Quantisierungskorrektur ohne belegt fehlerhaften Übergang. Gesammelte Diagnosedaten bedeuten keine Qualityfreigabe.
- H7 mit 1 s × 3 und D/H3/H4/H6c mit 30 s × 3 sind Integrationsprofile. Ihr Messumfang ist kein finaler Energievertrag.

## H32-7: wissenschaftlicher Finalumfang

Eigener vorab festgelegter Finalscope: Native Energy `measure`, FS/command, 60 s × 3, passende FS-Gain-/Idlebelege und bindungsgültige Endpunkte. B500 bleibt Screening-/Diagnoseumfang; die vereinbarte 5.000-Bilder-/5.000-Bootstrap-Auswertung wird getrennt festgelegt. Qualitygrenzen, Ranking und Referenzpolicy bleiben unverändert. Software-Closure, technische Hardwareabnahme und wissenschaftliche Freigabe sind getrennte Entscheidungen.

Nur Ergebnis-/Nachweisdateien direkt von Smartmirror2 ins vorhandene Ergebnis-Git sichern. Keine Modelle, Gewichte, Images, Roh-Tensoren, Venvs oder Delivery-ZIPs ins Ergebnisrepository.
