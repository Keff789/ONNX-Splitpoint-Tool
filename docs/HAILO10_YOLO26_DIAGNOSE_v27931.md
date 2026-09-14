# Hailo10 / YOLO26 – begrenzte Rohdatenprobe in 2.79.31

Diese Probe untersucht **YOLO26m b398** und **YOLO26s b364** aus `complete_set_20260907_161614`. Ein erfolgreicher Diagnoseexport ist keine erfolgreiche Runtime- oder Quality-Abnahme. Die betroffenen regulären Jobs bleiben bei ungültigen Ausgaben gesperrt. Es gibt keinen nachgewiesenen Runtimefix und keine geänderten Quality-Grenzen.

In den ursprünglichen generischen Ergebnissen sind explizite Part2-Engines mit Cache-Hit und Prüfsummen belegt. Die späteren nativen Fehlerobjekte enthalten dagegen keine Engine. Der generische Vorlauf verwendete außerdem ein anderes Bild als der native Fehlerlauf. Die Probe nimmt pro Modell genau das in dessen **generischem `run_cfg.image`** deklarierte Originalbild; sie ersetzt dieses nicht durch das erste Datasetbild.

Nach Installation auf Smartmirror2, ohne gleichzeitig laufende Messung:

```bash
bash ~/ONNX-Splitpoint-Tool/scripts/run_hailo10_yolo26_boundary_probe_v27931.sh --plan-only
bash ~/ONNX-Splitpoint-Tool/scripts/run_hailo10_yolo26_boundary_probe_v27931.sh
```

Der erste Aufruf löst die vorhandenen lokalen Dateien auf und zeigt den Plan. Er startet kein SSH, keine Compilerprobe und keinen Build. Der zweite Aufruf führt die Diagnose aus und meldet `DIAGNOSTIC_ZIP=...`. Die gewöhnliche GUI muss dafür keine Quality-Bindung erzeugen. Eine laufende native/Full-/Suite-Messung auf dem Gerät blockiert die Probe; fremde Prozesse werden nicht beendet.

Standardwerte sind `~/Models/EvaluationRuns/complete_set_20260907_161614`, `nx@192.168.0.145`, Port 22 und `/home/nx/venvs/hailo10/bin/python`. Abweichende Orte lassen sich über `--run-dir`, `--artifact-root`, `--ssh`, `--port`, `--remote-python` und `--output-dir` angeben. `ONNX_SPLITPOINT_TOOL_DIR` wählt eine anders installierte Toolwurzel. Fehlende oder widersprüchliche lokale Artefakte führen zu einem konkreten Setupfehler, nie zu einem automatischen Neubau.

Die HEF-Datei, genau ein Originalbild pro Modell und aktueller Python-Code werden in ein eigenes temporäres Remoteverzeichnis kopiert. Part1-/Part2-ONNX-Referenzen bleiben lokal. Die ursprüngliche Part2-Engine, ihr Bridgegraph und ihr Receipt werden unter Prüfung der gespeicherten SHA256 aus dem persistenten Gerätecache in dieses Verzeichnis kopiert. Ein alter temporärer Remote-Run wird nicht benötigt. Fehlt der exakte persistente Cache, lautet das Ergebnis ein Setupfehler; vorhandene Modelle werden nicht neu gebaut.

| Stufe | Herkunft |
|---|---|
| A | Tatsächlich vorbereiteter Runtimeinput im aktuellen nativen Einzelbildpfad |
| B | Physischer HEF-Outputname und unveränderte Slotwerte; Speicherform vor dem Kopieren separat dokumentiert |
| C | Genau befüllter gepinnter TensorRT-Input nach der bestehenden Mapping-/Kopierroutine |
| D | **Berechneter** Zwischenoutput des durch Hash gebundenen Bridgegraphen unter CPU-ONNX-Runtime; kein direkt beobachteter TensorRT-Zwischenoutput |
| E | Rohausgabe der bestehenden TensorRT-Engine vor Hostkorrekturen |
| F | Lokaler Part1-/Part2-ONNX-Referenzlauf mit demselben deklarierten Originalbild und dazu passender Inputvorbereitung |

Die Probe dokumentiert die Vorbereitung von A ausdrücklich. Sie behauptet keine byteidentische Reproduktion eines nicht archivierten historischen generischen Tensors. Der Vergleich von A mit F, Originalvertrag, Laufzeit-QuantInfo, gebundenen Bridge-Konstanten, Namen und Layouts bleibt sichtbar. Gleiche Shapes oder gute Korrelation setzen keine Semantik auf PASS. Box-, Score- und Klassenkanäle werden getrennt geprüft. Es werden weder Scores beschnitten noch Boxen umsortiert oder Sigmoids ergänzt.

Beide Modelle laufen **nacheinander**. Pro Modell gelten 90 Sekunden für den überwachten Remoteprozess und 5 Sekunden TERM-Nachfrist; verbleibende eigene Kinder werden beendet und eingesammelt. Transfer, Remoteausführung und lokale Referenzrechnung werden getrennt protokolliert. Maximal 64 MiB Diagnoseoutput pro Paket; NPZ enthält ausschließlich numerische Arrays und wird mit `allow_pickle=False` geladen. Budgetüberschreitungen ergeben einen unvollständigen Export; Arrays werden nicht abgeschnitten. Fehlgeschlagene Sammlung/Timeout ist kein Probe-PASS. Das Remoteverzeichnis wird nur nach bestätigter Prozessbereinigung entfernt; sonst steht sein erhalten gebliebener Pfad im Sammelbericht.

Alle Pakete und Sammlerberichte tragen `diagnostic_only=true`, `counts_as_benchmark=false`, `claim_eligible=false`. Auch vollständig exportierte Werte bleiben `regular_path_released=false`. Erst ein belegter falscher Übergang erlaubt einen gezielten Fix. Dafür werden dieselben Originalinputs vorher/nachher und eine frische zentrale Qualityauswertung mit unveränderten Margen benötigt. Diese Hardwareprüfung ist bei Auslieferung **NOT_RUN**.

## Softwareprüfgruppen

`tests/test_v27931_hailo26_boundary_diagnostics.py` enthält synthetische Prozess-/Layout-/Quantisierungs- und echte ONNX-Bridge-/Exporter-/Loaderprüfungen. Die physische Hailo-/TensorRT-Engine wird ausschließlich in Tests durch klar bezeichnete Handles ersetzt.

| Gruppe | Prüfungen / verbleibende Grenze |
|---|---|
| T06.1 | Beide Originalfälle, alte Remote-Run-Pfade fehlen, exakte lokale Auflösung, nur HEF/Bild/Code im Staging, fehlende/mehrdeutige Artefakte scheitern |
| T06.2 | Direkte Diagnose ohne erfundene Binding; echter Workerpfad bis zum NPZ-Sammler erhält alle drei Diagnoseflags |
| T06.3 | Falsche Namen, gleichförmige transponierte Werte und abweichende Layoutdeklarationen; kein Shape-PASS |
| T06.4 | Skalar-/Kanalquantisierung, falsche Achse, bereits dequantisierter Float, doppelte Dequantisierung; tatsächlicher gebundener Bridgegraph liefert ausdrücklich berechnetes D |
| T06.5 | NaN/Inf in numerischem NPZ und JSON-null, Objectarray-Verbot, Größenlimit, Hashmanipulation, realer Prozess-Timeout mit Cleanup, Sammlerfehler, aktive Messung |
| T06.6 | Starke ungültige BN6-Werte bleiben durch echten Diagnoseexport/-import und den unveränderten Produkt-Endpunktprüfer ungültig. **Original-Hardware Vorher/Nachher + zentrale Quality nach einem erst nachzuweisenden Fix: NOT_RUN** |
