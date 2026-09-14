# Testanleitung v2.79.34

Build: `v2.79.34-hailo-compiler-context-force-off`.

## Installation und Softwareabnahme

Der Bundle-Installer aktualisiert eine bestehende Installation über den vorhandenen
Updater. GUI und laufende Workflows vorher regulär beenden. Die bestehende Tool-Venv,
Vendor-Venvs, eigenen Profile, Registry, Caches, Kalibrierungen und Ergebnisse bleiben
erhalten. Es werden keine CUDA-, TensorFlow-, Torch- oder Hailo8-Pakete installiert.

```bash
bash install_v27934_and_collect_acceptance.sh
```

Die Softwareabnahme enthält die bisherigen verhaltensbezogenen v30–v33-Regressionen
und die neuen v34-Tests. Identitätstests beziehen sich auf v34; die ursprünglichen
v33-Tests und ihre Ergebnisse sind mit der Ausgangsbasis nachvollziehbar. Tests,
die einen produktiven Force-Start akzeptierten, werden an die ausdrücklich
geänderte Betriebsregel angepasst. Skips und Xfails sind kein Software-PASS.

## Normale Vorbereitung

Vorhandene passende Artefakte wiederverwenden, fehlende oder inkompatible bauen.
Hailo und DeepX Force AUS; Native-/TensorRT-Force ebenfalls AUS. CPU ist der
Default jeder DFC-Familie. Hailo10-GPU wird ausdrücklich familienbezogen ausgewählt;
das gibt Hailo8 nicht frei. Eine Computepräferenz allein invalidiert keinen
passenden vorhandenen HEF. Bei Cache-HIT erfolgt kein Compiler-GPU-Rechentest.

Hailo: relaxed, balanced, Opt1, Kalibrierung B500, Batch8. DeepX: B500, EMA, Opt0,
Klassifikation imagenet_mean_std; Detection unverändert. Final Quality verwendet
5.000 Validierungsbilder und 5.000 Qualitätsbootstrap-Wiederholungen, keine
Kalibrierung mit 5.000 Bildern. Ein explizites altes scale-only-Profil überschreibt
weiterhin den Modusdefault und muss bewusst korrigiert werden.

Eine neue Vorbereitungsprofilkopie wird mit
`scripts/create_preparation_profile_v27934.py` erzeugt; das Original bleibt erhalten.
Modellliste, tatsächliche Graphen und geplante Boundaries vor dem Lauf prüfen.
Der alte eigene v2771-Canary kann Cache und Artifact Store deaktiviert haben;
Force AUS allein repariert diese zusätzlichen Wiederverwendungshindernisse nicht.

## Hardwaregates und begrenzte Diagnosen

Die Softwareabnahme führt keine Hardwarejobs aus: **NOT_RUN**. Historisches
Hailo10-XLA-R2-PASS ist kein v34-Modellbuild. Der nächste reale Test ist ein
isolierter MobileNetV3-Full-Hailo10H-Build nach unverändertem CPU-Recipe und mit
16 vorab festgelegten Diagnosebildern. Der Diagnoseauftrag darf ausschließlich
in seinem privaten Arbeitsbaum schreiben. Ein privater HEF ist kein Beleg für
Publikation in den Produktivcache. Build, GPU-Ausführung, HEF-Lesbarkeit,
Runtime und Qualität werden getrennt ausgewiesen.

Die zugehörigen Einstiegspunkte sind
`scripts/hailo_model_build_probe_v27934.py` mit `prepare`, `validate` und `execute`
sowie `scripts/hailo_model_runtime_probe_v27934.py`. `prepare` bindet den vorhandenen
CPU-HEF samt Receipt, Original- und Compilergraph, die tatsächlichen 500
Kalibrationsdateien, 16 feste Bild-IDs und das ausdrücklich gewählte Zeitbudget.
`execute` liefert bei abgeschlossenem technischem Build RC0; das vollständige
G3-Gate bleibt bis zum nachfolgenden Runtimevergleich offen.
Der Runtime-Aufruf verwendet `--build-dir`, `--hardware-registry`, `--setup-id`
und `--remote-python`; `--plan-only` erstellt den Auftrag ohne Geräteausführung.
Er erfasst beide HEFs über den bestehenden Hailo-Backendpfad und berechnet
Original-/Compiler-ONNX mit CPU-ORT auf denselben tatsächlich erfassten logischen
Eingaben. Kompakte Evidence enthält keine Modelle, HARs, Bilder oder Roharrays.

Der vorhandene YOLO26-Bridge-Probe unterstützt zusätzlich `--images-json` für
vier vorab festgelegte Bilder und den Vergleich des gebundenen Part2-ONNX mit
TensorRT auf genau demselben Bridge-Eingang. Die aufgerufene CLI-Hilfe dokumentiert
die erforderlichen vorhandenen Artefaktpfade; es werden keine Pfade automatisch
geraten oder fehlende Modelle still ersetzt.

Danach folgt ein begrenzter normaler Build beziehungsweise Wiederverwendungslauf
mit der identischen Konfiguration. Die zweite Anfrage muss auch nach Prozessneustart
einen belegten HIT ohne Translate/Optimize/Compile ergeben. Ein erforderlicher erster
Mean/Std-Build darf einen alten scale-only-Bestand nicht als kompatibel ausgeben.

Die Hailo8-Ergänzung ist ein separater geprüfter Paketauftrag. Ein erzeugter Paketplan
ist weder eine Installation noch eine GPU-Freigabe. Bereits funktionierende
Hailo10-/DeepX-Frameworks werden dafür nicht ausgetauscht.

MobileNet/Hailo, YOLO26/Hailo10-Bridge sowie DeepX-Mean/Std/B500 bleiben eigene
Qualitätsgates. Fehlende HARs heißen not_available. Keine zusätzliche Normalisierung,
kein Blind-Clipping/Sigmoid/NMS, keine geänderten Qualitätsmargen. Ein korrekt
gemessenes negatives Qualitätsresultat bleibt negativ.

## Energie und Abschluss

Kurze Native-Integration 1 s × 3 beziehungsweise technische Zwischenabnahme
30 s × 3 ersetzt nicht den wissenschaftlichen Vertrag **FS/command, 60 s × 3**.
Generische/CPU-/ORT-Energie wird nicht aktiviert. Fehlende FPS oder CIs bleiben
fehlend; Rohenergie ist keine Quality- oder Claimfreigabe.

Das historische Replay muss 63 logische Fälle, 55 technische Erfolge, acht alte
Fehler und null zusätzliche Missing-Platzhalter erhalten. Terminal fast/strict
prüft weiterhin unabhängig die Abschlussintegrität; relaxed im Vorbereitungsprofil
schaltet diese Abschlusskontrolle nicht aus.

Exakte `PARSER_UNSUPPORTED`-/`COMPILE_INFEASIBLE`-Evidence bleibt wirksam.
GPU-/Loader-/Timeout-/OOM-/Abbruchfehler sind kein Beweis der Modellunrealisierbarkeit;
`TRANSIENT_INFRASTRUCTURE` und `ABORTED_UNKNOWN` bleiben getrennt und erneut versuchbar.
