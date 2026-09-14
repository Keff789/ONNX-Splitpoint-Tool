# Gebundene Klassifikationsdiagnose für 2.79.31

Ein Einstieg, dieselben aktuellen Produktfunktionen, vorhandene Artefakte und ein eigenes temporäres Verzeichnis. Die R1-Werkzeuge sind hier integriert; die ausgelieferte Source wird beim Start ausdrücklich kopiert. Ein alter Suite-Codefallback findet nicht statt. Full-/Quality-Feeds und Outputs werden direkt am tatsächlichen Inferenzaufruf beobachtet. Der normale Mess-Hotloop erhält keine Dateiarbeit. Top-k, Labelmapping, Mean/Std-Adapter, Geometrie und Qualitätsgrenzen bleiben die vorhandenen Produktpfade.

Vom installierten Tool:

```bash
bash scripts/run_classification_input_probe_v27931.sh --run-dir /pfad/zum/run --models mobilenet_v3_large resnet50 regnet_x_1_6gf --stage contracts
bash scripts/run_classification_input_probe_v27931.sh --run-dir /pfad/zum/run --models mobilenet_v3_large --stage cpu-paired
bash scripts/run_classification_input_probe_v27931.sh --run-dir /pfad/zum/run --models mobilenet_v3_large --stage hardware-paired
```

`inputs` sammelt Produktionsfeeds, `calibration-audit` rekonstruiert die genau gebundene Loaderfolge in der vorhandenen Runtime, sofern deren OpenCV-Abhängigkeit verfügbar ist. Dies ist ausdrücklich **kein** beobachteter interner DXNN-Tensor und kein Nachweis damaliger Kalibrierung. `--plan-only` prüft nur Auflösung/Staging ohne CPU- oder Geräteausführung. Bei abweichender Tool-Venv darf `ONNX_SPLITPOINT_PYTHON` auf einen bereits vorhandenen Python zeigen. Es werden keine Pakete installiert.

Die CPU-Auswahl sind die ersten 32 geordneten IDs des vorhandenen Quality-Auftrags; Gerätepfade verwenden die ersten 16 derselben Liste. Dieser historische B500-Teilreplay bleibt Diagnose und dient keiner nachträglichen Optimierung der Abnahmestichprobe. Fehlende/mehrdeutige Artefakte, Bilder, Labels und Quellfunktionen führen zum Fehler. Fehlender Split wird gesondert ausgewiesen und blockiert die Full-Diagnose nicht. Splitkontrolle verwendet die gleichen 16 Bilder, den vorhandenen Part1 und Float-CPU-Tail. Zusätzlich führt `hardware-paired` die setup-lokale TensorRT-Full-Kontrolle automatisch aus, wenn der exakt gebundene Full-Quality-Producer und seine vorhandenen Cacheartefakte verfügbar sind. Die aktuelle Native-Full-Identitätsprüfung verifiziert Source/Build/Engine/trtexec und Buildreceipt; die verifizierte Engine wird in das eigene Staging kopiert. Ausgeführt werden die unveränderten NativeTRT-Methoden und die normale Native-Full-Eingabevorbereitung. Fehlende/inkonsistente TRT-Bindung bleibt separat sichtbar; es gibt weder Compileraufruf noch Engine-Suche nach ähnlichem Dateinamen.

Ein Modell-/Backendprozess hat höchstens 180 Sekunden und fünf Sekunden TERM-Nachfrist. Der bestehende Full-Workflow-Supervisor verfolgt auch Enkelprozesse mit eigener Session. Transfer und Collection sind getrennt begrenzt. CPU- und Geräte-Rohtensoren haben jeweils 40 MiB Budget, ausgewählte Originalbilder 32 MiB. Alte Runs, Vendor-Venvs und Modellcaches werden nicht verändert; ONNX/DXNN kommen nicht ins Ergebnis-ZIP. Outputbuffer werden unmittelbar kopiert, NaN/Inf als Fehler behandelt. Der festgelegte erste Bildfeed wird zusätzlich genau dreimal ausgeführt.

`collection_status`, `pipeline_consistency` und `hardware_executed` sind getrennte Aussagen. Ein vollständig gesammelter Mean/Std-Konflikt bleibt `pipeline_mismatch` mit Exitcode 2. Ein ZIP allein ist kein Pass. Kein Diagnoseergebnis zählt als Benchmark, B500-Abnahme, FPS- oder Energieergebnis. Die v31-Geräteabnahme bleibt bis zur Ausführung offen. R1/R2-Originale liegen ausschließlich als gekennzeichnete Regressionsfixtures vor.

YOLO11/YOLO26 sind getrennte Detectionkontrollen: ihre Frozen-NMS-/Geometrie- und Epsilonregressionen bleiben erhalten. Diese Klassifikationshülle fügt keine ImageNet-Normalisierung in Detection ein. Der reguläre Detection-Kurzlauf und vier echte YOLO26s-Bilder sind eigene Hardwareprüfungen.

Für H6a kann jede Stufe einzeln gewählt werden: `contracts` untersucht Identität und ONNX-Numerik ohne Inferenz, `inputs` führt tatsächliche Vorbereitung ohne ORT-Inferenz aus, `cpu-paired` führt die CPU-Arme aus. `--offline-only` unterdrückt jeden SSH-/Geräteaufruf; `--plan-only` prüft ausschließlich Auflösung/Staging. Der Artefaktbestand bleibt unverändert. Die optional-exakte TRT-Kontrolle gehört zu `--stage hardware-paired`; kein Zusatzbefehl ist nötig. Ergebnis: `trt_full_control.status=observed` oder konkreter Kontrollfehler, eigene `hardware_executed`-Angabe und Roharrays. Die ursprünglichen R1/R2-Tensoren und Original-TRT-Producer sind Regressionsfixtures, keine neue v31-Hardwareevidenz.

S6 verwendet denselben Einstieg für **vier feste YOLO26s-Full-Bilder**:

```bash
bash scripts/run_classification_input_probe_v27931.sh --run-dir /pfad/zum/run --models yolo26s --stage detection-control --plan-only
bash scripts/run_classification_input_probe_v27931.sh --run-dir /pfad/zum/run --models yolo26s --stage detection-control
```

Die Auswahl wird vor der Inferenz festgeschrieben: die ersten vier unterschiedlichen Seitenverhältnisse unter den ersten 32 IDs des Original-Quality-Auftrags. Alternativ friert `--detection-image-ids ID1 ID2 ID3 ID4` vier darin enthaltene IDs mit unterschiedlichen Seitenverhältnissen ein. Fehlende Bilder, nicht passende Artefakthashes oder unvollständige Eingabe-/Output-Verträge stoppen den Lauf. Es gibt genau vier Inferenzaufrufe und keinen Zusatz-Wiederholungslauf in S6. Die versiegelten RGB/HWC/uint8-Feeds werden bytegenau mit dem tatsächlichen Quality-Feed verglichen.

Jeder echte Output wird sofort kopiert und als NPZ behalten. Derselbe Output durchläuft den vorhandenen Full-Completion-Runtime und den tatsächlichen Produktionspfad bis `detections.json`. Bei Raw-Heads und decodierten Vor-NMS-Ausgaben ist der bestehende Decoder/NMS erforderlich; bereits als NMS-complete attestierte BN6-Ausgaben werden ohne zweite NMS materialisiert. Klassen, Scores, Boxen, inverse Geometrie und Recordreihenfolge müssen übereinstimmen. Pro Bild bleiben beide Completion-Verträge, Vorbereitungsaudits und Rohdaten im Diagnosepaket. NaN/Inf, ungebundene Integer-Ausgaben, geänderte Eingaben oder abweichende Records ergeben einen Fehlerstatus.

`pipeline_consistency=consistent_same_raw_output_scope` bestätigt ausschließlich diese vier Bilder und dieselben beobachteten Rohoutputs. Das ist keine COCO-AP-, B500-, Modell-, FPS- oder Energieabnahme. Die normale Quality-Regel und das geschützte YOLO11-Epsilon bleiben unverändert. Bis zur Ausführung auf dem Zielgerät ist S6 **Hardware NOT_RUN**; die Softwaretests verwenden ausdrücklich synthetische Bilder und Engine-Ausgaben.
