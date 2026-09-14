# Testanleitung v2.80

Version `2.80`, Build `v2.80-hailo-reuse-env-cleanup`. Der Plan mit Zielvorschlag
v2.79.35 wird auf ausdrücklichen Benutzerwunsch als v2.80 umgesetzt.
Historische Schemas und v34-Diagnosen behalten ihre Vertragsnamen.

## Installation und Softwaretests

GUI und laufende Tool-/Compiler-/Messjobs vorher regulär beenden. Als Besitzer der
vorhandenen Installation ohne sudo ausführen. Locks nicht löschen oder umgehen.

```bash
(
  set -e
  cd -- "$HOME/Downloads"
  unzip -n ONNX-Splitpoint-Tool_v2.80_COMPLETE_DELIVERY_BUNDLE.zip
  cd -- ONNX-Splitpoint-Tool_v2.80_COMPLETE_DELIVERY_BUNDLE
  bash ./install_v280_and_collect_acceptance.sh
  bash ./run_v280_short_tests.sh
)
```

Erfolg: `INSTALL_ACCEPTANCE=PASS` und `SHORT_TESTS=PASS`, jeweils mit
`EVIDENCE_ZIP=…`. Der Installer führt die vollständige ausgewählte Softwareabnahme
aus. Fehlende Pflichtabhängigkeiten blockieren die Abnahme; keine Pflicht-Skips
oder Xfails als Erfolg. Die Kurztests enthalten alle neuen v2.80-Module und die
fortgeführten Konfigurations-/Compiler-/Diagnoseregressionen.

Ohne erneute Installation:

```bash
(
  set -e
  cd -- "${ONNX_SPLITPOINT_TOOL_DIR:-$HOME/ONNX-Splitpoint-Tool}"
  bash scripts/run_v280_short_tests.sh
)
```

Das Update verwendet die bestehende Tool-Venv. Es installiert keine DFC-, CUDA-,
TensorFlow-, Torch-, DeepX- oder Treiberpakete. Eigene Profile, bestehende Registry,
Kalibrierungen, Ergebnisse und Caches werden nicht still umgeschrieben.

## Geltende Buildpolitik

Produktiver Start, Resume und Vorbereitung: Force AUS; passende Artefakte
wiederverwenden, fehlende oder inkompatible nach bestehender Policy bauen. Seit
v34 ist produktives Force AN gesperrt, auch wenn ein älterer Plan eine Bestätigung
vorsah. Das entspricht dem ausdrücklichen Auftrag im Chat. Ein altes eigenes
Profil mit Force AN blockiert vor Compilerdispatch. Die bewusste Profilkopie ist
die Korrekturmöglichkeit; das Update verändert keine Benutzerdateien dafür.

CPU/GPU-Präferenz ändert keine Artefaktidentität. Ein gültiger HIT benötigt weder
GPU-Smoke noch nutzbare GPU-Toolchain. Hailo bleibt balanced/Opt1/B500/Batch8 mit
relaxed-Policy, DeepX B500/EMA/Opt0 und Classification `imagenet_mean_std`.
Ein ausdrücklich privater Diagnose-Kaltbuild bleibt erlaubt und nichtpublizierend.
Sein HEF ist dadurch kein produktiv freigegebenes Cacheartefakt.

## Kurzer Wiederverwendungstest am vorhandenen Bestand

```bash
(
  set -e
  cd -- "${ONNX_SPLITPOINT_TOOL_DIR:-$HOME/ONNX-Splitpoint-Tool}"
  bash scripts/run_hailo_reuse_probe_v280.sh \
    --request "$HOME/Downloads/v27934_mobilenet_prepare_k14w3svb/request.json"
)
```

Erwartet `REUSE_PROBE_STATUS=two_fresh_process_cache_hits_pass`. Der Starter prüft
das normale bestehende CPU-Artefakt mit GPU-Präferenz in zwei frischen Prozessen.
Er darf kein SDK importieren und keinen Compiler-/GPU-Kindprozess starten.
Bei MISS oder unbekannter Identität bleibt der Test negativ, ohne automatisch zu
bauen. Ausgaben/Materialisierung bleiben im neuen privaten Diagnoseordner; kompakte
Evidence enthält keine Modelle oder Roharrays. Dies ist ein begrenzter Lookup-
und Neustartnachweis, keine vollständige GUI-/Quality-FIRST-Abnahme.

## Kurzer Hailo10-Compute-Gegentest

Seriell nach abgeschlossener Softwareabnahme:

```bash
(
  set -e
  cd -- "${ONNX_SPLITPOINT_TOOL_DIR:-$HOME/ONNX-Splitpoint-Tool}"
  bash scripts/run_hailo_gpu_compute_v280.sh --families hailo10h --gpu 0 --timeout 180
)
```

Erwartet: `GPU_SMOKE_STATUS=compute_pass`. Dies prüft die tatsächliche GPU und
Assembler/XLA-Umgebung; kein Modellbuild und keine Accuracyfreigabe.
Hailo8 separat durch denselben Starter mit `--families hailo8` prüfen, wenn die
bewusst ausgewählte H8-Venv bzw. ihr geprüftes Overlay bereitsteht. Ein H10-PASS
ist keine H8-Freigabe. Ein fehlendes Overlay wird nicht automatisch installiert.
Die Auswahlvariable wird nur für den Kindprozess angewendet; keine globale
LD_LIBRARY_PATH-/CUDA-Änderung vornehmen.

## Fixed16-Runtime mit vorhandenem privaten Build erneut prüfen

Der bereits tatsächlich erfolgreiche v34-H10-Modellbuild muss für den Cleanupfix
nicht neu kompiliert werden. Falls die folgenden vorhandenen Pfade unverändert
bestehen, kann der neue Collector dieselben Artefakte verwenden:

```bash
(
  set -e
  cd -- "${ONNX_SPLITPOINT_TOOL_DIR:-$HOME/ONNX-Splitpoint-Tool}"
  .venv/bin/python -B scripts/hailo_model_runtime_probe_v280.py --help
)
```

Die vollständige gebundene Zielsystemzeile steht in der mitgelieferten
`HARDWARE_GATES_2.80.md`. Der alte Einstieg `_v27934.py` und der neue `_v280.py`
führen dieselbe korrigierte Implementierung aus.

`remote_process_cleanup_complete=true` belegt beendete Worker. Zusätzlich muss
`remote_staging_cleanup.status=pass` die erfolgreiche Löschung samt Prüfung auf
Abwesenheit belegen. Nur beide gemeinsam ergeben `remote_cleanup_complete=true`.
Fehlende Supervision blockiert die Löschung; der eigene Restpfad bleibt im Bericht.
Ein Runtimefehler bleibt bei zusätzlichem Cleanupfehler der primäre Fehler.
RC0 nur bei Erfolg und vollständigem Cleanup, reguläre Fehler RC2, Abbruch RC130.
Compact Evidence enthält keine Modelle, Bilder oder Roharrays.

## Grenzen der Qualitäts- und Vorbereitungsaussage

Numerischer Vergleich angelieferter Arrays ist möglich. Automatische HAR-
Full-Precision-/Quantized-Emulation ist nicht implementiert. Fehlende Stufen heißen
`not_available`; eine frei gesetzte `origin`-Beschreibung beweist keinen Erzeuger.
Fixed16 ist eine geöffnete Entwicklungsdiagnose, kein Holdout und keine allgemeine
Accuracy-Äquivalenz. Kein MobileNet-/YOLO26-/DeepX-Qualitätsfix wird daraus abgeleitet.

Die Vorbereitungskopie endet nach dem ersten ausgewählten Modell:
`first_selected_model_only`, `quality_first_trt_binding_ready=False`. Das ist
keine volle Sieben-Modell-Nachtserie. Der festgelegte Modell-/Boundarybestand und
nötige Runtime-/Quality-FIRST-/TensorRT-Bindungen bleiben vollständig zu prüfen.
Die nächste Nacht erfordert zusätzlich echte reguläre Publikation und Neustart-HIT
am Zielsystem. Finalenergie FS/command und der festgelegte 60-s-×-3-Vertrag bleiben
getrennte Gates; dieser Release führt sie nicht aus.

## Fortgeführte technische und finale Gates

Lokale Hardwareausführung: `NOT_RUN`. Buildresultate unterscheiden weiterhin
`COMPILE_INFEASIBLE` (gültige negative Recipe-Evidence) und
`TRANSIENT_INFRASTRUCTURE` (retryfähiger Umgebungs-/Transportfehler); ein unbekannter
Zustand ist weder belegter Coldbuild noch Cache-HIT. Ein verletzter Artifact-/
Source-/Compilervertrag wird nicht durch einen erfolgreichen GPU-Smoke geheilt.

B500 bezeichnet 500 Kalibrationsbilder. Standard+ verwendet 5.000 Validierungsitems
pro Aufgabe und 5.000 Bootstrapwiederholungen; die Kurzdiagnose mit 16 Bildern
erfüllt dieses Qualitätsgate nicht. Die vereinbarten Energie-Stufen bleiben
1 s × 3 (kurze technische Probe), 30 s × 3 (Zwischenprüfung), 60 s × 3 (Finalvertrag),
mit FS/command als tatsächlichem Messbezug. Keine dieser Messungen läuft automatisch
durch Installation, Softwaretests, Computeprobe oder Fixed16-Runtime.
