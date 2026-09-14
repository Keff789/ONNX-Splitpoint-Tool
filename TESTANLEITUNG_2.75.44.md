# Testanleitung 2.75.44

Release: `2.75.44`  
Build/Workflow: `v2.75.44-runtime-validation-cohort-projection`

## Zweck

v2.75.44 korrigiert ausschließlich die Validation-Authority-Prüfung des
ResNet50-DeepX-B1000-Preflights. Das vorhandene Quellmanifest beschreibt alle
50.000 ImageNet-Validation-Bilder. Die eingefrorene Laufzeitautorität beschreibt
dagegen die deterministisch daraus ausgewählte 500-Bilder-Kohorte. Der Preflight
prüft weiterhin das vollständige Quellmanifest, projiziert anschließend mit dem
eingefrorenen Auswahlvertrag die exakte Laufzeitkohorte und vergleicht erst
diese Projektion mit der 500-Bilder-Autorität.

Der Patch verändert weder Manifest noch Bilddateien, B500-Pin, B1000-Auswahl,
Seed, Calibration-/Validation-Trennung, Preprocessing, Quality-Schwellen oder
Hardwareverfahren. Der Ranker und der spätere Sieben-Modell-Abschlussplan sind
nicht betroffen.

## Lokale Release-Prüfung

Aus dem Tool-Ordner:

```bash
cd /home/kmika/ONNX-Splitpoint-Tool
PY=/home/kmika/ONNX-Splitpoint-Tool/.venv/bin/python \
  bash scripts/run_v27544_small_acceptance.sh
```

Diese Prüfung startet keine Hardware, kein SSH, keine Inferenz, keinen
TensorRT-/DX-COM-Build und keine Energieerfassung.

## Wissenschaftlicher B1000-Preflight

Der vorhandene B500-Pin und das unveränderte 50.000-Bilder-Quellmanifest werden
weiterverwendet:

```bash
/home/kmika/ONNX-Splitpoint-Tool/.venv/bin/python -B \
  /home/kmika/ONNX-Splitpoint-Tool/scripts/preflight_v27541_deepx_calibration_1000.py \
  --calibration-manifest /home/kmika/.onnx_splitpoint_tool/final_datasets/v27541_imagenet_n1000_s20260710/manifests/imagenet_train_calibration_manifest.json \
  --validation-manifest /home/kmika/.onnx_splitpoint_tool/final_datasets/manifests/imagenet_val_manifest.json \
  --baseline-run /home/kmika/Models/EvaluationRuns/resnet50_v27540_deepx_preprocess_b_imagenet_mean_std_20260814_155031 \
  --baseline-calibration-manifest /home/kmika/.onnx_splitpoint_tool/final_datasets/v27541_imagenet_n500_s20260710/manifests/imagenet_train_calibration_manifest.json
```

Nur `status=ready`, `ready=true`, `dataset_only_ready=true` und Exit-Code `0`
geben genau einen B1000-Canary-Lauf frei. Ein Ergebnis `blocked` startet keine
Hardware und darf nicht durch manuelle Manifeständerungen umgangen werden.

Nach erfolgreichem Preflight wird ausschließlich das unveränderte Profil
`profiles/resnet50_v27541_deepx_calibration_1000_imagenet_mean_std.yaml`
ausgeführt. B500 wird nicht erneut vermessen. Anschließend entscheidet der
vorab festgelegte Paarvergleich B500 gegen B1000; danach ist diese
Kalibrierungsfrage abgeschlossen.

Die v2.75.43-Smokes, Harnesses, Entry-Points, Anleitung und der Buildbericht
bleiben als historische Assets erhalten.

## Unveränderter allgemeiner Final-Pfad

Der normale Modus **Final Quality** bleibt der bestehende Standard-Pfad mit
5.000 Classification- und 5.000 Detection-Validation-Items. v2.75.44 ändert
daran nichts und erzeugt dafür **keinen Pflicht-Canary**. Die historischen
`run_v27528_*_final_canary.sh`-Werkzeuge bleiben optionale Archivdiagnostik und
sind kein Ersatz für den hier beschriebenen einzelnen B1000-Canary.
