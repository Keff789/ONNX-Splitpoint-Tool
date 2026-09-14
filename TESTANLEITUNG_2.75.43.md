# Testanleitung 2.75.43

Release: `2.75.43`  
Build/Workflow: `v2.75.43-legacy-portable-dataset-identity-compatibility`

## Zweck

v2.75.43 ist ein enger Kompatibilitäts-Patch für das bereits vorhandene
ImageNet-Validation-Manifest aus der v2.75.40-Kampagne. Preflight,
Management-Runner und vendored Split-Runner akzeptieren neben der
normalisierten Item-Identität genau die historische vierfeldrige
Hashdarstellung, die auch `verify_dataset_manifest` bereits akzeptiert. Für die
portable, eingefrorene Validation-Identität wird weiterhin ausschließlich die
normalisierte Darstellung verwendet.

Der Patch schreibt weder Manifest noch Bilder um und lockert keine Datei-,
Payload-, Inventar-, B500-, B1000-, Disjunktheits- oder Frozen-Authority-Prüfung.
B1000 muss weiterhin exakt 1.000 Klassen und 1.000 Bilder mit Seed `20260710`
enthalten; der echte B500-Satz muss eine echte Teilmenge davon sein.

## Lokale Release-Prüfung

Aus dem Tool-Ordner:

```bash
cd /home/kmika/ONNX-Splitpoint-Tool
PY=/home/kmika/ONNX-Splitpoint-Tool/.venv/bin/python \
  bash scripts/run_v27543_small_acceptance.sh
```

Diese Prüfung startet keine Hardware, kein SSH, keine Inferenz, keinen
TensorRT-/DX-COM-Build und keine Energieerfassung.

## Wissenschaftlicher Preflight

Der bestehende B500-Pin bleibt unverändert. Anschließend wird derselbe
read-only Preflight erneut ausgeführt:

```bash
/home/kmika/ONNX-Splitpoint-Tool/.venv/bin/python -B \
  /home/kmika/ONNX-Splitpoint-Tool/scripts/preflight_v27541_deepx_calibration_1000.py \
  --calibration-manifest /home/kmika/.onnx_splitpoint_tool/final_datasets/v27541_imagenet_n1000_s20260710/manifests/imagenet_train_calibration_manifest.json \
  --validation-manifest /home/kmika/.onnx_splitpoint_tool/final_datasets/manifests/imagenet_val_manifest.json \
  --baseline-run /home/kmika/Models/EvaluationRuns/resnet50_v27540_deepx_preprocess_b_imagenet_mean_std_20260814_155031 \
  --baseline-calibration-manifest /home/kmika/.onnx_splitpoint_tool/final_datasets/v27541_imagenet_n500_s20260710/manifests/imagenet_train_calibration_manifest.json
```

Nur `status=ready`, `ready=true`, `dataset_only_ready=true` und Exit-Code `0`
geben den einzelnen B1000-Canary-Lauf frei. Der dreimodellige Final-Lauf bleibt
bis zur Auswertung dieses Canary-Laufs gesperrt.

Der normale Modus **Final Quality** bleibt der bestehende Standard-Pfad mit
5.000 Classification- und 5.000 Detection-Validation-Items. Er erzeugt
**keinen Pflicht-Canary**. Die historischen `run_v27528_*_final_canary.sh`-Werkzeuge sind
nur optionale Archivdiagnostik; sie ersetzen weder den B1000-Preflight noch den
einzelnen B1000-Canary dieser Übergabe.

Die v2.75.42-Harnesses, Smokes, Entry-Points, Guide und der Buildbericht bleiben
als historische Assets erhalten.
