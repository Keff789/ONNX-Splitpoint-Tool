# Testanleitung 2.75.42

Release: `2.75.42`  
Build/Workflow: `v2.75.42-real-evidence-and-quality-companion-repair`

## Zweck dieses Reparatur-Releases

v2.75.42 ändert keine wissenschaftliche Achse des ResNet50-B500/B1000-
Canarys. Es schließt vier reale Integrationsfehler aus v2.75.41:

1. Eine installierte Tool-Kopie darf eigene Top-Level-Evaluation-Profile
   enthalten. Der neue Installed-Tree-Gate attestiert weiterhin jede
   Release-Datei bytegenau, meldet eigene Profile separat und blockiert
   zusätzliche Skripte oder andere Source-Dateien.
2. Der B500-Pin akzeptiert die tatsächlich vom Produktions-Provisioner
   geschriebene, unveränderliche ImageNet-Kernel-Identität. Die ausgelieferte
   B500-Kohorte bleibt über Manifest-, Selection-Receipt- und Item-Hashes
   vollständig gebunden; sie wird nicht neu erzeugt.
3. Das Source-Update vergleicht Dateien mit `rsync --checksum`, sodass auch
   geänderte Dateien mit identischer Größe und Zeitmarke sicher ersetzt werden.
   Die vorhandene `.venv` bleibt erhalten; `refresh_editable_install.py`
   erneuert anschließend offline und nur mit der Python-Standardbibliothek die
   Distribution-Metadaten, den Editable-Pfad und alle deklarierten
   Kommandozeilen-Einstiegspunkte dieses Projekts. Andere installierte Pakete
   und Venv-Daten bleiben unangetastet.
4. Ein setup-lokaler TensorRT-Quality-Companion erhält genau eine konkrete
   physische Setup- und Full-Quality-Endpunktidentität im Effective Plan.
   Diese Bindung bleibt über Dispatch und Management-Admission erhalten;
   fehlt das geforderte Companion-Ergebnis, endet Standard oder Final
   fail-closed. Damit kann der widersprüchliche v2.75.41-Zustand »Companion
   angefordert, aber keine Quality-ID vorhanden« weder ungebundene
   Remote-Aufträge noch einen unvollständig zugelassenen Lauf erzeugen.

Der Nachtlauf `resnet_yolo26s_yolo7_20260815_040255` ist kein gültiger
Final-Quality-Beleg. Alle neun Remote-Aufträge stoppten vor kanonischen
Benchmarkzeilen am fehlenden TensorRT-Quality-Endpunkt. Er wird jetzt noch
nicht wiederholt: zuerst wird der kontrollierte B1000-Canary abgeschlossen.

## 1. Hardwarefreie Release-Abnahme

Im aktualisierten Tool-Verzeichnis:

```bash
cd /home/kmika/ONNX-Splitpoint-Tool
PY=/home/kmika/ONNX-Splitpoint-Tool/.venv/bin/python \
  bash scripts/run_v27542_small_acceptance.sh
```

Der Installed-Tree-Gate darf eigene reguläre `profiles/*.yaml` und
`profiles/*.yml` separat ausweisen. Fehlende oder geänderte Release-Dateien,
Symlinks, zusätzliche Skripte und sonstige Source-Extras bleiben blockierend.
Der Gate führt keine Hardware-, SSH-, DX-COM-, TensorRT-, Energy- oder
Inferenzarbeit aus.

## 2. Bewiesenes B500-Manifest unveränderlich pinnen

Der bestehende erfolgreiche v2.75.40-B-Run und dessen ursprüngliches Manifest
bleiben die Autorität:

```bash
cd /home/kmika/ONNX-Splitpoint-Tool

B500=/home/kmika/Models/EvaluationRuns/resnet50_v27540_deepx_preprocess_b_imagenet_mean_std_20260814_155031
SOURCE500=/home/kmika/.onnx_splitpoint_tool/final_datasets/manifests/imagenet_train_calibration_manifest.json
PIN500=/home/kmika/.onnx_splitpoint_tool/final_datasets/v27541_imagenet_n500_s20260710/manifests/imagenet_train_calibration_manifest.json

./.venv/bin/python scripts/pin_v27541_deepx_calibration_baseline.py \
  --baseline-run "$B500" \
  --source "$SOURCE500" \
  --out "$PIN500"
```

Das Werkzeug prüft beide echten Quality-Endpunkte, Request-/Result-Bindungen,
DXNN- und Cache-Verträge sowie das Produktions-Selection-Receipt. Ein bereits
vorhandenes identisches Ziel wird wiederverwendet; andere Bytes werden nie
überschrieben.

## 3. Exakte B1000-Kohorte isoliert bereitstellen

Nur falls das B1000-Manifest noch fehlt, mit vorhandenem Kaggle/ImageNet-
Zugriff ausführen:

```bash
cd /home/kmika/ONNX-Splitpoint-Tool

./.venv/bin/python scripts/provision_final_datasets.py \
  --registry /home/kmika/.onnx_splitpoint_tool/final_datasets/v27541_imagenet_n1000_s20260710/dataset_registry.json \
  provision-imagenet-kaggle \
  --root /home/kmika/.onnx_splitpoint_tool/final_datasets/v27541_imagenet_n1000_s20260710 \
  --accept-terms \
  --calibration-items 1000 \
  --seed 20260710
```

Den v2.75.40-Bestand und den vollständigen ImageNet-Train-Root nicht als
Ziel verwenden. Der dedizierte Root muss exakt 1.000 reguläre Bilder in 1.000
Klassen enthalten.

## 4. Read-only Preflight bis `ready=true`

```bash
cd /home/kmika/ONNX-Splitpoint-Tool

B500=/home/kmika/Models/EvaluationRuns/resnet50_v27540_deepx_preprocess_b_imagenet_mean_std_20260814_155031
PIN500=/home/kmika/.onnx_splitpoint_tool/final_datasets/v27541_imagenet_n500_s20260710/manifests/imagenet_train_calibration_manifest.json
CAL1000=/home/kmika/.onnx_splitpoint_tool/final_datasets/v27541_imagenet_n1000_s20260710/manifests/imagenet_train_calibration_manifest.json
VALMANIFEST=/home/kmika/.onnx_splitpoint_tool/final_datasets/manifests/imagenet_val_manifest.json

./.venv/bin/python scripts/preflight_v27541_deepx_calibration_1000.py \
  --calibration-manifest "$CAL1000" \
  --validation-manifest "$VALMANIFEST" \
  --baseline-run "$B500" \
  --baseline-calibration-manifest "$PIN500"
```

Nur `ready=true` autorisiert den Lauf. Der Preflight hasht alle Dateien,
verlangt exakte Inventare und beweist anhand vollständiger Fünf-Feld-Items
`B500 ⊂ B1000`. Bei `blocked` nichts starten; der Report ist read-only.

## 5. Genau einen B1000-Hardwarelauf starten

In **Evaluation Workflow** dieses bestehende wissenschaftliche Profil wählen:

```text
/home/kmika/ONNX-Splitpoint-Tool/profiles/resnet50_v27541_deepx_calibration_1000_imagenet_mean_std.yaml
```

Der Effective Plan muss Classification Calibration `1000`, Validation `500`,
Bootstrap `500`, ImageNet-Mean/Std, EMA, DeepX-Opt-Level 0, den isolierten
`v2.75.41/.../b1000_imagenet_mean_std`-Cache und genau DeepX Full plus die
setup-lokale TensorRT-Full-Kontrolle zeigen. Native Splits, Energy, Ranking und
Performance-Claims bleiben aus. Den abgeschlossenen Run-Pfad als `B1000`
notieren.

## 6. B500 gegen B1000 fail-closed verifizieren

```bash
cd /home/kmika/ONNX-Splitpoint-Tool

B500=/home/kmika/Models/EvaluationRuns/resnet50_v27540_deepx_preprocess_b_imagenet_mean_std_20260814_155031
B1000=/home/kmika/Models/EvaluationRuns/DEIN_ABGESCHLOSSENER_V27541_B1000_RUN
PIN500=/home/kmika/.onnx_splitpoint_tool/final_datasets/v27541_imagenet_n500_s20260710/manifests/imagenet_train_calibration_manifest.json
CAL1000=/home/kmika/.onnx_splitpoint_tool/final_datasets/v27541_imagenet_n1000_s20260710/manifests/imagenet_train_calibration_manifest.json

./.venv/bin/python scripts/verify_v27541_deepx_calibration_size_canary.py \
  --baseline-b500 "$B500" \
  --candidate-b1000 "$B1000" \
  --baseline-calibration-manifest "$PIN500" \
  --candidate-calibration-manifest "$CAL1000" \
  --out /home/kmika/Models/EvaluationRuns/resnet50_v27541_deepx_calibration_size_canary.json
```

`standard_plus_ready=true` erfordert vollständige technische Evidenz, beide
TensorRT-Passes, den offiziellen B1000-DeepX-Pass und den direkten
B1000-vs-TensorRT-Pass mit unveränderter Policy. Schwellen werden nicht
gelockert und die v2.75.40-A/B-Läufe werden nicht wiederholt.

## 7. Erst danach Final Quality

Nur bei `standard_plus_ready=true` folgt einmal der normale
**Final Quality (Standard+)**-Bestätigungslauf mit 5.000 Validierungsbildern
und 5.000 Bootstrap-Wiederholungen. Vorher den dreimodelligen Nachtlauf nicht
wiederholen. v2.75.42 verhindert nicht nur dessen leere
TensorRT-Companion-Planung vor dem Dispatch: Die materialisierte physische
Setup-/Endpunktidentität wird bis in die Management-Zulassung geprüft, und ein
fehlendes erforderliches Companion-Ergebnis blockiert Standard und Final.

Bei `fail`, `inconclusive` oder unvollständiger Provenienz bleibt Standard+
blockiert. Die historischen `run_v27528_*_final_canary.sh`-Werkzeuge bleiben
optionale Diagnostik; v2.75.42 erzeugt für andere Backends **keinen Pflicht-Canary**.
