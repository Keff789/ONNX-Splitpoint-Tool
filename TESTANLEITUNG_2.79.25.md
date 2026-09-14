# Testanleitung 2.79.25

Build-ID: `v2.79.25-deepx-gpu-quality-followup`

Basis ist die ausgelieferte 2.79.24 und die beiden erneuten Abnahmen
MobileNet b027 / YOLO11l b003 vom 06.09.2026. Die Version korrigiert eng
begrenzte Compiler-Umgebungs- und Quality-Bindungspfade. Qualitätsgrenzen,
FS-Energiekalibrierung und die Native-Pipeline bleiben bestehen.

## Installation und Offline-Abnahme

Tool schließen, Lieferpaket nach `~/Downloads` herunterladen:

```bash
cd ~/Downloads
unzip -n ONNX-Splitpoint-Tool_v2.79.25_COMPLETE_DELIVERY_BUNDLE.zip
cd ONNX-Splitpoint-Tool_v2.79.25_COMPLETE_DELIVERY_BUNDLE
bash install_v27925_and_collect_acceptance.sh
```

Das Skript aktualisiert die vorhandene Installation `~/ONNX-Splitpoint-Tool`
über den bisherigen Updater. Bei abweichendem Installationsort `TOOL=/pfad`
voranstellen. Das exakte Source-ZIP wird geprüft; `.venv`, eigene Profile,
EvaluationRuns, tool-lokale `artifact_store` / `build_evidence` werden erhalten.
Zentrale Bestände außerhalb des Toolordners werden nicht bearbeitet. Der
Installer installiert keine Torch-, CUDA- oder Treiberpakete und startet
keinen echten Compiler oder Geräte-EvalRun.

Erwartet: `INSTALL_ACCEPTANCE=PASS`. Das ausgegebene `EVIDENCE_ZIP` enthält
Installationsidentität und Softwareabnahme. Wiederholung des Softwaregates:

```bash
cd ~/ONNX-Splitpoint-Tool
bash scripts/run_v27925_small_acceptance.sh
```

## Gezielte Geräteabnahmen

Die nachfolgenden Abnahmen dürfen fehlende, tatsächlich benötigte Artefakte
bauen. Der Preflight soll Modell, Boundary, Backend und Grund anzeigen;
ein Cache-Miss allein ist kein Anlass, den Lauf anzuhalten. Exakte bekannte
`COMPILE_INFEASIBLE`- bzw. `PARSER_UNSUPPORTED`-Befunde werden weiterhin
wiederverwendet. `TRANSIENT_INFRASTRUCTURE` bleibt wiederholbar. Alte
HEF-only-Generationen bleiben `legacy_unsealed` und werden nicht gelöscht.

1. MobileNet b027 und YOLO11l b003 erneut mit den korrigierten Profilen
   prüfen: vollständige CPU-/TRT-/Accelerator-Full- und Split-Quality-
   Projektion, ursprüngliche Oracle-Bindung und erfolgreicher Abschluss.
   Ein gemessener Accuracyverlust muss weiterhin als solcher erscheinen.
2. DeepX YOLO11l b003 in einer separat gestarteten GPU-Canary-Sitzung mit
   dem bereits getesteten cu126-Overlay ausführen. Der mitgelieferte
   Launcher begrenzt das Overlay auf DX-COM-Unterprozesse; keine globale
   `PYTHONPATH`-Änderung und kein neues Paketinstallieren. Ein erfolgreicher
   GPU-Matmul sowie DX-COM-Import beweisen noch keinen vollständigen Build.
3. Der echte DX-COM-Canary muss ein gültiges DXNN erzeugen und dessen
   Quality-/Native-/Energy-Folgepfad durchlaufen. CPU-Fallback nicht still
   einschalten. Fehlende DXNNs oder TRT-Varianten dürfen gezielt gebaut
   werden; vorhandene gültige Artefakte werden wiederverwendet.

Der bestätigte DeepX-Jetson-Zustand ist CPU 1984 MHz und GPU 1173 MHz mit
festen Sollgrenzen sowie EMC-Frequenzoverride. Nach einem Neustart prüfen
bzw. wieder anwenden. Thermisches/elektrisches Throttling bleibt möglich.

## Quality, Performance und Energie

Die 500-Bilder-Policy bleibt Screening. CPU Full, TRT Full, Accelerator Full
und Split müssen dieselben Eingaben vergleichen. Erst daraus lässt sich
zuordnen, ob ein Verlust bereits im quantisierten Full-Modell auftritt oder
am Split-/Boundary-/Postprocessing-Pfad entsteht. Kein technisch vollständiger
Lauf wird dadurch automatisch wissenschaftlich freigegeben.

Für Energie 30 Sekunden und drei Wiederholungen mit regulären Roh-Parquets
verwenden (`energy.include_raw_parquet_in_debug_pack: true` auf Profilebene). Ein repräsentativer 30-/60-s-Vergleich trennt den Einfluss des
Prozessstarts von der längeren Messung. FPS, echte Abschlusszähler und
Energie-Fensterbindung bleiben gemeinsam zu prüfen. Erst nach diesen kleinen
Abnahmen folgt der begrenzte Multimodell-Nachtlauf.

```text
REAL_DXCOM_GPU_BUILD=NOT_RUN
REAL_HARDWARE_ACCEPTANCE=NOT_RUN
REAL_ACCURACY_RECOVERY=NOT_RUN
REAL_30S_60S_ENERGY_COMPARISON=NOT_RUN
REAL_MULTI_MODEL_NIGHT_RUN=NOT_RUN
```
