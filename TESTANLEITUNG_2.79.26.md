# Testanleitung 2.79.26

Build-ID: `v2.79.26-deepx-full-diagnostics-fix`

Basis ist die ausgelieferte 2.79.25. Diese kleine Folgeversion korrigiert
DeepX Full für die vorhandenen YOLO11-Ausgaben, die Fehlerbehandlung beendeter
Quality-Worker und fehlende Diagnosedateien im Debug-Pack. Die bisherigen
Quality-Grenzen bleiben unverändert; `INCONCLUSIVE` wird nicht zu PASS erklärt.

## Installation

Tool schließen, Lieferpaket nach `~/Downloads` herunterladen:

```bash
(
  set -e
  cd ~/Downloads
  unzip -n ONNX-Splitpoint-Tool_v2.79.26_COMPLETE_DELIVERY_BUNDLE.zip
  cd ONNX-Splitpoint-Tool_v2.79.26_COMPLETE_DELIVERY_BUNDLE
  bash install_v27926_and_collect_acceptance.sh
  bash start_deepx_gpu_canary.sh
)
```

Der Installer aktualisiert die vorhandene Installation `~/ONNX-Splitpoint-Tool`
über den bisherigen Updater. Er erhält `.venv`, eigene Profile, EvaluationRuns,
HEF-/DXNN-Bestände sowie `artifact_store` und `build_evidence`. Zentrale Bestände
außerhalb des Toolordners werden nicht bearbeitet. Der Installer führt keine
Compiler-Builds oder Geräteaktionen aus. Erwartet: `INSTALL_ACCEPTANCE=PASS`.

Der korrigierte Launcher startet die GUI als importierbares Modul. Das bestehende
cu126-Overlay wird weiterhin ausschließlich für DeepX-Compilerprozesse verwendet.
Er installiert keine Pakete oder Treiber. `--probe-only` startet keine GUI.

## Abnahme D: YOLO11l b003 DeepX GPU

Das mitgelieferte Profil `v27926_acceptance_D_YOLO11l_b003_DeepX_GPU.yaml` laden.
Vorhandene passende DXNNs und TRT-Engines müssen als Cache-HIT wiederverwendet
werden. Fehlende benötigte Artefakte dürfen gebaut werden. Exakte negative
`COMPILE_INFEASIBLE`- oder `PARSER_UNSUPPORTED`-Befunde bleiben wiederverwendbar;
`TRANSIENT_INFRASTRUCTURE` wird weiterhin nicht als dauerhafter Modellfehler
behandelt. Kein Cache wird für diese Abnahme gelöscht.

Prüfen:

1. DeepX Full akzeptiert den korrekt gebundenen `decoded_pre_nms`-Output und
   erkennt die YOLO11-Modellidentität. Kein `native_full_raw_detection_model_identity_missing`.
2. Full-Eingaben werden vorbereitet; Quality erreicht den tatsächlichen
   Vergleich mit der CPU-Referenz. Fehlende oder widersprüchliche Verträge
   müssen weiter verständlich scheitern.
3. Alle drei nativen Pfade (DeepX Full, DeepX→TRT, TRT Full) liefern ihre
   Performance- und FS-Energie-Wiederholungen. Keine generische Energiemessung.
4. Das Debug-Pack enthält Backend-Ergebnisse, Runnerlogs, Validation-Matrix,
   relevante Verträge und die angeforderten Energie-Rohdaten. Fehlende
   erforderliche Dateien ergeben einen sichtbar unvollständigen Pack.
5. Ein Quality-Workerfehler beendet den betroffenen Auftrag und anschließend
   den Lauf; er darf keine endlose Fortschrittsanzeige verursachen.

Danach zunächst A/B erneut abnehmen, erst anschließend einen begrenzten
Multimodell-Nachtlauf starten. 500 Bilder bleiben Screening; Quality und
technischer Abschluss bleiben getrennte Bewertungen.

Softwaregate erneut ausführen:

```bash
cd ~/ONNX-Splitpoint-Tool
bash scripts/run_v27926_small_acceptance.sh
```

```text
REAL_V27926_DEEPX_FULL_ACCEPTANCE=NOT_RUN
REAL_V27926_ACCURACY_RECOVERY=NOT_RUN
REAL_V27926_MULTI_MODEL_NIGHT_RUN=NOT_RUN
```
