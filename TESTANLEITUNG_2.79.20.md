# Testanleitung ONNX-Splitpoint-Tool 2.79.20

Build-ID: `v2.79.20-artifact-reuse-closure`

Diese Version schließt die Wiederverwendung teurer Hailo-, TensorRT- und
DeepX-Artefakte. Sie ändert weder Ranking noch Pipeline-FPS, Quality-
Schwellwerte oder Energiemessung und führt keine neue Hash-, Manifest-,
Signatur- oder Versiegelungsebene ein. Vorhandene Backend-Identitäten und
Receipts werden nur konsequent vor einem Neubau ausgewertet.

## 1. Offline-Acceptance

Im installierten Tool-Verzeichnis:

```bash
cd ~/ONNX-Splitpoint-Tool
bash scripts/run_v27920_small_acceptance.sh
```

Erwartung:

```text
PASS v2.79.20 smoke
PASS v2.79.20 small acceptance (real cache preflight and hardware EvalRun: NOT_RUN)
```

Der Lauf prüft ohne Hardware insbesondere:

- YOLO-Raw-Head-Hailo-Fallback: erster Aufruf baut, zweiter identischer Aufruf
  nutzt den Cache und startet keine DFC-Phase;
- TensorRT: ein gültiges Engine-Receipt wird auch bei erlaubtem Build vor
  `trtexec` wiederverwendet;
- TensorRT Full: modellgebundener, splitunabhängiger Pfad; Part1/Part2 bleiben
  splitgebunden;
- ein teilweise gefüllter stabiler Namespace sucht weiterhin passende
  Legacy-Receipts und übernimmt nur fehlende Artefakte;
- Retention schützt aktuelle/aktive Artefakte und projiziert Gründe für
  frühere oder aktuelle Evictions;
- Hailo-, TensorRT- und DeepX-Cacheentscheidungen liefern explizite
  HIT/MISS-Gründe;
- die Cache-Preflight-Matrix trennt `HIT`, `MISS`, `UNKNOWN` und
  `NOT_APPLICABLE` und zählt erwartete sowie unerwartete Cold Builds.
- `generate_benchmark_set` darf vor der Matrix keinen DeepX-Prefetch starten;
  `build_backend_artifacts` folgt erst nach der Preflight-Barriere. Ein
  strikter Warm-Cache-MISS blockiert damit DX-COM vor dem ersten Aufruf.

## 2. Identität prüfen

```bash
cd ~/ONNX-Splitpoint-Tool
PYTHONPATH= PYTHONHOME= .venv/bin/python -I -B - "$PWD" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

print("VERSION=" + package.__version__)
print("RELEASE=" + package.__release__)
print("BUILD_ID=" + package.__build_id__)
print("WORKFLOW_VERSION=" + WORKFLOW_VERSION)
PY
```

Erwartung:

```text
VERSION=2.79.20
RELEASE=2.79.20
BUILD_ID=v2.79.20-artifact-reuse-closure
WORKFLOW_VERSION=v2.79.20-artifact-reuse-closure
```

## 3. Begrenzter Cache-Reuse-Test vor dem Langlauf

Für den ersten realen Test nur zwei bekannte Modelle und zwei unterschiedliche
Splitpunkte desselben Modells auswählen. Noch keine mehrstündige
Energiemessung starten. Nach der BenchmarkSet-Erzeugung die Dateien
`artifact_cache_preflight.json`, `artifact_cache_preflight.csv` und
`artifact_cache_preflight.md` im Laufverzeichnis prüfen.

Die Matrix muss pro Modell diese Rollen ausweisen:

```text
Hailo-8 HEF | Hailo-10 HEF | DeepX | TensorRT Full | TensorRT Part2
```

Dabei gilt:

- `HIT`: das vorhandene Backend-Receipt beziehungsweise der vorhandene
  Backendvertrag wurde erfolgreich geprüft;
- `MISS`: ein Neubau ist tatsächlich erforderlich;
- `UNKNOWN`: der Backend-Cache konnte nicht belastbar geprüft werden; dieser
  Zustand darf nicht als HIT ausgegeben werden;
- `NOT_APPLICABLE`: die Rolle gehört nicht zum gewählten Modell/Setup.

Bei einem absichtlich warm erwarteten Bestand soll gelten:

```text
unexpected_cold_builds=0
```

Ein sichtbarer erwarteter MISS ist zulässig. Entscheidend ist, dass das Tool
ihn **vor** `build_backend_artifacts` und der langen Hardwarephase meldet und
nicht still vorher einen Compiler startet. Im normalen diagnostischen Modus
darf der deklarierte Build anschließend laufen; im strikten Warm-Cache-Modus
bleibt er blockiert.

## 4. TensorRT-Wiederverwendung auf dem Jetson prüfen

Im kurzen ersten Lauf darf für einen realen MISS ein Build erscheinen:

```text
[trt-cache] MISS role=... model=... case=... reason=...
[trt-cache] BUILD role=... model=... case=...
```

Beim unveränderten zweiten Lauf muss dieselbe Engine ohne `trtexec` geladen
werden:

```text
[trt-cache] HIT role=... model=... case=... reason=verified ...
```

Für Full gilt zusätzlich: ein Wechsel beispielsweise von `b132` zu `b045`
darf nicht allein wegen des Splitpunkts eine neue Full-Engine erzeugen.
Part1/Part2 dürfen dagegen splitabhängige Engines besitzen.

Wenn der aktuelle stabile Namespace nur teilweise gefüllt ist, muss die
Diagnose eine artefaktweise Legacy-Suche beziehungsweise Migration zeigen.
`existing_stable_namespace` allein ist kein gültiger Grund mehr, die Suche zu
überspringen.

## 5. Hailo und DeepX prüfen

Für einen bekannten YOLO11-Raw-Head-Fallback gilt:

1. Ein echter Cold Run darf `translate`, `calibrate` und `compile` ausführen.
2. Der identische zweite Lauf muss den content-addressed HEF-Cache verwenden.
3. Die semantische Raw-Head-Validierung bleibt erforderlich; Cache-HIT ersetzt
   keinen Quality-Nachweis.

DeepX behält das vorhandene Cache-System. Der Test erwartet lediglich klare
Zeilen wie:

```text
[deepx-cache] HIT role=full model=...
[deepx-cache] MISS role=part1 model=... reason=...
```

Ein DeepX-MISS ist kein Anlass für eine neue Cache-Identität. Full und Part1
müssen weiterhin im persistenten `BackendArtifacts/deepx`-Bestand liegen.

## 6. Retention und dauerhafte Heimat

Ein EvalRun-Verzeichnis darf nur Projektion, Hardlink oder Kopie eines teuren
Compilerartefakts enthalten. Die dauerhafte Heimat bleibt:

- HEF/HAR: persistenter Hailo-Cache;
- DXNN: `BackendArtifacts/deepx`;
- TensorRT: persistenter Remote-TRT-Cache.

Wird eine TensorRT-Engine wegen Retention entfernt, muss der Grund in den
Retentiondaten beziehungsweise als `[trt-cache] EVICT ... reason=...`
sichtbar sein. Aktuell benötigte, aktive oder verifiziert kompatible Engines
dürfen nicht still verschwinden.

## 7. Statusgrenzen

Die Offline-Acceptance führt keine DFC-, TensorRT-, DeepX-, u.RECS- oder
Jetson-Hardwareaktion aus. Deshalb bleiben bis zur Ausführung auf Smartmirror2:

```text
REAL_SEVEN_MODEL_CACHE_PREFLIGHT=NOT_RUN
REAL_HAILO_RAW_FALLBACK_REUSE=NOT_RUN
REAL_TRT_CROSS_SPLIT_FULL_REUSE=NOT_RUN
REAL_DEEPX_REUSE_REPLAY=NOT_RUN
REAL_MULTI_HOST_EVALRUN=NOT_RUN
```
