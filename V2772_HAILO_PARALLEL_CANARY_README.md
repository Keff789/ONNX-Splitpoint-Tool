# Hinweis für v2.79.34

Der frühere Force-Canary-Starter ist stillgelegt und beendet sich vor jeder
Änderung mit `CANARY_RESULT=NOT_RUN` / `legacy_force_canary_retired`.
Normale Vorbereitung erfolgt mit Force AUS und aktivem Cache/Artifact Store.
Der historische Verifier bleibt ausschließlich zum Einlesen alter Canary-Evidence.
Die folgende frühere Anleitung beschreibt den damaligen Testvertrag; sie ist
keine aktuelle Startanweisung für v34.

---

# v2.77.2 Hailo-Parallelitätscanary

Dieser Canary ist ein enger technischer Rollouttest. Er prüft, ob das ONNX
Splitpoint Tool genau zwei unabhängige Compilerprozesse für denselben Split –
Hailo-8 und Hailo-10H – wirklich gleichzeitig ausführt und die beiden
Part-1-Artefakte sauber getrennt zusammenführt.

Der v2.77.1-Lauf hat den Parallelkern bereits positiv belegt (198,593211 s
echte Überlappung), aber zusätzlich zwei unerlaubte Full-Builds erzeugt. v2.77.2
schließt genau diese Scope-Lücke: `hailo_build.build_full: false` wird als
`full_hef_policy=skip` durch RunPlan, Generation Runtime und Orchestrierung
gereicht.

Der Canary erzeugt keine Performance-, Quality-, Ranking- oder
Energie-Evidenz. Er ist kein wissenschaftlicher Modelllauf und verändert den
eingefrorenen `cut_bytes_only`-Vertrag nicht.

## Exakter Umfang

- Tool: exakt v2.77.2
- Modell: `resnet50.onnx`
- Split: exakt `b052`
- Ziele: `hailo8_to_trt` und `hailo10_to_tensorrt`
- Builds: genau Hailo-8 Part 1 und Hailo-10H Part 1
- Cache und ArtifactStore: aus
- Runtime, Remote, Native, Quality, Ranking und Energie: aus
- Stop: direkt nach `build_backend_artifacts`
- Scheduler: zwei Worker, acht CPU-Tokens, 12.288 MiB RAM-Pool plus
  2.048 MiB Reserve

Ein serieller Fallback ist betriebssicher, gilt für diesen Canary aber als FAIL.

## Voraussetzungen

- `/home/kmika/Models/resnet50.onnx`
- der im Profil genannte ImageNet-Kalibrationsmanifestpfad
- beide verwalteten DFC-Venvs für Hailo-8 und Hailo-10H
- mindestens acht logische CPUs
- mindestens 14.336 MiB aktuell verfügbarer RAM

Die globalen Compiler-Caches und Venvs werden nicht gelöscht. Kälte und
Isolation entstehen durch `force_build: true`, deaktivierten Hailo-Cache,
deaktivierten ArtifactStore und einen neuen Output-Root.

## Start

Das vollständige v2.77.2-Quellpaket in ein neues Verzeichnis entpacken. Nicht
als Overlay über v2.77.1 kopieren. Danach:

```bash
cd /home/kmika/ONNX-Splitpoint-Tool_v2.77.2
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e .
bash scripts/run_v2772_hailo_parallel_build_canary.sh
```

Wenn eine bereits passend installierte Projekt-Venv verwendet wird, reichen:

```bash
cd /home/kmika/ONNX-Splitpoint-Tool_v2.77.2
PYTHON_BIN=.venv/bin/python \
bash scripts/run_v2772_hailo_parallel_build_canary.sh
```

Optionale Pfade:

```bash
MODELS_ROOT=/home/kmika/Models \
CANARY_BASE_ROOT=/home/kmika/Models/EvaluationRunsCanary \
PYTHON_BIN=.venv/bin/python \
bash scripts/run_v2772_hailo_parallel_build_canary.sh
```

## Harte PASS-Kriterien

- Der Preflight bestätigt v2.77.2, Profilvertrag, Modell, Manifest, beide
  verwalteten DFC-Venvs, Ressourcen und einen Full-freien projizierten RunPlan.
- Genau zwei Scheduler-Events existieren: Hailo-8 Part 1 und Hailo-10H Part 1,
  beide mit `status=ok`, je vier CPU-Tokens und 6.144 MiB RAM.
- Die beiden Prozessintervalle überlappen sich um mehr als eine Sekunde.
- Es gibt genau eine effektive Pair-Logzeile mit `backend_effective=venv` und
  `reason=resources_available`.
- Unter `b052/hailo/hailo8/part1` und `b052/hailo/hailo10/part1` liegt jeweils
  genau ein nichtleeres `compiled.hef`.
- Beide v2-Receipts binden Architektur, Größe, SHA-256 und Cache-Key korrekt;
  beide Build-Ergebnisse sind erfolgreich, ungekacht und nicht übersprungen.
- Es existiert kein Full- oder Part-2-HEF und es wurde keine Runtime-, Quality-
  oder Native-Stufe betreten.

Der beabsichtigte Stop direkt nach dem Build kann intern einen von null
verschiedenen Workflow-Rückgabecode erzeugen. Das ist allein kein FAIL; der
Verifier prüft stattdessen, ob wirklich keine spätere Stufe betreten wurde.

Der Launcher endet eindeutig mit `CANARY_RESULT=PASS` oder
`CANARY_RESULT=FAIL` und nennt `VERDICT_JSON` sowie `EVIDENCE_ZIP`. Die
Evidence-ZIP enthält diesmal auch beide kleinen Part-1-HEFs, die ausgeführten
Prüfskripte und ein SHA-256-Manifest. Für die Auswertung bitte die ZIP und die
letzten Terminalzeilen hochladen.
