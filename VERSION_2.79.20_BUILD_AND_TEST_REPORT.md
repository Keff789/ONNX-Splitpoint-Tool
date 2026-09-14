# ONNX-Splitpoint-Tool 2.79.20 — Build- und Testbericht

Build-ID: `v2.79.20-artifact-reuse-closure`

## Anlass

Ein realer Lauf zeigte zwei getrennte Reuse-Defekte. Der YOLO11-Raw-Head-
Fallback übergab `force=True` und umging dadurch den normalen Hailo-Cache. Bei
TensorRT wurde eine bereits bekannte RegNet-Full-Engine am identischen Pfad
erneut mit `trtexec` gebaut. Zusätzlich konnte ein bereits existierender, aber
nur teilweise gefüllter stabiler Namespace die artefaktweise Legacy-Suche
unterdrücken.

Die Korrektur ist deshalb eine zusammenhängende Artifact-Reuse-Closure und
kein weiterer Einzelflick.

## Implementierte Änderungen

### Hailo

- Der YOLO-Raw-Head-Fallback ruft den vorhandenen content-addressed Builder
  mit `force=False` auf.
- Ein zweiter identischer Aufruf kann damit HEF/HAR wiederverwenden und muss
  weder Translation noch Kalibrierung oder Kompilierung starten.
- Die bestehende semantische Raw-Head-/Output-Validierung bleibt unverändert
  erforderlich.

### TensorRT

- Vor jedem erlaubten Build wird zuerst eine vorhandene Engine mit ihrem
  existierenden Receipt, Source-ONNX und Builder-/ABI-Vertrag geprüft.
- Ein gültiger Treffer wird geladen; `trtexec` läuft erst nach einem
  begründeten MISS.
- Full-Engines liegen in einem modellgebundenen und splitunabhängigen Bereich.
  Part1/Part2 bleiben an ihren Split gebunden.
- Ein teilweise gefüllter aktueller Namespace beendet die Legacy-Suche nicht
  mehr. Fehlende kompatible Leaves werden artefaktweise übernommen, ohne
  vorhandene aktuelle Leaves zu überschreiben.
- Retention schützt den aktuellen und aktiv verwendeten Bestand und führt
  Eviction-Gründe in der Diagnose mit.
- HIT, MISS, BUILD, Migration und Eviction werden mit Rolle, Modell, Case,
  Grund, Identity und Pfad soweit jeweils verfügbar sichtbar protokolliert.

### Cache-Preflight und DeepX

- Ein read-only Cache-Preflight normalisiert die vorhandenen Backend-Probes in
  eine Modellmatrix mit Hailo-8, Hailo-10, DeepX, TensorRT Full und TensorRT
  Part2.
- Der Bericht unterscheidet bestätigte MISSes von unbekannten/unprüfbaren
  Zuständen und zählt erwartete sowie unerwartete Cold Builds.
- Der Preflight erzeugt keine neue Artifact-Identity und keinen neuen Hash-
  Vertrag.
- Die Barriere liegt nach `generate_benchmark_set`, aber vor
  `build_backend_artifacts`. Der opportunistische DeepX-Prefetch der
  BenchmarkSet-Erzeugung wird bis hinter die Barriere deaktiviert; die
  Preflight-Prüfung verwendet ausschließlich `cache_verify_only` und kann
  DX-COM nicht dispatchen.
- Ein strikter Warm-Cache-MISS beendet den Ablauf vor dem Backend-Build. Im
  diagnostischen Modus läuft derselbe Build erst nach der protokollierten
  Matrix weiter.
- DeepX behält sein vorhandenes persistentes Cache-System; ergänzt wurden nur
  nachvollziehbare HIT/MISS-Gründe und Reuse-Regressionstests für Full und
  Part1.

### Unveränderte Grenzen

- Ranking, `cut_bytes_only`, Pipeline-FPS, Quality-Gates und Energy-Semantik
  bleiben unverändert.
- EvalRun-Verzeichnisse sind nicht die dauerhafte Master-Ablage teurer Builds.
- Es entsteht **keine neue Hash-, Manifest-, Signatur- oder Versiegelungsebene**.
  Vorhandene Backend-Receipts und Identitäten werden lediglich vor einem
  Neubau konsequent geprüft.

## Softwareprüfung

Die hardwareunabhängige Acceptance umfasst:

- zentrale Release-/Workflow-/Packaging-Identität;
- Hailo-Raw-Fallback Build→HIT ohne zweite DFC-Phase;
- TensorRT gleicher Vertrag Build→HIT ohne zweiten `trtexec`;
- TensorRT-Full-Reuse über zwei Splitpunkte;
- partielle stabile Namespace-Migration aus kompatiblem Legacy-Bestand;
- Retention- und Eviction-Diagnose;
- DeepX Full-/Part1-Reuse und Cache-Logging;
- Sieben-Modell-fähige Cache-Preflight-Aggregation einschließlich
  `unexpected_cold_builds=0` für eine vollständig warme Testmatrix;
- Source-Manifest, Python-Kompilation und Shell-Syntax.

Die tatsächlich ausgeführten Ergebnisse stehen im maschinenlesbaren Bericht
von `scripts/run_v27920_small_acceptance.sh` und im finalen Liefernachweis.
Ein Offline-PASS belegt Kontrollflüsse und Verträge, aber keinen realen
Compiler- oder Hardwaredurchsatz.

## Hardware- und Laufstatus

| Prüfung | Status im Build-Umfeld |
|---|---|
| Sieben-Modell-Cache-Preflight gegen Smartmirror2/Jetsons | `NOT_RUN` |
| Hailo YOLO11 Raw-Fallback Cold→Warm | `NOT_RUN` |
| TensorRT gleicher Vertrag ohne zweiten `trtexec` | `NOT_RUN` |
| TensorRT Full-Reuse über unterschiedliche Splits | `NOT_RUN` |
| DeepX Full-/Part1-Reuse auf echter Hardware | `NOT_RUN` |
| Vollständiger Multi-Host-EvaluationRun | `NOT_RUN` |

Die reale Freigabe erfolgt erst nach dem kurzen Cache-Reuse-Test aus
`TESTANLEITUNG_2.79.20.md`. Ein erwarteter Cold Build ist kein Fehler; ein
unerwarteter Cold Build oder ein stiller Rebuild eines validen Artefakts ist
vor dem nächsten Langlauf zu klären.
