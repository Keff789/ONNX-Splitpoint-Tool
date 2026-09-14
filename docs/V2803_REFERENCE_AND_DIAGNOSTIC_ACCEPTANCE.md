# v2.80.3 – kurze Referenz-/Native-Abnahme und getrennte Diagnosen

Der vorbereitete Auftrag ist ein neuer normaler Workflow mit **MobileNetV3 b056
und YOLO11l b062 auf Hailo10H / TensorRT**. Die Grenzen stehen vor den neuen
Qualitätsergebnissen fest. Es werden weder b064 als Kaltbuild gewählt noch die
besten neuen Qualitätsergebnisse nachträglich selektiert. Alte Läufe bleiben
unverändert. Die Hailo8-GPU-Konfiguration wird nicht umgeschaltet.

Die Profilkopie hat einen eigenen **16-Bilder-Validierungsvertrag und 100
Bootstrapwiederholungen**. Ihre 16er-Referenz ist keine B500-/5.000er-Referenz.
Die Compilerrezepte bleiben Hailo B500/balanced/Opt1/Batch8 und DeepX
B500/EMA/Opt0/MeanStd. Native-Energiemessung und Generic-Energiemessung sind für diese kurze
Profilkopie ausgeschaltet; Native-Performance/Qualitätsbindung bleibt aktiv.
Die vorhandenen Native-Frame-/Replikatwerte bleiben erhalten. Die ursprünglichen
Benutzerprofile und die zentrale Modusregistry werden nicht verändert.

**Keine alternative Referenzimplementierung:** Der Helfer schreibt eine gebundene
Profilkopie. Der danach gestartete Standard-Workflow verwendet den produktiven
Generator, Managementlauncher, Suiteprozess, ONNX Runtime CPU und zentralen
Consumer. Das abschließende `inspect` startet selbst keine Inferenz; es prüft die
vorhandenen gebundenen Bytes mit dem tatsächlichen Referenzconsumer und strengen
TensorRT-Producerloader. Eine erfolgreich erzeugte Profilkopie ist kein G2-PASS.

## G1 → G2/G3: vorbereiten und normalen Workflow starten

GUI schließen und den aktuellen Workflow beenden lassen; die vorhandenen
Plattformlocks gelten weiterhin. Nach erfolgreicher v2.80.3-Installation:

```bash
(
  set -Eeuo pipefail
  GATE_TOOL="${ONNX_SPLITPOINT_TOOL_DIR:-$HOME/ONNX-Splitpoint-Tool}"
  GATE_ROOT="$(mktemp -d "$HOME/Downloads/v2803_reference_native_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")"
  cd -- "$GATE_TOOL"
  "$GATE_TOOL/.venv/bin/python" -I -B scripts/reference_workflow_gate_v2803.py prepare \
    --profile "$GATE_TOOL/profiles/CompleteSetDev.yaml" \
    --source-run "$HOME/Models/EvaluationRuns/completsetdev_20260910_210826" \
    --output-dir "$GATE_ROOT/prepared"
  printf 'GATE_ROOT=%s\n' "$GATE_ROOT"
  "$GATE_TOOL/.venv/bin/python" -B -m onnx_splitpoint_tool.workflow.run_evaluation \
    --profile-driven \
    --profile "$GATE_ROOT/prepared/profile_fixed16.yaml" \
    --out "$GATE_ROOT/prepared/runs" \
    --require-fresh-run
)
```

`prepare` prüft ONNX/ORT/PyCOCO in einem eigenen Prozess mit 30 Sekunden Budget,
die originalen Modelldateien und die vorab festgelegten historischen Case-IDs.
Die Datenpfade werden aus dem existierenden Profil/der normalen Datasetbindung
übernommen. Fehlende Daten/Modelle ergeben einen Fehlerbericht, keine erfundene
Referenz. Der normale Benchmark bekommt ein endliches Runnerbudget von 300
Sekunden. Die bestehenden Lauf-/Abbruchdiagnosen bleiben maßgeblich.

Für die Artefakte gilt der vorhandene **strikte Warm-Cache-Preflight**:
`default_expectation=warm`, `block_on_unexpected_cold_builds=true`. Das ist **kein
globales `cache_verify_only`**; die CPU-Referenz wird tatsächlich erzeugt.
Unerwartete MISS-/UNKNOWN-Ergebnisse blockieren vor einem unbegründeten Kaltbuild.
Native darf keine fehlenden TensorRT-Engines bauen. Ein Cacheproblem wird zuerst
anhand des konkreten Preflightberichts geklärt. Force bleibt AUS.

Der Lauf schreibt seinen normalen Debug-Pack. Diesen mit dem separaten
G2-Prüfbericht hochladen. Den bei `GATE_ROOT=` ausgegebenen Ordner behalten.
Der folgende Befehl findet unter einem ausgewählten GATE_ROOT genau einen Lauf:

```bash
(
  set -Eeuo pipefail
  GATE_TOOL="${ONNX_SPLITPOINT_TOOL_DIR:-$HOME/ONNX-Splitpoint-Tool}"
  GATE_ROOT="$("$GATE_TOOL/.venv/bin/python" -I -B -c 'from pathlib import Path; roots=[p for p in (Path.home()/"Downloads").glob("v2803_reference_native_*") if p.is_dir()]; print(max(roots,key=lambda p:p.stat().st_mtime) if roots else "")')"
  [[ -n "$GATE_ROOT" ]] || { echo 'Kein GATE_ROOT vorhanden'; exit 2; }
  mapfile -t GATE_RUNS < <(find "$GATE_ROOT/prepared/runs" -mindepth 2 -maxdepth 2 -name run_manifest.json -type f -printf '%h\n')
  [[ "${#GATE_RUNS[@]}" == 1 ]] || { echo 'Erwartet genau ein frischer Lauf'; exit 2; }
  cd -- "$GATE_TOOL"
  "$GATE_TOOL/.venv/bin/python" -I -B scripts/reference_workflow_gate_v2803.py inspect \
    --run-dir "${GATE_RUNS[0]}" --output-dir "$GATE_ROOT/acceptance"
)
```

`G2_STATUS=verified_existing_reference_and_consumer` bedeutet: beide tatsächlichen
CPU-Referenzen passen zur aktiven Suite, Originalmodell-/Datasetbindung und
Population; die zentralen Aufträge sind technisch abgeschlossen und an genau
diese Referenzen/Requests gebunden. **Quality-FAIL ist dabei ein zulässiges
technisch abgeschlossenes Ergebnis.** Fehler, Timeout oder fehlende Originale
bleiben blockiert.

`PRODUCER_BINDING_STATUS` zeigt den tatsächlichen strengen Loaderbefund.
`G3_STATUS=not_accepted_by_reference_check_alone` bleibt absichtlich bestehen:
Die Producerbindung allein beweist noch keine ausgeführte Native-Messung oder
null Compilerstarts. Für G3 werden zusätzlich der normale Preflight, die
gebundenen Native-/TRT-Ergebnisse und der vollständige Runbericht ausgewertet.
Eine unbekannte beobachtete Compilerzahl bleibt `null`; die Erwartung ist null.
Die ausgelieferte Softwareintegration verwendet echte ORT-Inferenz an kleinen
synthetischen ONNX-Modellen; ihre synthetischen Accelerator-Kandidaten werden
vom strengen Loader korrekt abgewiesen. Sie ersetzt keine Zielhardwareabnahme.

## AP5: fester Diagnoseumfang ohne numerischen Produktpatch

Die Maschinenfassung liegt in `docs/V2803_DIAGNOSTIC_SCOPE.json`.

- **MobileNet:** Full/b056/b135 bleiben gemeinsam im Bestand. Dieselben vorher
  bestimmten Developmentbilder und ihre tatsächlichen Eingaben verwenden.
  Jeden Part1-Output gegen seine **eigene passende Float-Part1-Referenz** prüfen;
  die zwei unterschiedlich geformten Part1-Tensoren nicht direkt gleichsetzen.
  Endvorhersagen zusätzlich bildweise vergleichen. Die alten Genericwerte
  (H8 60,62/71,38/60,68 %, H10 59,42/71,42/59,74 %, DeepX
  71,72/72,28/71,64 %) sind Diagnosewerte, keine zentrale Qualitätsfreigabe.
- **HAR:** Parsed-/Quantized-Emulation nur mit vorhandenen tatsächlich gebundenen
  HAR-Dateien. Eine Herkunftszeichenkette reicht nicht; sonst `not_available`.
  Kein Neubau allein für diesen Reportingrelease.
- **YOLO26:** b398/b364 samt `score_column_not_probability_like` und
  `coordinates_not_ordered_xyxy` erhalten. Passender Float-P1 → echter Hailo-P1
  → gebundene QuantInfo/Layout/Dequantisierung → tatsächlicher P2-Input.
  TensorRT-P2 und Float-P2 erhalten exakt **denselben erfassten Übergangstensor**.
  Keine Scorekorrektur, kein Clipping, kein zusätzliches NMS auf Verdacht.
- **Compute und Energie:** Wunsch GPU ist kein Aktivitätsnachweis;
  `gpu_execution_unproven` widerlegt keinen erfolgreichen Build. Fehlende
  Hailo8-Komponenten bleiben ein eigener Konfigurationsblocker. Die historische
  Nacht bleibt **1 s × 3, FS/command, Screening**. TPC/EMC/Soll/Ist/Before/After
  aus dem Original getrennt lesen, keine Takte/Treiber/Masken umschalten und
  keine Kernanzahl aus dem Label `MAXN_SUPER/0` erfinden.

Nach G2/G3 folgt MobileNet gepaart mit B500; die 5.000er-Auswertung und eine
prospektiv konfigurierte Final-Energiemessung sind spätere eigene Aufträge.
Negative numerische Ergebnisse bleiben erhalten und begrenzen ihre Claims.
