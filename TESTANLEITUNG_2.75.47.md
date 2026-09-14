# Testanleitung 2.75.47

Release: `2.75.47`

Build/Workflow:
`v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent`

## Zweck und verbindliche Reihenfolge

v2.75.47 bindet den YOLOv7-Decoder an Modellidentität, Inputgeometrie und
Ankertabelle. Außerdem trennt der Debug-Pack-Vertrag einen wirklich
angeforderten Ranking-Audit von bloß vorhandenen `not_requested`-Berichten.
Die Freigabe erfolgt bewusst in dieser Reihenfolge:

1. hardwarefreie Small Acceptance;
2. CPU-A/B-Probe des bisherigen und des modellgebundenen YOLOv7-Decoders mit
   offizieller COCO-Auswertung;
3. erst bei vollständig abgeschlossener Official-COCO-Auswertung
   (`status: ok` für alle drei Probe-Policies) ein frischer GUI-Anker;
4. danach ein separater 20er **Native-first** Ranking-Audit mit **Energy aus**;
5. Energy bleibt ein späterer, eigener Schritt.

Die CPU-A/B-Probe ist ein hartes Go-Gate vor dem GUI-Anker. Ein technisch
laufender Decoder oder eine interne Ersatzmetrik genügt nicht. Der Probe-
Summary muss top-level `status: completed` und `acceptance.status: accepted`
ausweisen. Der Probe- sowie der GUI-Lauf müssen außerdem den offiziellen
COCO-Vertrag vollständig abgeschlossen ausweisen.

## Installation und Pflichtabhängigkeit

Der Updater erhält die bestehende `.venv` und installiert absichtlich keine
Netzwerkabhängigkeiten. Deshalb nach dem Update im tatsächlichen Tool-Ordner
zuerst ausführen:

```bash
cd /home/kmika/ONNX-Splitpoint-Tool
./.venv/bin/python -m onnx_splitpoint_tool.dependency_bootstrap \
  --groups yolov7_probe \
  --python ./.venv/bin/python

if ! ./.venv/bin/python - <<'PY'
from importlib.metadata import version
import re

raw = version("pycocotools")
match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[.+-].*)?", raw)
assert match and tuple(map(int, match.groups())) >= (2, 0, 7), raw
PY
then
  ./.venv/bin/pip install --upgrade 'pycocotools>=2.0.7'
fi
```

Die Probe-Gruppe bindet nicht nur `pycocotools`, sondern auch `onnxruntime`,
`numpy` und Pillow. Das bedingte Upgrade läuft nur bei fehlender oder zu alter
pycocotools-Version: der Bootstrap erkennt vorhandene Module per Import und
würde eine bereits installierte, aber zu alte Version sonst nicht selbst
aktualisieren. Bei bereits erfülltem Pin gibt es keinen unnötigen Netzwerk-
oder Latest-Change. Anschließend im selben Interpreter den Import und die
Mindestversion `pycocotools>=2.0.7` final prüfen:

```bash
./.venv/bin/python - <<'PY'
from importlib.metadata import version
import re
import pycocotools

raw = version("pycocotools")
match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[.+-].*)?", raw)
assert match and tuple(map(int, match.groups())) >= (2, 0, 7), raw
print("pycocotools: available", raw, pycocotools.__file__)
PY
```

Ohne erfolgreichen Import kein Probe-Go und kein GUI-Start.

## Lokale Small Acceptance

```bash
cd /home/kmika/ONNX-Splitpoint-Tool
PY=/home/kmika/ONNX-Splitpoint-Tool/.venv/bin/python \
  bash scripts/run_v27547_small_acceptance.sh
```

Der Block startet keine Hardware, kein SSH, keinen Compiler-Build, keine
Inferenz und keine Energieerfassung. Er prüft insbesondere:

- die exakte v2.75.47-Identität und die sechs neuen Feature-Flags;
- den modellgebundenen Standard-/Legacy-Diagnostikvertrag;
- den frühen fail-closed Modell-SHA-256-Gate im `resolve_model`-Schritt;
- die CPU-A/B-Probe und ihre Official-COCO-Vertragsoberfläche;
- Debug-Pack-Vollständigkeit mit und ohne Audit-Intent;
- das frische YOLOv7-only-Profil und die v2.75.46-Regressionsverträge;
- Source-Provenance und den aktuellen Release-Smoke.

## Pflicht-Gate: CPU-A/B vor GUI

Die Probe verwendet exakt den eingefrorenen 500er Bild-ID-Cohort aus dem
realen v2.75.46-Lauf. Zuerst die vier gebundenen Inputs und einen neuen,
leeren Output-Pfad setzen:

```bash
cd /home/kmika/ONNX-Splitpoint-Tool
MODEL=/home/kmika/Models/yolov7_paper.onnx
VALIDATION_MANIFEST=/home/kmika/.onnx_splitpoint_tool/final_datasets/manifests/coco2017_val_manifest.json
SELECTION_REQUEST=/home/kmika/Models/EvaluationRuns/resnet_yolo26s_yolov7_v27546_standard_anchor_b500_20260817_201735/models/yolov7_paper/benchmark_results/quality_inputs/orin_nx_deepx_m1_01/results/b044/results_ort_tensorrt/task_quality_inputs/full_request.json
ANNOTATIONS=/home/kmika/.onnx_splitpoint_tool/final_datasets/coco2017/annotations/instances_val2017.json
IMAGES=/home/kmika/.onnx_splitpoint_tool/final_datasets/coco2017/val2017
OUT=/home/kmika/Downloads/yolov7_decoder_ab_v27547

test -f "$MODEL"
test -f "$VALIDATION_MANIFEST"
test -f "$SELECTION_REQUEST"
test -f "$ANNOTATIONS"
test -d "$IMAGES"
test ! -e "$OUT"
test ! -e "${OUT}.tar.gz"
test ! -e "${OUT}.tar.gz.manifest.json"

./.venv/bin/python -B scripts/probe_yolov7_decoder_ab.py \
  --preflight-only \
  --model "$MODEL" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --selection-request "$SELECTION_REQUEST" \
  --annotations "$ANNOTATIONS" \
  --images-root "$IMAGES" \
  --output-dir "$OUT"
```

Der Preflight muss `status: preflight_ok`, exakt 500 selektierte IDs, den
Validation-Manifest-SHA
`2de36f0f8949e4f1fcbd0eefbda1f4d18b411208fa91e2dd2bf985a56d0f22e2`,
den Selection-Request-SHA
`3d51d22511681cc28b99c2d927fe08003b92a234a4ec101683a0875d0ae72c32`,
den Annotation-SHA
`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`
und den Image-ID-SHA
`b8ac329f5d3e7e201a3938d36c63bfe96a13c8a2be6e4fe6ddfab89cbc135e45`
attestieren. Danach im selben Terminal dieselbe Probe ohne
`--preflight-only` ausführen:

Ein Fehler vor erfolgreicher Full-run-Output-Beanspruchung, insbesondere im
Preflight, endet mit Returncode 2 und JSON-Diagnose auf `stderr`; absichtlich
entsteht dabei **kein** Evidence-Archiv oder Sidecar. Erst wenn der Full Run
den frischen Output-Pfad erfolgreich beansprucht hat und danach fehlschlägt,
materialisiert die Probe den Failure-Output samt `.tar.gz` und
`.tar.gz.manifest.json`. Dadurch wird ein bloßer Preflight-Fehler nicht
fälschlich als uploadbares Ausführungspaket dargestellt.

```bash
./.venv/bin/python -B scripts/probe_yolov7_decoder_ab.py \
  --model "$MODEL" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --selection-request "$SELECTION_REQUEST" \
  --annotations "$ANNOTATIONS" \
  --images-root "$IMAGES" \
  --output-dir "$OUT" \
  --overlay-count 10
```

Der Nachweis muss das Schema
`onnx-splitpoint/yolov7-decoder-ab-probe`, den erwarteten Modell-SHA-256
`7a13e66f91047cce0e251c05f64159646847e842af31d60441c63dcdfad7825d`,
die Standard-Ankertabelle `yolov7_paper_standard_anchors_640_v1` und
den bestandenen vorregistrierten Gate enthalten. In `probe_summary.json`
müssen top-level `status: completed`, `acceptance.status: accepted` und alle
Einträge unter `official_coco` den Status `ok` haben; das umfasst
Legacy-Tiny-Production, Standard-Production und Standard-Upstream-Sanity. Ein
`unavailable`, `partial`, fehlender Policy-Arm, Hashfehler oder COCO-Fehler ist
ein No-Go. Die Legacy-Tiny-Ankertabelle
`yolov7_paper_legacy_tiny_anchors_640_v1` bleibt reine A/B-Diagnostik und darf
nicht als Produktionsvertrag registriert werden.

Dabei bleiben zwei verschiedene Grenzen absichtlich getrennt: der
Produktions-Decoder-/NMS-Vertrag behält `max_detections=300`; jede offizielle
COCOeval-Auswertung aller drei Arme muss dagegen die kanonischen
`maxDets=[1,10,100]` verwenden. `maxDets=300` in den Official-COCO-Metriken
wäre kein gültiger Probe-Nachweis.

Zusätzlich müssen `pre_registered_gate.passed` und ausnahmslos alle folgenden
Gates wahr sein: `exact_500_id_cohort`, `one_raw_ort_pass_per_image`,
`generic_native_standard_exact_parity`,
`official_coco_all_policies_completed`,
`paired_production_policy_equal_except_anchor_table`,
`paired_contract_diff_exactly_pre_registered`,
`standard_ap75_material_improvement`,
`standard_ap_no_regression_beyond_001` und
`standard_ap50_no_regression_beyond_001`. Der abgeschlossene
Standard-Production-Arm muss separat diese vier absolut vorregistrierten
Completed-task-Health-Floors bestehen:

- `standard_production_ap_50_95_at_least_020`: AP50:95 >= 0,20;
- `standard_production_ap50_at_least_035`: AP50 >= 0,35;
- `standard_production_ap75_at_least_020`: AP75 >= 0,20;
- `standard_production_ap75_ap50_ratio_at_least_045`: AP75/AP50 >= 0,45.

Der unabhängige Standard-Upstream-Sanity-Arm muss zusätzlich die strengeren
vier absoluten Floors bestehen:

- `upstream_sanity_ap_50_95_at_least_025`: AP50:95 >= 0,25;
- `upstream_sanity_ap50_at_least_040`: AP50 >= 0,40;
- `upstream_sanity_ap75_at_least_025`: AP75 >= 0,25;
- `upstream_sanity_ap75_ap50_ratio_at_least_050`: AP75/AP50 >= 0,50.

Beide Floor-Blöcke belegen ausschließlich Completed-task-/Upstream-Decoder-
Health beziehungsweise das Ausbleiben katastrophaler Korruption. Sie sind
**kein kanonischer Accuracy-Claim**; `canonical_accuracy_claim` muss deshalb
`false` bleiben.

Der Output-Ordner enthält mindestens `probe_summary.json`,
`decoder_contracts.json`, `input_provenance.json`, drei gebundene Prediction-
JSONs, die Official-COCO-Metriken und `SHA256SUMS`. Die Probe erzeugt nach
erfolgreichem Gate automatisch das deterministische, uploadbare Archiv
`$OUT.tar.gz` sowie dessen maschinenlesbares Sidecar
`$OUT.tar.gz.manifest.json`. Beides nur prüfen, nicht erneut überschreiben:

```bash
ARCHIVE="${OUT}.tar.gz"
SIDECAR="${ARCHIVE}.manifest.json"
test -s "$ARCHIVE"
test -s "$SIDECAR"
(cd -- "$OUT" && sha256sum -c -- SHA256SUMS.txt)

./.venv/bin/python -B - "$ARCHIVE" "$SIDECAR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import tarfile

archive = Path(sys.argv[1])
sidecar = Path(sys.argv[2])
payload = json.loads(sidecar.read_text(encoding="utf-8"))
digest = hashlib.sha256()
with archive.open("rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
actual_sha256 = digest.hexdigest()
with tarfile.open(archive, mode="r:gz") as bundle:
    actual_members = sum(member.isfile() for member in bundle.getmembers())

assert payload["schema"] == "onnx-splitpoint/yolov7-decoder-ab-probe-archive"
assert payload["archive_basename"] == archive.name
assert payload["archive_sha256"] == actual_sha256
assert payload["archive_size_bytes"] == archive.stat().st_size
assert payload["archive_member_count"] == actual_members
assert payload["probe_status"] == "completed"
assert payload["acceptance"] == "accepted"
print("archive sidecar: OK", actual_sha256, actual_members)
PY

ls -lh -- "$ARCHIVE"
sha256sum -- "$ARCHIVE" "$SIDECAR"
tar -tzf "$ARCHIVE" | sed -n '1,120p'
```

## Frischer YOLOv7-only GUI-Anker

Nur nach bestandenem CPU-Gate in der GUI
`profiles/yolov7_paper_v27547_standard_anchor_b500.yaml` laden. Die effektive
Zusammenfassung muss vor dem Start exakt zeigen:

- Modell/Fall: nur `yolov7_paper=b044`;
- Modell-SHA-256: der oben genannte v2.75.46-Erfolgs-Export;
- Generic, Native und Native Full: an;
- Native-Repetitionen: 3;
- neun Native/Full-Zeilen und 27 gültige Wiederholungen insgesamt: drei
  Setups mal Split/Accelerator-Full/setup-lokales TensorRT-Full mal drei;
- Native Energy und Generic Energy: aus;
- Ranking und score-unabhängiger Audit: aus;
- Detection Calibration/Validation und Bootstrap: jeweils B500;
- Official COCO: enabled und required.

Mit **Start Evaluation Workflow** einen neuen Results-Ordner erzeugen: Start,
nicht Resume und insbesondere nicht **Resume latest** eines v2.75.46-Runs. Der
frühe `resolve_model`-Gate muss vor Prepare/Analyze/Build/Remote-Mutation
abbrechen, falls der lokale ONNX-Inhalt nicht zum gepinnten Hash passt.

B500 bleibt für diesen Decoder-Treatment-Vergleich eingefroren; es gibt
**kein B1000** und keinen neuen B500/B1000-Canary. Resultate, Predictions,
Quality-Urteile und Stage-Checkpoints werden nicht versionsübergreifend
wiederverwendet. Normale receipt-/hashvalidierte TensorRT-, Hailo- und DeepX-
Build-Caches dürfen dagegen wiederverwendet werden.
Der DeepX-Build-Cache bleibt dazu exakt auf
`~/Models/BackendArtifacts/deepx/v2.75.44/thesis_standard_b500_imagenet_mean_std`
gebunden; ein Treffer ist nur mit dem bestehenden Receipt-/Hash-Vertrag
zulässig.

Der GUI-Anker ist erst bestanden, wenn die modellgebundene Decoderidentität,
Generic-/Native-Parität und Official COCO im finalen Results Bundle
vollständig und ohne blockierende Zeile materialisiert sind.

## Danach: 20er Native-first Ranking-Audit

Erst nach dem bestandenen B500-Anker einen gesonderten 20-Kandidaten-Lauf
starten. Dabei **Native-first**, Ranking-Audit an und **Energy aus** verwenden.
Der Audit darf nicht in den Anker hineingemischt und kein vorhandener
`not_requested`-Report als Audit-Intent interpretiert werden.

Wenn der Audit angefordert wurde, bleibt der Debug Pack fail-closed: Plan,
modellbezogene Scientific Reports und alle durch den Plan erwarteten
Geschwistermodelle müssen im Archiv vorhanden sein. Wenn der Audit nicht
angefordert wurde, sind leere erwartete/fehlende Memberlisten und ein
vollständiger Debug Pack korrekt, selbst wenn ein diagnostischer
`not_requested`-Report existiert.

Energy wird erst nach diesem Audit in einem eigenen Lauf aktiviert. So bleibt
der Decoder-/Ranking-Vergleich frei von einer zusätzlichen Energie-Achse.

## Erhaltener Final-Quality-Modus

Der normale Modus **Final Quality (Standard+)** bleibt unverändert und erhöht
Classification-/Detection-Validation sowie Bootstrap auf jeweils 5.000. Er
hat weiterhin keinen Pflicht-Canary. Die historischen optionalen Wrapper
`run_v27528_*_final_canary.sh` bleiben separate Werkzeuge und gehören nicht
zum v2.75.47 Decoder-Anker.

## Handoff

Die CPU-Probe als eigenen Evidence-Ordner samt deterministischem `.tar.gz`,
`.tar.gz.manifest.json` und SHA-256 aufbewahren. GUI-Anker und 20er Audit
dagegen jeweils als eigenes Evaluation Results Bundle samt Debug Pack,
jeweiligem Manifest-Sidecar und SHA-256 sichern. Das Release-ZIP und sein
finales `SOURCE_MANIFEST.json` werden erst nach Abschluss aller Produktions-
und Acceptance-Tests eingefroren; ein vorläufiges Archiv ist kein finaler
v2.75.47-Nachweis.
