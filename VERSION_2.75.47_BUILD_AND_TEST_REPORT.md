# Build- und Testbericht 2.75.47

Release: `2.75.47`

Build/Workflow:
`v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent`

## Änderungen

### Modellgebundener YOLOv7-Decoder

Der YOLOv7-Paper-Export wird nicht mehr nur aufgrund von Modellname und
Outputform interpretiert. v2.75.47 materialisiert einen kanonischen
Decodervertrag mit Modell-SHA-256, Inputgeometrie, Strides und Ankertabelle.
Die Produktionsregistrierung verwendet
`yolov7_paper_standard_anchors_640_v1`; die historische Tiny-Ankertabelle
`yolov7_paper_legacy_tiny_anchors_640_v1` ist ausschließlich ein expliziter
Diagnostikarm. Native-Full-Evidenz bindet Vertragshash und Ankeridentität.

Zusätzlich verifiziert der Workflow einen im Profil deklarierten
`model_sha256` bereits in `resolve_model`. Der Pin erzwingt die Hashbildung
auch bei einer allgemeinen No-Hash-Option. Ungültiger Pin, fehlende Datei oder
Mismatch enden vor Prepare/Analyze/Build/Remote-Mutation fail-closed; erwartete
und beobachtete normalisierte Hashes bleiben in der Diagnose erhalten. Das
etablierte Manifestfeld `model_sha256=sha256:<hex>` bleibt kompatibel.

### Official-COCO CPU-A/B-Gate

Die CPU-Probe vergleicht den Legacy-Diagnostikarm mit dem modellgebundenen
Standardarm auf demselben gepinnten ONNX und demselben COCO-Subset. Die Probe
ist kein optionaler Komforttest: der Summary muss `status: completed` und
`acceptance.status: accepted` ausweisen; jede Official-COCO-Policy muss
`status: ok` tragen, bevor der GUI-Anker gestartet werden darf. Ein
fehlendes `pycocotools`, `unavailable`, Partial oder Hash-/Vertragsfehler
bleibt ein No-Go.
Der Production-Decoder-/NMS-Vertrag behält seine Grenze von maximal 300
Detections. Davon getrennt wertet Official COCO alle drei Probe-Arme mit den
kanonischen `maxDets=[1,10,100]` aus; die pycocotools-Summary wird nicht auf
300 umgedeutet.
Der vorregistrierte Vergleich verlangt außerdem exakte Generic-/Native-
Parität, exakt einen Raw-ORT-Pass je Bild, den exakt gepinnten 500er Cohort,
nur die erlaubte Ankertabellen-Differenz, mindestens 0,05 AP75-Verbesserung
und höchstens 0,01 AP beziehungsweise AP50-Regression. Der abgeschlossene
Standard-Production-Arm bindet als eigenständigen Completed-task-Health-Block
AP50:95 >= 0,20, AP50 >= 0,35, AP75 >= 0,20 und AP75/AP50 >= 0,45. Ein
unabhängiger Standard-Upstream-Sanity-Arm bindet zusätzlich die strengeren
vier absoluten Floors:
AP50:95 >= 0,25, AP50 >= 0,40, AP75 >= 0,25 und AP75/AP50 >= 0,50. Diese
beiden Floor-Blöcke attestieren ausschließlich Completed-task-/Upstream-
Decoder-Health beziehungsweise keine katastrophale Korruption; sie begründen
ausdrücklich **keinen kanonischen Accuracy-Claim**
(`canonical_accuracy_claim=false`).
Bei Erfolg entstehen ein deterministisches Evidence-Archiv und ein separates
`.tar.gz.manifest.json`, das Archivname, SHA-256, Bytegröße, Memberzahl sowie
Probe-/Acceptance-Status bindet; die Testanleitung verifiziert das Sidecar vor
dem Upload gegen das tatsächliche Archiv.
Preflight- beziehungsweise sonstige Fehler vor beanspruchter Full-run-Output-
Ownership liefern Returncode 2 und `stderr`, aber bewusst kein Archiv. Ein
Failure-Archiv samt Sidecar entsteht nur nach erfolgreicher Output-
Beanspruchung durch den Full Run.

### Debug-Pack Audit-Intent

Ein vorhandener Scientific Report mit `status=not_requested` erzeugt nicht
mehr selbst einen Ranking-Audit-Intent. In diesem Non-Audit-Fall bleiben
`enabled=false`, erwartete und fehlende Pflichtmember leer und der
vollständige Debug Pack kann top-level `complete=true` sein. Sobald Profil
oder kanonischer Plan den Audit anfordern, bleibt die Archivierung dagegen
fail-closed und erwartet auch materialisierte Geschwistermodelle, die vor der
Report-Erzeugung abgebrochen sind.

### Frisches YOLOv7-only Anchor-Profil

`profiles/yolov7_paper_v27547_standard_anchor_b500.yaml` bindet genau
`yolov7_paper=b044`, den erfolgreichen Modellhash
`7a13e66f91047cce0e251c05f64159646847e842af31d60441c63dcdfad7825d`
und B500. Generic, Native und Native Full sind aktiv, alle Energy- und
Ranking-Achsen aus, Native hat drei Repetitionen und Official COCO ist
required. Der Lauf ist Start-only: Result-, Prediction-, Quality- und
Checkpoint-Reuse über die Releasegrenze ist ausgeschlossen. Die bewährte
Policy für receipt-/hashvalidierte Build-Caches bleibt unverändert
(`cache_integrity: relaxed`, `verify_on_reuse: metadata`). Es gibt keinen
B1000-Neulauf. Der bewährte DeepX-B500-Root
`~/Models/BackendArtifacts/deepx/v2.75.44/thesis_standard_b500_imagenet_mean_std`
bleibt explizit gebunden, damit ein gültiger DXNN-Receipt-/Hash-Treffer nicht
unnötig neu gebaut wird.

### Release- und Provenance-Schicht

Version, Workflow, Paketmetadaten, Updater, Smoke-Entrypoints, aktuelle
Dokumente und Acceptance-Harnesses sind auf v2.75.47 gebunden. Die
claim-kritische Paketprovenance umfasst nun explizit `campaign.py`,
`native_detection_postprocess.py`, `runners/harness/yolo.py`,
`validation/official_coco.py`, `execution_plan.py`, beide Debug-Pack-Module
und den Workflow-Runner. Der paketierte Remote-Validator
`resources/remote_scripts/native_producer_validate_visualize.py` ist
claim-kritisch gebunden und im Smoke byte-identisch mit dem Top-Level-Skript
bei SHA-256
`1e2e4f293a381338549a57da5ceab28c136fc330ef42ba4bec2632abf248080e`;
damit bleibt auch die fail-closed Ablehnung eines ungebundenen
`yolov7_paper`-Raw-Heads (keine Detections, `claim_capable=false`) im Release-
Vertrag. Damit sind außerdem die exakte Annotation-Nutzung und die kanonischen
COCO-`maxDets` claim-kritisch digest-gebunden. Der
Updater erhält die Venv, prüft aber `pycocotools` lesend und nennt bei Bedarf
den exakten `dependency_bootstrap --groups yolov7_probe`-Installationsschritt
für ONNX Runtime, NumPy und Pillow sowie den separaten expliziten
`pip install --upgrade 'pycocotools>=2.0.7'`-Schritt. Letzterer verhindert,
dass ein bereits importierbares, aber zu altes pycocotools vom importbasierten
Bootstrap übersprungen wird. Der Updater selbst führt keinen impliziten
Netzwerk-Install aus.

## Release-Vertrag

- Version/Release: `2.75.47`
- Lineage: `v2.75.47`
- Build/Workflow:
  `v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent`
- neue Features:
  - `model_bound_yolov7_anchor_contract`
  - `official_coco_yolov7_decoder_ab_probe`
  - `generic_native_yolov7_decoder_parity`
  - `debug_pack_ranking_audit_intent_fix`
  - `debug_pack_non_audit_completeness`
  - `early_declared_model_sha256_admission`
- neue Smokes: `onnx-splitpoint-smoke-v27547` und
  `onnx-splitpoint-smoke-v2-75-47`
- v2.75.46/v2.75.45 Features, Smokes, Harnesses und historische
  Release-Dokumente bleiben enthalten.

## Hardwarefreier Prüfstatus und Release-Freeze

Nach dem Produktions-Freeze wurden die hardwarefreien Blöcke seriell und mit
deaktiviertem externem Pytest-Plugin-Autoload wiederholt:

- vollständige v2.75.47 YOLOv7-/Native-/Completed-Endpoint-Regressionsmatrix,
  einschließlich modellgebundener Standard-Anker, Generic-/Native-Parität,
  Official-COCO-Probe, atomarem Evidence-Archiv und expliziter Ablehnung alter
  ungebundener Live-, Auto-Probe- und schema-v6-Cache-Evidence:
  **369/369 bestanden** (eine erwartete fp16-Overflow-Warnung im absichtlichen
  Mutationstest);
- alle neuen v2.75.47 Modellhash-, Profil-, Release-, Debug-Pack- und
  Decoderfälle gemeinsam: **62/62 bestanden**;
- historischer Versions-/Provenance-Wildcard-Block: **154/154 bestanden**;
- fokussierter Debug-Pack-Vertrag einschließlich Audit-/Non-Audit-Intent:
  **39/39 bestanden**;
- retained v2.75.46 Hailo-Interpreter-, Calibration-Canary-, Standard-Quality-
  und Release-Provenance-Block: **32/32 bestanden**;
- stdlib-Editable-Refresh und realer Updater-Integrationstest:
  **23/23 bestanden**;
- aktuelle CLI-/GUI-/Release-/Local-Harness-Verträge:
  **33/33 bestanden**;
- vollständiger Python-AST/Bytecode-Compile- und Shell-`bash -n`-Block:
  **bestanden**;
- aktueller v2.75.47 Smoke: **15/15 Prüfungen bestanden**;
- unabhängiger abschließender Science-/Fail-closed-Review:
  **59/59 Fokus- und 9/9 Archive-/Failure-Fälle bestanden**, ohne offenen
  Code- oder Science-Blocker.

Die ursprünglich verwendete isolierte Test-Venv zeigte beim direkten
Lazy-Import ihres lokalen `numpy.random`-Binary reproduzierbar einen
Prozess-`SIGBUS`, also vor einer Test-Assertion. Der retained v2.75.46-Block
wurde deshalb unverändert mit derselben Python-/Pytest-Schicht, aber der
sauberen NumPy-2.3.5-Runtime aus der System-Testumgebung wiederholt und bestand
32/32. Der Fehler ist damit als beschädigte Testumgebung eingegrenzt, nicht als
Produkt-Assertion umgedeutet.

Der abschließende Local-Acceptance-Lauf ergab zusätzlich:

- Stage 2 (Isolation, Clean-Source, Standard-Pfad und aktueller
  Integrationsblock): **1.485 bestanden, 4 erwartete Skips**;
- Stages 3 bis 6 (Quality-, Native-, Hotloop-, Decision- und Energy-
  Vertragsblöcke): **18/18, 220/220, 31/31 und 73/73 bestanden**;
- vollständige Suite gegen den eingefrorenen v2.73.8-Fehlerbestand:
  **3.448 Tests beobachtet**, davon **3.421 bestanden**, **6 Skips** und
  **21 bekannte Altfehler**; **0 neue Fehler** außerhalb des eingefrorenen
  Bestands;
- Small Acceptance aus dem frisch entpackten Release:
  **118/118 aktuelle Integrationsfälle**, **32/32 retained-v2.75.46-Fälle**
  und **15/15 Smoke-Prüfungen**;
- großer Extracted-Source-Acceptance-Satz aus demselben Release:
  **1.712 bestanden, 4 erwartete Skips, 0 Fehler**;
- synchronisierter Updater-Integrationstest: Venv, externe und lokale Caches,
  Evaluation-Runs, Logs und eigenes Profil blieben erhalten; ausgelieferte
  Dateien wurden aktualisiert und veraltete Source-Dateien gesichert/entfernt.

Der vollständige Testlauf wird damit weiterhin ehrlich als
Regressionvergleich und nicht als global grüne Suite bezeichnet. Die 21
beobachteten Fehler sind eine exakte Teilmenge des eingefrorenen
v2.73.8-Bestands; v2.75.47 führt keinen neuen Fehler ein.

## Noch ausstehende reale Gates

Nach dem Source-Freeze folgen getrennt vom hardwarefreien Release-Nachweis:

1. Official-COCO CPU-A/B-Probe mit Summary `completed`, Acceptance `accepted`
   und allen Official-COCO-Policies `ok`;
2. frischer GUI-Start des YOLOv7-only B500-Ankers, nicht Resume;
3. separater 20er Native-first Ranking-Audit mit Energy aus;
4. **Final Quality (Standard+)** mit 5.000 Validierungsitems je Task erst nach
   dem Ranking-Freeze und vor Energy;
5. Energy erst als späterer eigener Schritt.

Der B500-Datensatz bleibt eingefroren; ein neuer B1000-Lauf gehört nicht zu
v2.75.47. Der CPU-Gate muss vor dem GUI-Anker erfolgreich sein.

## Manifest und Archiv

Das finale `SOURCE_MANIFEST.json` bindet 934 Source-Dateien sowie Paket- und
Workflow-Identität; `SHA256SUMS.txt` wurde daraus deterministisch erzeugt und
read-only gegengeprüft. Drei unabhängige Archivbauten – zweimal aus dem
Arbeitsbaum und einmal durch den Builder des frisch entpackten Baums – waren
byteidentisch. Alle 936 ZIP-Member waren eindeutig, in kanonischer Reihenfolge,
CRC-sauber und hatten deterministische Metadaten.

Der endgültige ZIP-Hash wird außerhalb des Archivs in der ausgelieferten
Prüfsummendatei gebunden. Ein Archiv kann seinen eigenen endgültigen Hash
nicht inhaltlich attestieren.
