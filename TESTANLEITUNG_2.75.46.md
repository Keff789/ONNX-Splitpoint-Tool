# Testanleitung 2.75.46

Release: `2.75.46`

Build/Workflow:
`v2.75.46-native-full-onnx-attestation-standard-quality-projection`

## Zweck

v2.75.46 schließt drei voneinander unabhängige Verträge:

1. Die Native-Full-Attestierung eines Hailo-Source-ONNX läuft über den bereits
   ausgewählten, ONNX-fähigen Engine-Build-Interpreter. Der Hailo-8-
   Laufprozess muss deshalb nicht selbst das Python-Paket `onnx` importieren
   können. Import-, Lade-, Protokoll- und Prozessfehler bleiben fail-closed,
   werden aber mit Phase, Exception-Typ und Detail ehrlich protokolliert.
2. Der explizite Standard-Quality-Vertrag akzeptiert genau die neun
   setup-lokalen TensorRT-Full-Companions für drei Modelle und drei Setups.
   Die 24 normalen Generic-Quality-Zeilen bleiben vollständig im Bericht,
   werden aber als Diagnostics von der Acceptance-Aggregation getrennt. Bei
   neun gültigen PASS-Companions lautet das wissenschaftliche Ergebnis PASS.
3. Der DeepX-B500/B1000-Canary normalisiert
   kalibrierungsgrößenabhängige Output-Contract-Hashes und vergleicht die
   semantische Endpoint-Invariante. Unterschiedliche B500-/B1000-Artefakte
   werden nicht mehr fälschlich als unterschiedlicher Quality-
   Ausführungsvertrag verworfen; echte semantische Abweichungen bleiben
   blockierend.

Die v2.75.45-Verträge für große GUI-Ranking-Audits, getrenntes TensorRT-
Retentionsbudget, aktives Working Set und Resume-Cache-Reuse bleiben erhalten.

Der normale Modus **Final Quality (Standard+)** bleibt ebenfalls unverändert:
Er erhöht gegenüber Standard nur Classification-/Detection-Validation und
Bootstrap auf jeweils 5.000. Er verlangt keinen Pflicht-Canary; die historischen
optionalen Wrapper `run_v27528_*_final_canary.sh` bleiben gesonderte Werkzeuge.
Der hier beschriebene Drei-Modell-Anchor ist dagegen bewusst ein Standardlauf.

## Lokale Small Acceptance

Aus dem Tool-Ordner:

```bash
cd /home/kmika/ONNX-Splitpoint-Tool
PY=/home/kmika/ONNX-Splitpoint-Tool/.venv/bin/python \
  bash scripts/run_v27546_small_acceptance.sh
```

Der Lauf startet keine Hardware, kein SSH, keinen DX-COM-/TensorRT-/Hailo-
Build, keine Inferenz und keine Energieerfassung. Er prüft:

- die exakte v2.75.46-Build-Identität und alle neuen Feature-Flags;
- ausgewählten ONNX-Interpreter, Probe-Protokoll und ehrliche
  Exception-Diagnostik;
- den Standard-Fall mit 33 Quality-Zeilen: 9/9 Acceptance-PASS plus 24
  erhaltene Generic-Diagnostics;
- den B500/B1000-Output-Contract- und Endpoint-Invariantenvertrag;
- die v2.75.45-Retentions- und Working-Set-Regressions;
- Source-Manifest, Release-Dokumente und den aktuellen Smoke.

Die beiden Native-Full-Runner müssen bytegleich sein:

```bash
cmp -s \
  scripts/native_full_baseline_eval_runner.py \
  onnx_splitpoint_tool/resources/remote_scripts/native_full_baseline_eval_runner.py
```

## Referenzprofil für den frischen GUI-Lauf

In der GUI das mitgelieferte Profil
`profiles/resnet_yolo26s_yolov7_v27546_standard_anchor_b500.yaml` laden und
einen neuen Lauf mit **Start Evaluation Workflow** erzeugen.

Für den zeitsparenden Standard-/Quality-Anker vor dem Start in der GUI die
effektive Zusammenfassung kontrollieren:

- Run-Modus: `standard`;
- Modelle/Fälle: `resnet50=b052`, `yolo26s=b038`,
  `yolov7_paper=b044`, also genau ein Fall pro Modell;
- Generic und Native: eingeschaltet;
- Native Energy und Generic Energy: ausgeschaltet;
- Ranking-Audit: ausgeschaltet;
- DeepX-Kalibrierung: B500, `imagenet_mean_std`, `opt=0`,
  `exact-v2` und isolierter Cache-Namespace.

Die GUI darf diese Achsen nicht stillschweigend auf ein anderes Profil oder
eine andere Kalibrierungsgröße materialisieren.

## Frischer Lauf statt versionsübergreifendem Resume

Den fehlgeschlagenen v2.75.44-/v2.75.45-Anker nicht mit **Resume latest** in
v2.75.46 fortsetzen. Ein Resume bindet Stage-Checkpoints, Start-Snapshot und
Workflow-Provenance des alten Runs; es ist kein sauberer Nachweis für den neuen
Attestierungs- und Quality-Projektionsvertrag. Stattdessen nach Installation
von v2.75.46 einen neuen Results-Ordner mit **Start** erzeugen.

Ein frischer Run bedeutet nicht, dass valide Build-Artefakte künstlich neu
gebaut werden müssen. Bereits vorhandene, receipt- und hashgebundene
TensorRT-, Hailo- oder DeepX-Caches dürfen durch die normalen Cache-Gates
wiederverwendet werden. Alte Stage-Ergebnisse oder ein altes finales Quality-
Urteil werden dagegen nicht in den neuen Run übernommen.

Das mitgelieferte Profil verweist für DeepX deshalb absichtlich auf den bereits
vom korrekten v2.75.44-B500/Mean-Std-Anker verwendeten exact-v2-Cache-Namespace.
Nur ein vollständig passender Receipt-/Hash-/Build-Vertrag wird übernommen;
bei jeder Abweichung baut das Tool neu oder blockiert fail-closed. Der neue
Results-Ordner, die Attestierung, alle Messungen und das Quality-Urteil stammen
weiterhin vollständig aus v2.75.46.

## Erwarteter Native-Full-Nachweis

Für die YOLOv7-Hailo-8-Full-Zeile wird der in der Engine-Auswahl bestätigte
ONNX-fähige Interpreter an die Raw-Head-Attestierung übergeben. Ein gültiges
Source-ONNX mit drei kanonischen Multiscale-Heads darf nicht mehr allein deshalb
mit `hailo_source_raw_head_onnx_unreadable` enden, weil der Hailo-Prozess einen
anderen Python-Interpreter benutzt.

Falls die Attestierung wirklich fehlschlägt, enthält
`hailo_hef_build_receipt_diagnostics` mindestens den verwendeten
`onnx_python`, den Probe-Returncode, `failure_phase`, `exception_type` und
`exception_detail`; vorhandene stdout-/stderr-Tails bleiben erhalten.
`status_detail` beziehungsweise `error` nennt zusätzlich
`ExceptionType: Detail`. Die stabile Fehlerklasse bleibt fail-closed.
Der Child hasht und parst denselben einmal gelesenen Byte-Puffer; der Parent
verlangt den receipt-identischen Child-Hash und prüft die Datei zusätzlich vor
und nach der Probe. Abweichende Child-Bytes sowie eine während der Probe
geänderte oder nicht mehr lesbare Datei enden fail-closed mit
`hailo_source_raw_head_onnx_identity_drift` und der Phase
`validate_child_parsed_identity` beziehungsweise
`validate_parent_identity_post_probe`; erwarteter und beobachteter SHA-256
bleiben in der Diagnostik erhalten. Das Probe-Protokoll ist Schema-Version 2.

## Erwarteter Standard-Quality-Nachweis

Im `central_quality_summary.json` muss für den vollständigen 3x3-Anker gelten:

```text
technical_status = ok
result_count = 33
aggregate_expected_full_result_count = 9
aggregate_full_result_count = 9
aggregate_usable_full_result_count = 9
aggregate_excluded_diagnostic_full_result_count = 12
aggregate_identity_contract_complete = true
aggregate_identity_contract_issue_count = 0
aggregate_decision_counts = {pass: 9}
quality_decision = pass
scientific_pass = true
```

Die übrigen zwölf der insgesamt 24 Generic-Diagnostics sind Composed-Zeilen
und deshalb nicht in `aggregate_all_full_result_count` enthalten. Alle 24
bleiben dennoch unter `results` erhalten. Missing, Duplicate, Unexpected oder
Schema-/Scope-Fehler müssen das Urteil weiterhin fail-closed blockieren.

## DeepX-B500/B1000-Canary

Der Canary wird wie bisher mit getrennten B500-/B1000-Run-Verzeichnissen und
den gepinnten Manifesten ausgeführt. Erwartet wird kein pauschales
`quality execution contracts differ` mehr, wenn nur der artefaktbezogene
Output-Contract-Hash zwischen den Kalibrierungsgrößen variiert. Die
normalisierte semantische Endpoint-Invariante muss dagegen identisch sein.
Eine echte Änderung von Output-Anzahl, Namen, Shapes, Dtypes oder
Postprocessing-/Quality-Semantik bleibt ein harter Vergleichsfehler. Konkret
bleiben `endpoint.identity` einschließlich `stage`, `output_format`,
`tensor_signature` und Semantik, `endpoint_contract_hash`, beobachtete
QRE-Runtime-Outputs, Postprocessor, Runner, Preprocessing, Prepared-Input-
Evidenz, Modell/Dataset, Präzisionssemantik und Full-only-Identity streng.
Alle Arm-Seals sowie Request-/Candidate-Bindings werden vor der Projektion
validiert.

Der hardwarefreie fokussierte Vertragstest ist:

```bash
/home/kmika/ONNX-Splitpoint-Tool/.venv/bin/python -B -m pytest -q \
  -p no:cacheprovider --tb=short \
  tests/test_v27546_calibration_size_canary_output_contract.py
```

## v2.75.45-Retention

Der neue Release darf den v2.75.45-Cache-Vertrag nicht zurücknehmen. Für große
Audits bleibt:

```text
active_working_set_bytes
  = current_namespace_bytes + planned_current_growth_bytes

projected_managed_bytes <= effective_admission_max_bytes
```

Aktive, fremde, verlinkte oder nicht sauber attestierte Namespaces werden nicht
gelöscht. Ein wiederverwendbarer aktueller Namespace bleibt für einen
gleichartigen neuen Lauf verfügbar; die separate physische Speicherplatz-
Prüfung bleibt maßgeblich.
