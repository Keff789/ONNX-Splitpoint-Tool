# ONNX-Splitpoint-Tool 2.79.19 — Build- und Testbericht

Build-ID: `v2.79.19-calibration-warning-evalrun-closure`

## Anlass

Eine reale Full-System-Eingangskalibrierung lieferte technisch nutzbare
Messpunkte, verfehlte den sehr engen Richtwert für die Übereinstimmung beider
Punktfaktoren aber knapp. Die bisherige gemeinsame „Quality Gate“-Darstellung
behandelte diese numerische Plausibilitätsabweichung wie einen technischen
Messfehler und verhinderte das Speichern.

Der vorherige Complete-Set-EvaluationRun zeigte außerdem mehrere voneinander
unabhängige Softwarefehler: unvollständiges Remote-Modulstaging, bereits
berechnete aber nicht an der erforderlichen Stelle duplizierte
Quality-Vertragsfelder, vor Reject/Backfill versiegelten Case-Scope und einen
stale Artifact-Index nach dem absichtlichen Neuaufbau kanonischer Reports. Die
Debug-Pack-Nachanalyse identifizierte zusätzlich eng begrenzte Classification-,
Hailo-8-ResNet- und DeepX-Fehlerpfade.

## Implementierte Änderungen

### Kalibrierung: technische Gültigkeit und Plausibilität

- Das Messmodell bleibt der Ein-Faktor-Fit durch den Ursprung. Es wird keine
  freie Offset-Gerade eingeführt.
- Punktfaktor-Spread, enger erwarteter Faktorbereich und Abweichung des
  Ist-Stroms vom nominellen Sollpunkt werden als Warnungen gespeichert und in
  der GUI angezeigt. Sie verhindern das Speichern nicht.
- Zu kleiner Messsprung, hohe Idle-Drift, nicht endliche/nicht positive Werte,
  singulärer Fit, Verletzung der breiten harten Faktorgrenze und
  Acquisition-Integrity-Fehler bleiben Blocker.
- Der Operator kann reale Messwerte nicht durch das Gate verändern; Warnungen
  dokumentieren die Plausibilität des gemessenen Fits.

### EvaluationRun-Closure

- Remote Runtime Closure enthält nun die direkten Hailo-Backend-Abhängigkeiten
  `hailo_attempt_receipts` und `hailo_timeout_policy`.
- Generic TensorRT Quality Request und Candidate duplizieren
  `decoder_contract_sha256`, `nms_contract_sha256` und
  `quality_record_endpoint_contract_sha256` aus der bereits signierten
  Producer-Identität. Widersprüche bleiben fail-closed.
- Der ausführbare Required Case Scope entsteht aus der finalen Accepted-Liste
  des erzeugten BenchmarkSets nach Reject/Backfill und vor Runtime-Dispatch.
  Verworfene Kandidaten gelten nicht als fehlende Messungen.
- Nach erfolgreichem kanonischem Reportaufbau werden nur Indexeinträge für
  ausdrücklich ersetzte oder entfernte Reportflächen bereinigt. Sonstige
  registrierte, aber fehlende Artefakte bleiben terminale Fehler.

### Weitere Debug-Pack-Korrekturen

- Native Classification Split behandelt MobileNet und RegNet mit der passenden
  Policy und führt sie durch Contract-, Expected-Row- und Quality-FIRST-
  Orchestrierung.
- Hailo-8 ResNet trennt quantisiertes HEF-Speicherformat vom verlangten
  FLOAT32-VStream-/Runtime-Vertrag. Rohe Integerpfade prüfen zusätzlich die
  tatsächliche native HEF-Streambreite und verwerfen `UINT16`/`UINT8`-
  Verwechslungen.
- DeepX behält Fehler bei der Verarbeitung bereits dispatchter Ergebnisse als
  Post-Dispatch-Fehler bei, statt sie als fehlenden Dispatch zu melden.
- Neu erzeugte DeepX-Einzelverträge binden `model_id` bereits am Producer.
- Fehlende und widersprüchliche `input_contract.model_id`-Werte bleiben strikt
  abgelehnt. Die Bindungsprüfung liefert dafür jetzt konkrete Reason-Codes
  statt nur eines undifferenzierten Gesamtfehlers.

### Unveränderte Grenzen

- Ranking, `cut_bytes_only`, Quality-Schwellwerte, Pipeline-FPS-Definition und
  die vorhandene Energy-Plan-Semantik werden nicht geändert.
- Der vom Betreiber bereits behobene volle Datenträger ist eine operative
  Voraussetzung, keine Codeänderung.
- Es entsteht keine neue Hash-, Manifest-, Signatur- oder Versiegelungsebene.
  Die vorhandene Release-Manifestmechanik und vorhandene
  Kalibrierungsbindung bleiben unverändert.

## Softwareprüfung

Die hardwareunabhängige Acceptance umfasst Release-Smoke, fokussierte
Regressionstests der Kalibrierung und aller genannten EvaluationRun-Pfade,
Source-Manifest-Prüfung, Python-Kompilation und Shell-Syntax. Dieser
Quelldokument-Stand erfindet keine Testzahl. Die tatsächlich ausgeführten
Ergebnisse stehen im maschinenlesbaren Bericht von
`scripts/run_v27919_small_acceptance.sh` und im finalen Liefernachweis.

Ein Offline-PASS belegt die implementierten Kontrollflüsse und Verträge. Es ist
kein Ersatz für die reale Multi-Host-Ausführung.

## Hardware- und Laufstatus

| Prüfung | Status im Build-Umfeld |
|---|---|
| Reale u.RECS-/elektronische-Last-Kalibrierung | `NOT_RUN` |
| Kalibrierung mit speicherbarer Plausibilitätswarnung | `NOT_RUN` |
| Begrenzter realer Multi-Host-EvaluationRun | `NOT_RUN` |
| Hailo-8 ResNet realer Native-Split-Replay | `NOT_RUN` |
| DeepX YOLO11 End-to-End Hardware-Replay | `NOT_RUN` |

Insbesondere wird aus den softwaregeprüften DeepX-Zustands- und
Diagnosekorrekturen kein erfolgreicher DeepX-YOLO11-Hardwarelauf
abgeleitet. Die Hypothese muss mit dem in
`TESTANLEITUNG_2.79.19.md` beschriebenen begrenzten Replay geprüft werden.
