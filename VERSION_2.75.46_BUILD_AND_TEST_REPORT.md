# Build- und Testbericht 2.75.46

Release: `2.75.46`

Build/Workflow:
`v2.75.46-native-full-onnx-attestation-standard-quality-projection`

## Änderungen

### Native-Full-ONNX-Attestierung

Der v2.75.44-Anker zeigte 26 erfolgreiche von 27 geplanten Native-Zeilen. Nur
`native_full_hailo8 / yolov7_paper / full` scheiterte in allen drei
Wiederholungen mit `hailo_source_raw_head_onnx_unreadable`. Dasselbe
Source-ONNX mit identischem SHA-256 wurde im Hailo-10-Pfad erfolgreich
verwendet. Die Fehlerklasse belegte daher keine korrupte Modelldatei, sondern
die Attestierung im falschen Python-Prozess.

v2.75.46 führt die enge Raw-Head-ONNX-Probe als kanonisches JSON-Protokoll über
den bereits ausgewählten ONNX-fähigen Engine-Build-Interpreter aus. Der Child
liest die Source-ONNX-Datei genau einmal, hasht diesen Byte-Puffer und parst
exakt denselben Puffer. Der Parent akzeptiert ausschließlich das bekannte
Schema in Version 2, den receipt-identischen Child-Hash und primitive,
kanonische Output-Deskriptoren. Alle Signaturprüfungen bleiben fail-closed.

Import- und Ladefehler werden nicht länger zu einer begründungslosen Meldung
zusammengezogen. Der Receipt-Diagnostikblock bindet Interpreter,
Probe-Returncode, Fehlerphase, Exception-Typ und -Detail sowie vorhandene
stdout-/stderr-Tails. Die stabile äußere Fehlerklasse bleibt für bestehende
Consumer erhalten; `status_detail` und `error` tragen die konkrete Ursache.
Die Bindung von Child-Hash und Parsing an denselben Byte-Puffer schließt den
Time-of-check-/Time-of-use-Spalt. Parent-Hashes vor und nach der Probe bleiben
als zusätzliche Driftkontrolle erhalten. Abweichende Child-Bytes, Drift oder
ein nachträglich unlesbares ONNX enden mit
`hailo_source_raw_head_onnx_identity_drift` und gebundener
erwarteter/beobachteter Identität.

### Standard-Quality-Projektion

Der Workflow erzeugte bereits einen expliziten Standard-Quality-
Acceptance-Vertrag für die setup-lokalen TensorRT-Full-Companions. Die
wissenschaftliche Projektion akzeptierte jedoch ausschließlich das historische
Full-only-Canary-Schema. Außerdem wurden normale Generic-Full-Zeilen beim
expliziten Vertrag als unerwartete Acceptance-Evidenz gezählt. Dadurch konnte
ein technisch vollständiger Standardlauf trotz 9/9 Companion-PASS kein finales
PASS erhalten.

v2.75.46 akzeptiert genau zwei bekannte Schema-/Scope-Paare:

- `full-only-quality-acceptance-identity-contract` / `full_only`;
- `standard-setup-local-tensorrt-quality-acceptance-identity-contract` /
  `standard_quality_setup_local_tensorrt`.

Version und die sieben Identity-Key-Felder bleiben exakt geprüft. Nur
`variant=full`, `execution_role=full_quality_only` und das literale
`performance_claims_emitted=false` können Acceptance-Evidenz werden. Normale
Generic-Zeilen bleiben Diagnostics. Absichtlich fehlerhafte Full-only-
Deklarationen werden weiterhin als unerwartet gezählt und können den Vertrag
nicht durch Umklassifizierung umgehen.

Der reale 3-Modelle-x-3-Setups-Vertrag ergibt damit 33 erhaltene Ergebnisse:
9/9 setup-lokale TensorRT-Companions als Acceptance-PASS und 24 Generic-
Diagnostics. Zwölf Diagnostics sind Full-, zwölf Composed-Zeilen. Das
wissenschaftliche Urteil lautet bei vollständigem Vertrag PASS.

### DeepX-Kalibrierungsgrößen-Canary

Der B500/B1000-Vergleich brach vor der inhaltlichen Qualitätsauswertung mit
`B500/B1000 deepx_m1_full quality execution contracts differ` ab. Ursache war
eine artefaktbezogene Output-Contract-Hash-Komponente, die zwischen zwei
getrennten, korrekt gepinnten Kalibrierungsgrößen naturgemäß verschieden sein
kann, obwohl der semantische Output-Endpoint und der Quality-Vertrag identisch
sind.

v2.75.46 normalisiert diese Hash-Komponente für den Armvergleich und ergänzt
eine explizite semantische Endpoint-Invariante. Kalibrierungsabhängige
Artefaktidentitäten dürfen verschieden bleiben. Neutralisiert wird nur
`quality_record_endpoint.identity.output_contract_sha256`; anschließend werden
nur dessen deterministische QRE-/Quality-Contract-Container und projizierte
Top-Level-Mirrors neu berechnet. Laufzeit-Endpoint, beobachtete Outputs,
Postprocessor, Runner, Preprocessing, Prepared-Input-Evidenz, Modell/Dataset,
Präzisionssemantik und Full-only-Identity müssen weiterhin übereinstimmen.
Alle Arm-Seals sowie Request-/Candidate-Bindings werden vor der Projektion
validiert; echte Abweichungen bleiben fail-closed.
Der fokussierte Regressionstest ist
`tests/test_v27546_calibration_size_canary_output_contract.py`.

### Historische Retention

Die v2.75.45-Features bleiben vollständig Teil des Build-Vertrags:

- GUI-Bestätigung für große score-unabhängige Audits;
- gültige Audit-Minimum-Grenze;
- explizites DeepX-Classification-Preprocessing;
- getrenntes TensorRT-Retentionsbudget und aktives Working Set;
- Erhalt und Wiederverwendung des aktuellen Cache-Namespace.

## Release-Vertrag

- Version/Release: `2.75.46`
- Lineage: `v2.75.46`
- Build/Workflow:
  `v2.75.46-native-full-onnx-attestation-standard-quality-projection`
- neue Features:
  - `native_full_selected_onnx_interpreter_attestation`
  - `native_full_onnx_attestation_exception_diagnostics`
  - `standard_setup_local_quality_contract_projection`
  - `generic_quality_diagnostic_row_separation`
  - `deepx_calibration_size_output_contract_hash_normalization`
  - `deepx_calibration_size_semantic_endpoint_invariant`
- neue Smokes: `onnx-splitpoint-smoke-v27546` und
  `onnx-splitpoint-smoke-v2-75-46`
- mitgeliefertes frisches Anchor-Profil:
  `profiles/resnet_yolo26s_yolov7_v27546_standard_anchor_b500.yaml`
- v2.75.45-Smokes, Harnesses, Tests, Dokumente und Feature-Flags bleiben
  erhalten.

## Hardwarefreier Prüfstatus

Hardwarefrei bestätigt:

- fokussierter Native-Full-ONNX-Interpreter-/Diagnostikblock:
  **8/8 bestanden**;
- Native-Full-Fix zusammen mit den relevanten historischen Hailo-/Workflow-
  Regressionen: **94/94 bestanden**;
- Standard-Quality-Projektion zusammen mit den historischen v2.75.22- und
  v2.69d-Regressions: **51/51 bestanden**.
- fokussierter B500/B1000-Output-Contract-/Endpoint-Invariantenblock:
  **5/5 bestanden**;
- vollständiger v2.75.41-plus-v2.75.46-Canary-Regressionsblock:
  **74/74 bestanden**;
- alle drei neuen v2.75.46-Integrationsdateien gemeinsam:
  **22/22 bestanden**.
- v2.75.46-Release-Provenance einschließlich Profil- und Manifestparser-
  Vertrag: **10/10 bestanden**;
- vollständiger fokussierter Stage-3-Block einschließlich v2.75.45-
  Working-Set-Retention sowie historischer Quality-Projektionen:
  **80/80 bestanden**;
- breiter hardwarefreier Versions-/Updater-/GUI-Provenanceblock:
  **329/329 bestanden**;
- aktueller v2.75.46-Smoke: **15/15 Prüfungen bestanden**;
- vollständige v2.75.46 Small Acceptance einschließlich des 80er-
  Stage-3-Blocks: **bestanden**;
- strikte Installed-Source-Manifest-Verifikation: **923/923 Release-Dateien,
  keine Missing/Changed/Unexpected/Symlink-Befunde**.

Der deterministische Archiv-Doppelbau und der externe Sidecar-Hash gehören zum
Handoff nach dem letzten Manifest-Freeze. Der endgültige ZIP-Hash wird deshalb
nicht in diesen archivierten Bericht zurückgeschrieben.

Alle lokalen Release-Tests sind hardwarefrei. Ein frischer echter
Standard-/Quality-Anker ist ein gesonderter Hardware-Akzeptanzlauf. Dafür ist
ein neuer v2.75.46-Results-Ordner erforderlich; ein versionsübergreifendes
Resume eines v2.75.44-/v2.75.45-Runs ist kein gültiger Release-Nachweis.
Receipt- und hashgebundene Build-Caches können durch die bestehenden Gates
dennoch wiederverwendet werden.

Der bestehende Modus **Final Quality (Standard+)** bleibt mit 5.000
Classification-/Detection-Validationselementen und 5.000 Bootstrap-
Wiederholungen erhalten. Er hat keinen Pflicht-Canary; historische optionale
Wrapper `run_v27528_*_final_canary.sh` werden durch diesen Hotfix nicht
verändert.

Archivgröße und Archiv-SHA-256 stehen ausschließlich im externen Builder-
beziehungsweise Handoff-Nachweis und im `.sha256`-Sidecar. Ein Archiv kann
seinen eigenen endgültigen Hash nicht inhaltlich attestieren.
