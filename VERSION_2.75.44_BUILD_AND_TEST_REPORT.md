# Build- und Testbericht 2.75.44

Release: `2.75.44`  
Build/Workflow: `v2.75.44-runtime-validation-cohort-projection`

## Änderung

Der v2.75.43-B1000-Vorbereitungslauf hat ein gültiges ImageNet-Quellmanifest
mit 50.000/50.000 verifizierten Bildern gegen eine eingefrorene Autorität der
500 ausgewählten Laufzeitbilder verglichen. Diese beiden Identitäten
beschreiben verschiedene Ebenen und können deshalb nicht direkt gleich sein.

v2.75.44 hält die vollständige Quellprüfung unverändert aufrecht und ergänzt
die fehlende, deterministische Projektion vom Source-Manifest auf die
Laufzeit-Validation-Kohorte. Nur die projizierten 500 Zeilen werden gegen die
eingefrorene Manifest-, Image-ID- und Ground-Truth-Autorität geprüft. Fehlende,
doppelte oder abweichende Zeilen, Labels, Dateien, Hashes oder
Auswahlparameter bleiben blockierend.

Die Änderung ist ein enger Gate-Vertragsfix. Sie verändert keine
wissenschaftliche Hypothese, keinen Datensatzinhalt, keine Calibration-
Auswahl, kein Preprocessing, keine Quality-Grenze und keinen Ranker.
Der allgemeine **Final Quality**-Modus bleibt der bestehende **Standard**-Pfad
mit 5.000 Classification- und 5.000 Detection-Validation-Items; der
B1000-Canary ist ausschließlich die Freigabe für diese konkrete
Kalibrierungsfrage.

## Release-Vertrag

- Version/Release: `2.75.44`
- Lineage: `v2.75.44`
- Build/Workflow: `v2.75.44-runtime-validation-cohort-projection`
- Feature: `legacy_source_to_runtime_validation_cohort_projection`
- Neue Smokes: `onnx-splitpoint-smoke-v27544` und
  `onnx-splitpoint-smoke-v2-75-44`
- v2.75.43-Smokes, Harnesses, Entry-Points und Dokumente bleiben erhalten.

## Prüfstatus

Der hardwarefreie Release-Vertrag umfasst:

- exakte Package-, Build- und Workflow-Identität;
- Python-AST- und Shell-Syntax;
- fokussierte Projection-/Preflight-Regressionen;
- historische v2.75.43-/v2.75.42-/v2.75.41-Pfade;
- aktuellen v2.75.44-Smoke;
- read-only Source-Manifest- und Archivverifikation nach dem finalen Freeze.

Bestätigter Stand vor dem Manifest-Freeze:

- kompletter v2.75.44-Small-Acceptance-Pytestblock: **327 passed in
  788.53 s (0:13:08)**;
- v2.75.44-Smoke: **13/13** Prüfungen bestanden;
- fokussierter Preflight-/Kohorten-Projektionstest: **18/18** bestanden;
- Validation-Materializer-/Bundle-Regressionen: **13/13** bestanden;
- breiter Identity-/Provenance-Block: **204/204** bestanden;
- zusätzlicher kombinierter Kern-/Canary-Block: **100/100** bestanden.

Alle genannten Läufe waren hardwarefrei. Die Small Acceptance meldet explizit
`SKIP hardware, SSH, DX-COM, TensorRT build, Energy and inference`.

Diese Testzahlen werden nach dem einmaligen Manifest-Freeze auf dem gefrorenen
Baum erneut bestätigt.
Archivgröße und Archiv-SHA-256 stehen bewusst ausschließlich im externen
Builder-/Handoff-Nachweis beziehungsweise im `.sha256`-Sidecar; ein Archiv kann
seinen eigenen endgültigen Hash nicht inhaltlich attestieren. Keine
Release-Acceptance startet Hardware, SSH, DX-COM, TensorRT-Builds,
Energieerfassung oder Inferenz.
