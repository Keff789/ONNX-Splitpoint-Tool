# Build- und Testbericht 2.75.43

Release: `2.75.43`  
Build/Workflow: `v2.75.43-legacy-portable-dataset-identity-compatibility`

## Änderung

Der Patch ergänzt eine eng begrenzte Kompatibilität für historische
ImageNet-Validation-Manifeste: Der deklarierte Item-Aggregathash darf entweder
der normalisierten vierfeldrigen Darstellung oder der historischen Darstellung
mit unverändertem `sha256:`-Präfix entsprechen. Beide Formen werden auf dieselbe
normalisierte portable Identität abgebildet. Preflight, Management-Runner und
vendored Split-Runner verwenden denselben Vertrag; die Template-Parität prüft
`tests/test_v269a_deepx_full_central_quality.py`.

Unbekannte oder manipulierte Aggregathashes bleiben gesperrt. Die Prüfung der
500 Validation-Dateien, Payload- und Komponentenhashes, der eingefrorenen
v2.75.40-Authority, des B500-Pins, des exakten B1000-Satzes, der Disjunktheit
und der echten B500-Untermenge bleibt unverändert fail-closed.

Der wissenschaftliche Ablauf bleibt unverändert: **Final Quality** verwendet
den normalen Standard-Pfad mit 5.000/5.000 Validation-Items. Dieser Patch
ändert weder Budgets noch Hardwareverfahren und führt keinen neuen
Pflicht-Canary für den allgemeinen Final-Modus ein; der separate B1000-Canary
bleibt die vorgeschaltete Freigabe für diese konkrete DeepX-Fragestellung.

## Release-Vertrag

- Version/Release: `2.75.43`
- Lineage: `v2.75.43`
- Build/Workflow:
  `v2.75.43-legacy-portable-dataset-identity-compatibility`
- Feature: `legacy_portable_dataset_identity_compatibility`
- Neue Smokes: `onnx-splitpoint-smoke-v27543` und
  `onnx-splitpoint-smoke-v2-75-43`
- v2.75.42-Smokes, Harnesses, Entry-Points und Dokumente bleiben erhalten.

## Finale Messwerte

- v2.75.43 Small-Acceptance-Testblock: `316 passed` in `382.04 s`;
- Legacy-/Versions-Kompatibilität: `215 passed`;
- kombinierter v43/v42/v41/v40-, Manifest-, Harness- und Helper-Paritätsblock:
  `103 passed`;
- Preflight-, Generic-Runner- und DeepX-Template-Parität einschließlich
  produktionsnaher Legacy-/Normalisierungs- und Manipulationsfälle:
  `38 passed`;
- v2.75.43-Smoke: `13/13`; historischer v2.75.42-Smoke: `13/13`;
- Source-Manifest: `903` Release-Dateien; die abschließende Release- und
  Archivverifikation muss für alle davon erfolgreich sein;
- `uv lock --check --offline`, Python-AST und Shell-Syntax: PASS.

Deterministische A/B/C-Archivhashes und die Archivgröße werden erst nach dem
einmaligen finalen Manifest-Freeze extern gemessen. Die Größe steht im
Builder-/Acceptance-JSON; der SHA-256 zusätzlich im externen
`<ZIP-NAME>.sha256`-Begleitnachweis. Selbstreferenzielle Archivmesswerte werden
nicht in dieses enthaltene Dokument geschrieben.

Keine Release-Acceptance startet Hardware, SSH, DX-COM, TensorRT-Builds,
Energieerfassung oder Inferenz.
