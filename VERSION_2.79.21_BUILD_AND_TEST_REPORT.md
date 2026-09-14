# ONNX-Splitpoint-Tool 2.79.21 — Build- und Testbericht

Build-ID: `v2.79.21-cache-preflight-atomic-publication`

Ausgangsbasis ist das bereitgestellte vollständige Lieferpaket 2.79.20.

## Anlass

Für YOLO26m wurde `b398` ausgewählt. Vorhandene Builds anderer Boundaries
konnten den erforderlichen Hailo-10H-Part1-Build nicht ersetzen. Das Tool muss
diesen echten Cold Build unmittelbar nach der endgültigen Splitauswahl
melden. Zusätzlich dürfen unvollständige HEF-Generationen und widersprüchliche
Artifact-Store-Duplikate nicht als gültige Wiederverwendung erscheinen.

## Änderungen

1. Die vollständige Cache-Preflight-Matrix entsteht nach der finalen
   Splitauswahl und vor den regulären Backend-Builds. Bei normaler,
   compilerunabhängiger Auswahl startet zuvor kein Backend-Compiler.
   Angeforderte Full- und Split-Artefakte werden einzeln geprüft.
2. Cold-Build-Diagnosen nennen Modell, Boundary, Backend, Artefaktrolle und
   Grund. Ein nicht prüfbarer Cache bleibt `UNKNOWN` und wird nicht als
   bewiesener MISS dargestellt.
3. Hailo HEF, Receipt und Cache-Meta werden als eine vollständige Generation
   atomar veröffentlicht und persistent gesichert. Fehler beim Schreiben
   dürfen keinen unvollständigen Treffer produzieren oder einen vorherigen
   gültigen Stand zerstören.
   Die vollständigen Dateien werden vor dem atomaren Zeigerwechsel
   synchronisiert; vorherige Generationen bleiben erhalten.
4. HEF-only-Cachegenerationen ohne Receipt erhalten die eindeutige
   Klassifikation `legacy_unsealed`. Ein historisches HEF mit gültigem
   Receipt und fehlender Cache-Meta kann nach erfolgreicher Validierung
   migriert werden. Vorhandene Daten werden dabei nicht gelöscht und
   fehlende Nachweise nicht erfunden.
5. Artifact-Store-Duplikate werden deterministisch geprüft und ausgewählt.
   Ein unvollständiger oder widersprüchlicher Kandidat verdeckt keinen
   nachfolgenden gültigen Kandidaten. Die Auswahl hängt nicht von der
   zufälligen Dateisystem-Reihenfolge ab.
6. Bei geänderter Splitauswahl macht `selection_changed` die vorher gebauten
   Boundaries, die aktuelle Auswahl und die erwarteten Cold Builds sichtbar.

Bei compilerabhängiger Gate-A-Auswahl wird vor jeder nötigen
Kandidatenprüfung ein Cache-Preflight mit Scope `selection_probe` in
`selection_probe_cache_preflight.json` gespeichert und protokolliert. Erst
danach darf der Compiler zur Machbarkeitsprüfung starten. Im strikten Warm-Cache-Modus blockieren
unerwartete MISSes bei erwartetem warmem Cache sowie UNKNOWN die
Auswahlprüfung vor dem Compiler. Explizit erwartete Cold Builds bleiben
gemäß der bestehenden Policy zulässig; im diagnostischen Modus können
die vorher gemeldeten Cold Builds ebenfalls ausgeführt werden. Nach der
endgültigen Auswahl folgt zusätzlich die vollständige finale Cache-Matrix.
Damit bleibt Gate-A mit compilerabhängiger Auswahl unterstützt. Direkte
Builds im Benchmark-Tab behalten ihren Ablauf.

Die vorhandenen Cache-Schlüssel und Backend-Receipt-Verträge werden
weiterverwendet. Es wird keine zusätzliche wissenschaftliche Hash- oder
Signaturhierarchie eingeführt. Atomare Publikation schützt die bereits
erforderlichen Dateien zusammen.

## Release- und Installationsumfang

Paket, Workflow, GUI-Identität, `pyproject.toml`, `uv.lock`, aktueller Smoke,
Acceptance-Aliase und Source-Updater tragen durchgehend 2.79.21. Die bisherigen
versionierten Einstiegspunkte und historischen Berichte bleiben erhalten.
Die dedizierte Acceptance behält die funktionalen 2.79.20-Regressionsfälle;
die alte strikt auf 2.79.20 festgelegte Identitätsprüfung ist kein aktuelles
Release-Gate.

Der Installer erhält die vorhandene virtuelle Umgebung und benutzereigene
Profile. Source-ZIP und Installer werden über den vorhandenen
Release-Mechanismus auf einen konkreten ausgelieferten Stand gebunden.

Die Rollbackdiagnostik erhält den ursprünglichen Installationsfehler auch
bei einem zusätzlichen Rücksetzfehler. Wiederaufgetauchte Zielpfade und
zugehörige Backups bleiben erhalten; unabhängige Launcher und Editable-Pfade
werden weiter zurückgesetzt. Unvollständige Rücksetzungen nennen die
betroffenen Pfade, statt den ursprünglichen Fehler zu verdecken.

## Softwareprüfung

**Fokussierte Softwareprüfung: 192 bestanden, 1 übersprungen.**

Der übersprungene zusätzliche DeepX-ONNX-Runtime-Vergleich benötigt
`onnxruntime`, das im Build-Umfeld nicht installiert ist. Die neuen
Cache-, Publikations-, Auswahl- und Transporttests wurden ausgeführt.
Die abschließenden Source-Manifest- und Installationsnachweise stehen
in den maschinenlesbaren Berichten des vollständigen Lieferpakets.

Die ausführbaren Prüfungen befinden sich in
`scripts/run_v27921_small_acceptance.sh`. Der maschinenlesbare Bericht und
die Konsolenlogs des endgültigen Lieferpakets dokumentieren die tatsächlich
ausgeführten Ergebnisse. Der Prüfbereich umfasst:

- aktuelle Release-Identität und historische Alias-Erhaltung;
- finale Splitauswahl, vollständige Matrix, Cold-Build-Gründe und
  `selection_changed`;
- Gate-A-Kandidatenprüfung mit vorangestelltem `selection_probe` und
  anschließendem finalem Preflight;
- HEF-/Receipt-/Meta-Publikation und unterbrochene Schreibvorgänge;
- `legacy_unsealed` ohne erfundene Validierung;
- deterministische Duplikatprüfung einschließlich ungültiger Kandidaten;
- bestehende Hailo-, TensorRT-, DeepX- und Preflight-Barriere-Regressionsfälle;
- Source-Manifest, Python-Kompilation und Shell-Syntax;
- Rollback-Kollision mit erhaltener Fehlerursache, gesicherten Backups und
  fortgesetzter Wiederherstellung unabhängiger Installationsdateien;
- Installation über den ausgelieferten Updater mit erhaltener `.venv` und
  benutzereigenen Profilen.

## Hardwarestatus

| Prüfung | Status im Build-Umfeld |
|---|---|
| Endgültige Sieben-Modell-Matrix gegen Smartmirror2/Jetsons | `NOT_RUN` |
| YOLO26m b398 Hailo-10H vor echter Optimierung | `NOT_RUN` |
| Hailo Cold→Warm auf echter Hardware | `NOT_RUN` |
| TensorRT- und DeepX-Wiederverwendung auf echter Hardware | `NOT_RUN` |
| Vollständiger Multi-Host-EvaluationRun | `NOT_RUN` |

Ein Offline-PASS ist kein Nachweis für Hardwaredurchsatz oder gemessene
Energie. Die begrenzte Prüfung auf dem Gerät ist in
`TESTANLEITUNG_2.79.21.md` beschrieben.
