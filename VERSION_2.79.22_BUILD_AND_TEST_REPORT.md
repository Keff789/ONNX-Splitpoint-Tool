# ONNX-Splitpoint-Tool 2.79.22 — Build- und Testbericht

Build-ID: `v2.79.22-global-negative-build-evidence`

Ausgangsbasis ist das vollständige Lieferpaket 2.79.21.

## Anlass und Umfang

Die bestehende Negativ-Library konnte exakt attestierte Parser- und
Mappingfehler wiederverwenden. Ihre Anbindung war jedoch auf den ausdrücklich
aktivierten Hailo-8-first/Gate-A-Pfad begrenzt. Normale Runs konnten deshalb
bekannte teure Fehlversuche wiederholen; neue lokale Terminal-Receipts wurden
nicht automatisch zu einem global nutzbaren Bestand fortgeschrieben.

Version 2.79.22 verankert Prüfung und Fortschreibung im gemeinsamen
Hailo-Backend. Das gilt für Hailo-8 und Hailo-10H und für die normalen
Full-/Part1-/Part2-Buildpfade. Die Buildfunktion bleibt für unbekannte oder
relevant geänderte Identitäten erhalten. Die bestehende Evidenzidentität wird
weiterverwendet; es entsteht keine zusätzliche wissenschaftliche Hash- oder
Signaturhierarchie und keine behauptete neue TensorRT-/DeepX-Negativdatenbank.

## Änderungen

1. Vor teurer Hailo-Kompilierung werden zentrale und vorhandene kompatible
   historische Indizes auf die vollständige Buildidentität geprüft.
   `PARSER_UNSUPPORTED` und `COMPILE_INFEASIBLE` unterbinden einen identischen
   erneuten Fehlversuch. Ein bloß gleiches Modell-/Boundary-Label reicht nicht.
2. Neue Terminalbefunde werden im zentralen Bestand
   `~/.onnx_splitpoint_tool/build_evidence/live_build_evidence.json`
   persistiert. `ONNX_SPLITPOINT_BUILD_EVIDENCE_ROOT` erlaubt einen isolierten
   alternativen Stammordner. Historische Indexdateien bleiben lesbare Quellen.
3. `TRANSIENT_INFRASTRUCTURE` und `ABORTED_UNKNOWN` sind keine permanenten
   Negativtreffer. Positivartefakte bleiben an die vorhandenen
   Artefakt-/Receipt-Prüfungen gebunden. Interne Wiederholungen mit `force=True`
   übergehen einen exakten negativen Befund nicht.
4. Der Recovery-Importer versteht die atomaren HEF-/Receipt-/Cache-Meta-
   Generationen aus 2.79.21. Er liest ein validiertes Paket aus einer
   zusammengehörigen Generation, statt die vorgesehenen Symlinks pauschal als
   `unsafe_evidence_symlink` abzulehnen. Die Pfad- und Generationensicherheit
   bleibt Bestandteil der Prüfung.
5. Speicher-/CUDA-Ressourcenfehler wie `CUDA memory allocation failed: out of
   memory` werden als `TRANSIENT_INFRASTRUCTURE` behandelt. Ein solcher
   Infrastrukturfehler darf keine dauerhafte `COMPILE_INFEASIBLE`-Sperre
   erzeugen. Echte strukturelle `Agent infeasible`-Befunde bleiben negativ.
6. Die Cache-Preflight-Ausgabe unterscheidet negative Evidenz von einem
   erwarteten Cold Build. Gespeicherte `known_infeasible`-Entscheidungen werden
   vor der finalen Matrix erneut geprüft: Ein inzwischen wiederhergestelltes
   gültiges HEF ergibt `HIT`; eine geänderte relevante Compiler- oder
   Kalibrierungsidentität führt zum aktuellen `MISS`-/`UNKNOWN`-Ergebnis.
   Ändert sich die Entscheidung erst nach der Matrix, verhindert
   `cache_preflight_refresh_required` einen verspäteten Cold Build, bis eine
   aktualisierte Matrix vorliegt. Die Cache-Matrix nach finaler Auswahl,
   `selection_changed`, `legacy_unsealed`, atomare Veröffentlichung und
   deterministische Artifact-Store-Prüfung aus 2.79.21 bleiben erhalten.
7. Windows-Aufrufe erhalten den tatsächlichen Quellmodellpfad und das
   Split-Manifest. Der Linux-Helfer prüft die Source-Identität anhand der
   tatsächlichen Quelldateibytes. Eine aus Part-Dateien erfundene vollständige
   Modellidentität wird nicht verwendet. Widersprechen geänderte Quelldateibytes
   einer erhaltenen Source-Hashangabe, wird `CONFLICT` gemeldet; daraus entsteht
   weder ein vermeintlich exakter Negativtreffer noch ein regulärer Cold Build.

## Vorhandene und verlorene Evidenz

Diese Version führt keinen Cleanup und keine automatische Rekonstruktion
alter Runs aus. Bereits vollständig indexierte Negativbefunde können auch
unabhängig von ehemaligen HEFs wieder nutzbar sein. Nicht indexierte gelöschte
Befunde werden nicht erfunden. Ob die historischen Einträge für `b364`, `b365`
und `b398` auf Smartmirror2 vorhanden oder rekonstruierbar sind, bleibt durch
eine separate read-only Inventur und Rettung zu klären.

## Release und Installation

Paket, Workflow, aktuelle Smoke-Aliase, `pyproject.toml`, `uv.lock`, Updater
und Acceptance tragen 2.79.22. Historische versionierte Einstiegspunkte und
Release-Dokumente, insbesondere 2.79.21, bleiben erhalten.

Der Installer verwendet die bestehenden Quellmanifest-/ZIP-Prüfungen und
erhält die virtuelle Umgebung sowie benutzereigene Profile. Die vorhandene
Rollbackdiagnostik bleibt erhalten. Er startet keine Compiler oder Hardware.

## Softwareprüfung

Die ausführbare aktuelle Prüfliste befindet sich in
`scripts/run_v27922_small_acceptance.sh`. Die abschließenden tatsächlichen
Testzahlen, Ergebnisse und Installationsnachweise werden in den
maschinenlesbaren Berichten und Konsolenprotokollen des Lieferpakets
festgehalten. Dieser Bericht setzt dafür keine ungeprüften PASS-Zahlen voraus.

Der Prüfumfang umfasst:

- aktuelle Release-Identität und erhaltene historische Aliase;
- dauerhaften Evidenzbestand und exakte deterministische Wiederverwendung;
- normalen Hailo-Backend-Kontrollfluss und Buildkontext für Full/Part1/Part2;
- negatives Preflight-Ergebnis ohne irreführenden erwarteten Cold Build,
  erneute Prüfung gespeicherter Entscheidungen und Matrix-Aktualisierung bei
  später geänderter Entscheidung;
- reale Source-/Manifestweitergabe über Windows-/Linux-Helfer und Konflikte
  zwischen gespeicherter Source-Identität und geänderten Quelldateibytes;
- sichere Recovery atomarer Hailo-Generationen;
- Infrastrukturklassifikation und weiterhin retrybare temporäre Fehler;
- vorhandene Evidenz- und Gate-A-Kontrollflusstests;
- bestehende Cache-, Publikations-, Duplikat- und Transportregressionen;
- Source-Manifest, Python-Kompilation und Shell-Syntax;
- isolierte Installation über den ausgelieferten Updater als gesonderten
  Abschlusstest mit dokumentiertem Ergebnis.

## Hardware- und Rettungsstatus

| Prüfung | Status im Build-Umfeld |
|---|---|
| Inventur/Recovery auf Smartmirror2 | `NOT_RUN` |
| Reale Wiederverwendung eines negativen Hailo-Befunds | `NOT_RUN` |
| Dauerhafte Aufzeichnung eines echten neuen Hailo-Terminalbefunds | `NOT_RUN` |
| Vollständiger Multi-Host-EvaluationRun | `NOT_RUN` |

Die Offline-Tests verwenden temporäre Daten und simulierte Compiler. Sie
belegen keine auf Smartmirror2 vorhandenen Indexeinträge und keine realen
Hardwaredurchsatz- oder Energiemessungen. Der begrenzte spätere Kontrollablauf
ist in `TESTANLEITUNG_2.79.22.md` beschrieben.
