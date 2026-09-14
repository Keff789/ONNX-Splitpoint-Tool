# Build- und Testbericht 2.75.42

Release: `2.75.42`  
Build/Workflow: `v2.75.42-real-evidence-and-quality-companion-repair`

## Reale Ausgangsbefunde

Der v2.75.41-Source-Release selbst bestand in einer sauberen 892-Datei-Sicht
alle 185 fokussierten Tests und den 13/13-Smoke. In der installierten Kopie
schlug derselbe Manifest-Gate jedoch bei 17 absichtlich erhaltenen eigenen
Profilen fehl. Fünf nicht manifestierte historische Shell-Wrapper wurden
zusätzlich als unzulässige Source-Reste erkannt. Release-Dateien fehlten nicht
und keine war verändert.

Der anschließende B500-Pin wurde vor jeder Ausgabe fail-closed blockiert. Das
historische Manifest trug korrekt die Produktions-Kernel-Identität
`kevinmika/onnx-splitpoint-imagenet-calibration-export-v60j`; der v2.75.41-
Validator erwartete stattdessen ausschließlich den synthetischen Testwert
`private/imagenet-calibration`. Manifest- und Item-Hashes entsprachen bereits
der eingefrorenen B500-Autorität. Reprovisionierung wäre deshalb falsch.

Unabhängig davon startete der dreimodellige Nachtlauf neun Remote-Aufträge,
obwohl der materialisierte Plan keine Full-Quality-Endpunktidentität enthielt.
Alle neun brachen vor kanonischen Benchmarkzeilen mit
`expected exactly one setup-local TensorRT quality-canary endpoint identity`
ab. Hardware-, SSH- und Storage-Preflights waren nicht die Ursache.

## Implementierte Reparaturen

- `build_source_manifest.py --verify --scope installed` attestiert weiterhin
  jede manifestierte Release-Datei nach Pfad, Größe und SHA-256. Ausschließlich
  zusätzliche reguläre, nicht verlinkte Top-Level-YAML-Profile werden als
  nicht attestierter User-State gemeldet und toleriert. Zusätzliche Skripte,
  sonstige Source-Dateien, Symlinks, Missing/Changed und manipulierte
  `SHA256SUMS.txt` bleiben Fehler. Strict Release-/ZIP-Verifikation ist der
  unveränderte Default.
- Der Updater verifiziert Staging und synchronisierten Release-Baum strikt,
  stellt eigene Profile wieder her und prüft den tatsächlichen Endzustand im
  Installed-Scope. `rsync --checksum --delete --backup` entscheidet nach
  Inhalt statt nach Größe und Zeitmarke; stale Skripte werden entfernt und
  außerhalb des Tool-Verzeichnisses wiederherstellbar gesichert. Die erhaltene
  `.venv` wird anschließend ohne Netzwerk, `pip` oder Build-Backend durch den
  Standardbibliothek-Refresher aktualisiert: Nur Projekt-Metadaten,
  Editable-Pfad und alle deklarierten Console-Scripts werden atomar ersetzt,
  andere Distributionen und Venv-Daten bleiben erhalten.
- Der B500-Canary-Loader bindet die vom Produktions-Provisioner geschriebene
  Kernel-Identität über den kanonischen Slug, das echte Selection-Receipt und
  die bereits eingefrorenen Manifest-/Item-/Run-Verträge. Tests verwenden nun
  Produktionsform statt einer vollständig gemockten Fantasieidentität.
- Setup-lokale TensorRT-Quality-Companions erhalten im materialisierten
  Effective Plan jeweils eine eindeutige physische Setup- und Full-Quality-
  Endpunktidentität. Planner, lokaler Dispatch-Preflight, Runtime und
  Management-Admission teilen dieselbe fail-closed Invariante. Gefälschte,
  fehlende oder mehrdeutige Identitäten werden blockiert; fehlt anschließend
  das erforderliche Companion-Ergebnis, kann Standard oder Final nicht
  erfolgreich abschließen.
- Der ResNet50-B500/B1000-Versuchsvertrag, seine v2.75.41-Profilidentität,
  Mean/Std, EMA, Opt-Level, Validation und Quality-Policy bleiben unverändert.

## Release-Artefakte

- `onnx_splitpoint_tool/v27542_smoke.py`
- `scripts/run_v27542_small_acceptance.sh`
- `scripts/update_source_release.sh`
- `scripts/refresh_editable_install.py`
- `tests/test_v27542_release_provenance.py`
- `tests/test_v27542_installed_source_manifest_scope.py`
- `TESTANLEITUNG_2.75.42.md`
- reparierte Manifest-, B500-Provenienz-, Updater-/Venv- und TensorRT-
  Companion-Pfade samt fokussierter Regressionstests

## Finale lokale Verifikation

Die endgültigen Messwerte des finalen v2.75.42-Baums sind:

- v2.75.42 Small Acceptance: PASS; Installed-Scope-Manifestsicht grün,
  `283 passed`, aktueller v27542-Smoke `13/13`;
- fokussierte Reparaturblöcke: PASS; Updater/stdlib-Refresh `41 passed`,
  Quality-Companion/Management-Admission `64 passed` plus `144 passed`
  angrenzende Regressionen sowie B500-/B1000-Autorität `88 passed`;
- Source-Manifest: `898` Dateien; Release-Verify und Installed-Verify mit allen
  Checks wahr;
- Python-AST/Compile und Shell-Syntax: PASS; vollständiges AST-Parsing aller
  `onnx_splitpoint_tool`, `scripts` und `tests` Python-Dateien sowie
  `bash -n` für `run_v27542_small_acceptance.sh`,
  `run_local_acceptance.sh` und `update_source_release.sh`;
- deterministische A/B/C-Source-ZIPs: PASS; Archiv A/B/C werden auf dem finalen
  Baum byteidentisch gebaut und jeweils vollständig verifiziert.

Die Archivgröße steht nach dem Build im Builder-/Acceptance-JSON. Der SHA-256
steht zusätzlich im externen `<ZIP-NAME>.sha256`-Begleitnachweis; selbst-
referenzielle Archivmesswerte werden nicht in das enthaltene Dokument
eingebacken.

## Nächster Hardwarebeleg

Nach Update und hardwarefreier Abnahme zuerst den bestehenden B500-Run pinnen,
danach B1000-Preflight und genau einen B1000-Run ausführen. Die bisherigen
Preprocessing-A/B-Läufe nicht wiederholen. Nur ein vollständig erfolgreicher
Canary erlaubt anschließend einen neuen 5.000er **Final Quality (Standard+)**-
Lauf; der fehlgeschlagene Nachtlauf wird nicht als Ergebnis verwendet.
