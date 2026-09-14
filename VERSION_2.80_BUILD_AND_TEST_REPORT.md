# Build- und Testbericht v2.80

Basis ist das unveränderte v2.79.34-Sourcearchiv mit SHA256
`4262a4c2500ff99168dc89b15906c196156414bb89683623cac4e656a19bc78f`.
Das Ausgangsmanifest wurde vor Änderungen mit 1627 Dateien und ohne Abweichungen
geprüft. Versionsnummer `2.80` gemäß Benutzerauftrag; der beigefügte Originalplan
nennt v2.79.35 als damaligen Vorschlag. Die Entwicklungslinie bleibt historisch v2.79.

## Änderungen

- Hailo8: validiertes Dependency-Overlay wird an die echte Compute-Kindumgebung
  übergeben und vor dem Spawn erneut gebunden geprüft. Parent und H10 bleiben
  getrennt. Dokumentierte Umgebung entspricht dem tatsächlichen Worker-env.
- Runtime: Prozess- und Stagingcleanup werden getrennt berichtet. Löschung mit
  begrenztem Transport und Abwesenheitsprüfung. Primärfehler bleibt erhalten;
  Bericht, ZIP, G3 und Rückgabecode sind auch bei Fehler/Abbruch konsistent.
  Das vollständige ZIP wird erst nach Abschluss atomar veröffentlicht; kein finales
  Teilarchiv mit veraltetem PASS nach einem Schreibfehler.
- Hailo: Bei Cache-only mit fehlender Managed-DFC-Metadatenidentität wurde unnötig
  das SDK im Controller importiert. Dieser rote Sonderfall ist korrigiert:
  `cache_miss_blocked/compiler_identity_unavailable` ohne SDK-/GPU-/Compileraufruf.
  Der allgemeine normale Cachepfad war bereits auf v34 korrekt: Build mit echten
  Graphfixups, reguläre Publikation, anschließend zwei frische Prozesse mit
  GPU-Präferenz ergeben einen HIT. Dieser Nachweis wird als Regression fortgeführt;
  ein allgemeiner Cacheidentitätsfehler wird nicht behauptet.
- Ein begrenzter Reusestarter verwendet den bereits gebundenen v34-MobileNet-
  Request für zwei reine Cacheprüfungen. Keine private HEF-Promotion und kein Neubau.
- Buildberichte: strukturierte Ergebniszeilen werden nicht als Kalibrationsfortschritt
  fehlgedeutet. Private nichtpublizierende Diagnosejobs erhalten einen sachgerechten
  Grund statt einer falschen Meldung über fehlende Compileridentität.
- Aktuelle Starter, Smoke, Paketmetadaten und Abnahme führen v2.80. Historische
  Fixtures/Requestschemata bleiben erhalten. Keine zweite Runtimeimplementierung.

## Verbindliche Nachweise

Die endgültigen tatsächlichen JUnit-Ergebnisse, roten Ausgangsregressionen,
Source-Deltaliste, frische Paketprüfung, isolierte v34→v2.80-Installation und
Bestandserhaltung liegen im vollständigen Bundle unter `verification/` und im
`PRUEFBERICHT_2.80.md`. Testumfänge werden nicht über Wiederholungsläufe addiert.
Ein Source-Smoke oder synthetischer Compilergrenzentest ist kein Hardware-PASS.

Produktives Force AN bleibt gesperrt. Eigene Profile/Registry werden erhalten,
keine stille Profilreparatur. Der private Diagnosebuild publiziert nicht. HAR-
Emulationssammlung bleibt nicht implementiert, freie origin ist kein Herkunftsbeweis.
Die Vorbereitung ist weiterhin `first_selected_model_only` und besitzt keine volle
Quality-FIRST-/TensorRT-Bindung (`quality_first_trt_binding_ready=False`).

## Bereits vorhandene v34-Zielsystemnachweise

Hailo10-Rechenprobe und echter privater MobileNetV3-GPU-Build bestanden.
Fixed16 auf demselben H10-Setup: Source/Compiler-Float jeweils Top1 13/16,
CPU-/GPU-HEF jeweils Top1 12/16 und Top5 15/16. CPU/GPU-HEF unterscheiden sich
numerisch; eine Top1-Klasse wechselt zwischen zwei falschen Antworten.
Diese Befunde schließen den damaligen technischen G3, keine allgemeine
Qualitätsfreigabe. Die aktualisierten Cleanupfelder stammen erst aus v2.80 und
werden nicht rückwirkend in alte Ergebnisse hineingelesen.

Reale v2.80-Zielinstallation, H8-/H10-Gegentest, erneuter Runtimecleanupnachweis,
reguläre Publikation/Neustart-Reuse und Nachtvorbereitung bleiben bis zur tatsächlichen
Zielsystemausführung offen. Kein erneuter Coldbuild nur wegen geänderter
Computepräferenz. Bestehende gültige CPU-HEFs bleiben verwendbar.

## Fortgeführte Ergebnisidentitäten

`COMPILE_INFEASIBLE` bleibt eine belegte negative Recipe-Evidence,
`TRANSIENT_INFRASTRUCTURE` ein retryfähiger Umgebungs-/Transportfehler.
Weder unbekannte Zustände noch fehlende GPU-Toolchains dürfen gültige Cache-HITs
in Coldbuilds umdeuten. Reale neue Hardwareausführung in der lokalen Abnahme:
`NOT_RUN`. Releaseidentität: `v2.80-hailo-reuse-env-cleanup`.
