# Release scope v2.80.3

Build-ID: v2.80.3-build-readiness-native-not-started-debug-export.
Basis ist das gelieferte v2.80.2-Source, SHA
2f186954c27f2b28520c46cea1c86c83a200c4d7d4821e4b59bebaa7ec1eb465.

AP1 korrigiert Deferred-Buildbereitschaft und exakt gebundene Primärursachen.
AP2 erhält explizite Native-Nichtstarts mit strenger Versuchsevidence.
AP3 ergänzt sichere kompakte Runtime-Diagnosen und ehrliche Index-/Exportlimits.
AP4 prüft reale Generator-/CPU-ORT-/Consumerintegration und bereitet die kurze
Zielkette vor. AP5 zeigt archivierte Messkonfiguration; AP6 erhält Fachregressionen,
Installationsbestand und konsistente aktive Releaseidentität.

Die v2.80.2-Modellbindung, TRT-Full-Qualityblockierung, Dashboardzählung,
Remote-Closure, Hailo8-Kindumgebung und Runtime-Cleanup bleiben erhalten.
Force AUS, relaxed und passende Artefakte bleiben unverändert. CPU-HEF-Reuse
funktioniert vor unnötiger GPU-/Compilerprüfung. Keine neue Hash-/Seal-/Registry-,
Negative-Cache-, Decoder-, NMS-, Mean/Std-, Kalibrations- oder Energiepolicy.
COMPILE_INFEASIBLE und TRANSIENT_INFRASTRUCTURE bleiben unterschiedliche Befunde.

Zusätzlicher konkret reproduzierter Fehler: ein externer Scientific-Replay konnte
über den COCO-Posthook in den historischen Run schreiben. Der Posthook erhält
jetzt das explizite Reportziel; direkter Standardaufruf bleibt erhalten.

Die 64 Plananforderungen sind keine Testanzahl. Die finale Abnahme zählt tatsächliche
JUnit-Fälle auf frischer Quelle und nach echtem isoliertem 2.80.2→2.80.3-Upgrade.
Skips/Xfails und fehlende Pflichtabhängigkeiten gelten nicht als PASS.
Nur die versionsgebundene Closure wird fortgeschrieben, fachliche Regressionen
bleiben enthalten. Details stehen im Implementierungsabgleich und Prüfbericht.

Hardware execution: NOT_RUN. Physische Producer-/Native-Positive, B500/B5000,
Hailo8-GPU und finale Energie-/Rankingclaims bleiben getrennte Zielgates.
Quality-FAIL oder INCONCLUSIVE dürfen korrekte Ergebnisse sein.
1 s × 3 ist Screening; 30 s × 3 und 60 s × 3 brauchen eigene beobachtete Aufträge.
FS/command, GPUpräferenz, tatsächliche GPUausführung, Setupdaten und wissenschaftliche
Eignung bleiben getrennt. Keine automatische Geräteumschaltung.

Historische Originale behalten Version, Quelle und Werte. Der alte 63er-Replay
hat 55 erfolgreiche und acht negative Fälle; die neue 84er-Nachtprojektion
21 erfolgreiche und 63 negative Fälle. Die 42 Split-Nichtstarts werden weder
zu Teilmessungen noch zu Erfolg umgedeutet. Fehlende alte große Originalkörper
bleiben fehlend; 43 synthetische große Quellen prüfen die neue Writer-/Exportkette.

Kompakte Evidence für das Ergebnis-Git enthält keine Modelle, Bilder, HEFs,
DXNNs, Engines oder Roharrays. Die ursprünglichen ZIPs sind über ihre SHA
gebunden; erforderliche kleine Original-/Derived-Fixtures sind gekennzeichnet.
