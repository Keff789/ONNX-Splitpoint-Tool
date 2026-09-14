# v2.80.3 – Build- und Testumfang

Build-ID: `v2.80.3-build-readiness-native-not-started-debug-export`.
Ausgangsbasis: geliefertes v2.80.2-Source mit SHA
`2f186954c27f2b28520c46cea1c86c83a200c4d7d4821e4b59bebaa7ec1eb465`.
Der finale äußere `PRUEFBERICHT_2.80.3.md` enthält tatsächlich ausgeführte
Testzahlen und Quellprüfsummen. Dieses Quelldokument enthält keine vorgezogene PASS-Zahl.

## Enger Korrekturumfang

- AP1: Deferred-Pflichtjobs in die tatsächliche Buildbereitschaft einbeziehen;
  exakt gebundene Ursachen bis in Native-Prerequisites weiterreichen.
- AP2: explizite Vorabblockaden mit null Versuchen erhalten; Readerdefaults,
  widersprüchliche Messdaten und echte Teilreplikate getrennt behandeln.
- AP3: Indexgrößenlimits ehrlich benennen; bereits geladene Resultate kompakt
  projizieren und im tatsächlichen Debugexport getrennt von Originalen bilanzieren.
- AP4: bestehende CPU-Referenz-/Consumerkette wirklich ausführen und den kurzen
  Zielworkflow vorbereiten; kein vierter Referenzdispatcher.
- AP5: effektive Konfiguration sichtbar machen, numerische Diagnosen und
  wissenschaftliche Eignung getrennt halten.
- AP6: Fachregressionen fortführen, endgültige Sourcebytes und tatsächlichen
  isolierten 2.80.2→2.80.3-Installer samt ausgelieferten Startern prüfen.

Die v2.80.2-CPU-Modellbindung, TRT-Full-Qualityblocker und der Dashboardfix bleiben
erhalten. Produktives Force bleibt AUS. Compilerrezepte, relaxed,
Cacheidentitäten, Mean/Std, Decoder und Kandidatenauswahl werden nicht geändert.
Vorhandene CPU-HEFs bleiben auch bei GPU-Präferenz verwendbar.

`COMPILE_INFEASIBLE` benötigt passende negative Originalevidence;
`TRANSIENT_INFRASTRUCTURE` und unklare Zustände bleiben davon getrennt.
Quality-FAIL oder INCONCLUSIVE dürfen korrekt berechnete Ergebnisse sein.
Buildentscheidung, Artefaktverfügbarkeit, Runtime, Quality und Claim sind
unterschiedliche Aussagen.

## Prüfgrenzen

Die Softwareabnahme verwendet reale Generator-/Suite-/ONNX-Runtime-Inferenz,
Consumer, Reader, Workflowstufen und ZIP-Publikation. Externe Compiler,
SSH/Transport und physische Acceleratorgrenzen können kontrolliert ersetzt sein.
Pflichtabhängigkeiten, Skips/Xfails und nichtnull Returncodes sind Fehlergates.
Physische Hardwareabnahme in der Lieferumgebung: **NOT_RUN**.

Die Nacht `completsetdev_20260910_210826` lief unter 2.80.1.
Replays verändern weder Originaldateien noch ihren historischen Status.
Die 84er-Nachtmatrix und die frühere 63er-Matrix sind unterschiedliche Fixtures.
Erfolgreiche Vendor-Full-Replikate bleiben Messungen der alten Version.
B500/B5000-, GPU-, Native-Energie- und Rankingfreigaben bleiben getrennt offen.
1 s × 3 ist keine 60-s-Finalmessung; Profilnamen ersetzen keine Zeilenevidence.

Alle 64 Plananforderungen werden in `IMPLEMENTIERUNGSABGLEICH_2.80.3.md` mit
Nachweisen und Zielsystemgrenzen zugeordnet. Befehle und Reihenfolge stehen in
`TESTANLEITUNG_2.80.3.md`.
