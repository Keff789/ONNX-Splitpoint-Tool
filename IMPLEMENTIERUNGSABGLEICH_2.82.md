# Implementierungsabgleich beider Pläne – v2.82

Build: `v2.82-selected-energy-generic-roles-workspace-product-evidence`  
Basis: unverändert geprüfte v2.81-Lieferung; beide Implementationspläne vom 13.09.2026.

Der abschließende Prüfstand des exakt ausgelieferten Archivs, die Testzahlen und die isolierte Upgradeprüfung stehen im äußeren Lieferbundle in `VERIFICATION_V282.json` und `BUILD_AND_TEST_REPORT_v2.82.md`. Dieser Sourcebericht beschreibt die Korrekturen und deren fachliche Grenzen. Reale Hailo-/CUDA-/u.RECS-Ausführung während der Entwicklung: **NOT_RUN**.

## Umgesetzte Korrekturen

| Bereich | Verhaltensänderung | Nachweis |
|---|---|---|
| Energy-Retry | Logischer Repeat und gewählter physischer Versuch werden eindeutig verbunden; genau dessen stdout, Command-, Nonce-, Fenster- und Work-Unit-Belege werden geprüft | Beide historischen Fehler zuerst mit v2.81 reproduziert; korrigierter vollständiger Import aller 115 Original-Checkpoints bestanden |
| Energy-Reimport | Checkpoints und Originalaggregate bleiben unverändert; separate, idempotente Projektion mit belegtem historischen Start und Rückgabecode | Originalprojektion 113→115 importierbare Zeilen; keine neue Messung und keine Rohtrace-Neuberechnung |
| Generic-Ausschlüsse | Explizite Part2-only-Messung widerspricht einem exakten Composed-Compileausschluss nicht; Composed-Start/-Erfolg oder unklare Rollen bleiben streng | Echter Reader und produktive Pflichtmatrix mit Originaldaten: 6 Ausschlüsse, 3 verbleibende Missing; 140 Zeilen erhalten |
| Workspace | Tatsächlicher Cold-Auftrag liefert Shapes/Kalibration/Arbeitswurzel für frühe Prüfung; vor Dispatch erneute Prüfung | Vier Originalberichte, getrennte 640er-/224er-Anforderung, Warm-/Unknown-/Permission-/Inode-Regressionen |
| Cache/Auswahl | Fehler bleiben requestlokal; abhängige TRT-Prüfung wartet nur auf ausdrücklich erlaubten Producer-MISS; Fallzahl pro Modell | Gemischte Warm-/Cold-/Negative-Pfade und unterschiedliche explizite Fallzahlen |
| H10-Runner | Die erzeugte Session nutzt den korrigierten Paketadapter; veraltete Skripte werden erneuert, bestehende HEFs/Engines bleiben nutzbar | Tatsächlich erzeugte Sessionklassen und sechs originale NWC-Puffer; dies beweist keine AP-Verbesserung |
| H8-Dumps | Untimed finaler Snapshot nach Synchronisierung; getrennte Repetitionsdateien und eindeutige Referenzbindung | Pufferüberschreiben, unterschiedliche Frames/Repetitionen und negative Dumpprüfungen |
| H8-YOLOv7-Arithmetik | Reproduzierte Unterschiede zwischen NumPy-Dispatchvarianten werden durch explizite Berechnung und Materialisierung im vorhandenen Decodervertrag begrenzt | Alter Vertrag bleibt historisch rekonstruierbar; keine neue Toleranz oder veränderte Schwelle; betroffene neue Qualitätsbindungen erforderlich |
| Full-Qualität | Vorhandener zentraler Completed-Endpunkt wird zusätzlich im bestehenden Binding weitergereicht und gegen die attestiere Fortsetzung geprüft | Beide originalen YOLO26m-Full-Fälle; alte Rohendpunkthashes erhalten, ursprüngliche Qualitätsresultate unverändert |
| Status | Auswertung abgeschlossen und Quality-FAIL getrennt; gültige Negativergebnisse bleiben nutzbar; Fallfehler ergeben partial | Gemeinsame GUI-/CLI-/Manifest-/Reportprojektion; globale Abschlussfehler bleiben failed, Abbruch cancelled |
| Debug | Initiale und ausgewählte Retrylogs samt unveränderter Auswahlhistorie werden erfasst; fehlende Belege bleiben sichtbar | Tatsächlicher ZIP-Roundtrip; keine Modelle oder Rohtensoren als neue Diagnosepflicht |

## Was die Originaldaten-Replays beweisen

- Native unverändert: **126 = 115 erfolgreich + 6 ausgeschlossen + 5 blockiert**, davon 42 erfolgreiche Full-Baselines. Keine neuen Runtime-Erfolge wurden erfunden.
- Zentrale Qualität unverändert: **150 = 81 PASS + 46 FAIL + 23 INCONCLUSIVE**, aufgeteilt in 129 primäre und 21 begleitende Aufträge. Das ist Screening mit 500 Bildern, kein 5000er-Finalnachweis.
- Generic: sechs exakt passende Ausschlüsse werden zusätzlich richtig berücksichtigt; die vorherigen neun Missing sinken auf drei. Die zwei ungültigen H10-Endpunkte bleiben getrennte technische Befunde.
- Energy: vollständige ursprüngliche Completion-/Importkette aller 115 Aufträge offline geprüft. In der abgeleiteten Reparatur der beiden fehlerhaften Zeilen bleiben die anderen 113 Zeilen unverändert. Ursprüngliche Fehlversuche, alte negative Importentscheidungen und physischer Messscope bleiben erhalten.
- Full-Binding: zwei passende zentrale Resultate werden ohne Rechenwiederholung korrekt zugeordnet. Die historischen vier YOLOv7-Dumpkonflikte werden dadurch nicht gelöscht.

## Verbleibende Zielsystemgates

H10 b398/b364 benötigt weiterhin die echte gebundene A–F-Kette. Ein gültiger Layouttransport, vollständiger Capture oder vorhandener HEF bedeutet nicht, dass Scores und Boxen korrekt sind. Ungültige Endpunkte erhalten keine Performance-/Qualityfreigabe und werden nicht per Sigmoid, Clipping oder Thresholdänderung kaschiert.

Die drei noch fehlenden H8-Bauten benötigen genügend Platz im tatsächlichen Arbeitsbereich und anschließend ihren normalen Folgepfad. `COMPILE_INFEASIBLE` bleibt ein gültiger negativer Buildbefund; `TRANSIENT_INFRASTRUCTURE`, ENOSPC und Cancel bleiben erneuerbare technische Ursachen. Der erfolgreiche RegNet-b073-Build und die bereits bewiesene GPU-/PTXAS-Funktion werden nicht neu in Frage gestellt.

Bei YOLOv7 stimmen die in den Originalmanifesten erklärten Quelltensorhashes mit der Attestation überein. Da die Roharrays im Debugpack fehlen, beweist die Offline-Reproduktion der SIMD-Arithmetik nicht die exakte Ursache jedes historischen ARM-/Desktop-Vergleichs. Neue Ergebnisse mit dem korrigierten Rechenvertrag müssen über den normalen erzeugten Runner und Native-Dump geprüft werden; alte Qualitätswerte dürfen nicht darauf umetikettiert werden.

## Erhaltungsregeln

Force AUS; positive Artefakte wiederverwenden, nur zulässige echte MISS bauen. Keine neue Hash-/Seal-/Registryarchitektur. Keine individuellen Compiler-Sweeps, keine pauschale 10-%-Marge, keine Änderung von Opt1/B500/Batch8 oder DeepX Mean/Std. Qualitäts-FAIL/INCONCLUSIVE bleiben exakt erhalten. Gültige Performance-/Energiemessungen außerhalb der Qualitätszulassung bleiben beschreibend sichtbar; Qualitätsgleichwertigkeitsclaims behalten ihre bisherigen Kriterien.

Historische Dokumente und Fixtures bleiben als solche enthalten. Die neue Versionsidentität betrifft die aktive Software und neue Auswertungen, nicht die ursprünglichen Läufe. Neue Gateberichte kennzeichnen Originaldaten, synthetisch simulierte Hardwaregrenzen und tatsächlich ausgeführte Softwareprozesse getrennt.

## Zuordnung der Arbeitspakete

A1/H-01: generierter Hailo-Adapter und tatsächliche Sessions. A2: Snapshot-/Dumpbindung sowie versionierte YOLOv7-Arithmetik. A3/AP1: vollständige Energyauswahl, Recovery, Reimport. A4: Full-Completed-Endpunktbindung. A5/AP2: requestlokaler Cache, modellbezogene Auswahl und variantenbezogene Ausschlüsse. A6/AP5: gemeinsame Abschlussprojektion und Originalreplays. A7/AP3: früher und dispatchnaher Workspacepreflight. A8/AP6: Starter, finaler Archivtest, offizieller isolierter Upgrade. AP4/H-02–H-08: begrenzte echte H10-Diagnose geliefert; physischer Endpunktnachweis bleibt Zielsystemgate.

Die vollständige ID-Prüfmatrix mit konkreten Belegen steht im äußeren Bundle. Eine implementierte Diagnose ist kein bereits ausgeführter Hardwaretest.
