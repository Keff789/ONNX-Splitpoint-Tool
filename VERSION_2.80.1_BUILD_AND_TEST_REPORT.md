# Build- und Testbericht v2.80.1

Identität: `2.80.1` / `v2.80.1-cpu-reference-remote-closure-debugexport`.
Ausgangsquelle ist exakt das v2.80-Sourcearchiv mit SHA256
`4d25750c84f85d31c9eea03dc56ffd80ea7a317c67067b4d05daeacfe90193af`;
Ausgangsmanifest: 1.648 Dateien, keine Abweichungen.

## Reparaturen

1. CPU-Referenzrolle wird am Suite-Dispatch eng geprüft. Nur die vollständige
   semantische CPU-Full-Recipe wird vom Hardware-Kandidatengate ausgenommen.
   Modell-/Datensatz-/Quellbindung und normale Generic-Kandidatengates bleiben bestehen.
2. Native-Paketbestand umfasst die fehlende Qualitätsabhängigkeit und benötigte
   eigene Paketinitialisierer. Alle aktiven Transferpfade übertragen den vollständigen
   Bestand vor dem ersten fachlichen Import und prüfen Zielpfad/Bytes.
3. Referenzprozessfehler behalten konkrete Primärursache, Rückgabecode und Logpfad
   durch Status, normalen Logcallback und zentrale Ergebnisdiagnose. Timeout,
   Abbruch, fehlende Ausgabe und Publikationsfehler bleiben unterscheidbar.
4. Debugexport nimmt genau die kleinen direkten Referenzstatus-/stdout-Diagnosen
   auf, mit Originalbytes und begrenzten Budgets. Körper und Workspaces bleiben außen.

## Abnahme und Grenzen

Tatsächliche JUnitzahlen, rote Baselinebefunde, Neu-/Alt-Testauswahl, Source-Delta,
Manifest-/Spiegelprüfung und isolierter 2.80→2.80.1-Installer stehen im ausgelieferten
`PRUEFBERICHT_2.80.1.md`, `IMPLEMENTIERUNGSABGLEICH_2.80.1.md` und `verification/`.
Wiederholte Testläufe werden nicht zu einem größeren Testumfang addiert.

CPU-Inferenz verwendet reale ONNX Runtime und synthetische Softwarefixtures.
Externe Acceleratorgrenzen und Transport werden in den Offlineprüfungen ausdrücklich
lokal nachgebildet. Neue Zielhardwareausführung: `NOT_RUN`. Reale G1–G5 bleiben
Zielsystemgates; fehlende Gates gelten nicht als PASS. Vorherige v34-GPU-/Fixed16-
Nachweise werden weder wiederholt noch in neue Hardwarefreigaben umgedeutet.

Historische BiggerSet-Fehlerdateien bleiben unverändert. 60 abhängige Qualityfehler
waren keine 60 unabhängig gemessenen Accuracy-FAILs. Schlechte neue Kandidaten
bleiben als echte FAIL/INCONCLUSIVE sichtbar; keine Metriken aus fehlender Arbeit.

Force AUS/Sperre, H8-Computeenv, Cleanup, Reuse vor Compilerprüfung, `relaxed`,
Mean/Std und bestehende Energieabgrenzung bleiben Regressionen. `COMPILE_INFEASIBLE`
und `TRANSIENT_INFRASTRUCTURE` behalten getrennte Bedeutung. Keine Änderung von
Modellbuildidentität, Kalibration, Qualitätsmargen oder Decoder/NMS auf Verdacht.
