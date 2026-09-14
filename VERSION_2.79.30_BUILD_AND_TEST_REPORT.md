# Releasevertrag 2.79.30

Build: `v2.79.30-deepx-full-terminal-closure`.

Dieses Source-Dokument beschreibt den unveränderlichen Prüfumfang. Tatsächliche Prüfergebnisse des endgültigen Archivs stehen im Lieferbundle in `PRUEFBERICHT_2.79.30.md` und `verification/`; sie werden nicht nachträglich in den geprüften Sourcebaum geschrieben.

Erforderlich sind die 51 Prüfgruppen aus REV2, gesamte bisherige Verhaltensabnahme, frisch entpacktes Sourcearchiv, isoliertes Update 29→30 mit Erhalt der Venv, eigener Profile/Konfiguration und synthetischer Artefakt-/Evidenzbestände, sowie der echte lokale Abschluss-Smoke. Die Zählungen unterscheiden bestanden, fehlgeschlagen, übersprungen und separat ausgeführt. Keine fehlende Testabhängigkeit wird als erfolgreiche Abnahme gewertet.

`COMPILE_INFEASIBLE` bleibt dauerhaft gebundene negative Evidenz; `TRANSIENT_INFRASTRUCTURE` bleibt wiederholbar. Beide bisherigen Klassen werden weiter geprüft. Qualitätsregeln, Float32-Toleranz und Buildidentitäten bleiben erhalten.

Reale DeepX-Hardwareabnahme H2/H3, neuer B500-Lauf und 3/3 Full-System-Energiematrix: **NOT_RUN**, bis die entsprechenden neuen Gerätedaten vorliegen. Offline-Replays und synthetische Prozess-/Abschlussmessungen sind keine Hardwaremessungen.
