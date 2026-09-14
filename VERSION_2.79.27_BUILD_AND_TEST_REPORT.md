# Version 2.79.27 – Build- und Testbericht

Build: `v2.79.27-deepx-output-probe-result-fixes`.

Basis ist die unveränderte 2.79.26 mit der Diagnose `decoded_pre_nms_values_invalid`. Diese Version ergänzt gezielte Ausgabediagnostik und korrigiert die zugehörigen Ergebnis- und Statuszuordnungen sowie die Übergabe des ursprünglichen Full-Fehlers. Die vorbereitete Eingabe wird vor den Performance-Wiederholungen einmal geprüft und explizit weiterverwendet.

## Prüfstatus

- Software-Regression: Die Freigabe erfordert alle Prüfungen aus `scripts/run_v27927_small_acceptance.sh`. Die ausgeführten Ergebnisse stehen in `verification/small_acceptance.json` und dem zugehörigen Log des Lieferpakets.
- Isoliertes Update 26→27: Die Freigabe erfordert den echten Installerlauf auf einer separaten 26er-Testinstallation einschließlich Prüfung der erhaltenen Dateien. Das Ergebnis steht in `verification/upgrade_preservation.json` des Lieferpakets.
- Echte DeepX-Full-Ausführung, Full-Quality und Hardware-Energie: **NOT_RUN**.
- CPU/GPU-Compilerzeitvergleich: NOT_RUN.

Die Regression behält die relevanten 26er-Tests für decodierte Full-Ausgaben, nachgelagerte Prüfung, Quality-Worker und Debug-Packs sowie frühere Kalibrierungs-, Cache- und Negative-Evidence-Tests bei. Die alte Versionsidentitätsprüfung wird durch die 27er-Identitätsprüfung ersetzt.

Vorhandene `COMPILE_INFEASIBLE`-Evidenz bleibt erhalten. `TRANSIENT_INFRASTRUCTURE` bleibt ein Infrastrukturfehler und wird nicht als dauerhaft unmöglicher Build abgelegt. Historische Ergebnisse werden durch das Update nicht neu bewertet oder überschrieben. Quality-Toleranzen bleiben unverändert.

Die echte numerische DeepX-Ursache kann erst mit den auf dem Jetson erfassten Werten abschließend bestimmt werden. Diese Version weist kein vorweggenommenes numerisches Repair-PASS aus.
