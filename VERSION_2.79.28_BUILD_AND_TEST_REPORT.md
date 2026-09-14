# v2.79.28 – Implementierungs- und Testumfang

Build: `v2.79.28-decoded-score-roundoff-staged-probe`.
Basis: hochgeladenes vollständiges v2.79.27-Lieferbundle.
Hardwareausführung im Liefercontainer: **NOT_RUN**.

## Änderungen

- Feste, dtype- und endpointgebundene Float32-Randtoleranz für deklarierte
  `decoded_pre_nms`-Klassenscores; nur erlaubte Score-Randwerte werden auf einer
  Verarbeitungskopie normalisiert. Rohausgaben und Koordinaten bleiben erhalten.
- Dieselbe Funktion in FrozenDetectionPostprocessor, CompletionRuntime,
  schnellem Three-Stage-Postprocessing, Full-Probe und Full-Quality-Decoder.
  Korrekturzähler stehen in den vorhandenen JSON-Ergebnissen.
- Kleine zusätzliche Policy-Angabe im bestehenden Frozen-Contract und dessen
  bestehender Identität. Keine neue Dateiart oder Hash-Artefaktverwaltung.
  Alte v26/v27-Verträge bleiben mit exakt ihrer strikten Regel lesbar.
- FIX1-Staging im aktuellen v28-Launcher: vorhandenes Modell und Originalbild,
  eigener Remote-Ordner, begrenzter Lauf, korrekte Weitergabe von Setup-Fehlern.
- Release-Metadaten, aktuelle Smoke-/Acceptance-Aliase, Updater und README auf 28.

Die in v27 enthaltenen Eingabe-/Ergebnis-/Statuskorrekturen bleiben erhalten.
Keine Änderung an Compilerinstallation, Build-Key, Cache-/Retention-Politik,
Qualitätsmargen, NMS-Schwellen oder Messparallelität. `COMPILE_INFEASIBLE` und
`TRANSIENT_INFRASTRUCTURE` behalten ihre bestehenden Bedeutungen.

## Prüfnachweise

Die endgültigen ausgeführten Testzahlen, eventuelle Umgebungs-Skips und der
Installationsversuch stehen im vollständigen Lieferbundle unter
`verification/FINAL_VERIFICATION.json` und in den dortigen Originalprotokollen.
Der reale Tensor plus Eingabebild liegt ausschließlich als Replay-Fixture im
Lieferbundle, nicht im installierten Source-Inventar oder Modellcache.

Das Replay reproduziert zuerst den ursprünglichen strikten Fehler und prüft
anschließend beide Tensorlayouts bis zu den endgültigen NMS-Detektionen. Ein
unabhängiger Harness-Aufruf mit einer manuell korrigierten Verarbeitungskopie
muss exakt dieselben Datensätze liefern. NaN/Inf, echte Bereichsverletzungen,
negative Breiten/Höhen und manipulierte Policies müssen weiter scheitern.

Die Tests simulieren DXRT/Transport, wo Hardware notwendig wäre. Sie belegen
keine Live-FPS, keine Energiemessung und kein Quality-PASS für das Gesamtmodell.
