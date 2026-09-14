# v2.79.28 – DeepX-Full-Numerikkorrektur und Kurzprobe

Build: `v2.79.28-decoded-score-roundoff-staged-probe`.
Hardwareabnahme im Liefercontainer: **NOT_RUN**.

## Installation und nächster Hardwaretest

GUI und andere Messungen vorher schließen. Aus dem entpackten vollständigen
v2.79.28-Lieferbundle zuerst `bash ./install_v27928_and_collect_acceptance.sh`
ausführen, danach `bash ./run_deepx_full_probe_v27928.sh`.
Die Installation verwendet den vorhandenen Updater ohne pip-/Venv-Neuaufbau.
Der Hardwaretest ist ausdrücklich eine Diagnose, keine Energie-/Quality-Abnahme.
Er nutzt das bestehende DXNN und das ursprüngliche Bild 000000212226.jpg aus dem
v2.79.26-Lauf. Gelöschte Remote-Laufverzeichnisse sind nur Herkunftsangaben.
Benötigte lokale Dateien werden in ein neues temporäres Remote-Verzeichnis
kopiert. Es findet keine Kompilierung und kein Gesamtlauf statt.

Erwartetes Erfolgssignal: `PROBE_STATUS=diagnostic_pass`, NMS abgeschlossen und
`SCORE_NORMALIZATION` mit beobachteten Korrekturzählern. Die Zahl 21 ist nur für
die bisher aufgezeichnete Ausgabe belegt, nicht für jeden zukünftigen Aufruf
fest vorgegeben. `MODEL_ACCEPTANCE=NOT_EVALUATED_BY_DIAGNOSTIC` bleibt korrekt.

Das ausgegebene `DIAGNOSTIC_ZIP` enthält unveränderte Ausgaben und das Ergebnis.
Bei einem Fehler keine Caches löschen und nicht blind neu kompilieren.

## Numerikregel

Nur neu gebundene `decoded_pre_nms`-Float32-Klassenscores erhalten die feste
absolute Randtoleranz `2^-23` (0.00000011920928955078125). Werte darunter/oberhalb
der erweiterten Grenzen, negative Boxbreiten/-höhen und NaN/Inf bleiben Fehler.
Erlaubte Randreste werden vor dem Harness auf einer Verarbeitungskopie zu 0/1
normalisiert. Andere Datentypen und historische Verträge bleiben strikt.
Keine Änderung an AP-Margen, Confidence=0.25, IoU=0.45, max_det=300 oder NMS.
Kein zusätzliches Sigmoid und keine neue Pipeline-Parallelität.

## Offline-Abnahme

`bash scripts/run_v27928_small_acceptance.sh --report /tmp/v27928_acceptance.json`
führt die Release- und Verhaltensprüfungen aus. Für das reale Tensor-Replay setzt
das Lieferbundle `V27928_REPLAY_NPZ` auf seinen aufgezeichneten Testtensor.
Ohne diese externe Testdatei werden nur die explizit datenabhängigen Tests
übersprungen; synthetische Randwert- und Sicherheitsprüfungen bleiben aktiv.

Die Herstellermodelle sind nicht Teil der Offline-Fixtures. Bei den
Transport-/Hardwarepfadtests wird die DXRT-Engine ersetzt; Harness, NMS und
Suite-Code sind echt. Einzelbild-Replay ist keine B500-Nichtunterlegenheitsprüfung.

Cache-Verhalten bleibt unverändert: `COMPILE_INFEASIBLE` ist weiterhin negative
Build-Evidenz, `TRANSIENT_INFRASTRUCTURE` bleibt retrybar. Gültige DXNN-, HEF-
und TensorRT-Buildartefakte werden durch diese Numerikregel nicht neu definiert.
