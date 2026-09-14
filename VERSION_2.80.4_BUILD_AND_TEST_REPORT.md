# Build- und Prüfumfang v2.80.4

Identität: `v2.80.4-hailo8-context-quality-cancel-complete-debug`.

Dieser Sourcebericht definiert den Prüfvertrag. Tatsächlich ausgeführte Testergebnisse werden außerhalb der Quelle im Delivery-Verzeichnis `verification/` gespeichert; dadurch ändert das Testergebnis nicht nachträglich das bereits geprüfte Sourcearchiv.

Basis: v2.80.3-FIX5, Source-Manifest `dae6ef0c64b4fa20220f0baa312c8080de1060cc319d3ae2352e235893c7fcf4`, 1.861 Einträge ohne Abweichung vor dem Patch.

Pflichtumfang: gespeicherter Hailo8-Kontext mit realer Python-Kindprozessgrenze, CPU/GPU-Reuse ohne Compilerdispatch auf HIT, konkrete Qualitätsidentität, Cancel/Service-Closed-Abgrenzung, keine verlorenen fertigen Resultate, echte größere Debugexporte, fehlertreue Größen-/JSON-Diagnosen, Source-Manifest und Remote-Spiegel, frisch entpacktes finales Source sowie tatsächlicher isolierter .3→.4-Updater. Fortgeführte CPU-Referenz-, Native-, Remote-, Cache- und Fehlerregressionen bleiben enthalten.

Hardwareausführung in dieser lokalen Softwareabnahme: `NOT_RUN`. Hailo-GPU-Diagnoseergebnisse aus früheren Läufen bleiben historisch. Ein Quality-FAIL oder INCONCLUSIVE wird nicht als Toolfehler oder PASS umgedeutet. COMPILE_INFEASIBLE bleibt von TRANSIENT_INFRASTRUCTURE getrennt. Force AUS, feste B500-Rezepte und Qualitätsmargen bleiben unverändert.

Die Quellenlage zu den im Implementationsplan benannten Originalarchiven ist im Evidenceindex und finalen Bundlebericht ausdrücklich angegeben. Synthetische Reproduktionen werden als solche gekennzeichnet; insbesondere dürfen gleich große Testdaten nicht als die 123 Originalrequests bezeichnet werden.
