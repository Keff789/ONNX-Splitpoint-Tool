# Build- und Testumfang v2.80.2

Version `2.80.2`, Build-ID `v2.80.2-model-binding-native-quality-reporting`.

Dieser Patch setzt die drei belegten Korrekturen aus dem Lauf
`completsetdev_20260910_133033` um: Generator-/CPU-Modellbindung, konkrete
upstream-Quality-Blockierung von Native TensorRT Full und GUI-Modellzahlprojektion.
Das Original-Debugpaket und dessen fehlgeschlagene Ergebnisse bleiben unverändert.

Die CPU-Regression verwendet den tatsächlichen Generatorvertrag und echte kleine
ONNX-/ORT-Modelle. Negative Bindungen, Referenzreuse und die zentralen Übergänge
werden mitgeprüft. Native- und Dashboardregressionen beziehen sich auf Original-
Diagnosen beziehungsweise ausdrücklich gekennzeichnete strukturtreue Auszüge.
Erfolg der Softwaretests ist keine neue physische Hardware- oder B500-Abnahme.

Die finalen Testergebnisse werden erst nach Prüfung des frisch entpackten Source-
ZIPs, des tatsächlich ausgeführten isolierten v2.80.1→v2.80.2-Installers und des
ausgelieferten Kurzteststarters im äußeren `PRUEFBERICHT_2.80.2.md` veröffentlicht.
Dieser Bericht enthält die tatsächlich ausgeführten Fallzahlen, JUnit-Nachweise,
Prüfsummen und Prüfgrenzen. Wiederholte Läufe werden nicht addiert.

Produktives Force bleibt AUS, `relaxed` und alle bestehenden Build-/Cacheidentitäten
bleiben erhalten. `COMPILE_INFEASIBLE` und `TRANSIENT_INFRASTRUCTURE` werden weiter
getrennt behandelt. CPU-Referenzen bleiben außerhalb der Messpopulation; Quality-
FAIL und INCONCLUSIVE werden nicht zu PASS umgedeutet. Die bekannte Hailo10H-
YOLO26-Endpunktfrage bleibt separat offen. Neue Zielhardware: `NOT_RUN`.
