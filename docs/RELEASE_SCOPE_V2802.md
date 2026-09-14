# Releaseumfang v2.80.2

Build-ID: `v2.80.2-model-binding-native-quality-reporting`.

| Belegter Fehler | Korrektur | Fortgeltende Grenze |
|---|---|---|
| CPU-Referenz vergleicht ONNX-Dateipfad mit logischer Modell-ID | Tatsächlichen Generatorvertrag konsistent lesen und unabhängige Modellbindungen prüfen | Keine Freigabe falscher Modelle oder Sourcecontracts, auch bei Referenzreuse |
| Native TensorRT Full wegen fehlender Qualitybindung ausgelassen, später als Transferfehler ausgewiesen | Konkrete geplante blockierte Zeilen samt upstream-Ursache erhalten | Keine Messstarts, erfundenen Metriken oder Gateabschaltung |
| GUI zeigt models=0 aus veraltetem overview-Schema | Aktuelle summary-Modellzahl verwenden, Legacyformat weiterhin lesen | Unbekannte Werte bleiben unbekannt; Messzeilen sind keine Modellzahl |

Die drei funktionierenden v2.80.1-Änderungen bleiben erhalten: vollständige Remote-
Paketübertragung, primäre CPU-Referenzexception in Status/zentralem Ergebnis und
kleine Referenzdiagnosen im Debug-ZIP. Artefaktwiederverwendung, Force AUS, relaxed,
Native-/Energieverträge und historische Originale bleiben unangetastet.

Quality-FAIL/INCONCLUSIVE bleiben wissenschaftliche Ergebnisse; technisch nicht
ausgewertete Qualität bleibt nicht ausgewertet. CPU-Referenzen erhalten keine
Performance-/Energieclaims. `COMPILE_INFEASIBLE` und `TRANSIENT_INFRASTRUCTURE`
bleiben getrennt. Es werden keine neuen Modellbuilds oder GPU-Freigaben behauptet.

Die Softwareabnahme verwendet echte Generator-/ORT-Pfade und kontrollierte lokale
Native-/Transferkontexte. Physische Zielhardware ist `NOT_RUN`; B500, 5.000 Bilder
und spätere FS/command-Energiemessungen sind eigenständige Abnahmen.
