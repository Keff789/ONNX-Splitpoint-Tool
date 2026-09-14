# Implementierungsabgleich v2.80.4

Build: `v2.80.4-hailo8-context-quality-cancel-complete-debug`.

Die Umsetzung basiert auf dem vollständigen v2.80.3-FIX5-Source. Die Arbeitspakete des Plans vom 12.09.2026 werden anhand der tatsächlich ausgeführten Tests und verfügbaren Originalquellen im finalen Delivery-Bericht zugeordnet.

| Paket | Produktumfang |
|---|---|
| AP0 | Manifestgeprüfte .3-FIX5-Basis; Quelle und Reproduktion getrennt |
| AP1 | 512 MiB Requestsumme, 64 MiB Requestdatei, 256 MiB strukturierte Ergebnis-/Indexdatei; vollständige gebundene dekodierte Qualitydaten |
| AP2 | Terminal, ausgewertet, abgebrochen und technisch fehlgeschlagen getrennt; keine falschen Qualityentscheidungen durch Cancel |
| AP3 | Optionaler gespeicherter Hailo8-Overlaypfad, klare Vorränge, echte Kindumgebung |
| AP4 | Reuse vor GPU-Prüfung, Recipe- und konkrete Qualityidentität getrennt |
| AP5 | Historische Hailo8/Hailo10-Nachweise und negative Qualitätsbefunde korrekt dokumentiert |
| AP6 | Neue .4-Identität, vollständige Regressionen, Source-/Upgradeprüfung und zusammenhängender normaler Abnahmestarter |

Die historische Entscheidung gegen einen weiteren Level-/Kalibrierungssweep bleibt bestehen. Opt1/B500/Batch8 und Force AUS werden nicht verändert. YOLO11l/Hailo10 Full und b062 sind auf 5.000 Bildern zentral ausgewertet und negativ; daraus wird keine Pflicht zum Neubau abgeleitet.
