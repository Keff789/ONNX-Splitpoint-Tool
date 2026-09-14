# Implementierungsabgleich v2.81

Version **2.81**, Build `v2.81-hailo-cold-preflight-shared-input-layout-terminal-status`. Basis: vollständige verifizierte v2.80.4.

Produktives Force bleibt AUS; Hailo balanced/Opt1/B500/Batch8, DeepX EMA/Opt0/B500, Bootstrapmethodik und Qualitätsmargen bleiben erhalten. `COMPILE_INFEASIBLE` ist nur mit exakter vorhandener Evidenz eine abgeschlossene Nichtausführbarkeit; `TRANSIENT_INFRASTRUCTURE` bleibt technischer Fehler. Statusachsen trennen Ausführbarkeit, technische Ausführung, Qualityentscheidung und wissenschaftlichen Scope.

Neue Zielhardware-Ausführung: **NOT_RUN** bei Auslieferung. Die reale Zielabnahme führt der gemeinsame Starter durch. Energie-Finalfreigabe und erneute 5.000-Bilder-Kampagne sind nicht Teil dieses Korrekturlaufs. Tatsächliche Softwaretests, Gegenproben, Quellarchivprüfung und isoliertes Upgrade stehen in `VERIFICATION_V281.json` des Bundles mit Logs und JUnit-Dateien.

| Auftrag | Produktpfad / Nachweis |
|---|---|
| Compiler | H8-Konfigurationsübernahme und finale Kaltjob-Vorprüfung; keine GPU-Prüfung für HIT/gebundenen Reject |
| H8 Bildparität | Dual-Native-Runner und identischer Remote-Spiegel; einmalige RGB-Vorbereitung und SHA-Bindung |
| H10 Layout | Backendadapter; reale NWC-Puffer korrekt nach NCW transponieren, Quantisierung unverändert |
| Terminalstatus | Originale Build-Negative als explizite Nichtausführbarkeit, echte Fehler weiter blockierend |
| Ergebnisidentität | Äquivalente Aliasse/Digestdarstellungen, widersprüchliche Identitäten weiter ablehnen |
| Betrieb | Gemeinsamer Installer/Normalworkflowstarter, begrenzte Stufen, ein Evidence-ZIP |
| Knowledgebase | Vollständige REV3 fortgeschrieben, Scope-/Stopregel und historische Messwerte erhalten |
