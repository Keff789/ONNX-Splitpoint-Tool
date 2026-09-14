# Build- und Prüfbericht v2.81

Version **2.81**, Build `v2.81-hailo-cold-preflight-shared-input-layout-terminal-status`. Basis: vollständige verifizierte v2.80.4.

Produktives Force bleibt AUS; Hailo balanced/Opt1/B500/Batch8, DeepX EMA/Opt0/B500, Bootstrapmethodik und Qualitätsmargen bleiben erhalten. `COMPILE_INFEASIBLE` ist nur mit exakter vorhandener Evidenz eine abgeschlossene Nichtausführbarkeit; `TRANSIENT_INFRASTRUCTURE` bleibt technischer Fehler. Statusachsen trennen Ausführbarkeit, technische Ausführung, Qualityentscheidung und wissenschaftlichen Scope.

Neue Zielhardware-Ausführung: **NOT_RUN** bei Auslieferung. Die reale Zielabnahme führt der gemeinsame Starter durch. Energie-Finalfreigabe und erneute 5.000-Bilder-Kampagne sind nicht Teil dieses Korrekturlaufs. Tatsächliche Softwaretests, Gegenproben, Quellarchivprüfung und isoliertes Upgrade stehen in `VERIFICATION_V281.json` des Bundles mit Logs und JUnit-Dateien.

Die H10-Regression verwendet sechs originale physische Ausgabepuffer. Der Fehler wird mit unveränderter .4-Quelle reproduziert; v2.81 prüft den vollständigen Übergang durch beide Sessionklassen und die bestehende ONNX-Bridge. H8-Prüfungen sichern identische vorbereitete Bildbytes und deren Wiederverwendung über echte getrennte Laufzeitrepetitionen, einschließlich Fehlerclosure. Compilerprüfungen betreffen nur die finale Liste tatsächlich anstehender Kaltjobs.

Das ursprüngliche .4-Manifest mit 1.894 Dateien wurde vor Änderungen verifiziert. Neue Quellen werden erst nach Änderungen und auditierter Paketprüfung manifestiert. Historische Fixtures werden erhalten; neue Binärfixturefreigaben sind auf konkret benannte Dateien beschränkt.
