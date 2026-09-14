# Abnahme v2.81

Version **2.81**, Build `v2.81-hailo-cold-preflight-shared-input-layout-terminal-status`. Basis: vollständige verifizierte v2.80.4.

Produktives Force bleibt AUS; Hailo balanced/Opt1/B500/Batch8, DeepX EMA/Opt0/B500, Bootstrapmethodik und Qualitätsmargen bleiben erhalten. `COMPILE_INFEASIBLE` ist nur mit exakter vorhandener Evidenz eine abgeschlossene Nichtausführbarkeit; `TRANSIENT_INFRASTRUCTURE` bleibt technischer Fehler. Statusachsen trennen Ausführbarkeit, technische Ausführung, Qualityentscheidung und wissenschaftlichen Scope.

Neue Zielhardware-Ausführung: **NOT_RUN** bei Auslieferung. Die reale Zielabnahme führt der gemeinsame Starter durch. Energie-Finalfreigabe und erneute 5.000-Bilder-Kampagne sind nicht Teil dieses Korrekturlaufs. Tatsächliche Softwaretests, Gegenproben, Quellarchivprüfung und isoliertes Upgrade stehen in `VERIFICATION_V281.json` des Bundles mit Logs und JUnit-Dateien.

GUI und laufende Workflows vorher geordnet beenden. Bundle frisch entpacken und `bash ./install_and_accept_v281.sh` im Bundleverzeichnis ausführen. Der Starter sammelt alle Stufen in einem ZIP und gibt `UPLOAD_ONLY` aus. Die gebündelte reguläre Zielausführung baut ausschließlich tatsächlich noch fehlende benötigte Artefakte; bekannte exakte Rejects werden nicht erneut kompiliert.

Die fortgeltende volle Softwareabnahme läuft über `scripts/run_v281_small_acceptance.sh`; die historischen behavioral Regressionen sind vollständig erhalten, während der neue Release-Identitätstest den alten versionsgebundenen Closure-Test ersetzt. Ein gesonderter wiederholter Kurztest ist nach dem gemeinsamen Starter nicht erforderlich.
