# Quality-Resultatvertrag 3 – Unsicherheit und Altbestand

Die Quality-Berechnung ändert keine Marge, Seeds, Resample-Reihenfolge, AP-/Top-k-Metrik oder frühe Gesamt-Fail-Regel. Vertrag 3 trennt Punktentscheidung und Unsicherheitsnachweis. Eine berechnete Bootstrap-Komponente behält ihre echten Grenzen; `ci_computed=true`, `decision_basis=paired_bootstrap_lower_bound` und `gate_bound_value=ci_low` dokumentieren die Entscheidung.

Bei frühem Verlust-Fail sind `ci_low` und `ci_high` JSON-null, `ci_computed=false`, die wirkliche Replikatzahl ist null und der Gate-Wert ist das gemessene Delta. Andere Komponenten derselben abgebrochenen Auswertung bleiben `inconclusive` mit `decision_basis=bootstrap_not_computed_other_component_point_fail`. Sie erhalten weder eine erfundene Nullbreite noch einen statistischen PASS. Die gesamte Auswertung bleibt FAIL.

Identische Kandidaten-/Referenzvorhersagen werden anhand der gebundenen, ausgewählten Vorhersage-Payloads und Bild-IDs überprüft. Gleiche Metrikwerte oder fehlende beziehungsweise nur behauptete gleiche Hashes reichen nicht. Der nachgewiesene Fall hat einen deterministischen Gate-Wert null, aber kein empirisch berechnetes Bootstrap-CI.

## Inventar der Writer, Leser und Verbraucher

| Komponente | Behandlung |
|---|---|
| `quality_service.prepare_evaluation`, Worker, Shard-Aggregation, ManagementQualityService | Vertrag 3; vorhandene numerische Berechnung unverändert; explizite Entscheidung ohne Pseudo-CI; gebundene Identitätsprüfung vor dem Resampling |
| `quality_cache.PersistentQualityCache` | Resultat-/Cachevertrag und Quality-Fingerprint-Namespace 3; Writer und Loader prüfen Nullable-CI-Provenienz. Alte abgeleitete Cache-Dateien bleiben unberührt und werden nicht als Vertrag 3 ausgegeben. Vorhersagen und Runtime-Modellartefakte bleiben wiederverwendbar. |
| `quality_result_contract` | Gemeinsame, reine Darstellungsprojektion; Version des Originals und Version der Projektion getrennt; keine Dateiänderung oder Neuberechnung historischer Hashes |
| `quality_replay._write_csv` | Null als leere CSV-Zelle; decision_basis, gate_bound_value, ci_computed und uncertainty_status pro Primär-/AP50-/AP75-Komponente |
| `validation.accuracy_gates` | Gesamte registrierte Entscheidung bleibt maßgeblich; Legacy-Projektion vor Accuracy-/Energie-Bindings; keine Null-/Deltaersatzgrenzen |
| `workflow.scientific_reporting` | Zentrale Ergebnisse, Performance-/Energieprojektion, JSON, CSV, Markdown, LaTeX verwenden dieselbe Semantik. Explizites None wird nicht mit einem älteren numerischen Alias ersetzt. |
| `scientific_reporting._make_figures` | Punkte ohne Intervall bei fehlender Unsicherheit, bezeichnet mit „Unsicherheit nicht berechnet“. Tatsächliche Intervalle werden aus den vorhandenen Grenzen gezeichnet, auch wenn der Punktschätzer außerhalb eines Perzentilintervalls liegt. |
| `reporting_quality_decomposition` | Unberechnete Intervalle bleiben leer; Entscheidungsbasis und Unsicherheitsstatus werden in Vergleichsprojektionen erhalten. |
| `existing_evidence_verifier` | Unabhängige numerische und Vertragsprüfung unterscheidet Vertrag 3 von historischen Resultaten; akzeptiert keine Pseudo-CIs oder ungebundene deterministische Identität. Historische Entscheidungen und Originalbytes bleiben nachvollziehbar. |
| Native-Finalreport | Kein eigener Quality-CI-Leser; verwendet bestehende Quality-Entscheidungen. Seine FPS-Bootstrapintervalle bleiben unverändert. |
| GUI | Kein eigener Quality-CI-Rechner; zeigt den exportierten Text beziehungsweise öffnet Reports. Die reale GUI-Textmethode wurde mit der expliziten Unbekannt-Anzeige geprüft. |
| DeepX-Normalisierungs-/Kalibrierungsdiagnosen | Eigene tatsächlich ausgeführte Bootstrapverfahren; keine Null-Replikat-Skip-Pfade, unverändert |

Der zentrale Workflow-Summary-CSV-Writer übernimmt dieselben Primärfelder. Die Versionierung betrifft ausschließlich abgeleitete Qualityresultate. DXNN, HEF, TensorRT-Engines, Vendor-Venvs und CPU-Referenzvorhersagen werden dadurch nicht invalidiert.

## Historische Evidenz und Regression

`tests/fixtures/v27931_quality/PROVENANCE.json` benennt die Originalquelle, Prüfsummen und bewusst fehlenden großen Payloads. Vier ausgewählte Complete-Set-Resultate enthalten den MobileNet-Full-Point-Fail, die RegNet-Identitätsabkürzung, einen berechneten YOLO11-Bootstrap und den YOLO26s-Guardrail-Fail. Die originale YOLO11-Full-Qualityanfrage liegt zusätzlich bytegleich vor; der echte Vertragsvalidator und innere/äußere Mutationen werden geprüft. Diese reine Vertragsabnahme simuliert keine B500-Ausführung.

Ein separat als synthetisch markierter, mit unverändertem v30 erzeugter Golden-Fall enthält Eingaben, den PCG64-Plan, beide tatsächlichen Shardverteilungen und sämtliche Komponenten. Vertrag 3 stimmt für die berechneten Resultatfelder exakt damit überein. Testgruppen T08.1–T08.6 sowie T00.2 stehen in `tests/test_v27931_quality_uncertainty.py`; die abschließenden Testzahlen werden aus dem tatsächlichen Release-JUnit übernommen.
