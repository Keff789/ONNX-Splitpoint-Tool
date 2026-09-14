# Implementierungsabgleich v2.80.3

Build-ID: `v2.80.3-build-readiness-native-not-started-debug-export`.

Grundlage ist der unveränderte Originalauftrag vom 11.09.2026. Die 64 Anforderungen sind keine Testanzahl. Parametrisierte Fälle können mehrere Anforderungen abdecken. Die finale tatsächliche Testbilanz und Testidentitäten stehen im äußeren Prüfbericht und in JUnit. Hardware-NOT_RUN wird nicht als Software-PASS ausgegeben.

AP0–AP3 schließen die drei konkreten Softwarelücken. AP4 bewahrt die wirkliche Generator→ORT→Referenz→Consumer-Kette; positive physische Producer-/Nativeausführung bleibt ein Zielsystemgate. AP5 bereitet begrenzte MobileNet-/YOLO26-Diagnosen vor und zeigt effektive Konfiguration, ohne Decoder, Qualität, Energie- oder Compilerpolicy zu ändern. AP6 bindet die endgültige Quelle und den tatsächlichen isolierten Installer.

| ID | Anforderung | Nachweis / Prüfpfad | Grenze |
|---|---|---|---|
| T00.1 | Manifestbindung | `verification/v2803/baseline_manifest.log`, `verification/v2803/baseline_and_inputs.json` | Softwareprüfung / finale Ausführung siehe JUnit |
| T00.2 | Fixtureherkunft | `tests/fixtures/v2803_night_regression/PROVENANCE.json`, `tests/test_v2803_scope_and_diagnostic_claims.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T00.3 | Rote Sollfälle | `verification/v2803/ap1/baseline_red.json`, `verification/v2803/ap2/original_red.xml`, `verification/v2803/ap3_export/red.xml`, `verification/v2803/ap3_projection/red.xml` | Softwareprüfung / finale Ausführung siehe JUnit |
| T00.4 | Ehrliche Prüfumgebung | `tests/test_v2803_release_scope.py`, `verification/v2803/ap6/actual_failure_launchers.junit.xml` | Softwareprüfung / finale Ausführung siehe JUnit |
| T01.1 | Originale YOLO11l-Stufe | `tests/test_v2803_deferred_build_readiness.py` | Realer Producer und Workflowstufe; externe Vendorbuilder kontrolliert |
| T01.2 | Alle Artefakte vorhanden | `tests/test_v2803_deferred_build_readiness.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T01.3 | Bekannte Unrealisierbarkeit | `tests/test_v2803_deferred_build_readiness.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T01.4 | Nur ein Pflichtjob fehlerhaft | `tests/test_v2803_deferred_build_readiness.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T01.5 | Unbekannt und optional | `tests/test_v2803_deferred_build_readiness.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T01.6 | Boundary- und Stagebindung | `tests/test_v2803_deferred_build_readiness.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T01.7 | Echte Builderexception an externer Grenze | `tests/test_v2803_deferred_build_readiness.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T01.8 | Unabhängige Fortsetzung | `tests/test_v2803_deferred_build_readiness.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T01.9 | Globale Schutzfälle | `tests/test_v2803_deferred_build_readiness.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T01.10 | DeepX-Sammelrequest | `tests/test_v2803_deferred_build_readiness.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T01.11 | Dispatchzahl | `tests/test_v2803_deferred_build_readiness.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T01.12 | Ursache und Resume | `tests/test_v2803_deferred_build_readiness.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T02.1 | 42 Originalfälle | `tests/test_v2803_native_not_started.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T02.2 | 14 Fälle je Reader | `tests/test_v2803_native_not_started.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T02.3 | Gesamte 84er-Kette | `tests/test_v2803_native_not_started.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T02.4 | Unveränderte reale Messung | `tests/test_v2803_native_not_started.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T02.5 | TRT-Full-Regression | `tests/test_v2803_native_not_started.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T02.6 | Null allein reicht nicht | `tests/test_v2803_native_not_started.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T02.7 | Unbekannte Zählung | `tests/test_v2803_native_not_started.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T02.8 | Messwiderspruch | `tests/test_v2803_native_not_started.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T02.9 | Echte Teilreplikate | `tests/test_v2803_native_not_started.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T02.10 | Vollständige und gespiegelte Gruppen | `tests/test_v2803_native_not_started.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T02.11 | Identitätskonflikte | `tests/test_v2803_native_not_started.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T02.12 | Bericht und Energie | `tests/test_v2803_native_not_started.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.1 | Kleiner gültiger Index | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.2 | Größer als 32 MiB | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.3 | Groß und beschädigt | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.4 | Fehlend / falsches Schema | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.5 | Pfadsicherheit | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.6 | Teilabdeckung | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.7 | Runtimequellen vollständig gezählt | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.8 | Produktionswriter | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.9 | Normalisierter Writer | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.10 | Identitäten und Metriken | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.11 | Fehlende Felder / unbekanntes Schema | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.12 | Keine Rohkörper | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.13 | Budgets und JSONintegrität | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.14 | Historischer Run ohne Companion | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.15 | Provenienzgrenzen | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.16 | Archivpublikation | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.17 | Manifest/Consumer-Kompatibilität | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T03.18 | Ressourcen/Finalisierung | `tests/test_v2803_debug_pack_large_sources.py`, `tests/test_v2803_runtime_diagnostic_projection.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T04.1 | Echte Classificationreferenz | `tests/test_v2803_reference_workflow_gate.py`, `tests/test_v2802_cpu_reference_binding.py`, `tests/test_v2802_workflow_integration.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T04.2 | Echte Detectionreferenz | `tests/test_v2803_reference_workflow_gate.py`, `tests/test_v2802_cpu_reference_binding.py`, `tests/test_v2802_workflow_integration.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T04.3 | Modell/Source-Falschbindung | `tests/test_v2803_reference_workflow_gate.py`, `tests/test_v2802_cpu_reference_binding.py`, `tests/test_v2802_workflow_integration.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T04.4 | Kleiner eigener Datasetvertrag | `tests/test_v2803_reference_workflow_gate.py`, `tests/test_v2802_cpu_reference_binding.py`, `tests/test_v2802_workflow_integration.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T04.5 | Consumer und strenger Producer | `tests/test_v2803_reference_workflow_gate.py`, `tests/test_v2802_cpu_reference_binding.py`, `tests/test_v2802_workflow_integration.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T04.6 | Zielsystem-Positive | `tests/test_v2803_reference_workflow_gate.py`, `tests/test_v2802_cpu_reference_binding.py`, `tests/test_v2802_workflow_integration.py` | Zielsystem offen (NOT_RUN); Vorbereitung und Nachprüfer geliefert |
| T04.7 | Referenzfehler und Abbruch | `tests/test_v2803_reference_workflow_gate.py`, `tests/test_v2802_cpu_reference_binding.py`, `tests/test_v2802_workflow_integration.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T04.8 | Normalmodus und Reuse | `tests/test_v2803_reference_workflow_gate.py`, `tests/test_v2802_cpu_reference_binding.py`, `tests/test_v2802_workflow_integration.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T05.1 | MobileNet-Fallvergleich | `tests/test_v2803_scope_and_diagnostic_claims.py`, `tests/test_v2803_measurement_configuration.py` | Diagnoseumfang/Originalbefund geprüft; neue physische Ursachenprüfung offen |
| T05.2 | YOLO26-Fallvergleich | `tests/test_v2803_scope_and_diagnostic_claims.py`, `tests/test_v2803_measurement_configuration.py` | Diagnoseumfang/Originalbefund geprüft; neue physische Ursachenprüfung offen |
| T05.3 | Compute-/Energieanzeige | `tests/test_v2803_scope_and_diagnostic_claims.py`, `tests/test_v2803_measurement_configuration.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T05.4 | Setup- und Originalerhalt | `tests/test_v2803_scope_and_diagnostic_claims.py`, `tests/test_v2803_measurement_configuration.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T06.1 | Regressionenumfang | `tests/test_v2803_release_closure.py`, `tests/test_v2803_release_scope.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T06.2 | Remote-/Full-/Energieabschluss | `tests/test_v2803_release_closure.py`, `tests/test_v2803_release_scope.py` | Softwareprüfung / finale Ausführung siehe JUnit |
| T06.3 | Frisches endgültiges Sourcearchiv | `verification/v2803/fresh_source_acceptance.json`, `verification/v2803/source_delta.json` | Softwareprüfung / finale Ausführung siehe JUnit |
| T06.4 | Realer isolierter Installer | `verification/installer_v2803/v2803_small_acceptance.json`, `verification/v2803/preservation_after.json` | Softwareprüfung / finale Ausführung siehe JUnit |
| T06.5 | Starter/Fehlergates | `tests/test_v2803_release_scope.py`, `verification/short_tests/short_tests.xml` | Softwareprüfung / finale Ausführung siehe JUnit |
| T06.6 | Abnahmedokumentation/Git | `tests/test_v2803_release_closure.py`, `tests/test_v2803_release_scope.py` | Softwareprüfung / finale Ausführung siehe JUnit |

## Beibehaltene Grenzen

Force AUS, relaxed, B500/Opt1/Batch8 beziehungsweise DeepX EMA/Opt0/MeanStd bleiben erhalten. Ein geeigneter CPU-HEF bleibt bei GPU-Präferenz wiederverwendbar. Bekannte negative Builds, negative Quality, historische 1-s-Screenings und ursprüngliche Versionen bleiben negative beziehungsweise historische Evidence. Neue Modell-/Hardware-/5.000er-Freigaben werden nicht behauptet.

Die alten 63er- und neuen 84er-Replays sind getrennte Eingaben. Die 196 Runtimequellen mit 43 Größen-Auslassungen sind Originalinventurzahlen; 43 neue synthetische große Quellen prüfen den produktiven Writer und Export. Fehlende alte Originalkörper werden nicht nachgebaut. Original- und Derived-Abdeckung bleiben getrennt.

## Releasefortschreibung

Fachliche v2.80.2-Regressionen bleiben ausgewählt. Nur versionsgebundene Release-Closure und aktive Starter werden fortgeschrieben. Geänderte ältere Completeness-Erwartungen zum nun ausdrücklich unbekannten Indexumfang sind mit Begründung dokumentiert; ihre übrigen Schutzassertionen bleiben erhalten. Die genaue Auswahl- und Entry-Point-Liste liegt unter `verification/v2803/ap6`.
