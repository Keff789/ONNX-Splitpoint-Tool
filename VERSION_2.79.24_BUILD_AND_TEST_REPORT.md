# Build- und Testbericht 2.79.24

Build-ID: `v2.79.24-targeted-run-repairs`

Basis: 2.79.23. Anlass: `complete_set_20260905_164805`, Start 05.09.2026
16:48:05 und protokolliertes Ende 06.09.2026 02:33:40, jeweils lokale Zeit.
Alle 171 im Debug-Pack attestierten kritischen Module stimmen bytegenau mit
der ausgelieferten 23 überein. Der Befund ist damit keine Vermischung alter
Toolversionen. Das Original-Debug-Pack und die alten Ergebnisse bleiben
unverändert.

## Gezielte Änderungen

| Ursache | Korrektur im vorhandenen Ablauf |
|---|---|
| Finale Inventur lehnt die eigenen atomaren Hailo-Aliase ab | Streng validierte lokale Publisheraliase werden auf die bestehenden realen Generationsdateien abgebildet. HEF, Receipt und Cache-Meta bleiben zusammen geprüft. Allgemeine/externe/gebrochene Links bleiben abgelehnt. |
| Native-Transfer lässt genau diese Aliase weg | Der bestehende rsync-Dateiumfang enthält die validierten Aliase und den Generationszeiger. Ausgeschlossene alte Native-Produkte bleiben ausgeschlossen. |
| Full-Receipt-Promotion sucht Compiler-ONNX neben dem gepinnten HEF statt neben dessen öffentlichem Pfad | Beide bestehenden Ablageorte und die suite-relative Modellquelle werden berücksichtigt; signierter Compilerdateiname und exakte Byteprüfungen bleiben verbindlich. |
| TRT-LRU entfernt früh benötigte ResNet-Daten vor der späteren Native-Phase | Im vorhandenen Workflow-Lease-Scope wird Namespace-Retention zurückgestellt und sichtbar gemeldet. Reale freie Bytes/Inodes werden weiter geprüft. Standalone-Retention bleibt unverändert. |
| H8-Detection produziert keine vollständige Abschluss-Artefaktprojektion | Das nach bestandenem Postflight-Oracle bereits versiegelte Ergebnis wird im vorhandenen Persistenzvertrag weitergegeben. Keine zusätzlichen Hashes im Hotloop. |
| H10-YOLO26-Layout `[1,8400,84] → [1,84,8400]` wird verworfen | Die exakt passende Rank-3-Transposition wird in der vorhandenen ONNX-Bridge unterstützt und in deren bisherigen Identitäten gebunden. Falsche oder mehrdeutige Formen bleiben gesperrt. |
| DeepX-Childfehler erscheinen nach erfolgreichem Transfer als Transferfehler | Ursprung und Childgrund werden erhalten; fehlende DXNNs oder Enginevarianten sind keine rsync-Fehler. |
| Globales Quality-PASS betrachtet nur die 21 Full-TRT-Companions | Full-Referenzabnahme bleibt separat; die Standardkampagne berücksichtigt tatsächliche Fehlentscheidungen und fehlende erforderliche Quality-Zellen. Explizite Full-only-Canaries behalten ihren engen Vertrag. |
| Eine einzelne ungültige Full-Contract-Zeile heißt Duplikatkonflikt | Präzise Diagnose für einen ungültigen exakten Treffer; keine Zulassung gelockert. |
| Bereits vorhandene exakte Preflight-Probes verschwinden ohne Serviceplan-Zeile | Vorhandene exakt adressierte Probes ergänzen die Matrix unter den bisherigen Modell-/Boundary-/Backend-/Stage- und Negativprüfungen. |

Die allgemeine Negativ-Evidenz bleibt aktiv. `COMPILE_INFEASIBLE` und
`PARSER_UNSUPPORTED` werden ausschließlich mit exakt passender Identität
wiederverwendet. `TRANSIENT_INFRASTRUCTURE` bleibt wiederholbar. Tatsächlich
wurden im Nachtlauf die bekannten H8-Fehler b398/b364 nicht erneut kompiliert.
Alte HEF-only-Bestände bleiben `legacy_unsealed` und werden nicht gelöscht
oder ohne Receipt freigegeben.

Keine neue Datenbank, Cachehierarchie, Pipelineebene oder Quality-Grenze.
Keine Änderungen an den wissenschaftlichen Qualitätsanforderungen oder
automatische Installation anderer Compiler-/Torch-/Treiberpakete.

## Abnahme und Evidenz

Neue Tests reproduzieren den echten Publisher→Transfer→Fullfinder→Receipt-
Promotion-Ablauf, den Terminalabschluss, H8-Detection-Persistenz, siebenteilige
Modellfolgen mit erhaltenen Artefakten, globale Quality-Achsen und Negativ-
Preflight. Bestehende Cache-/Receipt-/Recovery-/Completion-/Energietests laufen
mit. Die Reprojektion der echten Quality-Ergebnisse ergibt global `fail`,
separat Full-Referenz `pass`, bei unverändert 33 FAIL, 9 inconclusive und
31 fehlenden erforderlichen Zellen.

Das Lieferpaket enthält das Ergebnis des vollständigen Softwaregates und
des isolierten 23→24-Upgrades in `VERIFICATION_V27924.json` und
`verification/`. Diese Nachweise werden nach dem Paketbau an die genaue
Prüfsumme des gelieferten Source-ZIPs gebunden. Die Prüfungen verwenden
temporäre Daten und simulierte Hardware. Ein lokaler rsync-Test führt das
tatsächliche Transferkommando aus, ohne einen Remotehost zu kontaktieren.

Der historische Test `test_v27516_primary_remote_error_cache_policy.py` wird nicht als
aktuelles Release-Gate verwendet; historische Vorbestandsfehler werden in
den Prüfprotokollen gesondert ausgewiesen, soweit geprüft. Das aktuelle
Regressionstor bewertet die produktiv betroffenen Pfade.

## Was damit noch nicht nachgewiesen ist

- Reale Accuracyverbesserungen: MobileNet, YOLO11/26 und einige DeepX-
  Varianten haben echte Verluste oder unentschiedene statistische Befunde.
  Diese werden durch den Ablauf-Fix nicht automatisch repariert.
- Gelöschte ResNet-Namespaces sind nicht wiederhergestellt. Warme Abnahmen
  können korrekt an fehlenden Dateien stoppen.
- Der DeepX-Compiler-GPU-Konflikt bleibt ein Umgebungsproblem. Mehrere
  Detection-DXNNs sowie daraus abhängige TRT-Vorbereitungen fehlen.
- Die korrigierte H10-Rank-3-Bridge kann einen gezielten neuen TensorRT-
  Part2-Build erfordern. Daraus folgt kein neuer Hailo-HEF-Build; ein warmer
  Preflight muss die fehlende passende TRT-Variante korrekt melden.
- 156 gültige Energie-Wiederholungen sind rechnerisch konsistent und laut
  Receipt kalibriert. Der 1-s-Hotloop lag jedoch im 2,5–4,4-s-Prozessfenster.
  30-/60-s-Kontrollen und reguläre Rohtraces sind für belastbare Energie-
  interpretation vorgesehen. Energie-Code und Gates wurden hierfür nicht
  verändert.
- Unterschiedliche Jetson-Taktpolicies verhindern einen sauberen direkten
  Hostvergleich; das Tool verändert diese Policies nicht automatisch.
- Die 500-Bilder-Abnahme bleibt Screening; vorhandene Quality- oder
  Vollständigkeitsfehler werden nicht nachträglich für wissenschaftliche
  Vergleiche freigegeben.

```text
HARDWARE_EXECUTION=NOT_RUN
REAL_MULTI_HOST_EVALRUN=NOT_RUN
REAL_ACCURACY_RECOVERY=NOT_RUN
REAL_ENERGY_REFERENCE_VALIDATION=NOT_RUN
```
