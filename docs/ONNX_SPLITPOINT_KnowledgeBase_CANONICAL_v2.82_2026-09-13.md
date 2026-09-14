# ONNX Splitpoint Tool – Knowledgebase v2.82

Build: `v2.82-selected-energy-generic-roles-workspace-product-evidence`  
Basis: unverändert geprüfte v2.81-Lieferung; beide Implementationspläne vom 13.09.2026.

Der abschließende Prüfstand des exakt ausgelieferten Archivs, die Testzahlen und die isolierte Upgradeprüfung stehen im äußeren Lieferbundle in `VERIFICATION_V282.json` und `BUILD_AND_TEST_REPORT_v2.82.md`. Dieser Sourcebericht beschreibt die Korrekturen und deren fachliche Grenzen. Reale Hailo-/CUDA-/u.RECS-Ausführung während der Entwicklung: **NOT_RUN**.

## Umgesetzte Korrekturen

| Bereich | Verhaltensänderung | Nachweis |
|---|---|---|
| Energy-Retry | Logischer Repeat und gewählter physischer Versuch werden eindeutig verbunden; genau dessen stdout, Command-, Nonce-, Fenster- und Work-Unit-Belege werden geprüft | Beide historischen Fehler zuerst mit v2.81 reproduziert; korrigierter vollständiger Import aller 115 Original-Checkpoints bestanden |
| Energy-Reimport | Checkpoints und Originalaggregate bleiben unverändert; separate, idempotente Projektion mit belegtem historischen Start und Rückgabecode | Originalprojektion 113→115 importierbare Zeilen; keine neue Messung und keine Rohtrace-Neuberechnung |
| Generic-Ausschlüsse | Explizite Part2-only-Messung widerspricht einem exakten Composed-Compileausschluss nicht; Composed-Start/-Erfolg oder unklare Rollen bleiben streng | Echter Reader und produktive Pflichtmatrix mit Originaldaten: 6 Ausschlüsse, 3 verbleibende Missing; 140 Zeilen erhalten |
| Workspace | Tatsächlicher Cold-Auftrag liefert Shapes/Kalibration/Arbeitswurzel für frühe Prüfung; vor Dispatch erneute Prüfung | Vier Originalberichte, getrennte 640er-/224er-Anforderung, Warm-/Unknown-/Permission-/Inode-Regressionen |
| Cache/Auswahl | Fehler bleiben requestlokal; abhängige TRT-Prüfung wartet nur auf ausdrücklich erlaubten Producer-MISS; Fallzahl pro Modell | Gemischte Warm-/Cold-/Negative-Pfade und unterschiedliche explizite Fallzahlen |
| H10-Runner | Die erzeugte Session nutzt den korrigierten Paketadapter; veraltete Skripte werden erneuert, bestehende HEFs/Engines bleiben nutzbar | Tatsächlich erzeugte Sessionklassen und sechs originale NWC-Puffer; dies beweist keine AP-Verbesserung |
| H8-Dumps | Untimed finaler Snapshot nach Synchronisierung; getrennte Repetitionsdateien und eindeutige Referenzbindung | Pufferüberschreiben, unterschiedliche Frames/Repetitionen und negative Dumpprüfungen |
| H8-YOLOv7-Arithmetik | Reproduzierte Unterschiede zwischen NumPy-Dispatchvarianten werden durch explizite Berechnung und Materialisierung im vorhandenen Decodervertrag begrenzt | Alter Vertrag bleibt historisch rekonstruierbar; keine neue Toleranz oder veränderte Schwelle; betroffene neue Qualitätsbindungen erforderlich |
| Full-Qualität | Vorhandener zentraler Completed-Endpunkt wird zusätzlich im bestehenden Binding weitergereicht und gegen die attestiere Fortsetzung geprüft | Beide originalen YOLO26m-Full-Fälle; alte Rohendpunkthashes erhalten, ursprüngliche Qualitätsresultate unverändert |
| Status | Auswertung abgeschlossen und Quality-FAIL getrennt; gültige Negativergebnisse bleiben nutzbar; Fallfehler ergeben partial | Gemeinsame GUI-/CLI-/Manifest-/Reportprojektion; globale Abschlussfehler bleiben failed, Abbruch cancelled |
| Debug | Initiale und ausgewählte Retrylogs samt unveränderter Auswahlhistorie werden erfasst; fehlende Belege bleiben sichtbar | Tatsächlicher ZIP-Roundtrip; keine Modelle oder Rohtensoren als neue Diagnosepflicht |

## Was die Originaldaten-Replays beweisen

- Native unverändert: **126 = 115 erfolgreich + 6 ausgeschlossen + 5 blockiert**, davon 42 erfolgreiche Full-Baselines. Keine neuen Runtime-Erfolge wurden erfunden.
- Zentrale Qualität unverändert: **150 = 81 PASS + 46 FAIL + 23 INCONCLUSIVE**, aufgeteilt in 129 primäre und 21 begleitende Aufträge. Das ist Screening mit 500 Bildern, kein 5000er-Finalnachweis.
- Generic: sechs exakt passende Ausschlüsse werden zusätzlich richtig berücksichtigt; die vorherigen neun Missing sinken auf drei. Die zwei ungültigen H10-Endpunkte bleiben getrennte technische Befunde.
- Energy: vollständige ursprüngliche Completion-/Importkette aller 115 Aufträge offline geprüft. In der abgeleiteten Reparatur der beiden fehlerhaften Zeilen bleiben die anderen 113 Zeilen unverändert. Ursprüngliche Fehlversuche, alte negative Importentscheidungen und physischer Messscope bleiben erhalten.
- Full-Binding: zwei passende zentrale Resultate werden ohne Rechenwiederholung korrekt zugeordnet. Die historischen vier YOLOv7-Dumpkonflikte werden dadurch nicht gelöscht.

## Verbleibende Zielsystemgates

H10 b398/b364 benötigt weiterhin die echte gebundene A–F-Kette. Ein gültiger Layouttransport, vollständiger Capture oder vorhandener HEF bedeutet nicht, dass Scores und Boxen korrekt sind. Ungültige Endpunkte erhalten keine Performance-/Qualityfreigabe und werden nicht per Sigmoid, Clipping oder Thresholdänderung kaschiert.

Die drei noch fehlenden H8-Bauten benötigen genügend Platz im tatsächlichen Arbeitsbereich und anschließend ihren normalen Folgepfad. `COMPILE_INFEASIBLE` bleibt ein gültiger negativer Buildbefund; `TRANSIENT_INFRASTRUCTURE`, ENOSPC und Cancel bleiben erneuerbare technische Ursachen. Der erfolgreiche RegNet-b073-Build und die bereits bewiesene GPU-/PTXAS-Funktion werden nicht neu in Frage gestellt.

Bei YOLOv7 stimmen die in den Originalmanifesten erklärten Quelltensorhashes mit der Attestation überein. Da die Roharrays im Debugpack fehlen, beweist die Offline-Reproduktion der SIMD-Arithmetik nicht die exakte Ursache jedes historischen ARM-/Desktop-Vergleichs. Neue Ergebnisse mit dem korrigierten Rechenvertrag müssen über den normalen erzeugten Runner und Native-Dump geprüft werden; alte Qualitätswerte dürfen nicht darauf umetikettiert werden.

## Erhaltungsregeln

Force AUS; positive Artefakte wiederverwenden, nur zulässige echte MISS bauen. Keine neue Hash-/Seal-/Registryarchitektur. Keine individuellen Compiler-Sweeps, keine pauschale 10-%-Marge, keine Änderung von Opt1/B500/Batch8 oder DeepX Mean/Std. Qualitäts-FAIL/INCONCLUSIVE bleiben exakt erhalten. Gültige Performance-/Energiemessungen außerhalb der Qualitätszulassung bleiben beschreibend sichtbar; Qualitätsgleichwertigkeitsclaims behalten ihre bisherigen Kriterien.

Historische Dokumente und Fixtures bleiben als solche enthalten. Die neue Versionsidentität betrifft die aktive Software und neue Auswertungen, nicht die ursprünglichen Läufe. Neue Gateberichte kennzeichnen Originaldaten, synthetisch simulierte Hardwaregrenzen und tatsächlich ausgeführte Softwareprozesse getrennt.

## Forschungsprämisse und fortgeltende Grenzen

Die automatisierte Partitionierung unter festem Rezept darf zulässige Qualitätsverluste, unsichere Urteile und Compile-Rejects ergeben. Individuelles Nachoptimieren jedes Netzes ist keine Abschlussbedingung. Eine Qualitätsdifferenz beweist nicht allein einen Compilerfehler; formell gültige AP=0 bleibt von ungültigen Scores, Boxen, NaN/Inf oder falschen Bindungen zu unterscheiden. Relative Prozentänderungen und Prozentpunkte werden getrennt bezeichnet.

Native-Energie und generische Energie bleiben getrennt. Kein Wechsel vom 1-s-/500-Bilder-Screening auf Finalwerte durch Reprojektion. Bestehende MobileNet-/YOLO26s-Float-/Backend-Full-Vergleiche im historischen Teil bleiben erhalten. DeepX-Detection-Split/Vendor-Full-Verlustzerlegungen bleiben bei abweichendem Decoder-/NMS-Vertrag unvergleichbar; geeignete einzelne Floatvergleiche bleiben davon unabhängig gültig.

## Historischer Stand v2.81 (unverändert)

Der folgende übernommene Originaltext dokumentiert den früheren Lieferstand. Bei Widersprüchen gelten die obigen v2.82-Korrekturen und der aktuelle Gatebericht. Historische PASS-Aussagen werden nicht auf neue Pfade übertragen.

# ONNX Splitpoint Tool – kanonische, konsolidierte Knowledge Base

## Stand v2.81 – Korrekturen aus Nachtlauf und gemeinsamem Smoke

| Feld | Arbeitsstand dieser Fassung |
|---|---|
| Dokumentstand | 13. September 2026; Fortschreibung der vollständigen v2.80.4-REV3 |
| Übernommener Evidenzstand | REV3 plus .4-Nachtlauf vom 12./13. September und kombinierter Smoke vom 13. September; historische Messungen bleiben ihrer Version zugeordnet |
| Ausgangsdatei | `ONNX_SPLITPOINT_KnowledgeBase_CANONICAL_v2.80.4_2026-09-13_REV3.md`, unverändert erhalten |
| Releasequelle | Originales vollständiges v2.80.4-Bundle wiederhergestellt; eingebettete Quelle mit 1.894 Manifestdateien geprüft; v2.81 baut darauf auf |
| Bereits auf Smartmirror2 beobachteter Stand | v2.80.4: gemeinsame Installation/Normalworkflow PASS mit Qualitäts-FAIL; anschließender größerer Nachtlauf technisch failed, Ursachen durch kombinierten Smoke eingegrenzt |
| Zweck von v2.80.4 | Historische Basis: H8-Kontext, Quality-Cancel und Debuglimits; v2.81 ergänzt die nachgewiesenen Produktions-/Zuordnungsfehler |
| YOLO11l/Hailo10H, 5.000 Bilder | Untersuchung abgeschlossen: TensorRT Full PASS; Hailo Full und b062-Split FAIL unter der unveränderten 1-pp-Marge |
| Hailo8/MobileNet GPU | Compute/XLA, echter Build, Fixed16-Hardware und erste HAR-Emulation abgeschlossen; keine B5000-Freigabe des privaten GPU-HEFs, kein Speedup belegt |
| Hailo8-Verlustlokalisierung | Erste große Abweichung zwischen Parsed-HAR und optimierter/quantisierter Darstellung; weitere Emulations-/Hardwareabweichung offen, keine ausschließliche Quantisierungsursache behauptet |
| Forschungsumfang | Automatisierte Partitionierung und heterogene Ausführung unter festgelegter Policy; keine manuelle Optimierung jedes Evaluationsfalls bis PASS |
| Vertiefte Hailo8-Diagnose | Zurückgestellt als optionale Folgearbeit; kein aktueller Arbeitsauftrag und keine Voraussetzung zum Berichten vorhandener Negativergebnisse |
| Ergebnis-Git | Übernommener Pushnachweis: Commit `5636017e5db3229862ba10c609b5f4b5f290e76b` / `main`; REV3 wurde hier nicht gepusht. Für spätere Dokumentergänzungen liegt kein neuer Pushnachweis vor; Live-HEAD nicht erneut geprüft |
| Hailo-Rezept | balanced / Opt1 / B500 / Batch8; kein zusätzlicher Opt2-/Opt3- oder B1024-Sweep |
| Dokumentrolle | Aktuelle Arbeits-, Methoden- und Evidenzreferenz; historische Ergebnisse behalten ihre ursprüngliche Identität |

**Entscheidung:** Bewertet wird die automatisierte Partitionierung mit der festgelegten Build- und Validierungspolitik, nicht die durch beliebig viel Einzelfalloptimierung maximal erreichbare Backendqualität. Ein korrekt ausgewerteter Qualitäts-FAIL bleibt ein abgeschlossenes negatives Ergebnis. Die vertiefte Hailo8-Ursachenanalyse wird als optionale Folgearbeit zurückgestellt. Bekannte eigene Implementierungsfehler werden weiterhin korrigiert oder als technische Einschränkung ausgewiesen; sie werden nicht pauschal dem Compiler zugeschrieben. Passende Artefakte weiterverwenden, fehlende benötigte Artefakte normal bauen; produktives Force bleibt AUS. **Abschlussziel: Für alle vorab festgelegten Fälle liegt ein korrektes, nachvollziehbares Ergebnis vor – nicht: alle Fälle bestehen.** [E-2804-PLAN] [E-2804-ABGLEICH] [E-SCOPE-REV3]

## Navigation

[0. Dokumentführung](#dokumentfuehrung) · [1. Methode](#methode) · [2. Fragenkatalog](#fragen) · [3. Setups](#endpunkte) · [4. Releasefortschritt](#v30) · [5. DeepX-Historie](#deepx-full) · [6. Energie](#energie) · [7. Modellqualität](#deepx-r1-r2) · [8. Cache und Abschluss](#abschluss) · [9. Historisches Complete Set](#complete-set) · [10. v2.81](#v31-plan) · [11. Abnahme](#gates) · [12. Aktuelle Aufgaben](#todo) · [13. Betrieb](#betrieb) · [14. Evidenz](#ablage) · [15. Klärungen](#grenzen) · [16. Übergabe](#uebergabe) · [17. Änderungen](#aenderungen) · [Quellen](#quellen)

<a id="dokumentfuehrung"></a>
## 0. Dokumentführung und Quellenrang

Diese Fassung erhält die vollständige REV3 und ihre verbindliche Untersuchungsgrenze. Neue Grundlage sind die tatsächlich gelesene v2.80.4-Quelle, der letzte Nachtlauf und der kombinierte Smoke. Abschnitt 12 bleibt die einzige aktuelle Aufgabenliste; die früheren .4-Installationsaufforderungen sind historisiert. Die v2.81-Softwareprüfung steht mit ihren tatsächlichen Ergebnissen in der beigefügten `VERIFICATION_V281.json`. Eine neue Hardwareabnahme wird erst aus dem Zielsystembericht des mitgelieferten gemeinsamen Starters abgeleitet. [E-281-BASE] [E-281-NIGHT] [E-281-SMOKE]

Originale Ergebnisdateien haben Vorrang vor Auswertungen und Chat-Zusammenfassungen. Ein Implementierungsplan belegt einen Auftrag und die dort benannten Befunde; seine Beschreibung ersetzt keine unabhängige Prüfung einer fehlenden Originaldatei. `SOURCE_INDEX` und die Verifikation im .4-Bundle halten die tatsächlich verfügbaren Quellen und Testausführungen fest. Nachgereichte Archive oder neue Reporter verändern weder das Alter noch den ausgeführten Scope einer Messung.

Software-PASS, technische Hardwareausführung, Qualitätsentscheid und wissenschaftliche Vergleichsfreigabe sind verschiedene Aussagen. Ein fertiges negatives Ergebnis bleibt auswertbar. Ein abgebrochener Auftrag ohne Qualitätsresultat bleibt nicht ausgewertet. Ein neuer Downloadname, ein `plan.json` oder eine GUI-Erfolgsmeldung ersetzt keinen terminalen Ergebnisbeleg.

Die Dokumentfortschreibung selbst führt keine GPU-, Compiler-, Inferenz- oder Energiemessung aus. Eine v2.81-Zielinstallation und ein normaler H8-Kaltbuild dürfen nur mit ihren tatsächlichen neuen Nachweisen als bestanden bezeichnet werden. Frühere H8-/H10-Smokes werden als historische Evidenz erhalten; die gezielte Abnahme der korrigierten Pfade ist in §11 beschrieben.

### 0.1 Quellenstatus des vorherigen REV2-Abgleichs – historisch übernommen

**In REV2 direkt gelesen:** das 236-Dateien-Archiv `har_and_git_evidence.zip`, seine originalen HAR-/Runtime-/Buildberichte, API-/HN-Snapshots, die zentrale Qualityübersicht und die bereits vorliegende Git-Konsole. Erneut nachgezählt: alle sechs Stufen-Top-k-Listen, Vorhersagewechsel sowie 123 Quality-Endzustände. Numerische Logitstatistiken und Feedgleichheit bleiben Angaben der auf Smartmirror2 ausgeführten Collector; es wurden hier keine fehlenden Roharrays rekonstruiert. [E-HAR-R1] [E-KB-REV2-CHECK]

**Übernommen, nicht neu als Originalprüfung ausgegeben:** die neueren .3-FIX1–FIX5-/v2.80.4-Lieferangaben und der separate YOLO11l/H10-5.000-Bilder-Abschluss aus der beigefügten KB. Die dort benannten Originalexporte bzw. das .4-Lieferpaket lagen diesem Dokumentabgleich nicht zusätzlich vollständig vor. Diese bereits dokumentierten Entscheidungen werden erhalten und nicht wieder als neue Testpflicht geöffnet. [E-KB-2804-INPUT]

**Historischer Chat-/Abnahmebefund:** frühere Sourcegegenproben, Hardwareläufe und Testzahlen in dieser Unterhaltung bzw. ihren Berichten. Sie werden mit Version und Scope zusammengefasst, nicht bei jeder KB-Revision neu ausgeführt oder zu einer kumulierten Testzahl addiert.

**In REV2 vorgeschlagen:** der vertiefende Diagnoseweg in §7.8 und im Begleitplan. Er wurde nicht ausgeführt und ist durch die nachfolgende Scopeentscheidung jetzt ausdrücklich zurückgestellt. Seine technische Beschreibung bleibt erhalten, nicht sein früherer Vorschlag als nächster Arbeitsschritt.

### 0.2 Quellenstatus und Verbindlichkeit der REV3

Grundlage der historischen REV3 waren die vollständig gelesene beigefügte REV2 und die ihr unmittelbar vorausgehende Diskussion zur automatisierten Partitionierung ohne modellspezifisches Nachoptimieren. Der Nutzer hat deren Aufnahme in die Knowledgebase angefordert. Neue Texte zur Forschungsfrage, Stopregel, Folgearbeit und Dissertation sind eine dokumentierte Projektentscheidung, **keine zusätzliche experimentelle Evidenz**. [E-KB-REV2-INPUT] [E-SCOPE-REV3]

In REV3 wurden keine Ergebnisarchive erneut numerisch ausgewertet, keine Hersteller- oder Runtimeinternas neu geprüft, keine Builds oder Inferenzen ausgeführt und keine Git-Änderungen veröffentlicht. Die Dokumentprüfung sichert Erhalt der bestehenden Abschnitte, Ergebnistabellen und Verweise. Bei einem Konflikt zur früheren Priorisierung des separaten Diagnoseplans gelten für die aktuelle Arbeit §§1.6–1.11, 7.8 und 12 dieser Revision. Eine Wiederaufnahme benötigt einen neuen, ausdrücklich abgegrenzten Auftrag.

<a id="methode"></a>
## 1. Verbindlicher wissenschaftlicher Leitpfad

### 1.1 Zweck und Fallauswahl

Das Werkzeug und die vorab festgelegte Methodik werden validiert, nicht nachträglich nur die günstigsten Kombinationen ausgewählt. Quality-`fail` entfernt einen Fall nicht aus dem Audit. `inconclusive` ist weder `pass` noch automatisch ein technischer Fehler. Compiler-Rejects, nicht unterstützte Schnittstellen, fehlgeschlagene Runtimes und fehlende Messungen behalten unterschiedliche, ehrliche Terminalzustände. [E-KB]

Der produktive Ranker bleibt **`cut_bytes_only`**, aufsteigend nach Cut-Bytes mit deterministischem Case-/Boundary-Tie-Break. Stratified-Windows-Auswahl und score-unabhängiger Generic-Audit bleiben erhalten. Kein neuer Ranking-Sweep, keine nachträgliche Grenzwertoptimierung und keine Auswahl der günstigsten Wiederholung zur Ergebnisverbesserung. [E-KB] [E-PLAN31]

### 1.2 Modellumfang und Rollen

| Methodische Rolle | Modelle |
|---|---|
| Development | `resnet50`, `yolo26s`, `yolov7_paper` |
| Transfer/Evaluation | `mobilenet_v3_large`, `regnet_x_1_6gf`, `yolo26m`, `yolo11l` |

`yolo26x` ist nur ein gesondert zu planender Größen-/Stresstest, nicht automatisch ein achtes Modell. Die Rollen sind die methodische Vorgabe aus der bisherigen Knowledge Base. Eine abweichende Rollenbelegung eines konkreten Diagnoseprofils wird als dessen effektive Konfiguration berichtet, nicht still rückwirkend geändert. [E-KB]

### 1.3 Eingefrorene Policies

| Bereich | Fortgeltender Vertrag |
|---|---|
| Generic | Single- und Multi-Tensor-Boundaries an **einer** Splitstelle |
| Native | Technisch kompatible Single-Tensor-Boundaries; keine erfundene Multi-Tensor-Unterstützung |
| Multi-Split-Begriff | Keine neuen seriellen Mehrfachsplitstellen innerhalb eines Modells |
| Hailo-Build | `balanced`, Optimization Level 1, 500 Kalibrierungsbilder, Kalibrationsbatch 8 |
| DeepX-Build | 500 Kalibrierungsbilder, `ema`, Optimization Level 0 |
| DeepX-Klassifikation | **`imagenet_mean_std`** für Modelle mit entsprechendem Referenzvertrag; `current_scale_only` nur ausdrücklich benannter Legacy-/A/B-Arm |
| Detection | Modellfamiliengebundene Vorverarbeitung, Decoder, NMS und Koordinatenrücktransformation |
| Energie | Primär kalibrierte Full-System-Eingangsgröße; Native-Full-Baselines eingeschlossen; keine ersatzweise Aktivierung generischer oder CPU-/ORT-Energie |
| Finale Qualitätsaussage | Soweit im Thesisvertrag beansprucht: 5.000 Validierungsbilder, 5.000 Bootstrap-Wiederholungen, Official COCO/Pycocotools für Detection |

**B500-Kalibrierung, B500-Qualitätsauswertung und ein 16-Bilder-Smoke sind verschiedene Rollen.** Dieselbe Zahl von Bildern macht Trainingskalibrierung und Validierung nicht zum selben Datensatz. Ein Standard-B500-Lauf ersetzt nicht automatisch den weitergehenden Finalvertrag mit 5.000 Bildern; bereits abgeschlossene Finalaufträge werden nicht nochmals als offen geführt. [E-KB] [E-PLAN31]

### 1.4 Vergleichbarkeit und Ranking

Fairness verlangt gleiche wissenschaftliche Strata, gebundene Inputs, passende Artefakte, Precision, Endpunkte, Work Units und Messbedingungen. Backendspezifische Low-Level-Implementierungen sind erlaubt; zusätzliche Parallelität darf nicht unbemerkt nur einem Backend zugutekommen. Hersteller-Optimization-Level sind nicht als numerisch äquivalente Qualitätsstufen zu interpretieren. [E-KB]

Drei natürliche, identische und vergleichbare Single-Tensor-Fälle pro Modell bleiben das Ziel für die vorgesehene modellinterne Generic↔Native-Korrelation. Weniger vorhandene Fälle werden ehrlich deskriptiv ausgewertet. Kein künstlicher Split und keine Vermischung unterschiedlicher Endpunkt-/Precisiongruppen, um die Mindestzahl zu erreichen. Ein Sieben-Modell-`Complete_Set` ist nicht automatisch ein hinreichendes Rankingexperiment. [E-KB] [E-V30, §8]

### 1.5 Final Quality ist kein impliziter Wechsel sämtlicher Messverträge

`Final Quality (Standard+)` erhöhte im beobachteten Nachtlauf Validierungsumfang und Qualitybootstrap auf jeweils 5.000; B500-Kalibrierung, balanced/Opt1/Batch8 und `relaxed` blieben unverändert. Die reale Profilquelle ist die aufgelöste Run-Mode-Konfiguration plus erhaltene explizite Profilwerte. Ein Modusname ersetzt weder Freeze-/Rollenprüfung noch den Energievertrag. Eine im Profil verbliebene Native-Energiedauer von 1 s wird durch diesen Namen nicht zu 60 s. [E-NIGHT-2803]

Die sieben Modelle wurden in den Diagnoseprofilen teilweise alle als `development` geführt. Das ist der archivierte tatsächliche Diagnoseumfang, keine rückwirkende Erfüllung einer ungeöffneten Hold-out-/Transferrolle. Weitere Ursachenprüfungen auf denselben 16 geöffneten Bildern sind Diagnose und kein neues unabhängiges Validierungsset.

### 1.6 Forschungsfrage und Bedeutung von „einfach so“

Die Hauptfrage lautet:

> Wie gut lassen sich die untersuchten vortrainierten Netze mit dem entwickelten Werkzeug und einer vorab festgelegten Verarbeitungspolitik automatisiert partitionieren und auf heterogener Hardware ausführen – hinsichtlich Ausführbarkeit, Qualität, Performance und Energie?

„Einfach so“ bedeutet **nach Einrichtung der unterstützten Backends, mit den implementierten allgemeinen bzw. modellfamilienbezogenen Regeln und ohne zusätzliche manuelle Optimierung jedes einzelnen Evaluationsfalls**. Es bedeutet nicht ohne Kalibrierung, ohne korrekte Vor-/Nachverarbeitung oder ohne Backendkonfiguration. Die im festen Rezept vorgesehenen automatischen Compileroptimierungen gehören weiterhin zum untersuchten Verfahren. Ihre Verwendung ist kein nachträgliches manuelles Tuning. [E-SCOPE-REV3]

Nicht Gegenstand der Hauptauswertung ist, für jedes Netz durch Architekturanpassung, individuelles Quantisierungsrezept, Compilerparametersuche oder Nachtraining die bestmögliche Herstellerimplementierung zu finden. Die Resultate beschreiben die Reichweite und Grenzen der untersuchten Tool-/Backendkonfiguration, weder eine universelle Eigenschaft aller Compiler noch die maximal erreichbare Modellqualität.

### 1.7 Fehlerkorrektur, Ursachenanalyse und Tuning getrennt behandeln

| Kategorie | Beispiel / Abgrenzung | Konsequenz |
|---|---|---|
| Implementierungskorrektheit | Fehlende vorgeschriebene Mean/Std-Normalisierung; falsches Tensorlayout; fehlerhafte Referenzbindung oder Mess-/Ergebniszuordnung | Den konkreten Fehler korrigieren oder den betroffenen Fall als technisch fehlerhaft bzw. nicht auswertbar kennzeichnen. Kein gewöhnlicher Quantisierungsverlust und kein Quality-PASS daraus ableiten. |
| Ergänzende Ursachenanalyse | Bestehende HARs vergleichen; optimierte Floatstufe oder Integer-I/O bei unverändertem HEF untersuchen | Kann die Diskussion vertiefen. Keine automatische Pflicht allein wegen einer großen Accuracyabweichung; für die derzeitige Hauptaussage zurückgestellt. |
| Modellspezifische Optimierung | Andere Kalibrationsmengen, Layerpräzision, Opt-Level, individuelle Compilerrezepte, Architekturänderung oder Nachtraining nach Betrachtung der Ergebnisse | Nicht Teil der eingefrorenen Hauptauswertung. Nur als eigenständiges, vorab beschriebenes Folgeexperiment mit getrennten Artefakten und Validierungsdaten. |

Ein korrekt eingerichteter Compilerkontext, Cache-Reuse und wahrheitsgemäße Berichte bleiben notwendige Softwarearbeit. Sie ändern nicht allein das wissenschaftliche Buildrezept. Eine Fehlerkorrektur wird versioniert und ihr betroffener Auswertungsumfang nachvollziehbar neu geprüft; alte Messungen werden nicht rückwirkend zu Ergebnissen des reparierten Pfads. Die durch frühere Ergebnisse beeinflusste Entwicklung wird transparent berichtet, nicht nachträglich als ungeöffnete Evaluation bezeichnet. [E-SCOPE-REV3] [E-KB]

### 1.8 Stopregel und Abschlussbedingung

**Die Größe eines korrekt zugeordneten Qualitätsverlusts ist allein kein Auftrag für weitere Tests, Neubauten oder numerische Änderungen.** Ein vollständiger Qualitäts-FAIL bleibt ein abgeschlossenes negatives Ergebnis. Zusätzliche Prüfung ist nur für einen konkreten neuen Fehlerverdacht, einen nachgewiesenen eigenen Fehler, noch fehlende erforderliche Evidenz oder eine ausdrücklich gewünschte stärkere Aussage zu planen. Ein neuer Verdacht ist selbst noch kein Fehlernachweis. [E-SCOPE-REV3]

Bekannte Implementierungsfehler werden nicht aus Zeitgründen zu „Compilerverlust“ umbenannt. Ungültige Detectionausgaben sind technische/semantische Misserfolge und keine regulären niedrigen AP-Werte. Ein nicht realisierbarer Split, eine Runtimeblockade, ein Qualitäts-FAIL, ein INCONCLUSIVE und ein Cancel ohne Ergebnis behalten ihre unterschiedlichen Zustände. Insbesondere werden die 60 Cancel-Fälle aus §7.9 durch diese Entscheidung nicht zu abgeschlossenen Qualitätsvergleichen.

Abschlussziel der Hauptauswertung ist die vollständige, überprüfbare Bilanz der vorab festgelegten Fälle: gemessene Resultate oder konkret belegte Nichtausführbarkeit/technische Einschränkung; verbleibende Evidenzlücken sichtbar. Ein bislang nicht gestarteter Messauftrag wird nicht allein durch einen Platzhalter abgeschlossen. Für positiv beanspruchte Qualitäts-, Performance- oder Energieaussagen bleiben sämtliche zugehörigen Gates bestehen. **Vollständig dokumentierte Ergebnislage ist nicht gleich vollständige positive Freigabe.**

Keine Lockerung der Qualitätsmargen, kein Austausch der Referenz, keine nachträgliche Auswahl besserer Splits, Seeds, Wiederholungen oder privater GPU-HEFs aufgrund bereits betrachteter Ergebnisse. Vorhandene erfolgreiche technische und negative fachliche Nachweise behalten ihren Scope; keine identische Testschleife bis PASS.

### 1.9 Zwei Vergleichsperspektiven auf den Nutzen des Splittens

**Gegen die kanonische Full-Floatreferenz:** Wie viel Qualität erhält die gesamte heterogene Umsetzung? Diese Referenz bleibt für das bestehende Qualitätsgate maßgeblich.

**Gegen die vollständige Backendumsetzung:** Was verändert die untersuchte Partitionierung gegenüber der Full-Ausführung unter dem zugehörigen festgelegten Backendrezept? Dieser ergänzende Vergleich hilft, den Gesamtverlust gegenüber Float nicht pauschal dem Splitten zuzuschreiben. Er ersetzt nicht die kanonische Referenz und ist keine neue Akzeptanzgrenze. Gleiches Dataset, Task-/Endpointvertrag und passende Artefaktzuordnung bleiben erforderlich.

Beispiel aus den bereits dokumentierten zentralen MobileNet-Ergebnissen (§§7.2 und 7.9): CPU-Float **73,70 %**, Hailo8 Full **60,62 %**, Hailo8 → TensorRT b027 **71,58 %** auf denselben 5.000 Bildern. Der konkrete frühe Split liegt damit **10,96 Prozentpunkte über Hailo8 Full**, aber weiterhin **2,12 Prozentpunkte unter Float** und bleibt nach der unveränderten 1-pp-Regel FAIL. Das ist eine deskriptive Einordnung bereits vorhandener Werte, kein neues gepaartes Signifikanzresultat zwischen Full-Hailo8 und Split und keine allgemeine Aussage für alle Grenzen. [E-NIGHT-QUALITY]

Der Fall illustriert einen Nutzen und zugleich eine Grenze der automatischen Partitionierung. Weder wird der frühe Split nachträglich als einziger Fall ausgewählt, noch wird aus einem schlechten Full-HEF geschlossen, sämtliche Splitpunkte müssten genauso schlecht sein. Die vorhandene HAR-Fallstudie unterstützt eine begrenzte Diskussion; sie liefert keine additive kausale Aufteilung des gesamten Accuracyverlusts.

### 1.10 Zulässige Tuning-Aussage und Grenze zukünftiger Arbeit

Zulässig ist: **Modellspezifische Anpassungen könnten einzelne Ergebnisse verbessern; Wirksamkeit, erreichbarer Umfang und zusätzlicher Aufwand sind für die hier untersuchten Fälle nicht systematisch bestimmt.** Die Herstellerbeschreibung benennt mögliche Optimierungsverfahren, aber keine nachgewiesene Verbesserung unserer Modelle. [E-HAILO-OPT-DOC]

Nicht belegt sind Aussagen wie „mit ausreichend Aufwand werden alle Netze gut“, „Opt2/Opt3 ist immer besser“ oder „der gesamte Verlust ist ausschließlich Quantisierung“. Mehr Aufwand garantiert keinen Erfolg; die methodische Entscheidung gegen neue Sweeps hängt nicht davon ab, ob eine künftige Verbesserung möglich wäre. [E-SCOPE-REV3]

Eine später ausdrücklich beauftragte Optimierungsstudie muss vom jetzigen festen Rezept und Ergebnisbestand getrennt bleiben. Bereits geöffnete Fixed16- und Evaluationsbilder sind keine unabhängigen Daten zur Bestätigung einer anhand dieser Resultate ausgewählten Konfiguration. Historische Opt2-Diagnosen und die abgeschlossenen GPU-/HAR-Smokes werden nicht verschwiegen: **Ausgeschlossen ist zusätzliches systematisches Einzelfalltuning der Hauptauswertung, nicht die tatsächlich erfolgte Entwicklung und Diagnose.**

### 1.11 Formulierungsbaustein für die Dissertation

> Die Evaluation untersucht die automatisierte Partitionierung und heterogene Ausführung vortrainierter Netze mit einer vorab festgelegten, backendspezifischen Build- und Validierungspolitik. Die innerhalb dieses Verfahrens vorgesehenen automatischen Optimierungsschritte bleiben Bestandteil der untersuchten Verarbeitungskette. Eine darüber hinausgehende manuelle, modellspezifische Optimierung von Netzarchitektur, Quantisierung oder Compilerparametern ist nicht Gegenstand der Hauptauswertung.
>
> Qualitätsverluste, nicht realisierbare Partitionierungen und technische Ausführungsfehler werden als unterschiedliche Ergebnisse vollständig berichtet. Ergänzende Diagnosen dienen der Prüfung ausgewählter Verarbeitungsschritte und der begrenzten Einordnung beobachteter Abweichungen; sie stellen keine vollständige Ursachenanalyse sämtlicher Compiler- und Runtimeeffekte dar.
>
> Die Ergebnisse beschreiben damit die Leistungsfähigkeit und Grenzen des implementierten automatischen Verfahrens unter der untersuchten Konfiguration, nicht die maximal erreichbare Qualität individuell optimierter Implementierungen. Ob und mit welchem Aufwand modellspezifische Anpassungen die verbleibenden Verluste verringern können, bleibt zukünftiger Arbeit vorbehalten.

Dieser Baustein dokumentiert den vereinbarten Untersuchungsumfang, keinen neuen Messbefund. Die tatsächliche Toolentwicklung, frühere Diagnosen, Datenöffnung und die unverändert fortgeltenden Modellrollen bleiben in der Arbeit transparent. [E-SCOPE-REV3]

<a id="fragen"></a>
## 2. Fortgeschriebener Fragen- und Boundarykatalog

Die folgende Arbeitsübersicht führt die dauerhaft relevanten Antworten fort und ergänzt die neuen DeepX-/Complete-Set-Fragen. Historische Detailbegründungen bleiben unter [E-KB] erhalten.

| Frage | Kanonische Antwort / heutige Grenze |
|---|---|
| Erreicht Native die ungefähr 97 FPS der handimplementierten YOLOv7-Referenz? | Für den exakten historischen Hailo-8-Pfad `yolov7_paper/b066` sind rund 97,077 P2- und 97,059 Completed-Detection-FPS dokumentiert. Kein allgemeiner Wert für andere Splitpunkte oder v31. |
| Ist dieselbe Implementierung für alle Backends erforderlich? | Nein. Derselbe wissenschaftliche Vertrag ist erforderlich; passende native I/O-, FIFO- und Runtimepfade sind erlaubt. |
| Bestimmt die langsamste Stage die Pipeline-FPS? | Sie beschreibt den idealen Flaschenhals. Maßgeblich ist die direkt gemessene Makespan-Rate abgeschlossener Work Units; Handoff und Backpressure können zusätzlich begrenzen. |
| Sind Hailo-8 und Hailo-10H mit „gleicher Qualität“ gebaut? | Gleiche angeforderte Recipe und derselbe Datenvertrag, nicht garantierte gleiche Accuracy. |
| Was ist bei DeepX qualitätsrelevant? | B500, EMA, Opt0 und der zum Modell passende numerische Eingabepfad. R1/R2 zeigen, dass ein vorhandener korrekter Pfad durch eine falsche Profilwahl wirkungslos bleiben kann. |
| Muss DeepX jetzt einen neuen Preprocessor erhalten? | Nein. Der vorhandene v30-Sub/Div-Buildadapter funktioniert im R2-Smoke. Der bestätigte vorhandene Pfad wird regulär geroutet; aktuelle .4-Regressionen erhalten Full-/Part1- und Cachebindung. |
| Ist MobileNet durch den Fixed16-Smoke bestanden? | Technische Ausführung im benannten Scope; keine B5000-Freigabe. Neuere abgeschlossene Full-Quality-FAILs stehen in Abschnitt 7. |
| Beweist R2, dass die alten 6,6 Prozentpunkte Verlust vollständig verschwinden? | Nein. Der Mechanismus ist bestätigt; die quantitative Wirkung auf den alten 500er-Umfang ist nicht neu gemessen. |
| Müssen Top-k, Softmax oder Labels geändert werden? | R1 liefert dafür keinen Anlass: die geprüften Rohlogits, Produktiv-Top-k und Zuordnungen stimmen im getesteten Umfang überein. |
| Sind 55 unqualifizierte Energiezeilen 55 Messfehler? | Nein. Alle 55 haben drei gültige Replikate. Globale Matrixunvollständigkeit sowie lokale Quality-/Endpointgründe verhindern die Freigabe. |
| Ist M.2-Idle vom Primärergebnis abzuziehen? | Primär bleibt die kalibrierte, nicht idle-subtrahierte FS-Gesamtenergie. Eine separate TensorRT-Full-Normalisierung darf den gebundenen Idlewert zusätzlich ausweisen. |
| Braucht jedes Modell zwingend drei neue Splits? | Nein. Vergleichbare natürliche Fälle nutzen; unzureichende Kandidatenzahl kennzeichnen. |
| Ist ein `KNOWN_INFEASIBLE`-Split ein Cachefehler? | Nein, wenn exakte negative Compile-Evidenz vorliegt. Er bleibt ein erklärter nicht ausführbarer Planfall. |
| Darf eine alte Messung durch einen neuen Reporter zu `pass` werden? | Ein Zuordnungsfehler darf read-only erklärt werden. Messstatus, Payloads und wissenschaftliche Freigabe dürfen nicht erfunden oder umetikettiert werden. |
| Brauchen wir neue Hash-/Seal-/Datenbankebenen? | Nein. Vorhandene notwendige Identitäten weiterverwenden; teure globale Cache-Neuschreibvorgänge entfernen statt neue Systeme einzuführen. |
| Sind die 56 neuen .4-Anforderungen schon 56 bestandene Tests? | Nein. Geplante Anforderungen, parametrisierte Testfälle und tatsächliche Ausführungen unterscheiden; die .4-Verifikation nennt den belegten Umfang. |
| Ist Hailo8 GPU erst noch grundsätzlich zu testen? | Nein: Compute/XLA, ein echter MobileNet-Build und 32 Fixed16-Geräteinferenzen sind abgeschlossen. Die nächste Abnahme betrifft den korrigierten normalen v2.81-Pfad (§11), nicht dieselbe Diagnosewiederholung. |
| Bedeutet Parsed-HAR = Float, dass nur Quantisierung schadet? | Nein. Der Vergleich bis Quantized umfasst auch Float-/Graphoptimierungen und quantisierungsbegleitende Anpassungen; §7.7–7.8 trennen belegt und vorgeschlagen. |
| Ist Quantized-HAR = GPU-HEF nachgewiesen? | Nein: vier unterschiedliche Top-1-Vorhersagen auf 16 Bildern; RMS 0,39264502. |
| Macht Quality-FAIL den HEF-Cache kaputt? | Nein. Technische Verwendbarkeit und Qualitäts-/Claim-Eignung sind getrennt. |
| Sind HAR-Ergebnisse bereits im Git? | Laut übernommener erfolgreicher Push-Konsole ja, einschließlich `REPORT.md` und `CLAIM_BOUNDARIES.md`; spätere ausführliche Interpretation und KB-REV2/REV3 sind nicht Teil dieses benannten Commits. Ein neuer Dokumentpush ist nicht belegt. |
| Sind 225 GPU-Samples oder 203 Assembleraufrufe 225/203 unabhängige Tests? | Nein. Das sind zugeordnete Beobachtungen innerhalb desselben Builds. |
| Was heißt „Splitten einfach so“ in der Dissertation? | Automatisierte Ausführung nach Backendeinrichtung mit den festgelegten Regeln und Recipes, ohne manuelle Optimierung jedes Evaluationsfalls; nicht ohne Vorverarbeitung oder Kalibrierung. |
| Müssen wir einen großen Qualitätsverlust vollständig erklären, bevor er berichtet werden darf? | Nein. Korrekt ausgeführte und gebundene negative Ergebnisse sind berichtsfähig; stärkere Ursachenbehauptungen benötigen eigene Evidenz. Bekannte eigene Fehler bleiben technische Fehler. |
| Ist die vertiefte H8-Analyse jetzt noch ein aktiver Auftrag? | Nein. Optimierte Floatstufe, Integer-I/O- und weitere interne Ursachenprüfung sind zurückgestellte optionale Folgearbeit (§7.8), keine Voraussetzung für den Integrationsrelease oder die jetzige Hauptaussage. |
| Sind Fehlerkorrektur und Tuning dasselbe? | Nein. Den festgelegten Vertrag korrekt umzusetzen ist Pflicht; das Netz nach sichtbaren Ergebnissen individuell zu optimieren ist ein anderes Experiment. |
| Dürfen wir Verbesserungen durch Tuning in Aussicht stellen? | Nur als unbestimmte Möglichkeit zukünftiger Arbeit, nicht als Nachweis, dass alle Modelle oder Qualitätsmargen damit erreichbar sind. |
| Bedeutet ein besserer Split als das Full-HEF automatisch Quality-PASS? | Nein. Der ergänzende Backendvergleich ersetzt das unveränderte Gate gegen die kanonische Floatreferenz nicht. |
| Wann ist die Hauptauswertung abgeschlossen? | Wenn die vorab festgelegten Fälle nachvollziehbar mit Ergebnissen oder belegten Grenzen bilanziert sind und benötigte Nachweise für die tatsächlich beanspruchten Aussagen vorliegen; nicht erst, wenn alle Qualitätsfelder grün sind. |


Quellen: [E-KB] [E-V30] [E-R1] [E-R2] [E-PLAN31] [E-SCOPE-REV3].

<a id="endpunkte"></a>
## 3. Setups, Work Units und Endpunkte

### 3.1 Dokumentierte Arbeitsumgebung

Controller/GUI: `Smartmirror2`, Tool unter `~/ONNX-Splitpoint-Tool`, bestehende Tool-Venv `.venv`. Modell-/Ergebnisablage unter `~/Models`, lokale Buildartefakte unter `~/Models/BackendArtifacts`. Die beobachtete lokale Compiler-GPU ist eine GTX 1080 Ti; sie ist **nicht** das Jetson-Messgerät. [E-R1] [E-R2] [E-OVERNIGHT]

| Setup-ID | Messsystemrolle |
|---|---|
| `orin_nx_hailo8_01` | Jetson Orin NX mit Hailo-8 und zugehöriger FS-Messkette |
| `orin_nx_hailo10_01` | Jetson Orin NX mit Hailo-10H und zugehöriger FS-Messkette |
| `orin_nx_deepx_m1_01` | Jetson Orin NX mit DeepX M1 und zugehöriger FS-Messkette |

Die historische Bezeichnung `hailo10` in Pfaden ist von der normalisierten Architektur `hailo10h` zu unterscheiden. Konkrete IPs, Runtime-Venvs und Powerzustände aus der zugehörigen Konfiguration lesen, nicht aus alten Kommandozeilen erraten. [E-CS] [E-OVERNIGHT]

### 3.2 Physischer Output und abgeschlossene Aufgabe

Bei Detection bleiben `raw_head`, `decoded_pre_nms` und `decoded_nms` getrennt. Bereits dekodierte Boxen/Scores vor NMS sind noch nicht die abgeschlossene Detectionaufgabe. Umgekehrt darf ein nachweislich integrierter NMS-Endpunkt nicht ungeprüft ein zweites Mal NMS erhalten. [E-V30] [E-PLAN31]

`endpoints/completed_task/` ist zunächst die **Rolle eines Evidence-/Ausführungspfads**. Nicht jede dort archivierte Tensor-Datei ist deshalb bereits ein Post-NMS-Detektionsarray. Physische Ausgabe und Completionnachweis bleiben getrennte Verträge. [E-PLAN31, AP3]

Performance und Energie verwenden denselben gebundenen Arbeitsendpunkt. Die reale Nachverarbeitung muss innerhalb des beanspruchten Messfensters erledigt sein. Aufwendige Oracle-, Hash- und Dump-Arbeit darf in Vor-/Nachprüfungen liegen; die **eigentliche Aufgabe pro Frame** darf dadurch nicht aus dem Messfenster verschwinden. Ein Oracle-PASS außerhalb des Zeitfensters allein beweist keine zeitlich überlappende Three-Stage-Ausführung. [E-PLAN31, AP4]

### 3.3 Pipeline- und Energieraten

Die Pipeline-Rate wird aus abgeschlossenen Work Units und der tatsächlich beobachteten Wandzeit bestimmt. P1-FPS, P2-FPS, Cycle-FPS, interne DXRT-Throughputzahlen und Completed-Detection-FPS werden nicht austauschbar benannt. Für Energie sind die Work Units der konkreten Energieausführung maßgeblich, nicht eine aus früheren FPS hochgerechnete Ersatzanzahl. [E-KB] [E-PLAN31]

### 3.4 Compilerfamilien und erfolgreich geprüfte lokale Kontexte

| Familie | Tatsächlich geprüfte Umgebung | Ursprüngliche Blockade | Positiver Nachweis / Grenze |
|---|---|---|---|
| DeepX | Compilerlokales Torch-cu126-Overlay für GTX 1080 Ti / SM 6.1 | cu130-Paket ohne passende GPUarchitektur | R1/R2 und spätere reguläre Mean/Std-DXNN-Builds; nicht mit Hailo-TensorFlow vermischen |
| Hailo10H | DFC 5.3.0; TensorFlow 2.19.1; vorhandener Triton-ptxas 12.8.93 plus zugehörige libdevice | Systempfad zeigte auf CUDA-13-ptxas, der `sm_61` ablehnte | GPU/XLA und echter MobileNet-Build mit 327 zugeordneten Aktivitätsmessungen; Fixed16 technisch bestanden |
| Hailo8 | DFC 3.33.1; TensorFlow 2.18.0; separates CUDA-12.5.82-Overlay mit zwölf NVIDIA-Paketen | benötigte Runtimebibliotheken und passender ptxas/libdevice-Kontext fehlten | GPU/XLA, Modellbuild mit 225 positiven eigenen GPU-Samples, 203 Assemblierungen und Fixed16 bestanden |

Hailo10-Freigabe deckt Hailo8 nicht ab. Der späte Threadsetterfehler im ersten Smoke war ein Helferfehler; `DLOPEN_UNRESOLVED` in einem neutralen Diagnoseprozess bewies bei Hailo10 keine neun fehlenden Compute-Libraries. Dort funktionierten normale GPUoperationen bereits, bevor XLA durch die passende Assemblerwahl repariert wurde. Ein Component-View aus ptxas/libdevice ist kein vollständiges CUDA-Toolkit. [E-GPU-HISTORY] [E-H8-COMPUTE] [E-H8-BUILD]

Der neue Hailo8-Zusatzbestand blieb außerhalb der Venv:

```text
~/.onnx_splitpoint_tool/hailo/hailo8_cuda_20260912T062240Z_hx7b7hcq/
  overlay_manifest.json
  packages/
```

Das Manifest allein reicht nicht; die referenzierten Bibliotheken behalten. Der Diagnoseprozess wählte ihn explizit. Eine GUI-GPUpräferenz alleine überträgt diesen Pfad noch nicht. Die .4-Integration sah dafür die bestehende Konfigurations-/Resolverstrecke vor. Der spätere Nachtlauf zeigte jedoch eine fehlende wirksame Auswahl; v2.81 korrigiert deren Übernahme und prüft den Kontext vor tatsächlich benötigten Kaltbuilds (§§7.11, 10.1), ohne Parentumgebung, Hailo10 oder DeepX zu verändern. Hailo8-Hardwaretest: VStreams, HailoRT 4.20.0, FLOAT32-Host-Ein-/Ausgaben; Hailo10-InferModel-Details nicht ungeprüft übernehmen. [E-H8-RUNTIME]

<a id="v30"></a>
## 4. Releasefortschritt bis v2.81

| Stand | Belegter Fortschritt und verbleibende Aussagegrenze |
|---|---|
| v2.79.30 | Historische positive fachliche Teilabnahme von DeepX-Full-Prüfübergängen, Merge und terminalem Abschluss; keine damalige Gesamtfreigabe |
| v2.79.33 | Force in 27 aktiven Nutzerprofilen ausgeschaltet; mitgelieferte Canary-Schalter wurden gesondert als Sourceauftrag verfolgt |
| v2.79.34 | Produktive Force-Einstiegspunkte bereinigt; Familienkontext, Diagnosebindung und Prozessbereinigung integriert. Auf Smartmirror2 laut originalen Abnahmen 2.561 Abnahme- und 388 Kurztests bestanden |
| v2.80 | Cacheprüfung ohne unnötigen SDK-Import, H8-Kindumgebung und Cleanup-/Archivabschluss korrigiert. Historisch je 2.659 Softwaretests aus Source und isoliertem Upgrade sowie 535 Kurztests bestanden; Wiederholungen nicht addieren |
| v2.80.1/.2/.3 | CPU-Referenzbindung, vollständige Remote-Abhängigkeiten, Buildbereitschaft, Native-Nichtstarts und Debugdiagnosen sind die fortgeltende .3-Basis; keine alten .2-Dateien zurückkopieren |
| v2.80.3 FIX1 | Umgebungsabhängigen Testfehler bei fehlendem Originalmodell isoliert; gezielte Reparaturabnahme PASS. Der vorherige Lauf hatte 3.122 bestandene Tests, einen Fehler und eine Deselektion; das ist kein vollständig grüner Erstlauf |
| v2.80.3 FIX2–FIX5 | Gezielte Folgekorrekturen für Referenzvorbereitung/Hardwarebindung und YOLO11l-Normalworkflow. Metadaten-/Source-PASS allein wurde nicht als Hardware-PASS gezählt |
| v2.80.3, 5.000-Bilder-YOLO11l | Normaler Workflow technisch abgeschlossen, vorhandene Artefakte wiederverwendet, vollständige Vorhersagen exportiert. Hailo-Qualitäts-FAIL bleibt bestehen |
| v2.80.4 | Liefert AP1/AP2/AP3 sowie Reuse-/Aussageabsicherung nach neuem Plan. Konkrete neue Testzahlen, erlaubte Nichtausführungen und Installerbelege ausschließlich aus der .4-Verifikation übernehmen |
| v2.81 | Kaltbuild-Kontext nach finaler Cacheauswahl prüfen; H8-Pfade mit identischem RGB-Puffer; H10 NWC/NCW-Übergang berichtigen; exakt bestätigte Nichtrealisierbarkeit und äquivalente Ergebnisidentitäten richtig bilanzieren. Software- und neue Zielhardwareabnahme getrennt. |

Diese Zeilen sind ein Fortschrittsindex. Alte Testzahlen werden nicht als neue .4-Tests ausgegeben. Die auf Smartmirror2 tatsächlich gestartete Version ist aus ihrem Modul-/Source-Manifest festzustellen; gleiche sichtbare Versionsnummern können unterschiedliche ausdrücklich benannte FIX-Stände haben. [E-2804-CHAT] [E-2804-PLAN]

### 4.1 Force-Ursprung und zentrale versus explizite Profileinstellungen

Der Audit fand echte YAML-Booleans: `standard` und `smoke` Force aus, gespeichertes `final` Hailo/DeepX Force an. Der historische Resolver reproduzierte den Konfigurationshash von 16 alten Profil-Snapshots bis zum 3. September; die Force-Werte waren somit schon vor v32 in der Gesamtregistry vorhanden. **Die erste Schreibaktion ist unbekannt; daraus folgt keine bewusste Änderung durch den Nutzer.** [E-FORCE-AUDIT]

Bloßes Öffnen/Speichern korrekt typisierter `false`-Werte reproduzierte keine automatische Aktivierung. Dagegen waren `bool("false")` und das Überschreiben neuerer Dateien aus einem alten offenen Panel echte reproduzierte Robustheitsfehler, aber nicht als konkreter historischer Auslöser belegt. Die späteren typisierten Konfigurations-/Speicherfixes bleiben Regressionen. Der v34-Stand führte zusätzlich eine strengere produktive Force-Sperre ein; private nicht publizierende Einzelfalltests bleiben davon getrennt.

Bei `follow_tool_config=true` kommt Force aus dem ausgewählten zentralen Modus. Eine ausdrücklich im Evaluationsprofil gesetzte DeepX-Normalisierung kann dagegen einen zentralen Mean/Std-Default übersteuern. Daher immer **effektive Werte und ihre Quellen** prüfen. Alte Profiles erhalten bedeutet nicht, ihre alten Einstellungen seien automatisch korrekt. Ein vollständiger Reset ist nicht erforderlich; keine heimliche Profildateiumschreibung.

### 4.2 Fehlerketten und reale Übergangsnachweise v2.80–v2.80.3

| Beobachteter Fehler / Lernpunkt | Späterer enger Fix bzw. heutige Einordnung |
|---|---|
| H8-Overlay akzeptiert, aber Librarydirs nicht im Compute-Kindprozess | v2.80 überträgt bestehende `child_library_environment()` tatsächlich; später positiver realer H8-Computeumfang |
| Worker beendet, Remoteverzeichnis aber nicht gelöscht; trotzdem Cleanup=true | Prozess- und Stagingcleanup getrennt; Primärfehler erhalten; ZIP nur vollständig publizieren |
| v2.80: `quality_result_contract.py` fehlt in Remote-Closure | gesamten Paketbestand vor Importprüfung übertragen; sauberer Zielpaketbaum als Regression |
| Management-CPU-Referenz verlangt Generic-Kandidaten-IDs | eigene enge Referenzrolle berücksichtigen, keine Jetson-ID erfinden |
| v2.80.1: `management_cpu_reference_context_invalid:model_binding` | logische ID aus `model_id → model_name → model` statt ONNX-Pfad; falsche explizite IDs weiter ablehnen |
| übersprungenes TRT Full erscheint als Transferfehler/Missing | explizite negative Zeile `blocked_upstream_quality`, null Versuche, unveränderte Planidentität |
| Deferred-Build `partial` verschwindet hinter Stufe `ok` | betroffene Pflichtjobs mit Modell/Boundary/Backend/Primärgrund in Readiness übernehmen; gültige andere Jobs fortsetzen |
| null gestartete Splits erscheinen als `partial_repetitions` | belegte Nichtstarts von echten Teilmessungen unterscheiden; unbekannte Versuchszahlen nicht auf 0 raten |
| Debugexport verliert kleine CPU-Logs oder nennt großen gültigen Index beschädigt | kleine Prozessdiagnosen zulassen, Größenlimit von Syntax-/Integritätsfehlern trennen |
| v2.80.3-Nacht: plötzlicher Fortschritt nach Cancel | 63 ausgewertet und 60 Abbruchfolgen, nicht 123 berechnete Vergleiche; .4-Berichtsauftrag |

Die vollständige v2.80.1-Zielsoftwareabnahme meldete 2.832 unterschiedliche Tests; 708 Kurztests waren eine Teilmenge. Trotzdem scheiterte danach der echte Generator-/Referenzübergang an `model_binding`. **Große Testzahlen ersetzen somit keine passende realistische Pipelinefixture.** Die .2-Prüfung verwendete die sieben tatsächlichen BenchmarkSet-Kontexte; im .3-Nachtlauf wurden schließlich alle sieben CPU-Referenzen mit 5.000 Einträgen erzeugt. Bereits erfolgreiche Remote-/Vendor-Full-/Energieteilstufen sind von fehlender zentraler Qualitätsauswertung zu trennen. [E-RELEASE-AUDITS] [E-WORKFLOW-HISTORY] [E-NIGHT-QUALITY]

<a id="deepx-full"></a>
## 5. Historische DeepX-Full-Fehlerlokalisierung und fortgeltende Regressionen

Die nachfolgenden Versionen v26–v30 sind historische Quellenstände. Ihre Fehlerbeschreibungen sind keine neue .4-Aufgabenliste; spätere Korrekturen und heutige Gates stehen in Abschnitt 4 und 10–12.

### 5.1 Fehlerfolge v26 bis v30

| Stand / Befund | Ursache und dauerhafte Konsequenz |
|---|---|
| v26 Native Full: 3 versucht, 0 gültig, 0 Frames | `deepx_shared_prepared_input_manifest_missing`; tatsächliche Eingabe vor Wiederholungen auflösen und explizit weiterreichen. |
| v26 separater Full-Test | `decoded_pre_nms_values_invalid`; Runtime-Laden war nicht gleich abgeschlossene Detection. |
| v27 erste Kurzprobe | Verwendete eine bereits gelöschte temporäre Remote-Suite; kein Modelltest. Eigene isolierte Staging-Verzeichnisse statt alter Remote-Arbeitskopien. |
| v27 FIX1 | Staging erfolgreich; echte Rohausgabe zeigt den numerischen Auslöser. |
| v28 | Begrenzte Float32-Score-Randtoleranz umgesetzt; lokale Abnahme durch unnötigen `cv2`-Import blockiert. |
| v29 | Pillow ersetzt diesen unnötigen Import im vorbereiteten Full-Pfad; Software-Abnahme und echte Einzelbildprobe erfolgreich. |
| v29 normaler D-Lauf | 3 × 3.000 Frames erfolgreich berechnet, anschließend von separatem Semantik-/Quality-Prüfpfad abgelehnt. |
| v30 | Die fehlenden Pre-NMS-Übergänge, Merge- und Vorprüfungsregeln implementiert und offline positiv teilabgenommen. |

Quellen: [E-D26] [E-D29] [E-PROBE29] [E-V30].

### 5.2 Numerikregel – kein allgemeines Clipping

Der aufgezeichnete YOLO11l-Tensor ist Float32 mit Form `[1,84,8400]`. Alle 705.600 Werte sind endlich. Es gibt keine negativen Breiten/Höhen und keinen Klassenscore über eins; **21 Klassenscores** sind exakt `−2⁻²⁴ = −5,960464477539063e−8`. Die frühere strikt negative Prüfung verwarf deshalb den ganzen Output. [E-D26] [E-NUMERIK]

Die produktive Korrektur verwendet für entsprechend deklarierte Float32-Wahrscheinlichkeits-Scores eine absolute Randtoleranz `ε = 2⁻²³ ≈ 1,1920928955e−7`. Nur innerhalb dieser engen Randzone außerhalb `[0,1]` erfolgt eine Normalisierung auf **einer Verarbeitungskopie**. Größere Bereichsverletzungen, NaN/Inf oder unzulässige Boxgeometrie bleiben Fehler. Kein zusätzlicher Sigmoid, keine gelockerten NMS-Schwellen und keine geänderten AP-Margen. Historische Verträge werden nicht rückwirkend auf die neue Numerikregel umgeschrieben. [E-NUMERIK] [E-V30]

Die reale v29-Kurzprobe verarbeitet denselben Rohtensor bis zu zwei Detektionen. Die 21 negativen Originalwerte bleiben im gespeicherten Rohoutput erhalten. Das bestätigt die beabsichtigte Verarbeitungskopie, nicht die gesamte Detectionqualität. [E-PROBE29]

### 5.3 Warum v29 trotz erfolgreicher Inferenz noch rot war

Im normalen D-Lauf waren die drei Performanceprozesse erfolgreich, je 3.000 abgeschlossene Frames und kein Timeout. Der separate DeepX-Semantikdump behandelte `decoded_pre_nms` aber nicht wie der Messpfad und lieferte einen leeren Frozen-Vertrag. Der Merge erzeugte dadurch irreführende negative übergeordnete Postprocessingfelder. Unabhängig davon lehnte der zentrale DeepX-Loader den physischen Stage-Namen ab. Beides sind Integrationsfehler, nicht ein erneut fehlendes OpenCV oder ein weiterer GPU-Ausfall. [E-D29]

Die dort aufgezeichneten Full-Raten von ungefähr 20,960 / 20,957 / 21,408 FPS sind deshalb **diagnostische Werte eines damals nicht abgenommenen Fullpfads**. Die akzeptierten v29-D-Medianwerte DeepX→TRT `b003` 34,681 FPS und TensorRT Full 36,901 FPS gehören zu ihren konkreten erfolgreichen Nativepfaden, nicht zu einem neuen v31-Lauf. [E-D29]

### 5.4 Release-/Installationsbeleg v29

Der v29-Installationsnachweis auf Smartmirror2 dokumentiert `INSTALL_ACCEPTANCE=PASS`, `FINAL_STAGE=complete`, 1.130 Tests plus einen separat ausgeführten GUI-Test, insgesamt **1.131**. Die bestehende Venv und 26 Benutzerprofile blieben erhalten. Dieser abgeschlossene Zwischenstand bleibt historischer Nachweis; die heutige Aufgabe ist nicht, v29 erneut zu installieren. [E-INSTALL29]

<a id="energie"></a>
## 6. Energy- und Kalibrierungsvertrag

### 6.1 Primärgröße und physische Grenze

Primär ist die **kalibrierte, nicht idle-subtrahierte Full-System-Eingangsenergie** am u.RECS. `FS` beschreibt die physische Messgrenze; `native_only` die Workflowstufe. Diese Begriffe sind keine Alternativen. Für den direkten DC-Gain-Abgleich beschreibt die bisherige KB den 20-mΩ-Shunt R16 zwischen `Vsense_Input` und `9V_20V_IN` mit INA225; ein 5-V-Verbraucher hinter einem Wandler ist kein direkter Ersatz für denselben Eingangsstromsprung. Die konkrete sichere Hardwarebedienung folgt dem vorhandenen Aufbau und Kalibrierungsdialog. [E-KB, §6]

### 6.2 Geführter Gain-Abgleich bleibt die Methode

Der bestehende Ablauf umfasst geordnetes Abschalten von M.2 und Jetson, Stabilisierung, `idle_before`, nominal 0,5 A mit **Ist-Strom und Ist-Spannung**, `idle_between`, nominal 1,0 A mit Ist-Werten, `idle_after`, ausdrückliche Bestätigung `0 A / Output OFF`, Wiederherstellung und anschließend Fit/Gate. Es gilt `P_ref = I_actual × V_actual`; die nominale Sollzahl ersetzt nicht die tatsächliche Last. Für die beiden Lastpunkte werden die benachbarten Idlefenster gemittelt. [E-KB, §6]

Es bleibt ein enger DC-Gain-Abgleich, kein neues höhergradiges Kalibrierungsmodell. Die ursprünglichen unskalierten Werte bleiben getrennt auditierbar, beispielsweise in `full_system_input_unscaled_avg_power_w` und `full_system_input_unscaled_energy_total_j`. Ein verifizierter Faktor wirkt vor einem gegebenenfalls zulässigen separaten Idle-Abzug. [E-KB]

### 6.3 Fortschreibung gegenüber dem alten „Kalibrierung noch offen“

**Die pauschale alte TODO „FS-Gain erst noch aufnehmen und danach M.2-Idle“ ist für den späteren Lauf überholt.** Die 55 vorhandenen Energieaggregate des v29-Complete-Sets melden bereits verifizierte und angewendete FS-Gain-Faktoren vom 4. September. Für die 21 setup-lokalen TensorRT-Full-Zeilen melden sie zusätzlich verifizierte, angewendete neuere Idlewerte. [E-CS] [E-DOC]

| Setup | FS-Gain-Faktor laut Aggregaten | Aggregate mit verifiziertem/angewendetem Faktor | Separater M.2-Idlewert bei TensorRT Full |
|---|---:|---:|---:|
| DeepX | 0,9035075097 | 17/17 | 2,4350704536 W, angewendet bei 7/7 TRT-Full-Zeilen |
| Hailo-8 | 0,9136343897 | 19/19 | 2,0079536423 W, angewendet bei 7/7 TRT-Full-Zeilen |
| Hailo-10H | 0,8963972780 | 19/19 | 0,9667883103 W, angewendet bei 7/7 TRT-Full-Zeilen |

Diese Tabelle ist eine **read-only Zusammenfassung gespeicherter Runtime-Verifikationen**, keine neue metrologische Abnahme. Die vollständigen ursprünglichen Last-/Kalibrierungsdateien sind im verkleinerten Complete-Set-Debug-Pack nicht enthalten. Die FS-Verifikation nennt vorhandene lokale Evidencepfade und übereinstimmende erwartete/tatsächliche Datei-Hashes; daraus wird keine hier neu durchgeführte Prüfung der elektronischen Last konstruiert. [E-CS] [E-DOC]

Leere allgemeine `energy_calibration_manifest`-Felder bedeuten in diesem Paket nicht automatisch, dass gar keine Gain-Kalibrierung angewendet wurde: Die spezifischen `full_system_current_scale_*`-Felder dokumentieren sie separat. Diese Unterschiede dürfen im Bericht nicht miteinander verwechselt werden. [E-CS]

### 6.4 Historische Gategrenzen nicht still mit späteren Ist-Werten versöhnen

Die KB vom 3. September nennt als damalige Gain-Gates unter anderem mindestens 2 W Leistungssprung, höchstens 2 % Unterschied der Einzelfaktoren, höchstens 0,5 W Idle-Drift, höchstens 10 % Abweichung des Ist-Stroms vom Sollpunkt und einen Faktorbereich **0,90 bis 1,10**, einschließlich sicherer Wiederherstellung. Der spätere Hailo-10H-Aggregatwert 0,8963972780 liegt knapp unter dieser damaligen Untergrenze, wird vom späteren Lauf aber als verifiziert ausgewiesen. [E-KB, §6.3] [E-CS]

**Dokumentationsgrenze:** Ohne den zugehörigen ursprünglichen Kalibrierungsbeleg und seine effektive Gateversion wird hier weder eine geänderte Grenze erfunden noch die spätere Messung pauschal verworfen. Vor einem darauf gestützten metrologischen Claim den bereits vorhandenen Beleg heranziehen. Dies ist kein Auftrag zu blindem Neukalibrieren oder nachträglicher Grenzwertlockerung. Die älteren v14-Idlewerte bleiben historische Werte und werden nicht mit den späteren Zahlen vermischt.

### 6.5 Messdauer: 165 gültige Replikate sind keine 60-Sekunden-Abnahme

Die 55 geplanten Energiezeilen des v29-Complete-Sets enthalten **durchgehend `duration_s=1.0`**, FS und `command`. Alle Aggregate melden drei gültige Replikate und eine konfigurierte Lastdauer von 1 s. Tatsächliche Command-, aktive Fenster- und Collectorzeiten sind eigene Felder und können davon abweichen. [E-CS] [E-DOC]

Damit ist die Sammlung von **165 gültigen kurzen Replikaten** belegt, nicht ein neuer Langzeit-/Stabilitätsnachweis mit 60 s × 3. Der bereits früher dokumentierte längere Energie-Abnahmeumfang bleibt ein eigener Vertrags- und Konfigurationscheck vor dem Final-Lauf; die kurze Diagnosekonfiguration darf nicht unbemerkt als dessen Ersatz gelten. Messdauer, Zahl der gültigen Wiederholungen und tatsächliche Work Units vor dem Start prüfen. [E-KB15] [E-PLAN31]

### 6.6 Methoden- und Freigabeebenen

Die betrachteten Aggregate nennen `command_marker_window` als wissenschaftliche Primärmethode und `chapter4_legacy_window` als Shadow-Methode; die Primärgröße heißt `calibrated_input_energy_unsubtracted`. Ein Shadow-/A/B-Fenster wird nicht automatisch zur besseren Primärmethode gewählt. Bereits eingefrorene Methodenentscheidungen bleiben bestehen. [E-CS]

Die 55 erfolgreichen Energiezeilen sind trotzdem `raw_energy_quality_not_qualified`. Ein gespeicherter globaler Grund ist `incomplete_expected_native_matrix`; bei einzelnen Fällen kommen lokale Quality-/Endpointprobleme hinzu. Deshalb getrennt ausweisen: **Messung erfolgt**, **Replikate vollständig**, **Zeilenqualität bestanden**, **Paarung passend**, **Kampagnenmatrix vollständig**, **wissenschaftlich freigegeben**. Keine dieser Aussagen ersetzt die anderen. [E-V30, §7] [E-PLAN31, AP9]

### 6.7 Power-Mode und Idle-Fairness

Vor dem final beanspruchten Vergleich die tatsächlich verwendeten `nvpmodel`-/Clock-/Governor-Einstellungen, thermische Bedingungen und relevante Peripherie aller Setups dokumentieren. Ein niedriger DeepX-Idlewert ist allein kein Messfehler. Neue Gain-/Idlewerte oder schnelle Smokes beweisen noch keine identischen Powerbedingungen aller späteren Energiepfade. Vorhandene gültige Kalibrierungen nutzen; nur bei geänderter Messkette, ungültiger Bindung oder konkretem Problem gezielt neu aufnehmen. [E-KB] [E-PLAN31]

### 6.8 Spätere Nachtläufe sind neue, getrennte Energiebelege

Im v2.80.1-Lauf vom 10. September liefen 21 Vendor-Full-Kombinationen mit 63 gültigen Performancewiederholungen und 63.000 gemessenen Frames. Die Energie sammelte 21/21 ausführbare Planzeilen mit 63 gültigen Replikaten; die 63er-Gesamtmatrix blieb unvollständig. Die folgende Zwei-Split-Nacht hatte eine 84er-Matrix und ebenfalls 21 erfolgreiche Vendor-Full-Fälle. Diese verschiedenen Nenner nicht vermischen. Beide kurzen Energieumfänge waren Screening, kein neuer 60-s-×-3-Nachweis. [E-WORKFLOW-HISTORY]

Der am 12. September abgebrochene .3-Nachtlauf wartete dagegen noch in der zentralen Qualityphase; seine anschließende Native-/Energiestufe war im beobachteten Abschnitt nicht gestartet. Das ist kein Verlust früherer Energieresultate. Gleiche `MAXN_SUPER`-Namen auf den Setups bewiesen außerdem keine identischen tatsächlichen Clock-/TPC-Einstellungen; die beobachtete TPC-Maskenabweichung bleibt für konkrete Setupvergleiche zu erklären, ohne aus ihr allein eine Kernanzahl oder Defektursache zu raten.

<a id="deepx-r1-r2"></a>
## 7. Abgeschlossene Qualitätsbefunde und eng begrenzte Diagnosen

### 7.1 DeepX-Preprocessing: kein weiterer Adapterprototyp

Die historische R1/R2-Reihe bestätigte den vorhandenen Mean/Std-Sub/Div-Buildadapter. Für passende Klassifikationsmodelle bleibt `imagenet_mean_std` der normale Arm; ein explizites `current_scale_only` in einem Profil übersteuert einen Modusdefault und ist weiter als Legacy-/Diagnosewahl zu erkennen. Das Zurücksetzen eines zentralen Modus korrigiert keine explizite Nutzerprofilwahl.

R1: drei Klassifikationsmodelle; Native-/Quality-Feed und Outputs auf den jeweiligen 16 Hardwarebildern gleich. R2: zwei echte private neue Mean/Std-DXNNs, vier Adapterkontrollinputs pro Modell exakt gleich; MobileNet im kleinen A/B 11/16 → 12/16 gegenüber CPU 13/16, ResNet 12/16 → 15/16 gegenüber CPU 14/16. Kleine Nenner und Logitähnlichkeit belegen keine allgemeine Accuracyverbesserung oder B5000-Freigabe. Die alte Forderung, zuerst nochmals denselben R1/R2-Adapter-Smoke zu wiederholen, entfällt. [E-R1] [E-R2]

### 7.2 MobileNet: B5000-FAILs statt pauschal „Qualität offen“

Der .4-Plan dokumentiert für die bereits abgeschlossenen zentralen 5.000-Bilder-Aufträge folgende Top-1-Ergebnisse des jeweiligen bisherigen Artefakts:

| Variante | Top-1 | Differenz zur CPU | Entscheidung |
|---|---:|---:|---|
| Originale CPU-Referenz | 73,70 % | — | Referenz |
| Hailo8 Full, auf CPU gebautes HEF | 60,62 % | −13,08 pp | FAIL |
| Hailo10H Full, auf CPU gebautes HEF | 59,42 % | −14,28 pp | FAIL |
| DeepX Mean/Std Full | 71,72 % | −1,98 pp | FAIL |

Die Werte wurden zunächst aus Plan/Q5 fortgeführt. Im jetzigen REV2-Abgleich liegt der ursprüngliche zentrale Qualitätssnapshot im `har_and_git_evidence.zip` vor: Ergebniszustände, Identitäten und gespeicherte Aggregate wurden direkt gelesen. Keine neue Berechnung aus Rohvorhersagen und keine nachträgliche Erweiterung des damaligen .4-Build-Prüfumfangs. [E-NIGHT-QUALITY] Die früheren „MobileNet erst noch erstmals größer bewerten“-TODOs sind mit diesen benannten abgeschlossenen Befunden überholt. Die genaue interne Verlustursache und die B5000-Qualität eines anderen privaten GPU-HEFs sind andere Fragen. [E-2804-PLAN, §8.2]

Frühe MobileNet-Splits b027/b056 sind laut Plan deutlich besser als Full/b135, aber nicht automatisch innerhalb der Marge. Keine nachträgliche Beschränkung der Finalmatrix auf die besseren Grenzen. Ein Stufenvergleich benötigt die passende Float-P1-Referenz für genau dieselbe Boundary.

### 7.3 Hailo8 Fixed16: Runtime-PASS mit unterschiedlichen Accuracyzahlen

| Variante auf denselben 16 IDs | Top-1 | Top-5 |
|---|---:|---:|
| Original-ONNX | 13/16 | 14/16 |
| Compiler-ONNX | 13/16 | 14/16 |
| CPU-HEF, ausgeführt auf Hailo8 | 8/16 | 14/16 |
| Privates GPU-HEF, ausgeführt auf Hailo8 | 10/16 | 15/16 |

Die Q2-Terminalquelle meldet G3-PASS, sechs geänderte Top-1-Klassen mit zwei zusätzlichen richtigen Treffern und sauberen Abschluss. Original-/Compiler-ONNX sind auf der Probe numerisch identisch. Gleichheit betrifft die tatsächlich erfassten FLOAT32-VStreaminputs; interne UINT8-Puffer wurden nicht beobachtet. Input-QuantInfo: `qp_scale=0.01872340589761734`, `qp_zp=114`. Der Plan benennt 32 erfolgreiche Inferenzen und einen unabhängigen Labelvergleich aus Q1. Nur soweit Q1 tatsächlich verfügbar ist, wird dessen Recount in der .4-Verifikation unabhängig bestätigt. [E-2804-H8-Q2] [E-2804-PLAN, §1.3]

Keine Behauptung „GPU verbessert generell Accuracy“ und kein automatischer Austausch des produktiven CPU-HEFs. Der private Build behält seine eigene Artefakt- und Qualitätsidentität. Die laut Plan etwa 520,2 s CPU-Bauzeit und 528,8 s GPU-Bauzeit belegen keinen Speedup. Ein Component-View mit ptxas/libdevice ist kein vollständiges CUDA-Toolkit.

**Fortschreibung:** Die erste Emulation der vorhandenen Parsed-/Quantized-HARs ist ausgeführt; siehe §7.7. Das frühere `not_available` im Fixed16-Runtimebericht bleibt als damaliger Zustand unverändert. Die zusätzliche optimierte Floatstufe und die kontrollierte Integer-I/O-Ursachenprüfung wurden nicht ausgeführt und sind jetzt ausdrücklich als optionale Folgearbeit zurückgestellt (§7.8). Es folgt daraus weder eine neue GPU-/G3-Releasepflicht noch eine Voraussetzung zum Berichten der vorhandenen Negativergebnisse. [E-SCOPE-REV3]

### 7.4 Hailo10 Fixed16 separat halten

Der frühere v2.79.34-MobileNet-Test auf Hailo10 lieferte für CPU- und GPU-HEF jeweils 12/16 Top-1 und 15/16 Top-5 gegenüber ONNX 13/16 und 14/16. Beide HEFs wurden technisch ausgeführt; der echte Hailo10-GPU-Build war zuvor erfolgreich. Diese Zahlen gehören nicht zum neuen Hailo8-Test und werden nicht vermischt. [E-2804-CHAT]

### 7.5 YOLO11l/Hailo10H: 5.000-Bilder-Untersuchung abgeschlossen

| Variante | AP50:95, Skala 0–100 | Differenz zur CPU | Zentraler Entscheid |
|---|---:|---:|---|
| CPU-Referenz | 46,4781 | — | Referenz |
| TensorRT Full | 46,4609 | −0,0172 pp | PASS |
| Hailo10H Full | 45,0829 | −1,3953 pp | FAIL |
| Hailo10H → TensorRT, b062 | 45,3656 | −1,1125 pp | FAIL |

Vollständige CPU-/Kandidatenvorhersagen umfassen dieselben 5.000 Bilder; Fingerprints und AP-Reproduktion wurden geprüft. Der Normalworkflow lief technisch durch, verwendete vier passende vorhandene Artefakte ohne Cold Build und schloss Native-/Remote-Prozesse ab. Der Hailo-Punktentscheid unterschreitet bereits die unveränderte 1-pp-Marge; 0 tatsächlich ausgeführte Bootstraps im dokumentierten frühen FAIL-Pfad sind kein fehlender Inferenzlauf und kein berechnetes Konfidenzintervall. TensorRT wurde mit dem vorgesehenen Statistikpfad bewertet. [E-2804-QUALITY]

Die AP-Verluste betreffen viele Klassen; ein allgemeiner Decoderfix ist aus diesen Daten nicht nachgewiesen. Der historische 500-Bilder-Lauf zeigte für dasselbe Full-HEF bereits ähnliche Verluste: Full −1,3337 pp, Split −1,1486 pp. Der neue Umfang bestätigt den Befund, er repariert ihn nicht.

Der zentrale Qualitätsvertrag ist abgeschlossen ausgewertet. Der ursprünglich nicht verfügbare zusätzliche offizielle COCOeval-Bericht bleibt ausdrücklich getrennt; eine eventuelle Ergänzung verwendet vorhandene Vorhersagen und Originalannotation ohne neuen Hailo-Build oder Hardwarelauf. Eine allgemeine wissenschaftliche Freigabe sämtlicher Modelle/Backends folgt daraus nicht.

### 7.6 Historischer Opt2-Versuch und verbindliche Entscheidung

Die Results-KB vom 3. September dokumentiert für YOLOv7/Hailo8 AP50:95 41,478 bei Opt1 und 40,984 bei Opt2; Opt2 war rund 0,494 pp schlechter. Der Opt1-Wert ist zusätzlich im archivierten Paper-Primärergebnis enthalten; ein vollständiger zugehöriger Opt2-Rohbericht wurde im bereitgestellten Archiv nicht gefunden. Der Befund ist daher kein neuer direkter A/B-Nachweis für YOLO11l/Hailo10H.

Die Projektentscheidung bleibt dennoch eindeutig: balanced/Opt1/B500/Batch8 beibehalten, kein pauschaler Optimierungs- oder Kalibrationsgrößen-Sweep auf den Evaluationsmodellen. Der zwischenzeitlich vorgeschlagene Opt2/B1024-Neubau und automatische Opt3-Folgeversuch wurden nach dem historischen Abgleich verworfen. Das ist eine abgeschlossene Entscheidung, kein noch auszuführender Test. [E-2804-ABGLEICH]

### 7.7 Hailo8 HAR-R1: erste große Verluststelle lokalisiert

Primärquelle: `har_and_git_evidence.zip`, Run `hailo8_har_git_20260912T171411Z_x0j2t2o1`. Die beiden HAR-Stufen liefen mit DFC 3.33.1 auf der CPU, dieselben bereits geöffneten 16 Bilder und tatsächlich aufgezeichneten FLOAT32-NHWC-Feeds. Neue HEF-Builds, Optimierung, Hardwareinferenzen, Energie und Bootstrap: **nicht ausgeführt**. Parsed-Emulation dauerte laut Prozessbericht 13,75 s, Quantized-Emulation 31,30 s; daraus folgt keine Leistungskennzahl für Hardware. [E-HAR-R1]

| Stufe | Top-1 | Top-5 |
|---|---:|---:|
| Original-ONNX | 13/16 | 14/16 |
| Compiler-ONNX | 13/16 | 14/16 |
| Parsed-HAR / `SDK_NATIVE` | 13/16 | 14/16 |
| Quantized-HAR / `SDK_QUANTIZED` | 9/16 | 13/16 |
| Privates GPU-Build-HEF auf Hailo8 | 10/16 | 15/16 |
| Historisches CPU-Build-HEF auf Hailo8 | 8/16 | 14/16 |

| Vergleich | Max. absoluter Logitfehler | RMS | Mittlere Cosine | Top-1-Wechsel |
|---|---:|---:|---:|---:|
| Original → Compiler-ONNX | 0 | 0 | 1,0 | 0/16 |
| Compiler-ONNX → Parsed-HAR | 0,000009775 | 0,000001469 | ≈1,0 | 0/16 |
| Parsed-HAR → Quantized-HAR | 4,919190 | 0,701965 | 0,721879 | 5/16 |
| Quantized-HAR → GPU-HEF | 2,394485 | 0,392645 | 0,904432 | 4/16 |

Die Klassenlisten wurden erneut unabhängig nachgezählt. Logitstatistiken bleiben Resultate des ausgeführten Collectors, weil Roharrays nicht im ZIP liegen. Parsed und Compiler-ONNX besitzen auf allen 16 Bildern dieselben geordneten Top-5. Parsed → Quantized verliert vier zuvor richtige Entscheidungen, ein weiterer Wechsel bleibt falsch. Quantized → Hardware korrigiert zwei Antworten, verschlechtert eine und wechselt eine andere falsche Klasse. Ähnliche Trefferzahlen beweisen folglich keine gleiche Ausgabe. [E-KB-REV2-CHECK]

**Befund:** Die wesentliche Abweichung ist bereits im DFC-Übergang zur optimierten/quantisierten Darstellung vorhanden. Tool-ONNX-Anpassung und Parsing erklären sie in dieser Probe nicht. **Keine stärkere Behauptung:** Der Übergang umfasst mehr als reine Rundung; eine separate optimierte Floatdarstellung fehlt noch. Der API-Snapshot zeigt 101 Parsed- gegenüber 89 Quantized-HN-Layern. Das kann Transformations-/Fusionsfolgen widerspiegeln und ist allein kein Defektbeweis.

**Emulation → Hardware bleibt relevant abweichend.** Keine pauschale Bitgenauigkeitsannahme und keine Behauptung, dieser Rest sei bereits vollständig als Rundung erklärt. Bisher beobachtet: identische FLOAT32-Hostfeeds und QuantInfo, nicht die internen UINT8-Eingänge. HailoRT 4.20.0: Inputscale 0,01872340589761734 / ZP 114; GPU-HEF-Outputscale 0,06651347130537033 / ZP 75. CPU-HEF-Outputscale nicht stellvertretend verwenden. [E-H8-RUNTIME]

HARs sind laut ursprünglichen Buildpfaden dem privaten GPU-Build zugeordnet; ihre Hashes wurden erst bei dieser Emulation aufgenommen. Das bestätigt verwendete Dateien und damalige Unverändertheit, aber keine rückwirkende Byteattestation zum Buildzeitpunkt. Für den älteren CPU-Build liegt kein zugehöriges Quantized-HAR vor. Kein Übertragen der neuen HAR-Ursachenlokalisierung als vollständige Erklärung seines 5.000-Bilder-Verlusts.

### 7.8 Weitergehende Ursachenklärung – zurückgestellte optionale Folgearbeit

**Status seit REV3: ZURÜCKGESTELLT, nicht beauftragt, nicht ausgeführt.** Die technische Hailo8-Vorabprüfung und die erste HAR-Fallstudie sind abgeschlossen. Die vorhandenen Ergebnisse reichen für die begrenzte Aussage über das automatische Verfahren unter dem festen Rezept; eine lückenlose interne Ursachenerklärung wird für diese Hauptaussage nicht verlangt. Die zusätzliche Emulations-/Hardwareabweichung bleibt ausdrücklich unerklärt. Das ist weder ein pauschaler Runtime-Fehlerfreiheitsnachweis noch ein Beleg für ausschließliche Quantisierungsursächlichkeit. [E-SCOPE-REV3]

Der frühere Begleitplan `DIAGNOSEPLAN_Hailo8_Optimierung_Quantisierung_Runtime_2026-09-12.md` bleibt unverändert als optionale technische Skizze erhalten. Er ist **kein nächster Pflichtschritt**, kein .4-Releasegate und kein Abschlussgate der derzeitigen Hauptauswertung. Erst ein neuer ausdrücklicher Auftrag zur stärkeren Ursachenfrage würde folgende Schritte reaktivieren:

| Schritt | Unveränderte Artefakte / konkrete Kontrolle | Aussageziel |
|---|---|---|
| A | Vorhandenen Quantized-HAR in einer optimierten Floatdarstellung auswerten (`SDK_FP_OPTIMIZED`, Verfügbarkeit und Zustand in DFC 3.33.1 vorher prüfen) | Floatoptimierung von anschließender Quantisierungsverarbeitung trennen |
| B | Dasselbe GPU-HEF und dieselben 16 Arrays über FLOAT32- und explizit quantisierte UINT8-Ein-/Ausgaben vergleichen | Hostquantisierung, Rundung/Clipping, Layout und Outputdequantisierung isolieren |
| C, nur bei weiterem Rest | Zugängliche korrespondierende Emulator-Zwischenoutputs und QuantInfo untersuchen; Herstellerfall bei nicht zugänglichen inneren Hardwarewerten | Erste verbleibende materielle Abweichung eingrenzen |

Bei einer später ausdrücklich beauftragten Wiederaufnahme wäre die erste 2×2-I/O-Kontrolle auf höchstens 64 neue kurze Inferenzen begrenzt; sie ist jetzt nicht auszuführen und keine neue G3-/Compute-/Buildschleife. **Die tatsächliche HailoRT-Rundungsregel prüfen**, nicht `numpy.round` ungeprüft verwenden. Keine doppelte Normalisierung oder Quantisierung; Emulatorkontext muss seine Eingangsdomäne explizit erklären. Ohne zugängliche passende optimierte Floatrepräsentation bleibt diese Stufe `not_available` – keine automatische Neuoptimierung.

Ein Stufendump des Softwareemulators ist nicht automatisch ein Hardware-Layerdump des unveränderten HEFs. Zusätzliche Outputendpunkte durch Neucompilierung könnten das Mapping verändern und wären ein gesonderter Eingriff. Ein solcher Build ist nicht Bestandteil des aktuellen Vorschlags.

**Endregel auch für eine spätere Wiederaufnahme:** Ein auf festen Inputs und einem konkreten Artefakt unauffälliger Runtimevergleich erlaubt eine auf diesen Scope begrenzte Aussage, keine universelle Gleichheit. Bleibt der Rest unerklärt, wird er dokumentiert; kein Nachoptimieren bis PASS. Numerische RMS-/Cosinewerte liefern keine additiven Prozentanteile des Accuracyverlusts. Diese Diagnose blockiert weder die kleinen .4-Softwarefixes noch die Dokumentation der vorhandenen korrekt ausgewerteten negativen Ergebnisse. [E-DIAGNOSE-PLAN] [E-HAILO-OPT-DOC] [E-SCOPE-REV3]

### 7.9 Abgeschlossene Qualitypopulation des abgebrochenen Nachtlaufs

Im originalen zentralen Report: 123 Requests, 63 vollständig ausgewertet (**43 PASS, 18 FAIL, 2 INCONCLUSIVE**) und 60 ohne fertigen Qualitätsvergleich (**4 CancelledError, 56 QualityServiceClosedError**). Alle sieben CPU-Referenzprozesse waren erfolgreich mit je 5.000 Einträgen. ResNet hatte 19 abgeschlossene PASS-Aufträge; YOLO11l in dieser Nacht 10 PASS, 6 FAIL, 2 INCONCLUSIVE. Der separate spätere YOLO11l/H10-Umfang aus §7.5 bleibt seine eigene Quelle. [E-NIGHT-QUALITY]

MobileNet-H8 b027: 71,58 %; H10 b027: 72,22 % gegenüber CPU 73,70 %, jeweils außerhalb der unveränderten 1-pp-Marge. Frühere Generic-5.000-Diagnostik b056: H8 71,38 %, H10 71,42 %; nicht rückwirkend als im gecancelten Run zentral abgeschlossene b056-Aufträge deklarieren. Full/b135 und frühe Splitpunkte dürfen nicht zu einem einzigen „Hailo-Splitwert“ vermischt werden.

Ein früher Punktwert-FAIL mit vollständigen Vorhersagen und ausgelassenem Bootstrap ist im bestehenden Vertrag fertig ausgewertet; das CI bleibt `null`. Statistischer PASS, Point-FAIL und INCONCLUSIVE sind getrennt. Der Cancel ändert die 18 fertigen FAILs nicht und beweist keine späteren Ergebnisse für die übrigen Modelle.

### 7.10 Aufwand der zurückgestellten Vertiefung – keine neue Terminplanung

Die folgende grobe Einschätzung stammt aus der Scope-Diskussion, nicht aus gemessenen Laufzeiten. Sie nimmt eine mit dem Projekt vertraute Person, vorhandene Artefakte und erreichbare lokale Umgebungen an. Arbeitsanteile überlappen; sie sind nicht exakt addierbar. Unzugängliche Internas oder notwendige Herstellerunterstützung können den Aufwand wesentlich erhöhen. [E-SCOPE-REV3]

| Optionaler Zusatzumfang | Grobe Arbeitsaufwandsschätzung, kein Auftrag / keine Zusage |
|---|---|
| Vorhandene optimierte Floatstufe prüfen und vergleichen | Einige Stunden bis etwa ein Arbeitstag, sofern die Darstellung zugänglich ist |
| Float-/Integer-I/O-Gegenprobe mit exakter Quantisierung, Tests und Auswertung | Etwa ein bis drei Arbeitstage |
| Verbleibende interne Abweichung layerweise oder mit Herstellerhilfe erklären | Mehrere Tage bis Wochen; bei fehlenden Internas nicht verlässlich begrenzbar |
| Modellspezifische Tuningstudie über mehrere Netze und unabhängige Bestätigung | Eigenes Experiment über mehrere Tage bis Wochen, ohne Erfolgsgarantie |

Kurze Einzelinferenzen bedeuten nicht automatisch geringen Gesamtaufwand: Diagnoseimplementierung, versionsgebundene API-/Numerikprüfung, Absicherung und Interpretation sind zusätzliche Arbeit. Für die aktuelle Hauptfrage wird dieser Aufwand **nicht vor den Abschluss der regulären Auswertung gestellt**. Die Aufwandsschätzung ist keine Begründung, bekannte eigene Fehler zu ignorieren; sie erklärt die Entscheidung gegen eine weitergehende unbestellte Ursachen- und Optimierungsstudie.

### 7.11 Nachtlauf und kombinierter Smoke vom 13. September 2026

Der .4-Nachtlauf `completsetdev_20260912_213922` enthält 126 geplante Native-Zeilen: 113 erfolgreich, 13 fehlgeschlagen/blockiert, keine fehlende Zeile; sämtliche 42 Full-Zeilen erfolgreich. 113 zugelassene Energieaufträge mit 339 Wiederholungen sind technisch vollständig, bleiben aber an ihren tatsächlich ausgeführten Screening-/Dauervertrag gebunden. Zentral sind 149 Vergleiche ausgewertet: 81 PASS, 45 FAIL, 23 INCONCLUSIVE; 12 weitere geplante Qualitätsfälle nicht verfügbar. Keine nachträgliche Freigabe des gesamten Nachtlaufs. [E-281-NIGHT]

Der kombinierte Smoke ist abgeschlossen: vier Stufen, keine Modellbuilds, keine Energiemessung, 77 kontrollierte Bestandsdateien unverändert, Prozessbereinigung erfolgreich. Das ZIP enthält 143 Mitglieder einschließlich 142 vollständig abgeglichener Manifestdateien. `COMPLETE` bezeichnet die erfolgreiche Diagnoseausführung, nicht alle Modelle oder jede Qualitätsentscheidung als PASS. [E-281-SMOKE]

| Fall | Belegter Befund | Konsequenz in v2.81 |
|---|---|---|
| H8-Compiler | Produktives Profil hat GPU, aber kein ausgewähltes Zusatzmanifest. Ursprünglicher Kontext meldet fehlende Komponenten. Derselbe vorhandene vollständige Overlaybestand besteht mit expliziter Auswahl SDK-Import und vier GPU-/XLA-Rechnungen, realer ptxas-Aufruf `sm_61`. | Auswahl bis in effektive Modi/Snapshots sichern; nur tatsächliche noch anstehende Kaltjobs prüfen. Kein Treiber-/SDK-Neuaufbau und kein GPUtest für Cachetreffer. |
| H8 YOLO11l b067 | Dasselbe JPEG; OpenCV- und Pillow-Aufbereitung ergeben unterschiedliche Pixel. Exakt gleicher roher RGB-Puffer ergibt hingegen 705.600 bytegleiche Float32-Ausgabewerte. | RGB einmal vorbereiten und beiden Native-Pfaden exakt übergeben; Messumfang ausdrücklich beschriften. |
| H10 YOLO26m b398 / YOLO26s b364 | Physisches NWC wird nach Entfernen der Batchachse irrtümlich nach NCW umgeformt. Die rohe Native-Gegenprobe mit gleichem B ist korrekt; P2-ORT und Bridge-ORT identisch, TRT-Abweichung höchstens 0,01838684. | Explizite metadata-gebundene Transposition vor dem bisherigen Reshapefallback; existierende Bridge und Artefakte wiederverwenden. |
| Dieselben H10-Fälle, Qualität | Alle 672.000 Klassenwerte pro Bild liegen bereits am jeweiligen Nullpunkt 39/35; nach Dequantisierung null, über sechs Bilder 4.032.000 Nullwerte. | Eigenen Layoutfehler beheben, belegten Qualitätsverlust trotzdem erhalten. Keine Garantiezusage besserer AP und kein Quantisierungstuning. |
| Sechs Compile-Rejects | H8 m b398/b399 und s b364/b365; H10 m b399 und s b365: zugehörige Primärlogs, Record-/Key-/Graphbindungen bestätigt, Compilephase rc3 ohne Timeout. | Explizite Nichtrealisierbarkeit unter festem Rezept, keine Neuversuche desselben exakten Falls; weder erfolgreiche Inferenz noch offene unbekannte Arbeit. |
| 59 Quality-Identitätsmeldungen | 28 DeepX-Split- und 24 H10-Run-Aliasse sowie sieben DeepX-Full-Digestdarstellungen sind äquivalent. | Nur erwiesene äquivalente Darstellung normalisieren; echte Widersprüche weiter ablehnen. |

Vier H8-Jobs wurden in der Nacht wegen des fehlenden Kontextes noch nicht kompiliert: RegNet b073, YOLO11l b064, YOLO26m b040 und YOLO26s b023. Der vorhandene Bestand wird erneut regulär aufgelöst; nur weiterhin benötigte MISS-Jobs sind Buildarbeit. Deren künftiger Compileausgang ist durch den synthetischen Smoke nicht vorweggenommen.

<a id="abschluss"></a>
## 8. Cache, Prozessgrenzen und terminaler Abschluss

Eine GPU-Buildpräferenz, GPU-UUID oder ein Overlaypfad sind Buildprovenienz, keine zusätzliche Modellcache-Identität. Der normale Builder prüft erst Modell-/Recipe-/Artefaktvertrag, dann positiven Cache bzw. genaue negative Compileevidenz. Nur ein tatsächlich erforderlicher Neubau benötigt den GPU-/Overlaykontext. Ein inzwischen fehlendes Overlay darf einen gültigen CPU-HEF-HIT nicht verhindern.

HEF, Receipt und Cachemetadata behalten ihre bestehenden atomaren Generationen-/Publikationsregeln; alte HEF-only-Bestände bleiben gegebenenfalls `legacy_unsealed`. Duplicate-/Generationauswahl bleibt deterministisch. Kein „neueste Datei gewinnen lassen“, kein Auswählen nach besserer Accuracy und kein automatisches Publizieren privater GPU-HEFs. Ein anderes tatsächlich verwendetes Artefakt braucht korrekt gebundene Runtime-/Qualityrequests. Identische Vorhersagen dürfen vorhandene mathematische Statistik wiederverwenden; fremde Hardwareprovenienz darf dabei nicht erfunden werden.

Bekannte `PARSER_UNSUPPORTED`-/`COMPILE_INFEASIBLE`-Fälle werden bei gleichem Vertrag nicht jede Nacht neu gebaut. Fehlender GPU-Kontext ist Infrastruktur und erzeugt keine dauerhafte negative Modellevidenz. `ABORTED_UNKNOWN` bleibt unvollständige Evidenz. Geänderte Splitauswahl muss alte/aktuelle Boundaries und erwartete Cold Builds erklären; ein neuer tatsächlich benötigter Split ist kein pauschaler Cacheverlust.

Der frühere globale Hashcache-Nachlauf ist als behobener und fortgeltend regressionspflichtiger Produktpfad dokumentiert. Keine zusätzliche Hash-/Seal-/Signatur-/Registryebene, kein neues Vollverzeichnis-Hashen und keine Metadatenarbeit in Native-Performancefenstern. Fehler beim Schreiben von Ergebnis oder Debug-ZIP dürfen kein finales Teilarchiv mit altem PASS publizieren. Primärfehler und Cleanupfehler bleiben getrennt.

Workflowlocks schützen aktive Prozesse und bleiben erhalten. Regulärer Cancel räumt eigene Worker/Supervisoren geordnet auf. Keine alten Chat-PIDs killen, keine Lockdateien als Cachefix löschen. Ein abgeschlossener Build ist noch kein Nachweis von Runtime, Qualität oder Energie.

### 8.1 Belegte Wiederverwendung und ehrlich begründete Neubauten

| Beobachtung | Belegter Scope |
|---|---|
| v2.80-Reusestarter zweimal je zwei frische Prozesse | Vier lokale exakte MobileNet-H10-CPU-HEF-HITs trotz GPUpräferenz; kein SDK-/Compilerdispatch, kein OS-Reboot-/Ganzworkflowclaim |
| BiggerSet v2.80 vom 10.09. 09:42 | 22 Hailo-HITs; 48 unterschiedliche TRT-Enginepfade wiederverwendet; sechs sinnvolle neue Mean/Std-DXNNs für Klassifikations-Full/Part1 |
| CompletSetDev v2.80.1 vom 10.09. 13:30 | 26 Hailo-HITs, 56 unterschiedliche TRT-Pfade; 13 DXNNs wiederverwendet, YOLOv7-Paper b044 neu nach `model_missing` |
| Zwei-Split-Nacht ab 10.09. 21:08 | 38 Hailo-HITs, ein H10-YOLO11-b064-Neubau, ein H8-Kontextfehler; 86 unterschiedliche TRT-HIT-Pfade, fünf TRT-Neubauten; sieben neue DeepX-P1-Boundaries |
| Drei-Split-Nacht ab 11.09. 21:35 | 49 Hailo-HITs ohne Compiler; 21 DXNNs wiederverwendet, sieben zusätzliche P1-Builds; 114 unterschiedliche TRT-HIT-Pfade und zwölf protokollierte Neubauten |

Zahlen gelten für ihren jeweiligen Log-/Snapshotumfang, nicht als zeitloses Cacheinventar. Wiederholte HIT-Logzeilen desselben Enginepfads wurden nicht zu neuen Artefakten addiert. Ein `MISS`, ein gestarteter Build und eine erfolgreich gespeicherte Generation sind verschiedene Zustände. Historische Receiptberichte können keinen heutigen Dateibestand garantieren. [E-REUSE-TARGET] [E-WORKFLOW-HISTORY] [E-NIGHT-2803]

In den hier historisch aufgeführten Nächten blieb H8/YOLO11 b064 wegen des damals noch nicht in den normalen Kontext eingebundenen Overlays blockiert; der nachfolgende .4-Nachtlauf und seine konkrete Korrektur stehen in §7.11. Hinzu kamen exakte negative späte YOLO26-Boundaries. Ein später bestandener privater H8-Smoke erzeugt weder dieses fehlende P1-HEF noch automatisch seine normale Profilbindung. Vor erneutem Kaltbuild den aktuellen tatsächlichen Bestand prüfen.

### 8.2 Modellcache ist nicht Qualitycache

Änderung nur des Buildgeräts oder Overlaypfads: kein neuer Modellvertrag. Wechsel von Validierung500/Bootstrap500 zu 5.000/5.000: vorhandenes HEF weiter nutzbar, aber nicht derselbe statistische Qualityauftrag. Wechsel DeepX `current_scale_only` → `imagenet_mean_std`: echter Build-/Numerikvertrag, alte DXNNs kein zulässiger HIT. Ein Quality-FAIL bleibt ein technisch verwendbares Artefakt mit eingeschränkter Qualitäts-/Claim-Eignung.

Im H8-CPU/GPU-Vergleich sind Cachepayload/Recipe gleich, die HEFs aber verschieden. Jede Runtime-/Qualityaussage bleibt an das tatsächlich ausgeführte HEF bzw. seine gebundenen Vorhersagen gekoppelt. Privates GPU-HEF nicht nach sichtbarer Fixed16-Accuracy automatisch bevorzugen oder über das bestehende Produktiv-HEF schreiben.

### 8.3 Finalisierung und Restzeit

Die alte JSON-Cache-Neuschreibschleife war anhand hoher Schreiblast und ständig wachsendem Cache belegt und wurde gezielt korrigiert. Spätere längere Finalisierungen bei 229.149 indizierten Pfaden sind nicht ohne gleiche Symptome erneut derselbe Bug. Nur quell-/prozessgestützte Diagnose, keine pauschale Cachelöschung.

In der .3-Nacht waren die langsamen Qualityabschlüsse ungefähr 22–23 Minuten pro Detectionvergleich; diese beobachtete Rate enthielt die vier Worker bereits. Sie wurde nicht als verlässliche Gesamt-ETA bestätigt: schnelle spätere Abbruchterminals dürfen die Rechnung nicht rückwirkend als normale Rechenabschlüsse verbessern. Unterschiedliche Taskarten, Cache-/Identitätskurzpfade und frühe Point-FAILs getrennt modellieren. [E-NIGHT-2803] [E-NIGHT-QUALITY]

<a id="complete-set"></a>
## 9. Historisches Complete Set v2.79.29: Ergebnis- und Fehleranker

Historischer Run vom 7. September. Die folgenden damaligen Fehlerlokalisierungen bleiben für Regression und Provenienz erhalten. Wörter wie „offen“ in dieser historischen Darstellung beschreiben den v29/v30-Stand; aktuelle Aufgaben stehen ausschließlich in Abschnitt 12. Insbesondere die hier genannte normale GPU-/Profilintegration wurde danach weiterbearbeitet und ist kein Auftrag zur Rückkehr auf v31.

### 9.1 Referenzlauf und Zählebenen

Run: **`complete_set_20260907_161614`**, tatsächlich **v2.79.29**. Es sind nicht 13 unabhängige Native-Geräteausfälle. [E-V30, §4]

| Ebene | Tatsächlich dokumentiert |
|---|---:|
| Native-Ergebnisobjekte | 63 |
| Technisch erfolgreich | 55 |
| Nicht erfolgreich | 8 |
| Alte Erwartungszuordnung | 55 erfolgreich + 3 fehlgeschlagen + 5 fehlend |
| Alter Kompaktbericht | 68 Zeilen, weil fünf vorhandene Fehlerobjekte zusätzlich als fehlend erscheinen |
| Energie | 55 erfolgreiche Zeilen × 3 gültige kurze Replikate = 165 |
| Zentrale Qualitätsaufträge | 70, davon 69 abgeschlossen |
| Quality-Entscheidungen | 40 `pass`, 22 `fail`, 7 `inconclusive`, 1 `not_evaluated` |
| Setup-lokale `native_full_tensorrt`-Qualityaufträge | Alle 21 `pass` |

Nach read-only Ergänzung **eindeutiger fehlender Planidentität**, ohne Änderung von Messwerten oder Status, ergibt der Gegenversuch 63 vorhanden, 55 erfolgreich, acht erfolglos, null zusätzliche Missing-Zeilen. `matrix_complete` bleibt falsch. Diese Zahlen sind die v31-Replay-Erwartung, nicht das vorweggenommene Ergebnis eines neuen Hardwarelaufs. [E-V30-N] [E-PLAN31, AP1]

### 9.2 Verteilung der acht erfolglosen Native-Fälle

| Anzahl | Fallgruppe | Status gegenüber v30/v31 |
|---:|---|---|
| 1 | DeepX Full YOLO11l | Bekannte Semantik-/Quality-Integrationslücke; v30 adressiert sie. |
| 3 | DeepX-Splits YOLO11l `b062`, YOLO26m `b398`, YOLO26s `b364` | Falscher Compilerkontext verhindert benötigte Part1-Builds; v31 AP5/AP2. |
| 2 | Hailo-10H-Splits YOLO26m/YOLO26s | Frühere ungültige Ausgaben und fehlende Quality-Bindung; v31 AP6/AP2. |
| 2 | Hailo-8-Splits YOLO26m `b398`, YOLO26s `b364` | Exakt bekannte Unrealisierbarkeit; als erklärter Planfall behandeln, nicht blind neu bauen. |

Die zusätzlichen Hailo-8-Manifest-/Energieprobleme betreffen dagegen zwei **laufzeitseitig erfolgreiche** Detectionfälle. Sie sind nicht bereits durch die Beseitigung der obigen acht Fehler erledigt. [E-V30, §§5–7]

### 9.3 CS1 – DeepX-Compilerumgebung

Der fehlgeschlagene normale Build verwendet PyTorch `2.12.0+cu130` mit Architekturen ab `sm_75`, während die GTX 1080 Ti `sm_61` benötigt. Der dokumentierte Primärfehler ist `deepx_compiler_cuda_architecture_unsupported`, nicht bloß eine später fehlende DXNN-Datei. R2 bestätigt die funktionsfähige compilerlokale **cu126-Overlaystrategie** mit echten Builds, setzt sie aber explizit. Ihre automatische Wahl im normalen GUI-/Workflowstart bleibt offen. [E-V30, CS1] [E-R2]

AP5 muss den geeigneten Childkontext eindeutig auswählen und protokollieren, ohne globales GUI-`PYTHONPATH`, Runtime-Venvs oder andere Vendorprozesse zu verändern. Ein vollständiger gültiger Cache-Hit darf keine unnötige Compilerprüfung/-installation erzwingen. [E-PLAN31, AP5]

### 9.4 CS2/CS3 – Hailo-8-Manifest und Three-Stage-Energie

Betroffen: **YOLO11l `b062` und YOLOv7 `b044`**. Die benötigten Manifeste existieren unter `endpoints/completed_task/native_fifo_outputs/native_fifo_output_manifest.json`; ihre Bytes stimmen laut unabhängigem Replay mit Command und Consumer überein. Der Leser sucht aber den alten absoluten Remote-Pfad beziehungsweise nur flache Fallbacks. Es ist zunächst ein **Auflösungsfehler**, kein nachgewiesener beschädigter Hash. [E-V30, CS2]

AP3 löst nur zur Job-/Setup-/Case-/Precision-/Endpointrolle passende Collectionpfade auf und prüft danach Manifest und Payloads. Keine beliebige rekursive Suche, keine Auswahl nach jüngstem Datum, keine Vertragsumsiegelung. Bei reduzierten Fixtures ohne Payload nur den tatsächlich erreichbaren Manifestnachweis melden. [E-PLAN31, AP3]

Unabhängig davon kennt der Energieprüfer den Modus `native_three_stage_fast_oracle_outside_timing` noch nicht. AP4 muss den **bestehenden vollständigen Fast-/Oracle-Verifier** integrieren. Eine zusätzliche erlaubte Zeichenkette genügt nicht. Falscher Count, unpassender Messendpunkt, falsche Eingabe oder fehlende Oraclebindung bleiben blockiert. Die vorhandene Angabe `three_stage_concurrency_directly_measured=false` bleibt erhalten: Eine direkt gemessene Three-Stage-Überlappung ist für diese historischen Fälle damit nicht belegt. Eine Berichtsreparatur darf das Feld nicht auf `true` setzen. [E-V30, CS3] [E-PLAN31, AP4]

### 9.5 CS4 – Hailo-10H/YOLO26: kein kleiner Rundungsrest

| Ursprünglicher Befund | YOLO26m | YOLO26s |
|---|---:|---:|
| Maximaler absoluter Schnittstellenfehler | 622,1212 | 635,1461 |
| Mittlerer absoluter Fehler | 54,8500 | 51,7847 |
| Gültige Wahrscheinlichkeits-Scores im BN6-Output | 0 % | 0 % |
| Geordnete xyxy-Boxen | 73 % | 75 % |

Der Vorlauf hatte jeweils eine explizite TensorRT-Part2-Engine als Cache-Hit. Wegen `score_column_not_probability_like;coordinates_not_ordered_xyxy` wurde der Qualityexport aber korrekt geschlossen abgelehnt. Anschließend fehlt die entsprechende Binding; ein konventioneller Native-Lookup meldet dann `missing_native_trt_part2_engine`. Die spätere Meldung beweist **keine Löschung der vorher vorhandenen Engine**. [E-V30, CS4]

Offen bleibt die konkrete Layout-/Quantisierungs-/Bridge-Ursache. AP6 soll vorhandene Artefakte an HEF-Ausgang und vor/nach der Part2-Bridge erfassen, mit Namen, Shape, Datentyp, Scale/Zero-point sowie getrennten Box-/Scorekanälen. Keine blinde Clip-/Sigmoid-Korrektur, keine zweite Parallelitätsebene und kein vorsorglicher Engine-Neubau. Eine implementierte Diagnose ist noch keine Reparatur ihres Untersuchungsgegenstands. [E-PLAN31, AP6]

### 9.6 CS5/CS6 – Identität, Ausschlüsse und bekannte Unrealisierbarkeit

Drei DeepX-Splitfehler verlieren `setup_id` und `comparison_backend`, zwei Hailo10-Fehler `comparison_backend`. Dadurch entstehen die fünf Doppelplatzhalter und ungültige Energie-Ausschlüsse. AP1 hält die geplante Identität **vor** der Ausführung fest und bewahrt sie auch bei frühen Fehlern. Widersprüchliche oder mehrdeutige Zuordnungen werden nicht geraten. [E-V30, CS5]

Für Hailo-8/YOLO26 `b398`/`b364` ist `KNOWN_INFEASIBLE / COMPILE_INFEASIBLE / exact_deterministic_outcome` vorhanden. AP2 reicht dieses Ergebnis als nicht ausführbaren, aber vorhandenen Planfall weiter. Kein neues identisches Compileexperiment und keine künstliche Missing-Kaskade. Andere Splitpunkte wären eine bewusst anders geplante Kampagne, keine stille Reparatur dieses Audits. [E-V30, CS6] [E-PLAN31]

### 9.7 Weitere Qualitätsbefunde bleiben eigenständig

MobileNet-Hailo-Splitverluste sind ebenfalls groß: CPU 73,6 %, Hailo-8 59,6 %, Hailo-10H 59,2 % im alten B500-Lauf. **Die DeepX-R1/R2-Erkenntnis behebt diese Hailo-Pfade nicht automatisch.** Deren tatsächliche Eingabe-/Kalibrierungs-/Schnittstellenverträge bleiben bei anhaltendem Verlust gesondert zu prüfen. [E-V30, §6]

Bei Detection bleibt DeepX YOLO26m Full `inconclusive`; YOLO26s Full insgesamt `fail`. Ein ungefähr 0,61-pp-Verlust in AP 50:95 versteckt nicht die etwa 1,50 pp Verlust in AP75. Hauptmetrik und Guardrails zusammen lesen. Ebenso bleibt ein technisch funktionierender DeepX-D-Split mit unzureichend abgesicherter AP50/AP75-Nichtunterlegenheit `inconclusive`. Keine Statusänderung durch einen reinen Softwarefix. [E-V30] [E-D29]

### 9.8 CS7/CS8 – Intervalle und Kandidatenzahl

Bei frühem Verlust-Fail wurden `bootstrap_repetitions=0` und dennoch `ci_low=ci_high=delta` ausgegeben. **Das ist kein berechnetes Konfidenzintervall.** AP8 kennzeichnet die ausgelassene Unsicherheitsschätzung und hält die bestehende Punkt-/Gateentscheidung davon getrennt. Der anders begründete Fall `candidate_reference_identical` wird nicht pauschal damit gleichgesetzt. [E-V30, CS7] [E-PLAN31, AP8]

Das Complete-Set-Profil enthält `max_accepted_cases_per_model: 1`; die sieben ausgewählten Grenzen sind `b119`, `b135`, `b132`, `b062`, `b398`, `b364`, `b044`. Der Plan meldet zu wenige vergleichbare Kandidaten für Rankingtransfer. `insufficient_candidates` wird nicht zu `pass`, weil alle sieben Modellnamen vorkommen. Die erfolgreiche historische YOLOv7-`b066`-Abnahme deckt nicht automatisch `b044` ab. [E-V30, §8]

<a id="v31-plan"></a>
## 10. Aktueller v2.81-Auftrag und erhaltene .4-Korrekturen

| Bereich | Umsetzung und Abnahmevertrag |
|---|---|
| Basis | Vollständige .4-Quelle einschließlich .3-FIX1–FIX5, Cancelzählung, Debuglimits, CPUreferenz und Remote-Closure erhalten. |
| Compiler | Gespeicherte H8-Auswahl zuverlässig übernehmen, typed GUIpfad respektieren; finale `cold_build_rows` nach Auswahl und Wiederverwendung vor Dispatch prüfen. |
| H8-Eingabe | Ein identischer RGB-Puffer für raw und completed sowie deren Wiederholungen; JPEG-/Pufferbindung und expliziter Timing-Scope. |
| H10-Layout | Reale physische Ausgaben durch beide Sessionklassen und vorhandene Bridge reproduzieren; singletongebundenes NWC↔NCW korrekt transponieren. |
| Terminalstatus | Bestätigte exakte Compile-Negative als explizite Nichtausführbarkeit erfassen; unbekannte, ungebundene und Infrastrukturfehler bleiben technische Fehler. |
| Qualityidentität | Äquivalente Run-/Precision-Aliasse und Hashdarstellungen vereinheitlichen; Quality-FAIL und echte Bindungskonflikte erhalten. |
| Lieferung | Neue Identität 2.81, gezielte echte Regressionsfixtures, vollständige fortgeltende Softwareabnahme, isoliertes Upgrade und ein gemeinsamer Zielstarter. |

Der Nutzer hat diese Korrekturen nach der vollständigen Smokeauswertung direkt beauftragt. Rezepte, Ranking, Qualitätsmargen, Bootstrapmethodik und Forschungsumfang aus REV3 bleiben verbindlich. Es wird keine weitere Optimierungsstudie und keine Wiederholung abgeschlossener 5.000-Bilder-Vergleiche eingeführt. [E-281-SMOKE] [E-SCOPE-REV3]

Die erhaltene .4-Basis liefert weiterhin 512 MiB Zentraldeskriptoren gesamt / 64 MiB je Datei und 256 MiB für freigegebene strukturierte Ergebnisse/Indizes, echte Cancelzählung und vollständige begrenzte Debugexports. Große erlaubte Daten heben Pfad-, Symlink- oder Atomaritätsregeln nicht auf.

### 10.1 Hailo8 im normalen Editor

Neben „Hailo8 compute for new builds“ stehen Manifestpfad, Dateiauswahl und lesende Prüfung. Ein fehlendes optionales Feld lässt die bisherige explizite Environment-Auswahl zu. „Explizit auswählen“ mit leerem Pfad bedeutet bewusst kein Overlay. Jobwert → Familienwert → bisherige explizite H8-Environment-Variable → kein Overlay. Ein Default enthält keinen maschinenspezifischen Pfad.

CPU verwendet den gespeicherten Pfad nicht und startet keine Overlay-/GPUprüfung. Lesende GUI-Prüfung ist keine GPU-/XLA-Messung. Sie prüft vorhandene Familie, ausgewählte Venv, Metadaten und Komponenten mit dem bestehenden Validator. Kein Paketdownload und keine Mutation von `os.environ` des GUI-Hauptprozesses. Hailo10, DeepX, Runtime und CPU-Referenz dürfen keine H8-Librarypfade erben. Die reine Quellinstallation erhält Profile und Registrys. Der gemeinsam gestartete v2.81-Konfigurationsschritt ergänzt anschließend ausschließlich die durch diesen Fixauftrag autorisierte fehlende H8-Auswahl mit geprüfter eindeutiger vorhandener Manifestdatei und Sicherung. Explizit leere oder abweichende Auswahlen bleiben erhalten; Mehrdeutigkeit wird nicht geraten.

### 10.2 Abbruchzählung und ETA

Der dokumentierte Q5-Sollreplay enthält 123 terminale Requests: 63 fertig ausgewertet mit 43 PASS, 18 FAIL, 2 INCONCLUSIVE sowie 60 Abbruchfolgen (vier `CancelledError`, 56 `QualityServiceClosedError`). `completed_count` zählt weiterhin ausgewertete Ergebnisse. Terminal ist nicht gleich erfolgreich berechnet und Requestzahl ist nicht Hardwaremesszahl.

Eine Service-Closed-Ausnahme ist nur mit passendem vorangehendem Run-Cancel als Abbruchfolge einzuordnen; ohne diesen Kontext bleibt sie technischer Fehler. Früher entstandene technische Fehler werden durch späteren Cancel nicht gelöscht. Fertige Cache-/Berechnungsergebnisse bleiben beim Cancelrennen einmalig erhalten; unvollständige Shards veröffentlichen kein CI/PASS oder erfolgreichen Cacheeintrag. Der alte abgebrochene Run bleibt cancelled und unvollständig, seine 18 beobachteten Qualitäts-FAILs bleiben fail.

Abbruchterminals, Cachetreffer und frühe Punktentscheidungen gehören nicht in die mittlere Laufzeit eines vollständigen Detectionbootstraps. Hailo-GPU beschleunigt nicht automatisch zentrale CPU-Statistik. Workerzahl, Methode, Wiederholungen und Margen werden für eine angenehmere ETA nicht geändert. [E-2804-PLAN]

### 10.3 Was die langen Qualityaufträge tatsächlich tun

Die Referenz-/Kandidateninferenz erzeugt zunächst Vorhersagen. Der zentrale Service bindet dieselben Bild-IDs/Labels, berechnet Taskmetriken und gegebenenfalls gepaarten Bootstrap aus diesen bereits vorhandenen Vorhersagen. Bei 5.000 Bildern und 5.000 Resamples laufen nicht 25 Millionen erneute Modellinferenzen. Klassifikation verwendet Trefferzählungen; Detection benötigt wiederholte gewichtete Precision-/Recall-/AP-Auswertung auf vorbereiteten Matches. Die vier Worker bearbeiten dabei häufig Teilstücke eines Vergleichs, nicht vier unabhängige vollständige Vergleiche.

Der bestehende zentrale AP-Evaluator und ein separat angeforderter Official-COCOeval-Bericht sind verschiedene Belege. Early-Fail, identische Vorhersagen und ein exakter Qualitycache-HIT können den vollständigen Bootstrap vermeiden, ohne Metriken zu erfinden. GPU für Hailo-Neubauten beschleunigt diese Management-CPU-Statistik nicht. Algorithmische Beschleunigungen wären nur nach nachgewiesener Metrik-/Seed-/Entscheidungsparität zulässig, nicht durch spontanes Reduzieren des Statistikumfangs. [E-QUALITY-EXPLAIN]

Der tatsächliche alte Exportblocker betraf **123 gültige Requestdateien mit 35.157.419 Byte**, die ein 32-MiB-Summenbudget um 1.602.987 Byte überschritten. Kein VRAM-/Modell-/Datenträgerfehler. Die .4-Limits 512/64/256 MiB gehören zur erhaltenen Quellbasis. Der ursprüngliche REV3-Dokumentabgleich war kein eigenständiger Implementierungsnachweis; maßgeblich bleiben die Softwarebelege der jeweiligen Lieferung.

<a id="gates"></a>
## 11. Eine gemeinsame Zielabnahme

`install_and_accept_v281.sh` installiert die geprüfte Quelle, führt die fortgeltende Softwareabnahme aus, prüft die gezielte H8-Konfigurationskorrektur und startet die betroffenen Fälle durch den normalen Workflow. Alle Stufen haben endliche Budgets; Erfolg und Fehler liefern ein zusammenhängendes Evidence-ZIP. Kein vorgeschalteter separater Metadata-Upload.

Zwei begrenzte Familienprofile laufen nacheinander innerhalb dieses einen Aufrufs: H8 mit vier bisherigen Kontext-MISS-Fällen, b067 und bestätigten H8-Negativen; H10 mit den zwei reparierten Übergängen und zwei bestätigten H10-Negativen. Die getrennten Profile verhindern ungewollte zusätzliche Builds auf der jeweils anderen Familie. Quellfälle und eingefrorene Buildrezepte bleiben erhalten; Force AUS, vorhandene Artefakte regulär wiederverwenden. Ist ein früher fehlender Job inzwischen warm, wird dieser HIT ehrlich berichtet, kein Kaltbuild erzwungen.

Erwartung ist technisch korrekter Abschluss aller ausführbaren Aufträge plus exakt belegte Nichtausführbarkeit der bekannten Rejects. Ein bekannter Qualitäts-FAIL darf erhalten bleiben. Neue unbekannte technische Fehler, fehlende Pflichtresultate, Cancel oder ungebundene Negativbehauptungen verhindern die technische Abnahme. Energie- und 5.000-Bilder-Finalvergleiche sind kein versteckter Teil dieses Korrekturlaufs. Die genaue CLI und Budgets stehen in `TESTANLEITUNG_2.81.md` und dem gelieferten Starter.

<a id="todo"></a>
## 12. Einzige aktuelle Aufgabenliste

### Abgeschlossen oder als dauerhafte Regression erhalten

- [x] Hailo10 isolierte Toolchain-/GPU-/Build-/Fixed16-Untersuchung im benannten v34-Scope.
- [x] Hailo8 isolierte Compute-/Build-/Fixed16-Untersuchung; Original-JSONs jetzt zusätzlich im 236-Dateien-Ergebnisarchiv verfügbar.
- [x] Hailo8 Parsed-/Quantized-HAR-R1 auf denselben 16 Inputs ausgeführt; erste große DFC-Verluststelle lokalisiert, verbleibende Emulations-/Hardwareabweichung offen.
- [x] 236-Dateien-Ergebnisnachtrag einschließlich HAR-Report und Claim-Grenzen laut Nutzerkonsole auf `main` gepusht (Commit `5636017…`); Live-HEAD nicht erneut gelesen.
- [x] YOLO11l/Hailo10H zentraler 5.000-Bilder-Vergleich mit vollständigen Vorhersagen: TensorRT PASS, Hailo Full/Split FAIL.
- [x] Historischen negativen Opt2-Befund berücksichtigt; zusätzlicher Opt2/B1024- und Opt3-Versuch verworfen.
- [x] Produktive Force-Politik AUS und Wiederverwendung passender Artefakte festgelegt; CPU/GPU-Präferenz ist kein Neubaugrund.
- [x] MobileNet-B5000-Befunde aus dem .4-Plan als bereits ausgewertete negative Resultate statt pauschal „noch nie bewertet“ eingeordnet.
- [x] Alte v31-Planlisten und Installationsaufforderungen aus der aktuellen Aufgabenliste entfernt; frühere Quellen bleiben historisch erhalten.
- [x] Forschungsumfang präzisiert: automatisierte Partitionierung unter festem Rezept, kein modellspezifisches Optimieren bis PASS.
- [x] Zusätzliche Hailo8-Float-/Integer-I/O-Ursachenanalyse zurückgestellt; erste HAR-Fallstudie und bestehende Qualitätsbefunde bleiben erhalten.

### Aus der neuen Softwarelieferung auf dem Zielsystem auszuführen

- [x] .4-Installation und gebündelten Normalworkflow auf Smartmirror2 ausgeführt; technischen PASS vom anschließenden größeren fehlgeschlagenen Nachtlauf getrennt.
- [x] Dessen vier Fehlergruppen gemeinsam diagnostiziert und konkrete Korrekturen in v2.81 umgesetzt.
- [ ] Den gemeinsamen v2.81-Installations-/Normalworkflowstarter einmal ausführen und den finalen Zielsystembericht prüfen.
- [ ] H8-Kaltbuild nur bei ohnehin benötigtem echten MISS beobachten; ein warmer Bestand bleibt warmer Bestand.
- [ ] Die aktuelle KB mit der erhaltenen REV3-Scopeentscheidung und die nachträgliche ausführliche HAR-Interpretation als getrennte Dokumentergänzung ins Ergebnis-Git aufnehmen; kein neuer Push ist hier erfolgt, der vorhandene Ergebniscommit muss nicht erneut erzeugt werden. Frühere KB-Snapshots bleiben erhalten.

### Nur für die ausdrücklich beanspruchte wissenschaftliche Aussage

- [ ] Die für die festgelegte Hauptauswertung noch tatsächlich fehlenden regulären Qualitäts-/Performance-/Energienachweise gezielt abschließen; abgeschlossene negative Qualitätsresultate nicht wiederholen. Nichtausführbare Fälle mit ihrem belegten technischen Grund erhalten; Cancel ohne Ergebnis nicht als fertige Auswertung zählen.
- [ ] Passende Power-/Clock-/TPC-, FS-/Idle- und Dauer-/Replikatbelege für den konkreten Energievergleich verwenden; 1-s-Screening nicht zu 60 s × 3 umetikettieren.
- [ ] Einen benötigten offiziellen COCO-Bericht aus vorhandenen Vorhersagen ergänzen; das erfordert keine erneute Hailo-Inferenz.
- [ ] Nur die neue technische Ausführung der in v2.81 korrigierten H10-Fälle im gemeinsamen Zielworkflow abnehmen. Der reine Bridgepfad ist im Smoke geprüft; der bekannte Klassenscoreverlust bleibt ein negatives Qualitätsergebnis und verlangt keinen weiteren Sweep.

### Zurückgestellt – keine aktuellen Arbeitsaufträge

| Thema | Status / Voraussetzung für Wiederaufnahme |
|---|---|
| Hailo8 optimierte Floatstufe, 2×2-Integer-I/O und innere Verlustlokalisierung | Optionale Folgearbeit nach §7.8; jetzt nicht ausführen. Nur mit neuem ausdrücklichem Auftrag zur stärkeren Ursachenfrage. |
| Modellspezifische Compiler-/Quantisierungs-/Architekturoptimierung | Außerhalb der Hauptauswertung; keine Opt2-/Opt3-/Kalibrationssweeps und keine Garantie besserer Ergebnisse. |
| Vollständige Erklärung aller Emulator-/Runtimeunterschiede | Für den jetzigen begrenzten Ergebnisclaim nicht erforderlich; offene Unterschiede in Diskussion/Limitationen nennen. |

**Keine aktuelle Pflicht:** weiteres identisches Metadata-Smoke, neuer pauschaler Hailo-Optimierungssweep, private GPU-HEFs wegen Fixed16 automatisch austauschen, komplette nächtliche Kampagne nach jedem Patch wiederholen oder ein abgeschlossenes Qualitäts-FAIL erneut erzeugen. Ein gezielter Diagnoselauf braucht einen konkreten Fehlerverdacht, eine benötigte Evidenzlücke oder einen neuen begrenzten Auftrag; ein numerischer Reparaturpatch braucht einen belegten Fehler. Die Größe eines Qualitätsverlusts allein reicht dafür nicht. [E-SCOPE-REV3]

<a id="betrieb"></a>
## 13. Betriebsregeln

GUI und laufende Workflows vor dem Update geordnet beenden. Der Installer erhält Tool-/Vendor-Venvs, Benutzerprofile, Run-Mode-/Hardware-Registry, vorhandene Overlays, Modell-/Qualitycache und Originalruns. Die H8-Auswahl erfolgt ausdrücklich im Editor oder durch die in diesem Fixauftrag autorisierte enge, gesicherte Migration einer eindeutigen vorhandenen Auswahl. Kein Treiber-, System-CUDA-, DFC-, TensorFlow-, Torch- oder DeepX-Upgrade in v2.81.

`relaxed` bleibt auch für Final der vereinbarte Repro-/Cachemodus. Hailo balanced/Opt1/B500/Batch8, DeepX EMA/Opt0/B500, ImageNet-Mean/Std, Qualitätsmargen, Seed, AP-/Top-k-Definitionen und Bootstrapmethode bleiben gleich. Der vorgeschaltete Standarddurchlauf und der separate Finalumfang dürfen beim Resume nicht als identischer Run mit verändertem Profil vermischt werden.

Generische Energie bleibt aus. Native-Energie ist ein eigener Scope; Dauer und Replikate werden nicht still geändert. Vorhandene alte Fehlermeldungen oder PIDs sind keine aktuellen Betriebszustände. Primärfehler, erwartete Nichtrealisierbarkeit, technische Ausführung, Qualitätsentscheid und Cleanup werden separat gelesen.

### 13.1 Was auf Smartmirror2 bleiben muss

Ergebnis-Git ist kein Modellbackup. Für bereits vereinbarte Ausführung und mögliche Ursachenprüfung die ursprünglichen Nacht-, privaten Build-, Runtime- und HAR-Diagnoseordner sowie den **gesamten** H8-Overlaybaum erhalten. Besonders das lokale `runtime_arrays.npz` lag absichtlich nicht im kleinen Ergebnis-ZIP. Nur ein Manifest ohne referenzierte Bibliotheken oder ein Receipt ohne benötigtes Modell reicht nicht zur erneuten Ausführung.

Keine gleichzeitige Installation von v2.81, Venvänderung, GPUkompilierung und neue Ursachenprüfung. Neue Diagnose-Supervisoren nutzen die bestehenden Workflow-/Plattformlocks; keine alten PIDs oder Lockdateien aus Chatbeispielen löschen. Erfolgreiche private HAR-/HEF-Tests werden nicht durch Wechsel des Releaseetiketts ungeschehen.

<a id="ablage"></a>
## 14. Evidenzablage und dauerhafte Referenzen

Git erhält Code, Dokumentation, kleine Profile, Logs und ausgewählte strukturierte Ergebnisse. Modelle, Bildcorpora, HEFs/DXNNs/Engines, CUDA-Bibliotheken und vollständige rohe Mess-/Vorhersagebestände bleiben im gesicherten Originalbestand. Ausdrücklich dokumentierte kleine Regressionsfixtures sind Teil der Quellprüfung; v2.81 enthält sechs vorhandene H10-B-Ausgangspuffer zur Layoutregression, keinen Ersatz des Modell- oder Messarchivs. Ein exportierter Ergebnisbericht ersetzt nicht die tatsächlich verwendeten Modelle oder Annotationen.

Das .4-Lieferpaket enthält den Planabgleich, konkrete Prüfprotokolle und Quellenindex. Original-Q1/Q3/Q5 werden nur als Originalfixture bezeichnet, wenn ihre Bytes tatsächlich verfügbar und geprüft sind. Abgeleitete kleine Cancel-Fixtures sind entsprechend benannt und ersetzen nicht den angeforderten Original-123-Request-Export. Ein Q2-Terminalreport ist keine unabhängig nachgezählte Q1-Rohdatei. Downloadfehler bleiben als Quellenlücke sichtbar.

Ein Replay schreibt neue Darstellung separat; vergangene `run_manifest`, Requests, Fingerprints und Qualitätsentscheidungen bleiben unverändert. Die alte kanonische KB vom 8. September ist die historische Quelle dieses Updates und bleibt als vorherige Fassung nachvollziehbar. Gleiche Dateinamen von Standalone-R1-Paketen beweisen keine gleichen Bytes.

### 14.1 Was der Git-Push tatsächlich gesichert hat

Die Nutzerkonsole belegt:

```text
Repository: Keff789/onnx-splitpoint-results
Branch: main
Commit: 5636017e5db3229862ba10c609b5f4b5f290e76b
GIT_PUSH=PASS
236 files changed
```

Das hochgeladene `har_and_git_evidence.zip` enthält denselben vorbereiteten **236-Dateien-Nachtrag**. Die Pushkonsole meldet genau diesen Auswahl-/Commitumfang; ihr diff-stat kürzt Dateinamen ab, weshalb daraus nicht jeder vollständige Pfad nochmals bytegenau rekonstruiert wird. Primär liegen die vollständigen Pfade und Bytes im Ergebnisarchiv. Ein neuer Liveabgleich des GitHub-HEAD war hier nicht verfügbar. Keine Aussage über unbekannte spätere Commits. [E-GIT-PUSH] [E-KB-REV2-CHECK]

**Ergebnisse und erste maschinell erzeugte Einordnung sind bereits gesichert:**

```text
results/hailo8/20260912_mobilenet_gpu/
  compute/
  build_metadata/
  runtime/
  har_emulation/hailo8_har_git_20260912T171411Z_x0j2t2o1/
    comparison.json
    comparison_request.json
    REPORT.md
    CLAIM_BOUNDARIES.md
    per_image.csv
    parsed_native/
    quantized/
experiments/hailo8/har_comparison_r1/
results/evaluation/completsetdev_20260911_213508/quality_snapshot/
```

`REPORT.md` enthält Stufentreffer, Maximal-/RMS-/Cosinewerte, Vorhersagewechsel und grundlegende Vergleichsgrenzen. `CLAIM_BOUNDARIES.md` nennt geöffnete Fixed16-Diagnose, späte HAR-Hashes und Cross-Build-Grenze. **Nicht Teil dieses Commits:** die erst danach geschriebene ausführliche Interpretation `AUSWERTUNG_Hailo8_HAR_Vergleich_2026-09-12.md`, KB-REV2 sowie REV3 und der nachfolgend zurückgestellte vertiefende Diagnoseplan; auch für die vorliegende v2.81-KB ist kein Push belegt. Eine Antwort im Chat wird nicht automatisch in Git geschrieben. Für spätere Dokumentcommits liegt in diesem Update kein neuer Nachweis vor.

### 14.2 Publikations- und Ablageregel

`build_metadata/` statt eines im Repository ignorierten `build/`-Verzeichnisses für kleine Receipts/Logs nutzen. Keine `git add .`- oder Force-Push-Anweisung als Standard. Neue Dokumentrevision neben Originalergebnisse legen, alte KB-Snapshots erhalten; Index/Claimstatus bewusst fortschreiben. Neu erzeugte Dokumente aus dieser Revision sind zunächst lokale Lieferdateien und **noch nicht gepusht**.

Der damalige Check meldete `fatal=0`, `review=159`, `PASS_WITH_REVIEW`: hauptsächlich lokale Pfade/private Adressen, zusätzlich eine heuristische password-like-Zeile in Diagnosesource. Das sind keine 159 fehlgeschlagenen Tests und auch keine vollständige Secret-/Lizenzfreigabe. Öffentliche Weitergabe bleibt eine bewusste Prüfung; keine Tokens/Schlüssel oder komplette lokale Arbeitsbäume ergänzen.

<a id="grenzen"></a>
## 15. Klärungen gegen wiederkehrende Fehlannahmen

| Verkürzung | Verbindliche Einordnung |
|---|---|
| „Für fertig müssen alle Qualitätsfelder grün sein.“ | Fertig ausgewertet kann korrekt FAIL sein; keine Margenlockerung oder Fallauswahl nach Ergebnis. |
| „123/123 terminal heißt 123 Qualitätsberechnungen.“ | Im Q5-Soll: 63 ausgewertet, 60 Abbruchfolgen; Requestzahl und unabhängige Hardwaremesszahl unterscheiden. |
| „Service geschlossen ist immer harmloser Cancel.“ | Nur mit passendem vorangehendem Run-Cancel; sonst technischer Fehler. |
| „GPU-HEF ist bei gleicher Recipe derselbe Qualitätsfall.“ | Konkrete Artefakt- und Vorhersageidentität zählt; keine fremden Ergebnisse übernehmen. |
| „GPU-Präferenz muss CPU-HEFs ersetzen.“ | Gültige vorhandene HEFs wiederverwenden, unabhängig vom Baugerät. |
| „Hailo8 hat jetzt 12/16 wie Hailo10.“ | H8: 8/16 CPU-HEF, 10/16 GPU-HEF; H10 historisch 12/16 für beide. |
| „Gleiche FLOAT32-Feeds beweisen gleiche interne UINT8-Puffer.“ | Interne UINT8-Puffer wurden in dieser H8-Probe nicht erfasst. |
| „Besseres Fixed16 rechtfertigt automatischen HEF-Tausch.“ | Kleine geöffnete Diagnoseprobe, kein allgemeiner Qualitätsvorteil und kein B5000-PASS. |
| „Opt2/B1024 sollten wir einfach nochmal versuchen.“ | Historischer Opt2-Befund und eingefrorene Transfer-/Evaluationspolicy schließen den pauschalen Sweep aus. |
| „Fehlender Debugexport ist ein Hailo-Modellfehler.“ | Q5 überschreitet das alte 32-MiB-Deskriptorsummenlimit mit gültigen Requests. |
| „AP-Fast-FAIL mit 0 Bootstraps ist nicht fertig.“ | Vollständige Vorhersagen und erlaubter früher Punktentscheid können abgeschlossen FAIL sein; kein fingiertes CI. |
| „Eine Sekunde × drei Replikate ist finale Energieabnahme.“ | Dauer-/Work-Unit-Vertrag und Scope bleiben zu prüfen. |
| „HAR-Emulation fehlt weiterhin vollständig.“ | R1 `SDK_NATIVE` und `SDK_QUANTIZED` sind ausgeführt; zusätzliche optimierte Floatstufe und Integer-I/O-Ursachenklärung sind nicht ausgeführt, aber ausdrücklich zurückgestellte Folgearbeit, keine aktuelle Pflicht. |
| „Alles nach Parsed ist reine Quantisierungsrundung.“ | Der tatsächliche Übergang umfasst Modell-/Floatoptimierung und Quantisierungsverfahren. |
| „9/16 Emulation und 10/16 Hardware sind nahezu identisch.“ | Vier verschiedene Top-1-Klassen und merkliche Logitabweichung; ähnliche Accuracy ist keine numerische Parität. |
| „Der spätere ausführliche Chattext ist mit dem Ergebniscommit gespeichert.“ | Automatischer Report und Grundgrenzen ja; spätere Interpretation/REV2/REV3 nein, solange kein weiterer Dokumentpush belegt ist. |
| „Ohne weitere Ursachenanalyse ist das negative Ergebnis nicht berichtsfähig.“ | Ein korrekt ausgeführtes und gebundenes negatives Resultat kann unter klaren Grenzen berichtet werden; eine stärkere Ursachenbehauptung ist eine andere Frage. |
| „Wir müssen alle Netze so lange verbessern, bis sie bestehen.“ | Nein. Bewertet wird das feste automatische Verfahren. Qualität, technische Misserfolge und Ausführbarkeit werden vollständig berichtet. |
| „Dann können wir bekannte Toolfehler als Compilerverlust stehen lassen.“ | Nein. Bekannte eigene Fehler korrigieren oder als technische Einschränkung kennzeichnen; kein regulärer Qualitätsverlust daraus machen. |
| „Mit genügend Tuning werden alle Netze gut.“ | Nicht belegt. Verbesserungen sind mögliche Folgearbeit mit unbestimmter Wirksamkeit und Aufwand, keine Garantie. |
| „Die Scopeentscheidung lockert die 1-pp-Marge oder den Messvertrag.“ | Nein. Qualitäts-, Daten-, Endpunkt-, Energie- und Claimregeln bleiben unverändert. |


<a id="uebergabe"></a>
## 16. Kompakte Übergabe

v2.81 baut auf dem unverändert verifizierten vollständigen .4-Bundle auf. Der kombinierte Smoke ersetzt weitere Metadatenrunden: fehlende H8-Overlaybindung, unterschiedliche Vorverarbeitung desselben JPEGs und falsche H10-Tensoranordnung sind konkret belegt. Die sechs exact gebundenen Compile-Rejects bleiben valide negative Ausführbarkeitsergebnisse. Die vorhandenen H10-Klassenscores sind bereits null; der Layoutfix ist keine Behauptung einer behobenen Modellqualität.

Der neue gemeinsame Starter umfasst Installation, Softwareabnahme, eng begrenzte Konfigurationskorrektur und normale Ausführung der betroffenen Fälle mit einer einzigen Ergebnislieferung. Neu bauen nur bei tatsächlich benötigtem MISS; Force aus. Bereits abgeschlossene .3-/ .4-Messungen behalten ihren Scope. Finaler neuer Hardwarestatus erst nach realem Zielbericht, keine automatische Freigabe aus lokalen Tests.

Die REV3-Stopregel gilt unverändert: automatisierte Partitionierung unter festem Rezept, negative Qualitätsresultate erhalten, eigene Fehler korrigieren, keine manuelle Modelloptimierung bis PASS und keine erneute vertiefte H8-HAR-/Integer-I/O-Studie ohne neuen Auftrag. Abschluss heißt nachvollziehbare Ergebnisse für die festgelegten Fälle.

<a id="aenderungen"></a>
## 17. Änderungsprotokoll

### 13. September 2026 – v2.81

Vollständige REV3 erhalten; .4-Installations-/Nacht-/Smokezustände nachgetragen. Fünf konkrete Korrekturverträge aufgenommen, H10-Layoutfehler und bereits vorhandenen Klassenscoreverlust getrennt, echte Bildbytes statt bloßer Dateinamen erklärt. Aufgabenliste und Übergabe auf einen gemeinsamen v2.81-Abnahmestarter aktualisiert. Keine nachträgliche Messwertänderung, kein neuer Qualityclaim, kein Gitpush. Softwarebelege gehören zur v2.81-Lieferung; historische Smokes bleiben .4. [E-281-BASE] [E-281-NIGHT] [E-281-SMOKE]


### 13. September 2026 – REV3: Untersuchungsumfang und Stopregel

Auf der unveränderten beigefügten REV2 fortgeschrieben. Forschungsfrage als automatisierte Partitionierung unter festgelegter Backend-/Build-/Validierungspolitik formuliert; „einfach so“ präzisiert. Implementierungsfehler, optionale Ursachenanalyse und modellspezifisches Tuning getrennt. Abschlusskriterium auf vollständige nachvollziehbare Ergebnisbilanz statt alle Fälle PASS festgelegt. Zusätzliche Hailo8-Optimized-Float-/Integer-I/O-Vertiefung ausdrücklich zurückgestellt und aus den aktiven Checkboxaufgaben entfernt; technische Skizze bleibt als optionale Folgearbeit erhalten.

Die ergänzende Vergleichsperspektive Split versus Full-Backend neben der kanonischen Floatreferenz aufgenommen, ohne neue Akzeptanzgrenze. Formulierungsbaustein für die Dissertation und zurückhaltende Aussage zu möglichem Tuningnutzen ergänzt; kein Erfolg für alle Netze garantiert. Grobe Aufwandsbereiche aus dem Chat als ungemessene, nicht beauftragte Planung eingeordnet. AP5, Abnahme, Fragenkatalog, Git-Dokumentstatus und Übergabe konsistent fortgeschrieben. Sämtliche vorhandenen Ergebnistabellen und deren Aussagegrenzen bleiben erhalten. Keine neue Messung, Produktimplementierung, Release-/Hardwareabnahme oder Gitveröffentlichung. [E-SCOPE-REV3] [E-KB-REV3-CHECK]

### 12. September 2026 – REV2 nach HAR-/Git-Abgleich

Auf der beigefügten .4-KB fortgeschrieben, nicht neu aus einem alten Release rekonstruiert. Quellenverfügbarkeit und .4-Implementierungsbehauptung getrennt. Forceherkunft, Familien-GPUkontexte, reale Release-/Workflow-Fehlerketten, Reuse-/Energie-/Nachtgrenzen ergänzt. Die Originalstufen des HAR-R1-Archivs erneut nachgezählt: 13/13/13/9/10/8 Top-1; erste große Abweichung und Restabweichung ehrlich getrennt. Noch ausstehende HAR-Emulationstexte auf den ausgeführten Stand berichtigt. Vertiefte Diagnose nur als neuer bedingter Vorschlag.

Gitcommit, 236-Dateien-Scope, tatsächlich gesicherten Report und nicht mitgespeicherte spätere Interpretation dokumentiert. Qualitäts-Cancelpopulation 63/60 direkt aus Original gelesen. Vorhandene Methoden-, YOLO11l-B5000- und Opt2-Entscheidungen erhalten; fehlende Originalexporte hierfür nicht erfunden. Kein Toolpatch, keine neue Messung, kein Push.

### 12. September 2026 – v2.80.4, übernommene Ausgangsrevision

Aktuelle Kopfidentität, Releasefortschritt und Aufgabenliste auf .3-FIX5/.4 fortgeschrieben; v31-Planaufträge historisiert. Hailo8/Hailo10-Fixed16 getrennt, MobileNet-B5000-FAILs aus dem neuen Plan eingeordnet, abgeschlossene YOLO11l-Qualität mit vollständigen Vorhersagen und historische Opt2-Entscheidung aufgenommen. Alte automatische Wiederholungs- und Optimierungsvorschläge gestrichen. Cancel-, Debugbudget-, Overlay-, Reuse- und Quellenregeln integriert. Originalergebniswerte und vorhandene wissenschaftliche Methodik unverändert.

### 8. September 2026 – konsolidierte Fortschreibung nach R2

Die bisherige v17-Kopfidentität wurde als historischer Stand abgelöst, nicht als aktuelle Installationsanweisung weitergeführt. Methodenrahmen, Fragen-/Boundarythemen, FS-Primärmessung, Generic-/Native-Scope, Rankinggrenzen und einfache Evidenzablage bleiben erhalten. [E-KB]

Neu integriert sind die v26–v30-DeepX-Full-Fehlerfolge, die unabhängige v30-Teilabnahme, der bestätigte Abschluss-Cachefehler und dessen enger Fix, die acht Complete-Set-Fehlergruppen beziehungsweise zusätzlichen Hailo8-Bindungslücken, korrekte Ergebnis-/Energie-/Qualityzähler und der aktuelle v31-Plan REV4 FINAL. [E-V30] [E-PLAN31]

R1/R2 ersetzen die allgemeine DeepX-Pre-/Postprocessing-Ursachensuche durch einen konkret bestätigten Build-/Profil-Normalisierungspfad. Die MobileNet-Klassifikationsqualität bleibt trotzdem offen: kleine Nenner, gleiche Bild-IDs, Logitähnlichkeit und Accuracy sowie historische B500-Werte sind jetzt ausdrücklich getrennt. Überzogene Aussagen zum bereits bewiesenen Anteil des B500-Gewinns sind eingegrenzt. [E-R1] [E-R2]

Zusätzlich wurden beim Dokumentabgleich die tatsächlich gespeicherten **1-s-Energieaufträge** und die spätere FS-Gain-/M.2-Idle-Anwendung aufgenommen. Nicht vorhandene ursprüngliche Kalibrierungsbelege werden nicht als hier geprüft ausgegeben; die Abweichung zwischen einer alten Gateuntergrenze und einem später verifizierten Faktor bleibt sichtbar. [E-CS] [E-DOC]

Diese Beschreibung betrifft das historische Dokumentupdate vom 8. September, nicht die neue .4-Softwarelieferung. Originale Messdaten und frühere Snapshots bleiben unverändert.

<a id="quellen"></a>
## Quellen- und Fundstellenverzeichnis

Die Kennungen verweisen auf vorhandene Dateien beziehungsweise klar benannte Chatbeobachtungen. Innerhalb von ZIPs sind die angegebenen Pfade relativ zur Archivwurzel. Frühere KB-Begleitpakete enthalten die dort benannten Dokumente und read-only Projektionen. Das historische REV3-Begleitpaket enthielt ausschließlich die aktualisierte KB, Änderungsnotiz, Textdiff und Dokumentprüfbericht, **nicht** die früheren Ergebnisarchive, Modelle oder erneut ausgeführte Projektionen. Die neue v2.81-Lieferung enthält zusätzlich Quelle, Installer, gezielte Regressionsfixtures und ihre tatsächlichen Softwareprüfnachweise.

**[E-KB]** `ONNX_SPLITPOINT_KnowledgeBase_CANONICAL_v2.79.17_2026-09-03.md`, aus der File Library herangezogene kanonische Vorgängerfassung. Insbesondere Dokumentführung/Evidenzklassen, wissenschaftlicher Leitpfad und Fragenkatalog, Energie/Kalibrierung, Backend-/Fairnessregeln, historische Claim-Map, Evidenzablage und Übergabestand. Historische Fassung bleibt unverändert; sie ist nicht als neue Datei diesem Paket beigelegt.

**[E-KB15]** `ONNX_SPLITPOINT_KnowledgeBase_v2.79_Three_Stage_2026-09-03_UPDATED_v2.79.15.md`, historischer Hinweis auf FS/command und den längeren 60-s-×-3-Abnahmeumfang. Die späteren v17-Klarstellungen zu Gain-/Idle-Skalendomänen haben gegenüber früheren pauschalen Wiederverwendungsaussagen Vorrang.

**[E-V30]** `ABNAHME_v2.79.30_und_Complete_Set_v2.79.29.md`, §§1–3 für Release-/Testumfang, §4 für Zähler/Closure, §§5–8 für CS1–CS8 und Qualität/Energie.

**[E-V30-N]** `ABNAHME_v27930_NACHWEISE.zip`, unter dem Archivpräfix `v27930_audit/evidence/`, insbesondere `independent_packaging.json`, `independent_targeted.log`, `terminal_closure_smoke_fast.json`, `terminal_closure_smoke_strict.json`, `independent_complete_replay.json`, `independent_hailo_replay.json`, `BEFUNDE_Complete_Set_v27929.json` und `independent_supervisor_diagnostic.json`. Beschreibt die bereits ausgeführte unabhängige Abnahme, keine neue Ausführung bei diesem KB-Update.

**[E-CS]** `complete_set_20260907_161614_debug_pack.zip`, insbesondere `reports/native_producer_summary.json`, `reports/native_stage_concise_summary.json`, `reports/native_evidence_status.json`, `quality_management/central_quality_summary.json`, `reports/artifact_index_closure.json` und `reports/native_energy_measurements/native_producer_energy_results.json`. Dort: `rows[*].row.duration_s`, `rows[*].run.energy_aggregate`, `full_system_current_scale_*`, `accelerator_idle_*`, Primär-/Shadow-Methode und Replikatzähler.

**[E-PLAN31]** `IMPLEMENTATIONSPLAN_v2.79.31_Complete_Set_Integration_und_Qualitaetsdiagnostik_REV4_FINAL.md`, verbindlicher Auftrag; AP0–AP11, §§16–20 für 103 geplante Prüfgruppen, Gates und Freigabe. Maßgeblich für die Umsetzung, nicht als bereits erfüllte Evidence zählen.

**[E-PLAN30]** `IMPLEMENTATIONSPLAN_v2.79.30_DeepX_Full_Normalworkflow_REV2.md`, ursprünglicher Sollstand für Full-Prüfübergänge und terminalen Abschluss. Bereits erfüllte Aufgaben in v31 als Regression erhalten, nicht nochmals als neue offene Diagnose erfinden.

**[E-R1]** `deepx_prepost_smokes_r1_20260908T080238Z_vo3whdeu.zip`: `collection_summary.json`, je Modell `analysis.json`, `cpu/cpu_result.json`, `cpu/graph.json`, gespeicherte `cpu/cpu_*.npz` und `remote/native_*.npz`/`quality_*.npz` sowie Split-/Kalibrierungsbeobachtungen. Herkunftsrun `complete_set_20260907_161614`.

**[E-R1-A]** `AUSWERTUNG_DeepX_PrePost_Smokes_R1.md`, vorhandene Auswertung und Scopegrenzen.

**[E-R1-LOCK]** `deepx_prepost_smokes_r1_20260908T065807Z_dct6j689.zip`, einzig `collection_summary.json`: blockierter Start, leere Modellliste. Dazu die im Chat gezeigte Lock-/Prozessbeobachtung. Kein Modelllauf.

**[E-R2]** `deepx_meanstd_ab_r2_20260908T084314Z_wwtqdejm.zip`: `collection_summary.json`, je Modell `analysis.json`, `build/build_summary.json`, `build/adapter_ort_parity.json`, Compilerprotokolle und `remote/arm_a_*.npz`/`arm_b_*.npz`. CPU-Gegenpopulation aus den exakt passenden R1-Bild-IDs.

**[E-R2-A]** `AUSWERTUNG_DeepX_MeanStd_AB_R2.md`, vorhandene A/B-Auswertung und ausdrücklich offene B500-/RegNet-/Splitgrenzen.

**[E-R2-PKG]** `DeepX_MeanStd_AB_Smoke_R2.zip` und zugehörige `README.md`/`TEST_REPORT.md`; Scope, Staging, Lock-Preflight und Diagnoseabgrenzung.

**[E-D26]** `v27926_DeepX_YOLO11l_Diagnose.md` und `v27926_acceptance_d_yolo11l_b003_deepx_gpu_20260907_101443_debug_pack(1).zip`; ursprüngliche Eingabe-, Postprocessing-, Matrix- und Energiefehler.

**[E-NUMERIK]** `deepx_full_probe_v27927_fix1_20260907T104449Z_2_jd9v0e.zip`, insbesondere `results/deepx_output_value_probe.json` und Rohoutput-NPZ; außerdem `PRUEFBERICHT_2.79.28.md` und `PRUEFBERICHT_2.79.29.md` für die begrenzte Numerik-/Importkorrektur.

**[E-D29]** `v27929_acceptance_d_yolo11l_b003_deepx_gpu_20260907_151417_debug_pack.zip` und `v27929_D_diagnose/DIAGNOSE_v27929_D_Normalworkflow.md`: reale Replikate, fehlende Semantik-/Completionintegration und zentrale Qualityannahme.

**[E-INSTALL29]** `v27929_install_acceptance_20260907_125759_497558239_2858752.zip`, `full_console.log` und `dedicated_acceptance.log` im benannten Installationsverzeichnis.

**[E-PROBE29]** `deepx_full_probe_v27929_20260907T130020Z_jjl3xxus.zip`, `results/deepx_output_value_probe.json` und `collection_summary.json`; ergänzend `v27929_auswertung/vergleich_FIX1_v27929.json` für den bereits dokumentierten Rohtensorvergleich.

**[E-ABSCHLUSS]** `v27929_Nachlauf_nach_finished_Codestellen.txt`, `v27929_afterrun_inspect/QUELLBELEGE_Hash_Cache_Nachlauf.md` und die im Chat gezeigten Prozess-/Dateistatistiken vom 7. September, zusammen mit dem tatsächlichen Closure-Bericht aus [E-CS].

**[E-OVERNIGHT]** `_latest_evaluation_workflow(20260908-070755).log`, interner Run `complete_set_20260908_030632`, Version v30. Maßgeblich sind die im Inhalt stehenden Zeitstempel, nicht allein der Exportdateiname.

**[E-CHAT]** Im vorliegenden Gespräch eingefügte Konsolenausgaben und Nutzerentscheidung zum Overnight-Artefaktlauf, insbesondere die nachgereichten Bias-Correction-Zeilen bis 09:32:52 sowie die MobileNet-Rückfrage. Es wird kein nicht hochgeladener Endabschluss ergänzt.

**[E-DOC]** Im Begleitpaket `checks/document_evidence_checks.json` und `checks/energy_existing_evidence_projection.json`. Read-only Abgleich archivierter R1-/R2-NPZs, gleicher Bild-/Labelpopulationen, R2-Zähler sowie der 55 vorhandenen Energieaggregate; außerdem reine Dokumentzählung der 103 geplanten Gruppen. Keine neue Inferenz, keine neue Kalibrierung, kein wissenschaftlicher PASS.

**[E-2804-CHAT]** Projektunterhaltung vom 8.–12. September 2026: v33 Force-OFF, v34/v2.80-Abnahmen, .3-FIX1–FIX5-Zielausgaben und Hailo10-MobileNet-Build/Runtime. Terminal-/Testzahlen gelten nur für ihren benannten Lauf; Zusammenfassungen ersetzen keine neue Originalprüfung.

**[E-2804-PLAN]** `IMPLEMENTATIONSPLAN_v2.80.4_Hailo8_GPU_Produktivpfad_Quality_Cancel_und_Debuglimits(1).md`, 12. September 2026, AP0–AP6, 56 geplante Prüfanforderungen. Insbesondere §1.3 H8-Fixed16, §5 Q5-Cancel-Soll und §8.2 MobileNet-B5000-Werte. Planrolle und tatsächlich verfügbare Originalnachweise unterscheiden.

**[E-2804-H8-Q2]** `Eingefügter Text(20260912-121049).txt`, tatsächliche H8-Runtime-Konsole; im Bundle als `Q2_terminal_report.json` abgeleitet, Run `hailo8_mobilenet_runtime_8FehkBqO`. Der Quellenindex kennzeichnet separat, ob Q1 `runtime_evidence(1).zip` für unabhängigen Recount verfügbar war.

**[E-2804-QUALITY]** `v2803_yolo11l_quality5000_6yzd81fj.zip`, vollständiger Ergebnisexport `v2803_quality5000_export_zu5ufxbf.zip`; Lauf `v2803_yolo11l_quality_5000_20260912_084434`. Nachgereichter Export bestätigt vollständige kanonische CPU-/Kandidatenvorhersagen, Identitäten und AP-Reproduktion; verändert den ursprünglichen Run nicht.

**[E-2804-ABGLEICH]** `ABGLEICH_Hailo_Qualitaet_2026-09-12.md` und `onnx-splitpoint-results-main(4).zip`: frühere KB `docs/ONNX_SPLITPOINT_KnowledgeBase_v2.79_Three_Stage_2026-09-03_UPDATED_v2.79.16.md`, §3.4, Opt1/Opt2; Paper-Level1-Primärbericht und v29-Complete-Set-Qualität. Opt2-Rohbericht nicht gefunden; kein erfundener direkter YOLO11l-A/B-Test.

**[E-2804-SOFTWARE]** Zugehörige v2.80.4-Lieferung: finaler Source-Manifest-/Archivabgleich, konkrete Software-/Upgradeprüfung und Quellenindex. Diese Dateien bestimmen, welche .4-Tests tatsächlich gelaufen sind. Zielhardware wird dadurch nicht automatisch abgenommen.


## Zusätzliche Quellen der REV2

**[E-KB-2804-INPUT]** Unveränderte hochgeladene `ONNX_SPLITPOINT_KnowledgeBase_CANONICAL_v2.80.4_2026-09-12(1).md`, ursprüngliche 578 Zeilen. Trägt u.a. die neueren .3-FIX5/.4- und separaten YOLO11l-/Opt2-Aussagen. Keine neue Originalprüfung dieser nicht zusätzlich gelieferten Release-/Qualityarchive in REV2.

**[E-HAR-R1]** `har_and_git_evidence.zip`; `results/hailo8/20260912_mobilenet_gpu/har_emulation/hailo8_har_git_20260912T171411Z_x0j2t2o1/`: `comparison.json`, `comparison_request.json`, `REPORT.md`, `CLAIM_BOUNDARIES.md`, `controller_summary.json`, `parsed_native/api_and_model.json`, `quantized/api_and_model.json` und Stufenergebnisse. SDK 3.33.1; HARpfade aus ursprünglichem GPUbuild, Hashes erst beim HAR-R1 erfasst.

**[E-H8-COMPUTE]** Im selben Archiv `results/hailo8/20260912_mobilenet_gpu/compute/`: Paket-/Overlaymanifest, Venvinventare, tatsächliches Computeergebnis und ptxas-Trace. Historischer Lauf `hailo8_gpu_test_20260912T062240Z_hx7b7hcq`.

**[E-H8-BUILD]** Im selben Archiv `.../build_metadata/`: ursprünglicher Request, CPU- und privates GPU-Receipt, Builderargumente, Phasen, eigene GPUaktivität, Compilerlog, Supervision. Lauf `hailo8_mobilenet_gpu_20260912T083456Z_a_10a8wv`. Historischer CPUzeitvergleich zusätzlich im bereits erstellten H8-Modellbuild-Abnahmebericht; kein neuer kontrollierter Zeitbenchmark.

**[E-H8-RUNTIME]** Im selben Archiv `.../runtime/{comparison.json,runtime_request.json,results/runtime_result.json}`; tatsächliche HailoRT-4.20.0-/VStreammetadaten und 32 abgeschlossene Inferenzen. Runtime-Roharrays nur mit Pfad/Hash referenziert, nicht im ZIP. Run `hailo8_mobilenet_runtime_8FehkBqO`.

**[E-NIGHT-QUALITY]** Im selben Archiv `results/evaluation/completsetdev_20260911_213508/quality_snapshot/quality_management/central_quality_summary.json` und sieben Referenzstatus/-logs, dazu alle 123 Requestdateien. 43/18/2 fertige Qualityentscheidungen; 4/56 Cancel-Ausnahmearten. Alte Datei bleibt technisch `failed`; neue Dokumentprojektion beschreibt ihren belegten Cancelkontext ohne Originalumschreibung.

**[E-NIGHT-2803]** `evaluation_workflow(20260912-050559).log`, `evaluation_workflow(20260912-054051).log`, `profile(2).yaml`, `run_manifest(3).json`; bereitgestellte Auswertungen unter `night_current_20260912/`. Beobachteter Lauf `completsetdev_20260911_213508`; aktueller Snapshot und historische Beendigung nicht gleichsetzen.

**[E-FORCE-AUDIT]** `AUSWERTUNG_Force_Ursprung_v27932_2026-09-09.md`, `force_audit_analysis_20260909/REPLAY_FORCE_ORIGIN.json`, ursprüngliche Registry und Verlauf. Ein gespeichertes true belegt keine bewusste Nutzeraktion.

**[E-GPU-HISTORY]** Ursprüngliche GPU-/Toolchainarchive und die im Chat erzeugten Abnahmen: `AUSWERTUNG_Hailo_GPU_Rechensmoke_R1_2026-09-09.md`, `AUSWERTUNG_Hailo_GPU_FIX1_und_v27933_2026-09-09.md`, `AUSWERTUNG_Hailo_Toolchain_2026-09-09.md`, Hailo10-XLA-R2- und v34/v2.80-Build-/G3-Abnahmen. Historische Familientests, kein neuer .4-Hardwarelauf.

**[E-RELEASE-AUDITS]** `ABNAHME_v2.79.31_Implementierung_REV4_2026-09-08.md`, `ABNAHME_v2.79.34_Hailo_GPU_und_Artefaktvorbereitung_2026-09-09.md`, `ABNAHME_v2.80_Hailo_Environment_Cleanup_und_Reuse_2026-09-09.md` sowie `v2801_target_review_20260910/`-Zielabnahme. Eigene vs. gelieferte Tests, Software-/Hardware-Scope und fehlende Auditabhängigkeiten bleiben wie dort dokumentiert.

**[E-WORKFLOW-HISTORY]** `biggerset_v280_analysis_20260910/`, `completsetdev_v2801_review_20260910/` und `review_v2802_overnight_20260911/`: Originalfehlerquellen, zugehörige Reports und gezielte Replays. Keine nachträgliche Messfreigabe aus einem Reportfix.

**[E-REUSE-TARGET]** `v280_target_review_20260910/independent_review.json` und die zwei originalen `v280_hailo_reuse_*`-Archive. Vier frische Controllerprozesse für genau den gebundenen MobileNet-H10-Request.

**[E-QUALITY-EXPLAIN]** `quality_explanation_20260912/AUSWERTUNG_Qualitaetsauftraege_und_Debugexport_2026-09-12.md`, damals gegen Run-Modulprüfsummen abgeglichene `quality_service.py`/`quality_metrics.py` und der reale Cancel-Snapshot. Neue .4-Implementierung in diesem KB-Update nicht neu ausgeführt.

**[E-GIT-PUSH]** `Eingefügter Text(20260912-171533).txt`, finale Pushbestätigung und Commit-ID; inhaltlich zugehöriger 236-Dateien-Payload `har_and_git_evidence.zip`. Keine automatische Aussage über den späteren Remote-HEAD.

**[E-DIAGNOSE-PLAN]** Früherer Begleitplan `DIAGNOSEPLAN_Hailo8_Optimierung_Quantisierung_Runtime_2026-09-12.md`. Vorgeschlagene zusätzliche Stufe `SDK_FP_OPTIMIZED` mit lokaler Verfügbarkeitsprüfung, versionsgebundene Integer-I/O-Kontrolle und bedingte innere Diagnose. Nicht implementiert oder ausgeführt. **Seit REV3 ausdrücklich zurückgestellte optionale Folgearbeit; keine aktive Abnahme-/Dissertationspflicht.** Die Originaldatei bleibt unverändert; aktuelle Priorisierung nach [E-SCOPE-REV3] und §12.

**[E-HAILO-OPT-DOC]** Externe Herstellerbeschreibung, Hailo Model Zoo `docs/OPTIMIZATION.rst`, Introduction und Optimization Workflow; abgerufen am 12.09.2026. Belegt die begriffliche Trennung Full-Precision- und Quantisierungsoptimierung, keine konkrete Ursache unseres Modells und keine API-Freigabe für DFC 3.33.1. URL zur Quellenauflösung:

```text
https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/OPTIMIZATION.rst
```

**[E-KB-REV2-CHECK]** Im vorherigen REV2-Update erzeugtes `evidence/SOURCE_AUDIT.json`, `GIT_STATUS.json`, `QUALITY_SNAPSHOT_COUNTS.json` und `DOCUMENT_CHECKS.json`. Read-only Archiv-/Text-/Klassen-/Dokumentabgleich; keine DFC-/ORT-/Hardwareausführung und keine Repositoryänderung. Begleitdiff dokumentiert sämtliche Änderungen zur unveränderten Eingabe-KB.


## Zusätzliche Quellen und Entscheidungen der REV3

**[E-KB-REV2-INPUT]** Im damaligen REV3-Auftrag beigefügte `ONNX_SPLITPOINT_KnowledgeBase_CANONICAL_v2.80.4_2026-09-12_REV2(1).md`, 845 Zeilen. Unverändert erhalten; vollständige Textgrundlage dieser Fortschreibung. Historische Mess- und Releaseaussagen werden daraus übernommen, nicht als neu ausgeführte Prüfungen bezeichnet.

**[E-SCOPE-REV3]** Unmittelbar vorausgehende Projektdiskussion und anschließender Nutzerauftrag zur Aufnahme in die KB: Forschungsfrage „wie gut funktioniert das Splitten eines Netzwerks einfach so mit meinem Tool“, Einwand gegen tiefe Einzelnetzoptimierung, Frage nach Aufwand und zulässiger Tuning-Future-Work-Aussage. Dokumentiert wird die danach bestätigte Abgrenzung: festes automatisches Verfahren bewerten; korrekte Negativergebnisse erhalten; eigene Fehler nicht dem Compiler zuschreiben; vertiefte Hailo8-Ursachenklärung zurückstellen; keine Verbesserung aller Netze garantieren. Die genannten Aufwandsbereiche sind grobe Chat-Planungsschätzungen, keine Messdaten. Dokumentiert am 13. September 2026; kein nachträglicher experimenteller Nachweis.

**[E-KB-REV3-CHECK]** `REV3_DOCUMENT_CHECKS.json`, `REV2_to_REV3.diff` und `AENDERUNGEN_KB_v2.80.4_REV3_2026-09-13.md` im REV3-Begleitpaket. Prüfen ausschließlich die Textfortschreibung, den unveränderten Eingabestand, bestehende Ergebnisabschnitte, Überschriften, Verweise und die konsistente Zurückstellung. Keine ONNX-/DFC-/Runtime-/Hardwareausführung, kein Produktpatch und kein Gitpush.

## Zusätzliche Quellen der v2.81

**[E-281-BASE]** Wiederhergestelltes originales `ONNX-Splitpoint-Tool_v2.80.4_COMPLETE_DELIVERY_BUNDLE.zip`, SHA256 `47c7a1e56ae9d3d094b5dcc55dc49aaa489e0e7b83e89aebf465f13c69ad750a`; eingebettetes Sourcearchiv SHA256 `23e98c6ce3a4fa8ce42005ba02ea2e189dddafc56b0e6dd3f7bc3470ee8f2527`. Sämtliche 1.894 Quelldateien und bestehendes Manifest unabhängig geprüft. Vollständige REV3 ist die Dokumentbasis, kein älterer KB-Ersatz.

**[E-281-NIGHT]** `completsetdev_20260912_213922_debug_pack.zip`, SHA256 `6c4cbe46bd794b5eb328acba0e8f962d097bdd7ae87970dfc7aa1108f1d80e75`; originale Native-/Quality-/Energieresultate und Compilerlogs. Vorherige .4-Zielabnahme laut Nutzerkonsole `v2804_install_and_accept_wr2fqm7i.zip`, kombinierter PASS mit Quality FAIL, SHA256 `9f6d18dfb6289cf052a984e9159766947e0b294cdf47c38fa808842601ea7fd5`.

**[E-281-SMOKE]** `v2804_night_smoke_r1_x12jzgo2.zip`, SHA256 `30ee88431ac0ec7e0697f1df17b911e57fee2461968917dd5f517ec649d1f2ac`; vier Diagnosebereiche Compiler, exact Compile-Infeasible, H10-Splits und H8-Imageparität. Primärdateien, rohe H10-Outputs und Herkunft in den v2.81-Regressionsfixtures erhalten. Keine Builds und keine Energieausführung. Die neue Releaseverifikation dokumentiert tatsächliche lokale Gegenproben und den gesonderten Hardware-Scope.
