# ONNX Splitpoint – fortlaufender Arbeitsstand

## R9J – abschließende lokale Übergangsprüfung nach HOST-A5 (20.09.2026)

A5 ist historisch vom HOST akzeptiert; B2 bleibt historisch negativ. Die bereits
umgesetzten B2-Korrekturen sind durch Original-Dateiimport und echte
Dispatch-Argumentbildung bis zur kontrollierten Remotegrenze bestätigt: vier
Hailo-Fullzeilen behalten Completion und Accuracyverlust, H8 erhält die gebundenen
TRT-Cases. Vier nie gemessene TRT-Ergebnisse bleiben fehlend.

Zusätzliche belegte Negativlücke: Die Normalisierung überschrieb explizit negative
Completionfelder und ignorierte widersprüchliche Full-Completion-Aliase sowie den
vorhandenen gebundenen Hosttail-Hash. Eine gemeinsame Prüfung vor beiden
Projektionen weist diese Konflikte jetzt ab. Nicht erforderliche Legacy-Hosttails
bei Classification/integrierten Outputs bleiben N/A; keine neuen Hashes/Decoder.
15 neue Übergangsfälle; acht Negativfälle vor Fix FAIL. Erster erweiterter Vorlauf
deckte eine zu strikte N/A-Prüfung und zwei unzutreffende Fixture-Erwartungen auf;
beides korrigiert, sämtliche Vorläufe/JUnit bleiben erhalten.

Software: sechs vereinbarte Dateien **250 PASS**, ein echter Tk-Test wegen
GUI-Startverbot deselected; Consumer/Mirrors zusätzlich **25 PASS**. P1/P2/P3,
tatsächlicher ONNX-/DXNN-Graph, SDK3.3.2 und effektive A/B/EVAL-Scopes geprüft.
Belege: Fortsetzungsordner/local_ertznmyv. Policy, Profile, Guards und Originalruns
unverändert; Bilder/GT/Nenner und pfadspezifische Confidence bleiben vollständig.

Keine eigene Hardwareausführung. Gemeinsame Sourceänderung invalidiert A5
konservativ; für diesen Source keine Hardwareclaims. A1–A5/B1–B2 verbraucht,
lokale Runde kumulativ sechs von sechs; keine zusätzliche Startfreigabe und kein
Budgetreset. EVAL ohne aktuelle A/B-Abnahmen gesperrt. HOST kann den begrenzten
Auftrag mit offenem Hardware-/EVAL-Nachweis abschließen. Langzeit-, Ranking- und
historische Ursachenlimits bleiben bestehen.

## R9J – lokale Korrektur nach HOST-B2 (20.09.2026)

B2: vier generische Hailo-Fullzeilen besitzen gemessene Decoderabschlüsse und
zentrale Quality, wurden aber vom ausschließlich Native-verstehenden
Hostabschlussprüfer abgewiesen. Der gemeinsame Prüfer liest jetzt den bestehenden
generischen Timerbeleg samt verifiziertem Frozen-Vertrag, Modell, Producer,
Fullrolle, Framezahl und Aliaswidersprüchen. Keine Änderung der Decoder,
Confidence, Bilder/GT oder Accuracyentscheidung; die vier Accuracyverluste bleiben.

Zweite Ursache: TensorRT war im Modellplan an H8 gebunden, der Dispatcher wählte
wegen der Target-Reihenfolge H10. H8 übersprang Performance als Quality-only;
H10 hatte keine passenden Cases. Der Dispatcher übernimmt jetzt die vorhandene
Planbindung; widersprüchliche oder fremde Setups sperren vor Ausführung. Fehlende
Messungen bleiben fehlend. Keine Nativewerte als Genericersatz.

Software: 235 PASS, ein echter Tk-Test HOST vorbehalten; zusätzlich25 PASS für
Dispatcher, Native-Hostabschluss und Mirrors. Neue B2-Übergänge:20 PASS, darunter
vier Originalzeilen und echter Timer/Decoder mit simulierter SDK-Grenze. Vor Fix
17 der ersten19 neuen Fälle FAIL; zwei Reihenfolgevarianten bereits PASS.
DeepX-ONNX/DXNN-TopK und SDK3.3.2 erneut lokal geprüft; bisherige P1/P2/P3-Tests
grün. Installed-Prüfung und genaue Befehle im Fortsetzungsordner/local_rmbggtax.

Hardware: keine eigenen Starts. A4 war auf vorigem Source akzeptiert; dieser
gemeinsame Codewechsel verlangt konservativ neue A/B-Abnahmen. A1–A4/B1–B2
bleiben verbraucht, B hat keinen Start mehr. EVAL ohne aktuelle A/B gesperrt;
kein Budgetreset. Guards, Policy, Profile, HOSTsteuerung und Originalruns bleiben
unverändert. Wissenschaftliche Langzeit-/Ranking-/historische Grenzen bleiben.

## R9J – lokale Korrektur nach HOST-A3/B1 (20.09.2026)

A3 und B1 bleiben technisch negativ. Alle sieben Modell-/Setup-Logs belegen
`runtime_endpoint_identity_hash_mismatch` im vorgeschalteten TRT-Qualityexport;
die fehlenden Generic-Zeilen und Native-Qualitybindungen sind Folgefehler.
Die gemeinsame bestehende V3-Projektion wird jetzt auch von beiden Exportern
verwendet, einschließlich Graphmetadaten und Classification-Signatur. Die
unabhängige Hashprüfung bleibt bestehen. DeepX-TopK-Kandidaten werden durch den
gebundenen Graphvertrag erkannt und nicht mehr wegen fehlender NMS als Raw-Head
exportiert. Auswahl, Confidence, Bilder/GT, Policy und Guards bleiben unverändert.

Software: acht neue Reproduktionen vor Korrektur FAIL, danach PASS. Weitere
Regressionen prüfen echte Graph-/Requestübergänge, negative Bindungen und den
gemeinsamen Perf-/Energie-/32er-Qualitätsendpunkt. Exakte Abschlusszahlen und
Installed-Verifikation: `local_5gbcllbi` im Fortsetzungsordner. Graph/SDK lokal
erneut gelesen; proprietäre niederkonfidente Arithmetik nicht als repariert behauptet.
Die HOST-Berichte enthalten keine UNKNOWN-IDs. Keine eigenen Geräte-/GUIstarts.
HOST-Abnahme offen: B2 unabhängig von A, dann A4; EVAL erst nach aktuellen A/B.
A1–A3/B1 und bisherige lokale Runden bleiben verbraucht; keine Budgetrücksetzung.

## R9J – gezielte Fortsetzung nach A1/A2 (20.09.2026)

LOKAL_GEPRUEFT: Native-Dump und Hailo-Smoke konstruieren denselben V2-Vertrag
wie Performance; historische V1-Belege bleiben V1. Gebundener ONNX-Graph und
CPU-Teil des vorhandenen DXNN (Compiler 2.3.0-rc.5) belegen feste TopK-Kandidaten
vor Consumer-Confidence, ohne NMS. Die bisherige Schwelle bleibt unverändert
(hier auch kanonische A2-Referenz: 0.25). Sechs rohe Inversionen werden erfasst;
übernommene Boxen bleiben strikt geprüft, ohne geometrische Reparatur.
Full-Matrix bindet die Ausführungsrolle getrennt vom Completed-Endpunkt.
Unsicherheit wird zusätzlich zur Accuracy gezählt; Sentinelabschluss prüft
Request, Candidatebytes, 32 beobachtete Records und kanonische Referenz je Consumer.

Software: 236 Tests bestanden; echter Tk-Test für HOST ausgespart. Die 32er-Kette
verwendet echte Bilder/GT und vier archivierte SDK-Tensoren (zwei Kontrollen
wiederverwendet), echten Export, zentralen Evaluator und unveränderten Hostguard.
Sie ist kein Nachweis von 32 neuen Hardwareinferenzen. A2-Replay: sechs Fullzeilen
erkannt; 13 referenznah, ein Verlust, zwei überlappend unsicher, 14 completed.
Ein zusätzlicher Legacy-BN6-Test scheitert schon am Fortsetzungseinstieg unverändert.
Belege: Fortsetzungsordner `v283_R9J_Fortsetzung_20260920_103442_70_d56qj`,
`local_qek_0l39/final.xml`, `graph_sdk_audit.json`, `legacy_test_comparison.json`.
GUI_ABNAHME OFFEN: keine Geräte-/GUI-/Energie-/Compilerstarts durch Codex.
HOST prüft zuerst B, dann A; EVAL erst nach aktuellen A/B. A1/A2 und Budgets
bleiben unverändert; alte Hardwarebelege geben diesen Quellstand nicht frei.
Stationarität, Ein-Split-Ranking, Paper-b066 und historische Crashursachen offen.

## R9J – Reparatur nach realem HOST-A-Fehler (19.09.2026)

Accuracy-Reporting v1, gepaarte relative95%-Intervalle und offizielle COCO-AP sind
implementiert; Loss ist Klasse/Warnung, technische Gates bleiben verbindlich.
Generische Fullproducer führen Taskabschluss im gemessenen Callback aus und
reichen Counter/Phasenzeiten weiter. A/B/EVAL sind normal geladen und scopegeprüft;
A enthält beide unveränderten Sentinelbilder plus30 deterministische Kontrollen.
Softwarebelege und genaue Dateiliste stehen im R9J-Ordner `CLOSURE.json` und
`local_4bu6zygs/final.xml`; der erste reale HOST-A-Lauf ist technisch negativ.
Keine neue Sourcefreigabe aus R9I-Hardwarebelegen. Archivierte finale DeepX-Boxen
bleiben technisch ungültig; Ursache nicht als behoben erklärt. R9H-Ranking hat
keine passenden generischen Completed-Task-Zeiten; b066 bleibt historisch eng gebunden.

HOST-A `r9j_a_20260919_220725`:7/9 Nativezeilen,21/21 gestartete Energiereplikate
gültig,4 erforderliche Qualitätsresultate fehlen. Der neue Full-Detection-Aufruf
verletzt die keyword-only-Signatur des Preprocessingresolvers und stoppt YOLO26s
vor Quality-Request/Native-Bindung. Aufruf korrigiert; Regression umfasst nun auch
Initialisierung, echten Decoder und Timercallback. Zusätzlich wählen generische
BN6-Fullpfade den vorhandenen strikten Materialisierer ohne zweite NMS (expliziter
v2-Vertrag); historische v1-Verträge bleiben lesbar. Kein Filter ungültiger finaler
Boxen. CPU-Replay mit archivierten Antworten verarbeitet32 Eingaben, meldet beide
Fehlbilder technisch negativ und30 gültige Antworten; kein realer Bild-PASS.
Neue lokale Tests/Installedprüfung: `local_na5w3dkp`; aktuelle Hardwareabnahme der
Reparatur ausschließlich durch HOST ausstehend. Keine bestandene R9J-Zelle vorhanden.


Stand: 17.09.2026, R9C plus einmaliger H10-Nachtest abgeschlossen: ursprüngliche R9C-Matrix unverändert 8/9 Native-Zeilen technisch akzeptiert, 9/9 Timingreihen. Zusätzliche normale H10H/MobileNet-b135-GUI-Nachabnahme des Capturefixes 3/3 technisch PASS, je100 Top-k-Paare; Quality FAIL bleibt erhalten. Historische finale Software 1012/1012 und Tk29/29 PASS; neue direkte Regression1/1 PASS, Produktcode unverändert.
Historische R8-Softwaregegenprüfung:678/678 Regressionen und6/6 echte Tk-Tests PASS. Historische Einträge behalten ihren damaligen Scope; die spätere Ausgabekorrektur wurde nicht erneut physisch ausgeführt.

## Priorität

Zuerst normaler GUI-Workflow, Collector-/Schutzintegration, Modusbudgets und lesbare Fortschritte. Qualitätsoptimierung zurückgestellt. Gültige negative Qualitätsresultate erhalten; technisch ungültige Outputs nicht als gültig ausgeben. Kein neuer Full-/Final-/Night-Run in R8.

## Statusdefinitionen

OFFEN / IN_ARBEIT / IMPLEMENTIERT / LOKAL_GEPRUEFT / GUI_ABGENOMMEN / BLOCKIERT / ZURUECKGESTELLT. Historische ABGESCHLOSSEN-Einträge nennen ihren Scope. „Getestet“ ohne Angabe des verwendeten Einstiegspunkts reicht nicht.

## Jetzt bearbeiten

| ID | Thema | Aktueller R8-Status | Beleg / konkrete Abschlussbedingung |
|---|---|---|---|
| GUI-01 | Tests bilden normalen GUI-Ablauf ab | GUI_ABGENOMMEN | Normales start_gui.sh → Profil → Queue → echte Prozesse → Reports → Qualitätswarndialog; H8-MobileNet-b135, drei Nativezeilen. Tk-/Negativtests ergänzen die physische Abnahme. |
| EN-01 | Abgenommenen Collector stabil normal bereitstellen | GUI_ABGENOMMEN | Stabiler R6-Collector über normale Registry; neun tatsächliche Spawns mit identischer SHA256, Protokollende jeweils bestätigt. |
| EN-02 | Quellen-/Retrypolicy in normalen GUI-Läufen | LOKAL_GEPRUEFT + GUI_ABGENOMMEN | Fehlende Policy normal aufgelöst; persistenter Quellenstopp lokal über mehrere Zeilen/Probe-Leaf. Physischer Smoke: ausdrücklich Retry0, neun Erstaufnahmen. |
| EN-03 | Kleine Energieabnahme über normalen GUI-Pfad | GUI_ABGENOMMEN | r8_gui_smoke_mobilenet_h8_20260915_200453: drei vollständige Energiezeilen, je3/3, Collector9 gestartet/0 fehlgeschlagen, kein Retry. |
| MODE-01 | Gesamtmodus steuert native UND generische Budgets | LOKAL_GEPRUEFT + GUI_ABGENOMMEN | Native Standard100/10/1, Final1000/100/3; normale Config eng migriert. Tk Save/Reload/Custom/Frozen und tatsächlicher Standard-Smoke. |
| MODE-02 | Summary-/Spawn-Wiederholungswiderspruch | GUI_ABGENOMMEN | Snapshot, reale Full/Split-Kommandos und drei Reports: Performance100/10/1; separate Energie3 je Zeile. |
| LOG-01 | Vollbefehle und Roh-JSON aus Normalausgabe nehmen | IMPLEMENTIERT + LOKAL_GEPRUEFT | Kurze GUI-Ereignisse und getrennte vollständige redigierte Diagnosen. Nach Smoke zusätzlicher Drain-Fix gegen langsame Handler/offene Pipe; gezielt35/35 und finale678/678 PASS. Keine zweite physische Ausführung. |
| LOG-02 | Native-Full-Fortschritt verständlich | GUI_ABGENOMMEN | Reale Full-Fortschritte nennen Modell, Setup/Backend, Rep1/1, 100 Frames +10 Warmup und Phase. |
| REPORT-01 | Runtime, Vertrag, Qualität, Energie getrennt darstellen | GUI_ABGENOMMEN | Runtime/Vertrag vollständig, Qualität FAIL separat, Energie3/3 Zeilen und9/9 Replikate, Artefaktabschluss PASS; Dialog Abgeschlossen mit Qualitätswarnungen. |
| RELEASE-01 | Finale Hauptinstallation und normale Config gemeinsam geprüft | LOKAL_GEPRUEFT + GUI_ABGENOMMEN | Hauptrepo-Imports und normale Config in echter GUI belegt; finale678 Regressionen +6 Tk-Tests PASS, Source-/Installed-PASS. Nachgelagerter Drainfix nur lokal erneut geprüft; normaler GUI-Neustart bestätigt. |
| LAT-01 | Klassifikation bis Top-1/Top-5 und echte Requestlatenz | IMPLEMENTIERT + LOKAL_GEPRUEFT; MOBILE_NET_GUI_SCOPE_ABGENOMMEN | R9C: H8/H10/DeepX Full/Split und vorhandener TRT-Prepared-Hotloop, alle9 Timingreihen; ursprüngliche Matrix bleibt8/9 technisch. Separater H10H/MobileNet-b135-Nachtest3/3 technisch PASS,100/100 Paare je Zeile; Capturemanifeste, Consumer-Join und GUI-Cleanup bestätigt. Qualitäts-FAILs bleiben bestehen; keine allgemeine Klassifikationsmodellfreigabe. |

## Im dokumentierten Umfang abgeschlossen – erhalten

| ID | Thema | Status / Scope | Beleg und verbleibende Grenze |
|---|---|---|---|
| H8-01 | Wiederholungs-/Nachweisweitergabe | ABGESCHLOSSEN für YOLOv7 b044 und YOLO11l b062 | R2 unabhängige Referenz, R6/R7-Regressionen, aktueller GUI-Lauf erneut gültiger nativer Vertrag. Kein allgemeiner Quality-PASS aller Modelle. |
| REPORT-02 | Zentrale Ausschluss-/Missing-Zählung | ABGESCHLOSSEN im aktuellen GUI-Lauf |56 primär:54 ausgewertet,2 Buildausschlüsse,0 fehlend. |
| CACHE-01 | Artefaktwiederverwendung / negative H8-Buildbelege | ABGESCHLOSSEN für aktuellen warmen Lauf |26 Hailo-Buildresultate Hit/Compilerdispatch0; bekannte H8-YOLO26-Ausschlüsse. Keine Kaltbuildabnahme daraus ableiten. |
| CLEANUP-01 | Terminaler Fehler→Management-Shutdown | IMPLEMENTIERT + LOKAL_GEPRUEFT; normaler GUI-Abschluss gelungen | R7 lokale Fehler-/Cancel-/Quarantänetests; letzter GUI-Lauf regulär abgeschlossen. Historische Remote-rc124-Ursache offen. |
| RELEASE-02 | Installed-Extras/Sourceumfang | ABGESCHLOSSEN im R7-Umfang | R7 Release-/Installed-PASS; nach neuen Dateien erneut prüfen. |
| EN-04 | Student-t-Intervallfaktor | IMPLEMENTIERT + LOKAL_GEPRUEFT | R1/R3-Regressions-/numerische Gegenprüfungen; kein Ersatz für reale Energieaufnahme. |
| H10-01 | Diagnose-Bindingresolver | IMPLEMENTIERT + LOKAL/GEZIELT GEPRUEFT | R2 Schemafilter für verschachtelten Beweis; Capture durchgeführt. Das repariert NICHT H10-02. |

## Bekannt offen, jetzt zurückgestellt oder beobachten

| ID | Thema | Status | Grenze / Wiederaufnahme |
|---|---|---|---|
| H10-02 | YOLO26m b398 / YOLO26s b364 Nulloutputs | R9C_OFFLINE_GEPRUEFT – NICHT REPARIERT | Vorhandene native UINT8-Outputs bereits mit Nullpunktscores; aktuelle Adapter +4 ORT-P2-Replays reproduzieren sie. Kein eigener Hostfix belegt, kein neuer HEF/Output-GUI-Lauf. Historische500/500-Leeroutputs nicht als neue Vollprüfung ausgeben. |
| DX-01 | YOLO26s Full, zwei XYXY-Semantikfehler | R9C_OFFLINE_GEPRUEFT – NICHT REPARIERT | Vier gespeicherte Outputs durch aktuelle Kette:052891/395801 bereits nativ invertiert,139/285 gültig; Exportsperre bestätigt. Keine Paddingregel/Adapterursache belegt, kein neuer DXNN/Output-GUI-Lauf. |
| QUALITY-01 | MobileNet-/Detektor-AP-/Accuracyverluste | ZURUECKGESTELLT nach Nutzerentscheidung | Werte/Gates erhalten; keine Qualitätsoptimierung bis stabiler Workflow. |
| TRANSPORT-01 | Historische rc124-Speicherprobe | BEOBACHTEN | Ursache nicht bewiesen; aktuelle Runde ohne diesen Abbruch. R7 Diagnostik erhalten. |
| EN-05 | Historischer R5-Transport-/Endpaketausfall | BEOBACHTEN | R6 funktionsfähig im Smoke; Ursache historisch offen. Neue normal eingebundene Ausfälle mit neuen Diagnosen bearbeiten. |
| SCI-01 | Finale Langzeitenergie / Ranking mit mehreren Splits | ZURUECKGESTELLT |1-Split-Screening bleibt kein Ranking-/Langzeitnachweis. Kein Scopewechsel in R8. |

## Änderungsjournal

| Datum / Runde | ID | Alter → neuer Status | Beleg / geprüfter Scope / Restgrenze |
|---|---|---|---|
|2026-09-15 / vor R8|alle|Bestandsaufnahme|Vorliegende R2/R6/R7-Reports und GUI-Lauf124548. Keine neue Umsetzung behauptet.|

Bei jedem Ergebnis ergänzen, nicht Einträge löschen. Neue Teilprobleme unter eigener Unter-ID führen. Private konkrete Pfade/Configänderungen in lokalen Testberichten referenzieren; keine Secrets in dieser versionierbaren Übersicht.

|2026-09-15 / R8 Start|EN-01, EN-02, MODE-01, MODE-02|OFFEN → IN_ARBEIT|Hauptrepo c5eb66e, R1–R7 gesichert; Hostpreflight grün, echter Tk-Zugriff bestätigt. Noch kein Hardwarestart.|

|2026-09-15 / R8 A|EN-01, EN-02|IN_ARBEIT → IMPLEMENTIERT|Verifizierte stabile Collectorkopie, normale zentrale Bindung mit Backup/Locks/ACL-Erhalt; absoluter Startsnapshot, gemeinsame Native-/Probe-Registry; fehlende Policy im Resolver. Produkt-GUI-Abnahme weiterhin offen.|
|2026-09-15 / R8 B|MODE-01, MODE-02|IN_ARBEIT → IMPLEMENTIERT|Schema14 migriert nur Standard1000/100/3; Standard100/10/1, Final1000/100/3, explizite Herkunft und echte Editorvariablen. Fünf Modusregressionen bestanden, Tk-Matrix noch offen.|
|2026-09-15 / R8 C/D|LOG-01, LOG-02, REPORT-01|OFFEN → IMPLEMENTIERT|Separate redigierte Native-/SSH-Diagnosen; kurze Fortschritte; Zeilen, logische Replikate und Collectorversuche getrennt. Zwölf neue lokale Tests bestanden; keine Hardwareabnahme.|

## R8 – Arbeitspaket E, lokale Abnahme (2026-09-15)

- Build `v2.83-r8-gui-product-integration`, Hauptrepo: 670/670 Regressionen bestanden, alle 647 R7-Knoten enthalten (`acceptance_r8_second.json`). Echte Tk-Modus-/Save-/Reloadmatrix: 3/3 bestanden (`junit_r8_tk4.xml`). Hardware wurde noch nicht gestartet.
- Gegenprüfung Logging: Bearer-Redaktion, vollständige getrennte SSH-Ausgabe und Abschlussdiagnose bei ungeklärtem Cleanup nachgebessert; zusätzliche Fehler-/Paralleltests werden erneut ausgeführt.
- Echte Queue-/Runanlage/CPU-Subprozesse und Reportabschluss wurden erreicht. Der erste Testaufbau wartete am echten modalen Dialog; Dialogsteuerung wurde nach regulärer App-Initialisierung korrigiert. Kein Workflow-PASS daraus.
- R8-GUI-CPU.1: BLOCKIERT als positiver CPU-Full/Split-Test. Bestehende physische Scopeauflösung verlangt Accelerator-Setup und ergänzt TensorRT; lokale CPU-only-Auswahl wird technisch abgewiesen. Kein Profiltrick und keine Erweiterung des Architekturauftrags. Der negative GUI-Abschluss wird als solcher getestet.
- Erweiterte Tk-Gegenfälle (Full/Split, Frozen Resume, veraltete Summary, falscher/fehlender Collector, persistenter Quellenstopp mit lokalen Prozessfixtures) in Prüfung. Ein Wrapper-/Leaf-Test ist keine vollständige Window-Probe-Coordinatorabnahme.
- Normale GUI-Einstellungen nach kontrolliertem Schließen des ersten Testfensters byteidentisch aus vorab gesicherter Datei wiederhergestellt. Originalprofile und Originalruns unverändert.

### R8 – Stufe 2 und Vorbereitung des einzigen Hardware-Smokes

- 2026-09-15, Build R8: fünf Tk-Matrixtests plus ein echter negativer GUI-Queue-/CPU-Prozess-/Report-/Terminaltest bestanden (5 PASS in `junit_r8_gui_fullmatrix.xml` plus korrigierter Tab-Lazyload-Test 1 PASS in `junit_r8_gui_source2.xml`). 16 gezielte R8-Softwaretests bestanden. Der negative Workflow weist die bestehende lokale CPU-Scopebarriere ausdrücklich als technischen Fehler aus; sein Artefaktabschluss ist PASS, sein Workflowstatus bleibt FAILED.
- Collector-/Modus-/Loggingintegration damit LOKAL_GEPRUEFT. Quellenstopp belegt an echten lokalen Prozessfixtures über Full/Split/Probe-Leaf und Reentrance; kein neuer vollständiger Probe-Coordinator-Nachweis.
- Ein eigenes normales Smokeprofil `profiles/r8_gui_smoke_mobilenet_h8.yaml` vorbereitet: MobileNetV3, H8-Setup, b135, Full H8 und lokales TRT Full, Standard100/10/1, Energie3 je Zeile, keine Retries, Windowprobe aus. Es erbt den normalen Collector; kein privater Registry-/Collectorpfad. Alte materialisierte Multi-Setup-Felder wurden nur in dieser eigenen Profilkopie auf H8 begrenzt.
- Historische passende Cachebeobachtungen: 4 HIT, keine MISS/UNKNOWN für genau diesen Umfang. `ort_tensorrt.variants:[full]` vermeidet zusätzliche generische TRT-Splitanforderungen. Aktuelle normale `require_warm_cache:true`-Barriere bleibt vor Dispatch erforderlich; kein Coldbuild genehmigt.
- Echte physische Abnahme weiterhin IN_ARBEIT, bis Mess-/Cleanup-/Abschlussbelege vorliegen. H10-/DeepX-Numerik bleibt zurückgestellt.

### R8 – einzige reale GUI-Abnahme und nachgelagerte Ausgabekorrektur

- Normaler Start `./start_gui.sh` aus Hauptrepo, einmaliger GUI-Start am 15.09.2026 20:04:14, terminal 20:17:15 (ca.13 Minuten). MobileNetV3Large/b135, H8-Setup, Full H8, setup-lokales Full TensorRT und Split H8→TRT. Warmmatrix4 HIT, kein Compilerdispatch. Keine zweite Hardwareausführung.
- Drei Nativezeilen jeweils100/10/1, neun Energie-Erstaufnahmen, neun gültige Endprotokolle, keine Transportfehler oder Retries. Runtime und semantischer Vertrag vollständig, Artefaktabschluss PASS. Zwei von vier zentralen Qualitätsauswertungen FAIL bei unveränderten Grenzen: gültiger Qualitätsbefund, kein Workflowfehler. Energie bleibt kurzes Screening.
- Nach diesem abgeschlossenen physischen Lauf zeigte die verbreiterte Softwareprüfung bei langsamer Diagnosedatei einen bestehenden Drainfehler: die 250-ms-Waisenschutzfrist schnitt nach Prozessende noch gepufferte Zeilen ab. Native-/SSH-Reader leeren nun abgeschlossene endliche Queues vollständig; offene geerbte Pipes bleiben zeitlich begrenzt. Observer-Nachlauf löst keinen Prozess-Timeout aus. Gezielt33 Tests bestanden, vollständige Gegenprüfung folgt. Diese letzte Korrektur wurde nicht erneut physisch ausgeführt.
- Das Abnahmefenster wurde regulär über eigene WM_DELETE_WINDOW-Ereignisse geschlossen; ursprüngliche GUI-Einstellungen danach byteidentisch unter Produktlock wiederhergestellt. Der Launcher-Shell-Abbruch war kein Nachweis des Fensterschlusses und wird nicht als Workflowstatus gewertet.

- Reviewnachtrag LOG-01/CLEANUP-01: Offene geerbte Ausgabepipe nach erfolgreichem Elternprozess meldet jetzt `pipe_drain_incomplete`/rc70 und erzwingt den vorhandenen Fehler-Cleanup; zwei reale lokale Prozessregressionen sichern den Randfall ab. Keine neuen Hardwareaufrufe.

### R8 – finale Regression, Testfixture-Provenanz

- Echte finale Tk-Matrix inklusive negativem Queue-/CPU-Prozess-/Report-/Terminalpfad:6/6 PASS (273,96s). Normale GUI erneut über start_gui.sh gestartet, CompleteSetDev und ursprünglicher Ergebnisordner wieder sichtbar, kein neuer Workflow. Sichtbare Summary100/10/1, Energie3, Produktbudget und verwalteter R6-Collector bestätigt.
- R8-TEST.2: Zwei Gesamtprüfungen678 Knoten zeigten jeweils674 PASS/4 FAIL an der unveränderten10s-Schranke alter Terminalfixtures; dieselben vier Fälle einzeln4 PASS. Ursache: nur runner.package_build_snapshot war synthetisch ersetzt, environment_snapshot rief weiterhin artifacts.package_build_snapshot auf. Wiederholte volle Provenanzzugriffe auf den auf2,9MB gewachsenen gemeinsamen Testhashcache verzögerten den Eintritt vor der Finalisierung. Keine Pipe-/Cleanup-Blockade. Der bestehende Provenanzstub wird auch am zweiten Einstieg vollständig angewendet; echte Runner-/Queue-/Lock-/Abschlussprüfungen und alle647 Knoten erhalten. Kein Cache gelöscht, keine Zeitgrenze erhöht, keine Produktlogik geändert. Finale Regression nach diesem Fixturefix läuft.

### R8 – Abschluss (15.09.2026)

- Hauptrepo installiert: Version2.83 / v2.83-r8-gui-product-integration; finale Source-/Installedprüfung PASS, reale GUI-/Core-Imports im Hauptrepo. Normale Config aktiviert und Neustart mit unverändertem CompleteSetDev-Profil/Ergebnisordner belegt.
- Software:678/678 PASS,0 FAIL,0 SKIP,476,70s; alle647 R7-Knoten erhalten (`acceptance_r8_complete.json/xml`). Echte separate Tk-Matrix6/6 PASS,273,96s. Gezielte Native-/SSH-Drain-/Cleanupregression35/35 PASS; vorhandene Terminaltests nach vollständigem Provenanzstub12/12 PASS mit demselben gewachsenen Testcache. Keine Zeitgrenze erhöht, kein Cache gelöscht.
- Reale GUI: genau ein H8-MobileNet/b135-Workflow, drei Nativezeilen100/10/1, Energie3/3 Zeilen und9/9 Erstaufnahmen; Runtime/Vertrag vollständig, Qualität FAIL separat, Artefaktabschluss PASS. Letzte Änderungen an native_progress und SSH-Ausgabepfad entstanden nach diesem Smoke und wurden nur lokal erneut geprüft; Dateihashes im externen Beleg.
- R8-TEST.2: LOKAL_GEPRUEFT/ABGESCHLOSSEN im Fixtureumfang. R8-GUI-CPU.1 bleibt als positiver CPU-only-Workflow blockiert; negativer echter GUI-Endzustand geprüft. Vollständige neue Probe-Coordinatorabnahme, H10-/DeepX-Numerik und Langzeitenergie bleiben außerhalb dieses Abschlusses.
- Lieferung: ABSCHLUSSBERICHT_R8.md, TEST_RESULTS_R8.json, JUnit/Logs, Config-/Import-/Hashnachweise, R8-Diff gegen Start und kumulativer Diff gegen c5eb66e, kleiner tatsächlicher GUI-Debugpack, GUI_BEDIENUNG.md und ERGEBNISSE_R8.zip. Kein Commit/Push; GUI bereit, kein neuer Workflow gestartet.


### R9A – G0 und lokale Fehlerreproduktion (16.09.2026)

- BACKFILL-01: IN_ARBEIT. Hauptrepo/Imports und kumulativer R1–R8-Stand gesichert; Hostpreflight berücksichtigt, Produktinterlock regulär gehalten. Echte lokale YOLO26s-/YOLO26m-Analyse und Splitgeneration mit gespeicherten originalen Negativbeobachtungen reproduziert: b364/b398 global akzeptiert, kein nächster H8-Kandidat. Keine Compiler-/Hardwarestarts.
- PERF-01: IN_ARBEIT. Originalreport-Replay reproduziert P2 104,311640/90,350410 als Hauptwerte mit fremden Completed-Intervallen. Zentrale endpunktgebundene Projektion ergänzt; erste18 lokale Regressionen bestanden. Keine neue Inferenz. Historische Full-Replikate ohne erhaltene Zeitserie bleiben unavailable.
- BACKFILL-01: erste7 lokale Regressionen bestanden: tatsächlicher ONNX-Splitgenerator, H8-Ersatz/H10-/DeepX-Beibehalt, Matrix samt Fullreferenzen, persistente Poolgrenze und historische/deaktivierte Policy. Kontrollierte Cache-/Compilerleaves sind keine physische Compilerabnahme. Weitere Budget-/GUI-/Integrationsprüfung ausstehend.
- BACKFILL-02: OFFEN; noch kein echter GUI-Workflow gestartet. LAT-01: ZURUECKGESTELLT, Einbildlatenzinstrumentierung nicht ausgeführt.
- Nachterkenntnisse aus dem R9A-Auftrag (Originalrun completsetdev_20260915_211333):61 vollständige Energiezeilen/183 gültige Erstaufnahmen, kein Retry. Dies ersetzt nicht rückwirkend den historischen R8-Scope; wieder fehlendes DeepX-5000er-Resultat separat offen. Keine neue Energieaufnahme.
- H10-Nulloutputs, DeepX-Zweibilderfall,97-FPS-b066-Abgleich und RegNet-Parallelität bleiben offen; Qualityoptimierung, Ranking/Paper und Langzeitenergie bleiben zurückgestellt.

### R9A: lokale Integration A/B vor GUI-Hardware (16.09.2026)

- BACKFILL-01: backendbezogene Auswahlzustände aus der ursprünglichen Reihenfolge, exakte Hailo-Negative als Auditfälle, getrennte Setup-/Richtungsquoten und Planmatrix. Kein Ersatz bei Infrastruktur-/Qualitätsfehlern. Kaltstarts vor Dispatch im bestehenden Generation-State bzw. TRT-Runcheckpoint reserviert; Remoteersetzung erhält den TRT-Zähler. Fehlende Resume-Verbrauchsevidenz sperrt neue TRT-Builds. DeepX hat keinen wiederverwendbaren exakten Negativstore: ursprünglicher Fall bleibt; gewöhnliche Compilerfehler berechtigen nicht zum Nachrücken.
- Normale neue Config-Policy in Standard/Final identisch: 16 untersuchte Kandidaten je Backendvertrag, 8 Split-Kaltstarts je Modell, höchstens 4 Hailo-Part1- und 4 TRT-Part2-Starts, Hailo 3600 s/TRT 7200 s. Full-Referenzen behalten ihren vorhandenen Pfad. Hardwarefreigaben dieser Sitzung werden ausschließlich im eigenen Profil auf 4/2/1/1 und 1200/300 s eingeengt. Sonstige Mode-, Collector-, Energie- und Kalibrierungsfelder bleiben unverändert.
- PERF-01: Rohreport → normaler Reader → normalisierte Summary/Matrix → GUI-Text/Export bindet Completed-Task-FPS und deren Intervalle an die eigene Count-/Zeit-/Replikatserie; P2 separat. Unbelegte Completion/Latenz bleibt unavailable. Keine neue Inferenz für Original-Replays.
- Original-Generatorreplay: YOLO26s b364 → b365, YOLO26m b398 → b399; echter ONNX-Analyse-/Split-/Exportpfad, gespeicherter Original-Negativbefund als physischer Cache-Leaf. Nächster MISS mit absichtlich null lokalen Coldstarts endet build_budget_exhausted, kein erfundener Compiler-PASS.
- Lokale Zwischenabnahme: 49 gezielte Tests grün; echter Tk-Editor/Save/Reload/Summary/Snapshot 1 PASS. Sechs zusätzlich untersuchte Cachetests scheitern identisch auf G0 und R9A (6 FAIL/24 PASS auf G0); bestehende Fehler, keine Cache-/Retentionsneugestaltung in R9A. Breitere finale Suite und BACKFILL-02 Hardwareabnahme stehen noch aus.
- Quellen: externes R9A-Verzeichnis v283_R9A_20260916_151759_Nqk9xZ, GREEN_A_ORIGINAL_GENERATOR.json, junit_focused_eighth.xml, junit_gui_third.xml, junit_g0_cache_control.xml. LAT-01 bleibt nächste Runde, nicht ausgeführt.

### R9A: Gegenprüfung der Ausführungsgrenzen

- BACKFILL-01: Setup und kanonische Ausführungsrichtung werden auch am Native-Dispatcher und in der exportierten Benchmark-Suite gefiltert. Leere Auswahl bleibt leer; fehlende Setupidentität bei mehreren Setups blockiert. Echte lokale Tk-Queue deckte zusätzlich eine leere Accelerator-Quote im CPU-/Full-Container auf: aktivierter Default darf dessen normale Generation nicht unterdrücken. Diese R9A-Regression wurde vor Hardwarestart korrigiert und separat getestet.
- PERF-01: echter Ergebnis-Textwidget zeigt beide Original-Endpunkte einschließlich ihrer Intervalle. Original-Rohreports vollständig separat durch Reader, Summary, Matrix und wissenschaftlichen Export replayt. Keine Originaldatei neu geschrieben.
- Nachtbeleg erneut direkt gelesen: 61 vollständige Energiebeobachtungen, 61 referenzierte Aggregate, 183 gültige Erstaufnahmen, 0 Retries und 0 ausgelassene Replikate. Alle ausgewählten Versuche tragen Index0. Methodentest-Probes sind nicht mitgezählt.
- Vorhandene Screeningenergie H8→TRT, jeweils n=3: RegNet X1.6GF/b132: 39,395178 / 40,299430 / 39,315151 J, Mittel39,669920 J, Stichproben-s0,546638 J, CV1,377967%; YOLO11l/b062: 53,686374 / 53,542164 / 52,781159 J, Mittel53,336566 J, s0,486370 J, CV0,911889%; YOLOv7/b044: 56,279315 / 56,497792 / 55,719764 J, Mittel56,165624 J, s0,401281 J, CV0,714460%. Quellen: Originalrun `reports/native_energy_measurements/measurements/`, jeweilige `energy_aggregate.json`, Feld `scientific_primary_energy_statistics.energy_j`; unabhängige Rechnung mit ddof=1 stimmt überein.
- Messmethode `command_marker_window`, primär `calibrated_input_energy_unsubtracted`, Tier `screening`; die vorhandenen Fenster dauern ungefähr2,56–2,94 s. Scheduling ist ein möglicher Einfluss, keine bewiesene Ursache. Keine neue Energieaufnahme oder Streuungsoptimierung.
- DeepX-YOLO26s-Full: wieder genau ein fehlendes 5000er-Qualitätsergebnis (`central_quality_summary.json`: applicable54/completed53/missing1). Technische Ausführung und rohe Repräsentation vorhanden, Qualitätsentscheidung unavailable und claim_eligible=false. Nicht mit dem historischen R8-500er-Ergebnis verwechseln. H10-Nulloutputs, DeepX-Zweibilderfall, b066-Abgleich, RegNet-Parallelität und LAT-01 bleiben offen.

- R9A-Budgetgegenprüfung vor Hardware: TRT-Part2-Startanteile summieren sich nun über alle Setups zum Modelllimit. Tatsächliche Workflowfortsetzung wird getrennt von der normalen Remote-Reuse-Erlaubnis transportiert; der Erststart erhält sein genehmigtes Budget, fehlender Verbrauchsnachweis im Resume sperrt neue Starts. Echte lokale Builderprozesse mit absichtlichem Compiler-Rückgabecode17 belegen einen verbrauchten Erststart und null Starts bei fehlendem Resumecheckpoint; kein künstlicher Compiler-PASS.
- Explizite vorhandene `benchmark_execution.extra_args` und die reine Remote-Timeoutangabe überleben die normale Modeauflösung. Das eigene Diagnoseprofil trennt Einzelinferenz120s von Remotegruppe900s einschließlich TRT-Build; äußere GUI-Abbruchanforderung bei2500s mit2700s Gesamtgrenze. Keine Änderung von Hosts, Venvs oder allgemeinen Laufzeitdefaults.
- Breiter Zwischenstand:763 PASS/6 bereits auf G0 bestehende Cache-FAIL/0 SKIP; alle647 R7-Knoten enthalten. Echter lokaler Tk-Scope3/3 PASS inklusive Startbutton, realem CPU-Prozess, Runanlage, Fehlerdialog und erfolgreichem Artefaktabschluss. Der CPU-only-Workflow bleibt als bekannter technischer Negativfall ausgewiesen. Nach den letzten Budgetkorrekturen folgt die finale Wiederholung auf identischen Produktdateien.

### R9A – einziger realer GUI-Lauf und genaue Abnahmegrenze

- BACKFILL-01: IMPLEMENTIERT + LOKAL_GEPRUEFT. Vor Hardware:768 PASS/6 identische G0-Cachefehler/0 SKIP, alle647 R7-Knoten; echte separate Tk-Matrix einschließlich R8:9/9 PASS. Hauptinstallation mit Build `v2.83-r9a-backfill-task-fps` und Source-/Installed-PASS unmittelbar vor Start belegt.
- BACKFILL-02: BLOCKIERT vor Inferenz. Genau ein tatsächlicher Startbutton-/Queue-Workflow am16.09.2026,16:51:43, YOLO26s/H8. Original b364 exakt COMPILE_INFEASIBLE, nächster b365 ebenfalls exakt COMPILE_INFEASIBLE, danach b021 vorhandener gültiger HEF-HIT. Drei von höchstens vier Kandidaten geprüft; zwei Audit-Ausschlüsse, ein ausgewählter Ersatz. Keine Kreuzverteilung auf H10/DeepX. Tatsächlich0 H8-Compilerstarts,0 TRT-Compilerstarts,0 Native-Inferenzzeilen,0 Energieaufnahmen.
- Der normale Warmcache-Preflight meldete fälschlich zwei bereits erfolgte Kaltbuilds, darunter Full-H8, und sperrte Dispatch. Die aktuellen Serviceplan-Attempts normalisieren ein fehlendes Receiptfeld zu `skipped=false`; der unveränderliche Receipt belegt zugleich `compiler_dispatch_count=0` und `cache_hit=true`. Der Consumer ignorierte den ausdrücklichen Nullzähler. Keine Compilerinfeasibilität und kein Quality-FAIL als Ursache dieses Workflowfehlers.
- Nach dieser einzigen Hardwareausführung wurde ausschließlich die betroffene Hailo-Cachebilanz funktional korrigiert: expliziter ganzzahliger Dispatchcounter0 verhindert die falsche Kaltbuildwertung; fehlende, ungültige oder positive Counter erhalten die konservative Warmbarriere. 48 gezielte lokale Tests bestanden; der unveränderte reale Run wird separat read-only durch den Reporter replayt. Diese Korrektur ist NICHT erneut auf Hardware abgenommen. Kein zweiter Start, kein Retry und kein Build als Fehlerumgehung.
- Der tatsächliche GUI-Ergebnisdialog meldete „Lauf fehlgeschlagen“, zentrale Qualität0 Auswertungen. Das externe Bedienprogramm erkannte den eigenen benutzerdefinierten Dialog, fand jedoch keinen Standard-OK-Widgetpfad; nach terminalem Workerende schloss es das eigene Fenster über seinen Cleanup-Fallback. Ca.194s bis Beobachterende, kein Erreichen des2500/2700s-Stopbudgets. Beobachter-Exit0 bedeutet keinen Workflow-PASS.
- PERF-01: vollständige Original-Reader-/Summary-/Matrix-/GUI-Widget-/Exportreplays LOKAL_GEPRUEFT; Completed-Task und P2 jeweils eigene Intervalle. Kein neuer physischer FPS-Nachweis. LAT-01 und die oben getrennt aufgeführten Numerik-/Quality-/Energiethemen bleiben zurückgestellt.

### R9A – Abschlussstand

- Hauptrepo INSTALLIERT:2.83 / `v2.83-r9a-backfill-task-fps`; normale Config AKTIV mit gesicherter, alleiniger Ergänzung der endlichen Nachrückpolicy.69 ursprüngliche Benutzerprofile und alle übrigen gesicherten Configdateien bytegleich. Originalrun:234622 Datei-/Linkeinträge bei identischem lstat-Inventar; ausgewählte Originalreport-Hashes separat belegt. Kein Reset, Commit oder Push.
- Finale Software nach der Cachebilanzkorrektur:798 PASS/6 bereits auf G0 identische Cache-/Retention-FAIL/0 SKIP,804 Knoten,346,10s einschließlich Runner-Overhead (`acceptance_r9a_delivered.json/xml`). Alle647 R7-Knoten erhalten; zusätzlich betroffene Cachepreflight-Gegenfälle enthalten. Diese Suite ist ausdrücklich nicht insgesamt grün. Enger Nachtest48/48 PASS; separate echte Tk-Matrix vor dem nachgelagerten Consumerfix9/9 PASS. Danach nur diese Abschlussdokumentation und die vorhandenen Source-/Release-Manifeste erneuert, keine weitere Produktfunktion geändert.
- Reale GUI bleibt BLOCKED: Auswahlkette b364→b365→b021 und0 tatsächliche Compilerstarts belegt, danach technischer Cachebilanzfehler vor Inferenz. Artefaktabschluss PASS mit1292 Einträgen,0 fehlend/Hashabweichungen/Verifikationsfehlern; Workflowstatus FAILED. Keine erneute Hardwareabnahme des finalen Consumerfixes. Keine Energie-/Collector-/Windowprobe-Ausführung.
- Lieferung im externen R9A-Ausgabeverzeichnis: `ABSCHLUSSBERICHT_R9A.md`, `TEST_RESULTS_R9A.json`, `GUI_ABNAHME_R9A.json`, vollständige G0-/HEAD-Diffs einschließlich untracked Source-/Tests, Source-/Config-/Originalerhaltbelege und `ERGEBNISSE_R9A.zip` mit Inhaltsprüfung. Große alte Fixture-Rohdaten bleiben im vollständigen lokalen kumulativen Diff; das kleine ZIP enthält dessen Hash/Index und einen entsprechend gekennzeichneten Code-Review-Auszug.
- Nächste Handlung: gelieferte Änderungen und Belege gegenprüfen. Eine neue begrenzte GUI-Laufzeitabnahme wäre ein neuer Auftrag; R9A startet sie nicht automatisch. BACKFILL-01/PERF-01 lokal geprüft, BACKFILL-02 bleibt offen/blockiert; LAT-01, H10-/DeepX-Numerik, Qualitysuche, Papervergleich und Nachtlauf bleiben zurückgestellt.

## R9A-Nachabnahme – lokale Korrekturen (16.09.2026)

- Sechs G0-Regressionen konkret korrigiert: vier Namespacefixtures mit notwendiger Owner-ABI, gültiger direkter Part2-Migration und explizit abgewiesener ungebundener Bridge; vollständige Workflowinitialisierung samt realem Managementshutdown; ABIvergleich unter identischem Runtimevertrag. Keine Receipt-/GPU-/Cleanupgate-Lockerung.
- Cachebilanz: expliziter nichtnegativer Compilerzähler bindet sowohl null als auch echte Starts; reale Receiptprüfung bleibt wirksam. Fehlende/ungültige Counter werden weiterhin konservativ behandelt.
- PERF-01: alle61 erfolgreichen Originalzeilen offline gelesen.19 Splitserien count-/zeitbelegt;42 historische Full-Dreierserien ohne126 Replikatzeiten bleiben beschriftete Diagnose inklusive alter FPS/Intervalle/Endpunkte. Kanonische Sammlung nach exakter Identität, keine letzte Wiederholung als Dreierserie. Aktuelle Fulladapter und Aggregation separat geprüft; vorhandene DeepX-Wandzeit und TRT-Classification-Tracezeit werden weitergereicht, keine neue Latenzinstrumentierung.
- BACKFILL-01/BACKFILL-02: bisheriger Auswahlbeleg bleibt erhalten; neuer GUI-Nachtest noch ausstehend. Softwareprüfung und echte GUI-Laufzeitabnahme bleiben getrennt. LAT-01 bleibt nächste Runde; Energie/H10-/DeepX-Hardware/Collector unverändert außerhalb des Auftrags.


## R9A-Nachabnahme – Fortsetzung nach Host-Lockklärung (16.09.2026)

- Eigener G0-Lockhelfer hostseitig eindeutig zugeordnet (PID556465, start_ticks55141395) und per SIGTERM beendet; Lockdatei erhalten. Die normale Produktsperre wurde für die Patchübernahme neu erworben und danach nachweislich freigegeben. Der externe Aufgabenhelfer hält keine wartende Dauersperre mehr: Arbeit und Sperre liegen in einem begrenzten Prozess, mit finally-Freigabe, Signal-/Timeoutbehandlung, nicht vererbtem Deskriptor und Parent-Death-Signal. Acht externe Prozessprüfungen PASS einschließlich normalem Ende, Ausnahme, SIGINT, SIGTERM, SIGKILL, Timeout, Elternprozessende und belegter Fremdsperre. Keine Änderung der Produktlockregeln; die echte GUI muss ihre Sperren selbst erwerben.
- Enger vorbereiteter Patch nach identischen acht Vorherhashes übernommen: vorhandene BUILD_ID `v2.83-r9a-nachabnahme` konsistent fortgeführt, fehlender pytest-Import ergänzt, konkrete Receipt-/GPU-Ablehnungsassertionen verschärft. Keine neue BUILD_ID, keine Gate-Lockerung.
- Zuerst exakt die13 zuvor fehlgeschlagenen Knoten:13 PASS. Anschließend vollständige877er-Auswahl:877 PASS/0 FAIL/0 SKIP,196.10s; alle bisherigen804 und alle647 R7-Knoten enthalten. Die sechs ursprünglichen G0-Testknoten bleiben PASS. Aktueller TRT-Classification-Adapterpfad jetzt ebenfalls positiv geprüft.
- PERF-01 offline erneut bestätigt:61 Originalzeilen,19 belegte Completed-Task-Splitserien,42 historische Full-Diagnosen mit erhaltenen FPS/Intervallen/Endpunkten.33 einzelne Full-Rohreports mit Count/Zeit ersetzen keine126 Replikatnachweise. Kein Latenzwert aus1/FPS und keine historische Qualitätsaufwertung.
- Reale GUI-Nachabnahme weiterhin AUSSTEHEND:0 Hardwareworkflowstarts. Die Fortsetzungssitzung erlaubt derzeit nur Hauptrepo-/tmp-Writes und eingeschränktes Netzwerk; ursprünglicher Ausgabeordner, normale Laufzeitconfig und Models sind schreibgeschützt. Tk erreicht den Displayanschluss nicht:aktuelle9er-Matrix vollständig SKIP; zusätzlicher61-Zeilen-Widgetcheck FAIL bei Tk-Erzeugung. Der ältere9/9-Tk-PASS und ältere61-Zeilen-Widgetbeleg bleiben historische lokale Nachweise, keine neue Hardwarefreigabe. Keine geänderten Produktfunktionen nach einem Hardwaretest, da keiner gestartet wurde.
- BACKFILL-01 lokal geprüft; BACKFILL-02 wartet auf die freigegebene einmalige normale GUI-Ausführung. LAT-01, Energie, H10-/DeepX-Hardware und Collector bleiben außerhalb des Auftrags. Fortsetzungsausgaben wegen aktueller Sandboxgrenze unter `/tmp/v283_R9A_Nachabnahme_FINAL_xWRKPf`; ursprüngliche Belege bleiben unverändert.


## R9B – erster unveränderter R9A-GUI-Nachtest (16.09.2026)

- Sandbox-Datei-/Loopback-/Tk-Probe PASS; effektive Launcherangabe gpt-6-astra/ultra dokumentiert. Genau ein normaler GUI-Workflow, R9A-Produktquelle vor/nach unverändert. Auswahl b364/b365 exakt ausgeschlossen, b021 warm automatisch nachgerückt. Generische Splitinferenz erreicht; Native Full H8 100 Frames/10 Warmup/1 Wiederholung, 21,937 FPS.
- BACKFILL-02 bleibt OFFEN: Native Split und Full TRT durch zentrale Qualityfehler blockiert. Ursache konkret im übernommenen externen Bedienhelfer: ungeschützter Modulstart erzeugt beim multiprocessing-spawn erneut Tk, scheitert am exklusiven Startmarker und beendet die Qualityworker (BrokenProcessPool). Kein belegter Quality-Produktfehler, kein zweiter R9A-Workflow.
- Regulärer terminaler Fehlerdialog bestätigt, Worker/GUI beendet, Produktinterlock frei, ursprüngliche GUI-Einstellungen byteidentisch wiederhergestellt. Keine Energie/Collectorstarts. Run: r9a_nachtest_yolo26s_h8_20260916_200325; externe R9B-Belege getrennt.
- LAT-01: Messvertrag vor Code dokumentiert. IMPLEMENTIERUNG beginnt; lokale und reale Abnahme noch ausstehend. Fehlendes Top-k in Klassifikationshotloops bleibt explizite Completed-Task-Grenze, kein zusätzliches Postprocessing.

### R9B / LAT-01 – Implementierung und lokale Prüfung

Build einmalig `v2.83-r9b-request-latency`, Paket 2.83. Gepaarte monotone Requests in Python-Full-/Splitpfaden, H8-C++-Hostoutputs und Three-Stage-Callback; begrenzte Puffer, Fehlernachweis, vollständige gepoolte Quantile und Reportingweitergabe. Klassifikation ohne tatsächliches Top-k bleibt explizit Hostoutput; TRT-trtexec ohne Admissionpaar bleibt unavailable. Three-Stage verschiebt vorhandenes Preprocessing vor Admission; keine zusätzliche Arbeit/Synchronisation, Hardwarewirkung ungeprüft.

Gezielte lokale Tests laufen; echte GUI-/Hardwarecoverage noch ausstehend. H8 b135/b062 derzeit unter strengem Buildverbot NOT_RUN: normaler GUI-Pfad kompiliert immer den C++-Wrapper, Leaf-`--no-build` wird dort nicht durchgereicht. Keine Gateumgehung. BACKFILL-01/02 und PERF-01 nicht durch lokale Tests schließen. H10-Nulloutputs, DeepX-Zweibilderfehler, allgemeine Quality, b066-Forschung, Langzeit/Ranking bleiben zurückgestellt; Energiestreuung unverändert dokumentiert, keine neue Messung.

### R9B / LAT-01 – abgeschlossene begrenzte GUI-Matrix

Software vor Hardware: 911 PASS (877 erhaltene Knoten + 34 neue), Tk 26 PASS ohne Skips. Anfangs-Tk-Fehler war die HOME-Isolation des Testprozesses bei impliziter Xauthority; expliziter, von libXau ermittelter aktueller Authority-Pfad nur im Testprozess, keine neue Freigabe. Laufzeit-GUI benutzt unveränderte Sessionwerte.

R9B: genau 3 Starts, keine Wiederholung. H10H MobileNet b135: 3/3 Native erfolgreich, Split und Full H10H je 100/100 Hostoutputpaare, Tasklatenz ohne Top-k unavailable; Full TRT nur unveränderte Legacygrenzen. H10H YOLO11l b062: 3/3 Native erfolgreich und je 100/100 vollständige Tasklatenzpaare; Mean Split 264,389 ms, Full H10H 182,748 ms, Full TRT 25,839 ms. Beide technisch OK, Quality FAIL, Finalisierung PASS, normale Dialoge bestätigt und Cleanup/Lockfreigabe belegt. Alle warm, Null-Kaltbuildbilanz.

DeepX MobileNet b135: vor Runtime durch die normale Warm-Cachebarriere blockiert, 0 Native-Inferenzen/Compilerdispatches. Full DXNN und TRT Full HIT; DeepX Part1 UNKNOWN (required_artifact_cache_probe_missing), abhängiges TRT Part2 UNKNOWN (native_part1_identity_unavailable). Gemeinsame Ursache: Part1 wird vor Preflight nur deferred, der lokale DeepX-Preflight materialisiert lediglich Full. Derselbe Pfad gilt für Detection; YOLO11l wird daher mit 0 Starts NOT_RUN_SHARED_DEEPX_PART1_PREFLIGHT_GAP geführt. Das beweist weder physisches Fehlen eines DXNN noch ein Detection-Runtimeversagen. Keine Gate-/Cacheänderung. H8 beide Tasks 0 Starts wegen verpflichtendem Wrapperbuild im normalen Pfad.

Letzte lokale Reportingkorrektur nach Hardware: informationsloses FPS-Wiederholungs-CI bei n=1 wird nicht neu projiziert; Rohfelder und Rate erhalten. Kein Runner-/Timing-/Qualitätswechsel, keine zweite Hardwarematrix. Abschließende Testzahlen und Bindung stehen in acceptance.json, gui.json und POST_HARDWARE_REPORTING.json im Ausgabeordner. LAT-01: Implementierung und lokale Abnahme getrennt von obiger begrenzter Hardwarecoverage. BACKFILL-02 bleibt wegen des einzigen R9A-Bedienhelferfehlers offen; keine pauschale Schließung BACKFILL-01/PERF-01. H10-/DeepX-Qualitätsforschung, b066, Langzeit/Ranking und Energie bleiben zurückgestellt.


### R9B – finale Lieferprüfung

Abschließende Produktauswahl: **913 PASS / 0 FAIL / 0 SKIP**, 221,24 s pytest; alle 877 Ausgangsknoten enthalten. Davon 36 neue Requestlatenztests. Separate echte Tk-Auswahl: **26 PASS / 0 FAIL / 0 SKIP**, 171,53 s einschließlich Starter. Enger Schlusslauf Request-/FPS-Tests: 67 PASS; keine additive Gesamtzahl. Zwei gespeicherte H10H-Hardwareruns im normalen Ergebniswidget erneut geöffnet: 100/100 und Mean erhalten, n=1-CI unterdrückt, null neue Workflows.

Finalisierung der beiden H10H-Runs und des vor Inferenz blockierten DeepX-Runs jeweils aus artifact_index_closure.json PASS; Runtime und Quality weiterhin getrennt. Eigene GUI-Fenster geschlossen, Settings byteidentisch wiederhergestellt, Startjournal unverändert bei R9A 1 / R9B 3. Nach diesen Tests nur dieser Arbeitsstand und die vorhandenen Source-/Release-Manifeste aktualisiert. Auftragsdiff, vollständige Vorher-/Nachherindizes, Installed-/Mirror-/Importbelege und ERGEBNISSE_R9B.zip im R9B-Ausgabeverzeichnis.

### R9B-Abschluss – lokale Korrekturen (16.09.2026)

Sandbox einschließlich aktuellem Tk-Display geprüft. Spawn-sicherer Bedienhelfer mit echtem ManagementQualityService-Kindprozess und beendetem Worker lokal PASS. DeepX-Part1 wird vor abhängigem TRT-Part2 aus dem gespeicherten ausgewählten Request cache-only aufgelöst; MISS stoppt vor Compiler/SDK und überschreibt veraltete Statusbelege. H8-Wrapper erhält Source-/Binary-/CMake-Bindung im bestehenden Configreceipt, begrenzte Vorbereitung und anschließend --no-build. Explizite Klassifikations-Hostoutputpaare erhalten separat beschriftete FPS; keine zusätzliche Top-k-Arbeit. Gezielt in Prüfung; noch null neue Hardwarestarts und keine neue Energieabnahme.

Lokale Gegenprüfung R9B-Abschluss: 97 gezielte Tests, erweiterte Auswahl 99 Tests und abschließende neue Fälle 23/23 PASS; Teilmengen nicht addieren. Echte Tk-Auswahl 26/26 PASS ohne Skips. Zwei gespeicherte H10-Runs im realen Widget replayt, sechs Raten-/Latenzzeilen numerisch erhalten, Classification als Hostoutput beschriftet. Erste vollständige Auswahl enthält alle 913 Ausgangsknoten; drei veraltete Classification-Completed-Task-Assertions werden gezielt an den Logitvertrag angepasst, ursprüngliche Rate-/Zeit-/Runtimeassertionen erhalten. Vollständiger Schlusslauf folgt vor Hardware. Keine H10-/Energieausführung.


### R9B-Abschluss – begrenzte GUI-Matrix und Restblocker (16.09.2026)

Softwareabschluss vor Hardware: **939 PASS / 0 FAIL / 0 SKIP**, 367,31 s einschließlich Starter; alle 913 bisherigen Knoten direkt verglichen und erhalten. Separate **26 Tk-Fälle PASS / 0 FAIL / 0 SKIP**, 160,78 s, alle bisherigen Tk-Knoten erhalten. Die neue echte Barrierenregression hat zusätzlich eine ungültige DeepX-Parentidentität aufgedeckt; Status, DXNN-Hash und Größe werden jetzt vor P2 geprüft. Die Remote-Lieferung enthält process_control/native_progress; der unveränderte Requestrecorder wird vor seinen Importeuren geliefert. Kein Recorder-/Timing-/Numerikwechsel. Build-ID bleibt v2.83-r9b-request-latency.

Genau drei neue normale GUI-Starts, keine Wiederholung, eine H8-C++-Wrappergeneration (9,44 s), null Modell-/HEF-/DXNN-/TRT-Enginebuilds, null Energie-/H10-Läufe:

- H8 / YOLO26s: Native Full H8 und setup-lokales TRT Full erfolgreich, je 100/100 Taskpaare; Mean 46,156 bzw. 6,747 ms. Technisch OK, Quality FAIL, Finalisierung PASS. **BACKFILL-02 bleibt OFFEN**: Der auftragslokale Bedienhelfer verwendete irrtümlich forced_cases=b364. Dieser exakte Scope unterdrückt den normalen Backfill; b364 wurde korrekt ausgeschlossen, b365/b021 wurden in diesem Lauf nicht nachgerückt. Kein Artefaktfehlbestand behauptet. Korrigiertes Profil ohne forced_cases lokal am Produktresolver geprüft (Pool b364, b365, b021), aber kein zweiter Start genehmigt oder ausgeführt. Eigener Vorbereitungsfehler, keine Quality-/Compilerfehlklassifikation.
- H8 / MobileNet b135: vier warme Artefakt-Hits, alle drei Native-Zeilen erfolgreich. C++-Generation mit vorhandenen Vendorlibraries und anschließend gebundenem --no-build-Reuse tatsächlich ausgeführt. Split und Full H8 je 100/100 Hostoutputpaare, Mean 8,249 bzw. 8,341 ms; Tasklatenz ohne Top-k unavailable. TRT Full hat Hostoutputrate, weiterhin keine gepaarten Requestzeitstempel. Technisch OK, Quality FAIL, Finalisierung PASS.
- DeepX / MobileNet b135: reparierte Part1-Auflösung tatsächlich HIT, ebenso Full DXNN und Full TRT. Abhängiges TRT P2 jetzt deterministisch MISS statt UNKNOWN: special_precision_direct_source_not_allowed. Keine Inferenz/Compilerdispatches. Technisch fehlgeschlagener Warm-Preflight, Quality nicht ausgewertet, Finalisierung PASS. Vorhandene Native-Bindings enthalten den passenden Part1-Hash und eine unveränderte FLOAT/as_input-Brücke; der bestehende P2-Preflight lehnt identische Source-/Buildhashes für diese Spezialprecision vor vollständiger Bindingprüfung ab. Das ist eine verbleibende Preflightgrenze, kein Beweis fehlender Engines oder Compiler-Qualitätsverlust.
- DeepX / YOLO11l b062: **BLOCKED_SHARED_TRT_P2_PREFLIGHT, 0 Starts**. Begrenzte reine Receipt-Leseprüfung über bestehenden SSHTransport belegt dieselbe unveränderte FLOAT/as_input-Brücke. Den bekannten gemeinsamen Fehler gemäß Auftrag nicht durch einen vierten Workflow wiederholt; keine Gates gelockert und kein Nachbau.

Alle drei gestarteten GUI-Ketten zeigen terminalen Dialog, beendete Worker, geschlossene GUI, freigegebenen Produktinterlock und bytegleiche Wiederherstellung der normalen Settings. Ein eigenes hängendes Vorschaufenster wurde anhand seines auftragslokalen Profils eindeutig erkannt und über normales WM_DELETE_WINDOW geschlossen; kein Fremdprozess beendet. Alle vier neuen vollständigen Latenzreihen aus Rohpaaren nachgerechnet; tatsächlicher CSV-Export und Ergebniswidget für sechs Zeilen geprüft. Alte Startmarker/Budgets bleiben erhalten.

LAT-01 bleibt task-/runnerbezogen: neuer H8-Full-Detectionbeleg und H8-Classification-Hostoutputbelege vorhanden; H8-Detection-Split b021 und beide DeepX-Taskpfade weiter offen. Historische H10-Detectionbelege bleiben unverändert, nur lokale Reportreplays. Classification-Top-k, TRT-Classificationpaare, H8-b062/b066, H10-/DeepX-Numerik, Qualityforschung und lange/finale Energie bleiben offen. Kein pauschales Schließen von BACKFILL-01/PERF-01.

Ausgabe: /home/kmika/.local/share/onnx-splitpoint-codex/v283_R9B_Abschluss_20260916_223540_VcJK7R. Vollständiger Auftragsdiff, neue Regressionen, Test-/GUI-/Receiptbelege, Hauptinstallationsprüfung und ERGEBNISSE_R9B_ABSCHLUSS.zip. Empfohlene nächste Arbeit: separat begrenzter Backfill-Nachtest mit korrigiertem Profil und eng validierte Auflösung der unveränderten DeepX-FLOAT-Brücke unter allen bestehenden Bindinggates; erst danach erneut genehmigte DeepX-Hardwareabnahme. Kein Commit/Push/Reset/Versionssprung.


### R9B-Restabnahme – Fortsetzung und lokale Vorprüfung (17.09.2026)

Derselbe Auftrag nach Tokenlimit fortgesetzt; Sandbox/Tk PASS, keine eigenen aktiven Produktprozesse, normale Sperren frei. Originale Startjournale/Marker erhalten. Bilanz vor Hardware: GUI 0/3, DeepX-P2-Preflights 0/2, H8-Wrappergeneration 0/1. Rekonstruktion unter fortsetzung_20260917_090623_xcXwYn/FORTSETZUNGSSTAND.md im maßgeblichen Restabnahme-Ausgabeordner.

DeepX-Part1 bleibt abgeschlossen. Enger P2-FLOAT-No-op-Fix führt identische Sourcebytes ausschließlich in den vollständigen bestehenden Native-Bindingvalidator; keine allgemeine UINT8-Ausnahme. Bisher 32 neue lokale Tests PASS, zwei Fixture-Schwächen in lesender Gegenprüfung erkannt und gezielt verbessert. H8-Profil ohne forced_cases vorhanden; alter Helferwiderspruch und Startgrenze lokal korrigiert. Editor-/Queue-/Generatorbelege und finale Gesamtprüfung noch offen. BACKFILL-02 und beide DeepX-Laufzeitabnahmen weiterhin offen, noch keine neue Geräteausführung. Der API-Versionsalias ist bereits an die kanonische Version 2.83 gebunden und frisch importiert PASS; kein Versionswechsel nötig.


R9B-Restabnahme – finale lokale Basis: **971 PASS / 0 FAIL / 0 SKIP**, 219,475 s, alle 939 Ausgangsknoten erhalten. **26 echte Tk-Knoten PASS / 0 FAIL / 0 SKIP**, 178,835 s. Neuer H8-Test prüft echte Editorvariablen, Snapshot, Queueoptionen ohne Workerstart und Produkt-Candidatescope; separater realer Generator nutzt diese aufgelöste Reserve und Originalnegativbelege: b364/b365 ausgeschlossen, b021 selektiert, andere Backends erhalten. Der volle Materializer-Handoff ist ergänzend statisch geprüft. Import-/Spawnprüfung sowie echter ManagementQualityService-Worker mit exitcode 0 PASS.

DeepX-P2-Fix jetzt lesend auf beiden Gerätenecken belegt: MobileNet b135 HIT (2,426 s), YOLO11l b062 HIT (4,883 s), jeweils vollständiger strikter Nativevalidator mit neu gehashten Dateien und exakten Crosslinks. 2/2 erlaubte P2-Preflights verbraucht, keine Inferenz/Builds. Drei GUI-Vorschauen mit normaler Config PASS, weiterhin 0/3 Workflowstarts. H8-Wrapper-Receiptprüfung in vorhandenem Remote-Ergebnisroot findet keine erhaltene gebundene Generation; maximal eine notwendige neue Generation bleibt erlaubt. Reale Split-/Latenzabnahmen und BACKFILL-02 noch offen.


R9B-Restabnahme – **BACKFILL-02 real geschlossen**: genau ein neuer H8/YOLO26s-GUI-Start (538,969 s), b364/b365 mit exakter Negativevidenz ausgeschlossen, b021 warm nachgerückt und tatsächlich inferiert. Vier Artefakt-HITs, null Modellbuilds; eine notwendige Wrappergeneration 9,318 s, danach Reuse. Native 3/3 erfolgreich, Split/H8 Full/TRT Full jeweils 100 Latenzpaare. Split Mean 26,753 ms; Quality FAIL bleibt wissenschaftliches Ergebnis, technische Achse OK und Finalisierung PASS. Terminaler GUI-Dialog, beendete Worker, freie Sperre und identische Settings belegt. Export-/Rohpaarnachrechnung wird in der Abschlusslieferung geführt. P2-Preflightbudget 2/2 und Wrapperbudget 1/1 verbraucht; zwei DeepX-GUI-Zellen noch offen. Ergänzend drei unveränderte transformierte Bridge-/Bindingregressionen PASS, keine nachträgliche Produktcodeänderung.


R9B-Restabnahme – DeepX-P2 MobileNet und reale Hostoutput-Abnahme abgeschlossen: DeepX/MobileNet: genau ein normaler GUI-Start, 474,833 s, b135 und beide Fullpfade technisch OK (3/3), vier warme HITs, null Modellbuilds. Quality INCONCLUSIVE, Finalisierung PASS; Dialog bestätigt, Worker/GUI beendet, Sperre frei und Settings bytegleich. Split/DeepX Full je 100 Hostoutput-Paare, Mean 2,09295367/1,50990052 ms. Tasklatenz wegen fehlendem Top-k im Messpfad unverfügbar; TRT Full ohne Requestzeitstempel bleibt unverfügbar. Kein gemeinsamer P2-/Runtimeblocker. Originale GUI_R9B_deepx_classification.json und HARDWARE_deepx_classification.json plus hardware_evidence/deepx_classification. Noch offen: einziger erlaubter DeepX/YOLO11l-GUI-Start und finale Rohpaar-/Exportgegenprüfung.


### R9B-Restabnahme – Abschluss der realen Matrix (17.09.2026)

DeepX/YOLO11l: genau ein normaler GUI-Start, 510.226 s; Native 3/3 technisch erfolgreich, einschließlich b062. Technik ok, Quality fail, Finalisierung pass. Vier warme HITs, null neue Modellartefakte. Terminaler Dialog, beendete GUI/Worker, freie Produktsperre und bytegleiche Settings belegt.

LAT-01: 8/9 vollständige Latenzreihen aus je 100 eindeutigen Rohpaaren unabhängig nachgerechnet, tatsächliche CSV- und GUI-Widgetwerte für alle neun Zeilen geprüft. H8 Detection (Split b021 und beide Fullpfade) sowie DeepX Detection (Split b062 und beide Fullpfade) liefern Tasklatenz; DeepX MobileNet Split/Full liefern Hostoutputlatenz. TRT-Classification bleibt ohne Requestpaare, Classification-Top-k bleibt außerhalb des Messpfads. Quality H8 FAIL, MobileNet INCONCLUSIVE und YOLO11l fail bleiben getrennt von technischem Erfolg; keine wissenschaftliche Gesamtfreigabe. Vorhandene optionale Legacy-TRT-Full-Validierungswarnung bei MobileNet bleibt dokumentiert.

BACKFILL-02 real geschlossen; DeepX-Part1 unverändert abgeschlossen, DeepX-P2-Fix lokal, lesend und in beiden normalen GUI-Ketten strikt belegt. Kein pauschales Schließen von BACKFILL-01/PERF-01. Software final: 971 Produkttests, 26 echte Tk-Fälle, drei zusätzliche Bridge-Bestandsfälle PASS ohne FAIL/SKIP; fünf einzigartige H8-Lokalfälle PASS nach dokumentierter Testassertionskorrektur. Wiederholungen nicht addiert.

Kumulativ GUI 3/3, dedizierte P2-Preflights 2/2, notwendige H8-C++-Wrappergeneration 1/1 (9,318 s); RESTbudget überall 0. Normale Cachegates innerhalb der GUI bleiben separat erhalten. Kein Modellbuild, keine Energie, kein Retry oder Nachtlauf. Produktquellen nach Einfrieren unverändert; nur Arbeitsstand und vorhandene Manifeste finalisiert. Keine weiteren Hardwarearbeiten. Classification-Top-k-/TRT-Paarlücken, H10-/DeepX-Numerik, Qualitätsforschung, Papervergleich, Ranking und finale Langzeitenergie bleiben spätere Aufgaben.

Lieferung: fortsetzung_20260917_090623_xcXwYn/ERGEBNISSE_R9B_RESTABNAHME_FORTSETZUNG.zip im maßgeblichen Restabnahme-Ausgabeordner. Enthält kumulativen Auftragsdiff, alte/neue Belege und Abschlussbericht. Kein Commit/Push/Reset/Versionssprung.

### R9C – Klassifikationsabschluss und getrennte Outputbefunde (17.09.2026)

- LAT-01: Top-1 aus einmaligem stabilem Top-5 des aktuellen Outputs im vorhandenen Messpfad ergänzt: H8 C++/Python-Hook, H10 async/sync, DeepX Split und Vendor Full. TRT Full nutzt den vorhandenen Prepared-Input-Hotloop; trtexec-Zahlen bleiben separat. Request-IDs, Queue/Inflight und Warmuptrennung erhalten. Ende nach Hosttransfer und Top-k; kein Datasetlesen/Bilddekodieren/Modellinitialisieren eingeschlossen. Keine neue Cache-/Hash-/Runnerarchitektur.
- Lokal: 36 neue Regressionen PASS, einschließlich C++-Vektoren, aktueller Outputs je Request, kleinem K/Batch/Dtypes, Full-TRT-Aufruf und DeepX-Performance-/Semantikmerge. Vorige fokussierte Probe: 42 PASS, 1 noch unsynchronisierter Mirror und 5 tasklokale tmp-Elternverzeichnisfehler; beides im Testaufbau korrigiert. Vollständige Abschlussauswahl ausstehend. Noch kein Hardware-PASS.
- H10-02: vorhandene Roharrays offline über aktuelle Native-/Generic-Adapter und vier ORT-P2-Aufrufe geprüft. Scoreverlust spätestens im nativen UINT8-HEF-Ausgabevertrag; kein Hostadapterfix belegt. NICHT REPARIERT, kein Output-GUI-Nachtest.
- DX-01: vier vorhandene SDK-Rohoutputs über aktuelle Semantik-/Decoder-/Exportkette replayt: zwei Kontrollen gültig, zwei bekannte positive Scoreboxen bereits nativ invertiert. Exportsperre bestätigt. NICHT REPARIERT, kein Output-GUI-Nachtest.
- BACKFILL-02 und DeepX-P2 bleiben im abgeschlossenen R9B-Scope erhalten. Energie-/Qualitätsoptimierung zurückgestellt. R9C bisher: 0 GUI-Starts, 0 Hardwarecalls, 0 Wrapper-/Modellbuilds, Energie AUS.

### R9C – finale lokale Abnahme, reale GUI-Matrix und Grenzen

LAT-01 in normaler Hauptinstallation aktiviert: aktueller Output jedes Requests wird im bestehenden Messfenster einmal bis stabilem Top-5 und daraus Top-1 abgeschlossen. H8 C++/Python, H10 async/sync, DeepX Split/Full, Hailo Full und vorhandener TRT-Full-Prepared-Hotloop abgedeckt. FPS aus Abschlusszahl/Makespan; Latenz aus gepaarten Admission-/Top-k-Zeitstempeln. Historische Hostoutputmessungen und trtexec-Diagnostik bleiben getrennt. Kein Ersatzrunner, keine neue Registry/Cache-/Hasharchitektur.

Finale Software: **1012 PASS /0 FAIL /0 SKIP**, alle971 Basisknoten +41 neue erhalten. Separate echte Tk-Abnahme: **29 PASS /0 FAIL /0 SKIP**, alle26 Basisknoten +3 neue. Gezielte Metadatenkorrektur95/95 und H10-Capturefix87/87 PASS, nicht addieren. Ursprüngliche aktuelle DeepX-Hostoutputfixture ausdrücklich an den nun tatsächlich im Loop ausgeführten Top-k-Abschluss angepasst; historische Negative erhalten.

Genau drei normale GUI-Starts, je MobileNet/b135 und drei Nativezeilen,100/10/1, Quality32/Bootstrap100. H8:3/3 technisch OK, Quality FAIL,486,473s. H10H:2/3 technisch akzeptiert, Gesamtstatus partial/Quality FAIL,456,413s. DeepX:3/3 technisch OK, Quality INCONCLUSIVE,488,487s. Alle neun Messreihen haben100/100 eindeutige Start-/Top-k-Endpaare und100 Abschlüsse; Mean/P50/P95 und FPS unabhängig nachgerechnet, tatsächliches GUI-Widget und Reporter-CSV9/9 geprüft. Das ist keine vollständige9/9 technische oder wissenschaftliche Freigabe.

H10H-Split: eigener R9C-Einbaufehler im ungemessenen `_capture_raw_hailo10_sample` (unzulässiger Klassifikationsblock mit undefiniertem `task_value`) verhinderte nach erfolgreicher Messung das semantische Manifest. Der strikte Consumer-Join blockierte korrekt. Vier fehlerhafte Zeilen entfernt, direkte Raw-Slot-/Diagnoseregression und vorhandene Fälle lokal PASS. Kein zweiter H10-Start: finaler Capturefix noch ohne eigene normale H10-GUI-Abnahme, Split-Quality unavailable. Dieser Implementierungsfehler ist unabhängig von H10-02 und wird nicht als Compilerverlust bezeichnet.

Alle drei Enddialoge bestätigt, Worker/GUI beendet, Produktsperren frei, Settings bytegleich wiederhergestellt. Artefaktabschluss je PASS. Vier warme Artefakte je Setup, null Modellbuilds. Eine notwendige H8-Wrappergeneration9,760s unter290s-Limit, danach gebundener Reuse. Kumulativ GUI3/3 verbraucht, Wrapper1/3, Output-GUI0/3, zusätzliche Rohdiagnose-Hardware0. Keine weiteren Starts, keine HEF/DXNN/TRT-/Firmwarebuilds, keine Energie/Power-/Kalibrierungsarbeit.

H10-02 und DX-01 bleiben durch getrennte Offlinebefunde eingegrenzt, nicht repariert; keine Score-/Box-/Schwellenwertkorrektur und kein Tuning bis PASS. BACKFILL-02 und DeepX-P2 bleiben abgeschlossen. Qualitätsforschung, größere Populationen, Ranking und finale Langzeitenergie bleiben zurückgestellt. Künftige gemeinsam genutzte Klassifikations-Energierunner schließen Top-k ein; vorhandene Endpoint-/Commandbelege wurden korrekt fortgeführt, alte J/Bild-Belege bleiben historisch.

Finale Hauptinstallation/Scriptmirrors/Source-Manifeste werden mit der Lieferung geprüft; Build-ID unverändert v2.83-r9b-request-latency. Source vor H8/H10 und nach dem lokalen Capturefix vor DeepX getrennt dokumentiert. Auftragsdiff gegen gesicherte Sitzungsanfangsbytes, nicht gegen altes HEAD. Keine Commits/Pushes/Resets oder Löschung von Laufzeitdaten.

Lieferung: /home/kmika/.local/share/onnx-splitpoint-codex/v283_R9C_20260917_110622_ciFKaG/ABSCHLUSSBERICHT_R9C.md und ERGEBNISSE_R9C.zip. Nächster sinnvoller eigener Auftrag: einmalige normale H10H/MobileNet-b135-GUI-Abnahme des lokal korrigierten Capturepfads mit vorhandenen Artefakten. H10-/YOLO26-Vertragsentwicklung und DeepX-/YOLO26s-SDK/Compilerursache davon getrennt, keine automatischen Folgeaktionen.


### R9C H10-Nachtest – normale MobileNet/b135-GUI-Nachabnahme (17.09.2026)

Genau ein zusätzlicher normaler GUI-Workflow auf unverändertem finalem R9C-Produktcode: MobileNetV3 Large, orin_nx_hailo10_01, feste Boundary b135.100 Frames/10 Warmup/1 Wiederholung, Quality maximal32 Bilder/100 Bootstrap. Native H10H Full, setup-lokales TRT Full und H10H→TRT-Split: **3/3 technisch erfolgreich**, je100/100 eindeutige Start→Top-k-Paare und100 tatsächliche Top-k-Abschlüsse. Dauer454,997s; GUI „Abgeschlossen mit Qualitätswarnungen“. H10H Full/Split Quality FAIL, TRT Full PASS; keine wissenschaftliche Gesamtfreigabe.

Die konkrete Capturelücke ist für diesen normalen GUI-Scope geschlossen: kein task_value-/Dumpfehler, Output- und Boundarymanifest vorhanden und hashgleich zur bestehenden Consumer-Attestierung; exact_quality_native_engine_command_and_boundary_match. Mean/P50/P95 aus gespeicherten Paaren unabhängig nachgerechnet und mit Reporter-CSV und tatsächlichem GUI-Widget abgeglichen. Split25,378450/26,216321/26,381196ms; H10H Full9,168241/9,210503/9,336424ms; TRT Full1,287758/1,286464/1,300421ms.

Normale Warmprüfung4 HITs, null Kalt-/Modell-/Wrapperbuilds, keine Zusatzdiagnose-/Direktinferenz, Energie AUS, keine Poweränderung. Enddialog bestätigt, alle Worker beendet, regulärer Remote-Cleanup rc0, Artefaktabschluss846/846 PASS. Workflow- und H10-Produktsperren frei; Settings, normale Konfiguration/Profile sowie alte R9C-Journale/START-Marker bytegleich. Neues Journal1/1 verbraucht, kein Retry.

Softwarebelege getrennt: neue direkte Capture-Regression1/1 PASS (0,74s), einmalige lokale Produkt-Spawnprobe PASS; unveränderter R9C-Source/Installedstand vor Hardware PASS.1012/29 nicht wiederholt; sie bleiben historische R9C-Nachweise. Produktcode unverändert, nur dieser Arbeitsstand und seine bestehenden Manifestzeilen gepflegt. Nichtfatale Matplotlib-Threadwarnungen im Reporter; GUI und Finalisierung erfolgreich. H10-/YOLO26-Nulloutputs und DeepX-/YOLO26s-XYXY bleiben offen, keine erneute Untersuchung.

Lieferung: /home/kmika/.local/share/onnx-splitpoint-codex/v283_R9C_H10_Nachtest_20260917_133258_oHq55p/ABSCHLUSSBERICHT.md und ERGEBNISSE_R9C_H10_NACHTEST.zip. Nächste Handlung: Ergebnisbelege prüfen; kein weiterer automatischer Workflow.


### R9D – Phase 0: begrenzte H10-/YOLO26-Vertragsprüfung (17.09.2026)

Auftrag und Vorbefunde gelesen; tatsächliche Sandboxprobe mit Tk/Loopback/Schreibrechten PASS. Ausgangsstand und uncommittete Vorarbeit gesichert. Ausschließlich H10 YOLO26s/b364, danach bedingt m/b398; zuerst lokale Compilerfähigkeit. Noch kein neuer Vertragsweg belegt, keine Produktcodeänderung, kein Hardware-/GUI-/Buildstart. Energie AUS; Finalvorbereitung bleibt Planung. Fortschritt und kumulative Zähler im externen R9D-Auftragsordner.


### R9D – Phase 1: belegter terminaler Präzisionsweg, noch kein HEF

DFC und HailoRT5.3.0 konkret aufgelöst; native alte HEF-Ausgänge UINT8/FCR mit skalarer QuantInfo. Ein lesender SSH-Metadatenaufruf, keine Inferenz. Für s/b364 einmalige lokale Übersetzung sowie positive/negative SDK-Konfigurationsprüfung:206 Layer, exakt8 terminale Präzisionsänderungen, unveränderte Anschlüsse/Convs/Backbone. Einziger Kandidat: Ausgang a16 plus terminaler EW-Mul a16/OnMac;15 Aktivierungsbits sind nicht16 unabhängige Bits. Bestehendes extra_model_script trennt Recipe/Cacheidentität; Opt1/B500/Batch8/Seed/Input bleiben. Keine Auswahl nach AP.

Fokussierte Regression180 PASS/1 falsch positiver AST-Stringtest; Identifierprüfung im bestehenden Test eng korrigiert, direkter Nachtest1 PASS. Produktlaufzeitcode unverändert. Kein gebauter Kandidat, keine normale Integration, GUI-/Quality-Abnahme offen. Build-/Geräteinputs bisher0; kontrollierter erster s-P1-Versuch wird separat journalisiert. Full/DeepX/Energie/Finalkampagne unverändert.


### R9D – Phase 2: P1-Versuch am Sandbox-GPU-Zugriff gestoppt

Genau ein bestehender Buildpipelineaufruf für YOLO26s/b364, explizit neue terminale Präzisionsrecipe bei unverändertem Opt1/B500/Batch8/Seed/Input. Vor dem DFC-Worker in compiler_context abgewiesen: hailo_gpu_selection_unresolved; NVIDIA-Abfrage rc9, /dev/nvidia0, /dev/nvidiactl und /dev/nvidia-uvm in der tatsächlichen Sandbox nicht vorhanden. Kernel-Versionsdatei lesbar. Infrastrukturblocker dieser Sitzung, keine belegte allgemeine Compiler-Unfähigkeit oder Treiberstörung des Hosts.

Pipeline0,027s, kontrollierter Worker0,785s/Exit2, kein Timeout, keine eigenen Prozessüberlebenden. s-P1-Budget1/1 verbraucht, gesamt1/2.0 neue HEFs/Engines; m nach fehlendem s-Nachweis nicht gestartet. TRT/Wrapper/Diagnoseinputs/GUI/Energie/Windowprobe jeweils0. Kein Retry oder Buildausweichen. Alte negative UINT8-Belege erhalten, keine neue Qualityauswertung.

### R9D – Phase 3: lokale Abschlussbelege, normale Abnahme offen

Lokaler DFC5.3.0-Vertragsnachweis für genau einen s-Kandidaten:206 Layer,8 terminale Präzisionsänderungen, positive und negative Konfigurationsprüfung. Keine16-Bit-/Scoreerhalts- oder Kompilierbarkeitsbehauptung. Produktlaufzeitcode/BUILD_ID unverändert; einziger Testfix unterscheidet AST-Identifier von Diagnosestrings. Initial180 PASS/1 FAIL, korrigierter Knoten separat1 PASS;181 verschiedene Knoten mit passenden PASS-Belegen, kein gemeinsamer finaler181er-Lauf. Zusätzliche AST-Positiv-/Negativprobe3/3. Historische1012/29 wegen unveränderter Produktsource nicht erneut ausgeführt und nicht als neue Abnahme gezählt.

Kandidat noch nicht normal integriert; neuer HEF, TRT-P2, numerischer Nachweis, Preview, normale GUI und Quality für beide Modelle offen.0 Settings-/Profiländerungen; keine Wiederherstellung erforderlich. Zwei direkt betroffene alte P1-HEFs gegen vorhandene Receipts geprüft, zwölf vorgegebene Sourcezeilen erhalten. Fullartefakte unangetastet. Source-Manifeste nur für Test/Arbeitsstand fortgeschrieben; keine neue Hash-/Cacheebene.

Lieferung im R9D-Auftragsordner: ABSCHLUSSBERICHT_R9D.md, TEST_RESULTS_R9D.json, H10_FAeHIGKEIT.md, GUI_COVERAGE.json, kumulatives Journal, vollständiger Diff gegen Startbytes und ERGEBNISSE_R9D.zip. Finaltest nur vorbereitet, kein automatischer Vorbereitungstest/Final-/Nachtlauf. Nächster möglicher Auftrag erst nach Ergebnisreview und Klärung der Sandbox-GPU-Sicht; erneuter s-P1-Versuch bedarf neuer ausdrücklicher Freigabe, ohne dieses verbrauchte Journal zurückzusetzen. DeepX/Collector/Classification/Latenz/Energie unverändert.


### R9E – allgemeine Builderfixes und lokale Fehlerpfadabnahme (18.09.2026)

Im Hauptrepo implementiert, VERSION/BUILD_ID unverändert: Der gemeinsame Hailo-Builder hält erfolgreich gespeicherte parsed/quantized-HARs in den vorhandenen Ergebnisfeldern auch bei Fehler, fehlender Kindantwort und nativem Managed-Timeout fest. Das bestehende Phasenprotokoll bindet abgeschlossene Speichervorgänge an aktuelle SDK-PID, Startzeit, Quell-/Cacheidentität, Pfad und Dateigeneration. Atomare Speicherung übernimmt keine liegengebliebene alte Datei. Container-/Metadaten-/NPZ-CRC-Prüfung ohne SDK unterscheidet fehlende, leere, defekte, fremde und nur geparste Zwischenstände. HAR ist kein HEF-HIT; SDK-Ladefähigkeit und vollständiger Resume-Vertrag bleiben eigene offene Nachweise.

Phasen stammen aus den tatsächlichen Builderereignissen: optimize completed / compile started bleibt bei Gesamtfristablauf sichtbar, statt einer ungenauen SDK-Prosephase compile_prep. Explizite kurze Hard-Timeoutwerte werden nicht mehr auf60s angehoben; Defaults und Unlimited-Vertrag bleiben bestehen. Bestätigter Timeout wird trotz SIGTERM/SIGKILL beim kontrollierten Cleanup als TRANSIENT_INFRASTRUCTURE eingeordnet; echte deterministische Unrealisierbarkeit bleibt wiederverwendbare Negativevidenz. Keine automatische Backfill-Freigabe durch Timeout.

Lokale gemeinsame Endprüfung: **70 PASS /0 FAIL /0 SKIP**,46,82s Pytest, Exit0. Enthält31 neue R9E-Fälle, bestehende Builderregressionen, Outcome-/Negativspeicher-/Backfill-/Timeoutverträge und alle bestehenden Script-Mirrors. Reale lokale Kindprozesse verwenden ausschließlich ein kontrolliertes Ersatz-SDK; beide4-s-Timeouts schließen ihren Prozessbaum. Normale GUI-Diagnosehandler, Live-Diagnoselog, gespeichertes Resultat und Verlauf mit ersetzten Dateidialog-/Fenstergrenzen geprüft; keine Tk-Instanz, kein Workflowstart. Erster28er- und zwischenzeitlicher69er-Lauf sind getrennte Vorprüfungen, keine zusätzlichen Endprüfungsfälle. Beide vorhandenen echten HAR-Container außerdem rein lesend ohne SDK geprüft; keine Übernahme als aktueller Checkpoint.

Hostbefunde getrennt: Der alte Hostbuild bleibt Timeout1750s nach erfolgreicher Optimierung1661,665s und begonnener Kompilierung. Der bereits vorliegende einmalige Compile-only-Helfer lud quantized_model, translate0/optimize0/compile1, scheiterte nach89,57s/Exit2 mit BackendAllocatorException/Agent infeasible; **kein HEF**. Keine Wiederholung. Es gibt keinen normalen Produkt-Compile-from-HAR-Einstieg; der externe SDK-Helfer ist ausschließlich Diagnose. Scoreerhalt und reale normale GUI-/Qualitätsabnahme bleiben offen; H10-/DeepX-Outputfragen bleiben offen. Windows/WSL-Timeoutrettung ohne nachweisbare Kind-PID/Dateigeneration übernimmt keine HARs; keine WSL-Abnahme in diesem Linux-Auftrag.

Keine GUI-/Eval-/Nacht-/Fallbackstarts, SSH-/GPU-/Compiler-/Energieaktionen durch Codex, keine neue Sperrhalterschleife, keine Subagenten; normale Konfiguration und Artefakte erhalten. Lieferung unter /home/kmika/.local/share/onnx-splitpoint-codex/v283_R9E_Fixes_Tests_iojb77up: ABSCHLUSS_FIX.md, TEST_RESULTS_FIX.json, vollständiger PATCH_R9E.diff gegen die gesicherten Anfangsbytes einschließlich neuer Tests und vorhandener Source-Manifeste. Danach ausschließlich fokussierte Hosttests/Installed-Prüfung und Archivierung, dann Ende. Kein späterer Benchmark autorisiert.

### R9F – technischer Outputvertrag und begrenzte automatische Auswahl (18.09.2026)

H10-02: Allgemeine Prüfung im vorhandenen Native-Vertragsresolver und Backend-Backfill implementiert. Nur standardmäßige ONNX-Graphsemantik (Sigmoid, identische Kanaltrennung, tatsächliche Scoreauswahl bis zum Ausgang) plus exakte HEF-/P1-/P2-Bindung und native QuantInfo können einen vollständigen Kollaps des deklarierten Wahrscheinlichkeitsbereichs auf den Nullpunkt beweisen. Modell-, Layer-, Boundary- oder feste Klassenanzahlbedingungen gibt es nicht. Fehlende/unklare Angaben bleiben UNKNOWN, AP-FAIL, leere Detektionen und Timeout sind keine technischen Ausschlüsse. Ein vorhandenes HEF bleibt BUILT/Cache-HIT; fremde a16-Compilefehler werden nicht übertragen. Deterministische Rangfolge, MISS-Stopp, endliche Budgets und bestehende Forced/Frozen/Resume-Bindungen bleiben erhalten. UNKNOWN beendet nur den betroffenen Backendvertrag; unabhängige Cases werden konsistent commitet.

Normale Config: ausschließlich Schema15/Outputpolicy v1 mit externem Backup migriert, sämtliche Run-Moduswerte erhalten. Profileditor/Summary zeigen die wirksame Policy; explizite Warm-only-Admission und build_missing_engines=false bleiben beim Speichern erhalten. Keine Produktversion-/BUILD_ID-Änderung. R9E-Builderfixes, Top-k, Latenz, H8-Backfill, DeepX-P2 und Collector sind unveränderte abgeschlossene Vorarbeiten; DX-01/DeepX-YOLO26s-XYXY bleibt eigener Folgefall.

Lokaler Archivnachweis für beide alten H10-HEFs: vorhandene portable Quality-Bindings und aktuelle Receipt-/Quellbindung ergeben INCOMPATIBLE. Die normale lokale Auflösung hat dagegen kein HailoRT und keine benachbarte gebundene Metadatendatei: required UNKNOWN. Echte Tk-Profileditor-/Summary-/Snapshot-Vorschau plus normaler lokaler Selektor auf vorhandenen Predictionbelegen: s beginnt b364 (Folgerang b365/b021/b023), m beginnt b398 (b399/b038/b040). Keine erzwungenen Cases. Tasklokaler vorhandener Aufwandssnapshot100/10/1, Quality32/Bootstrap100; Kandidatenwahl nicht eingefroren, normale Hardware-/Datasetregistry. Jeweils am ersten Kandidaten gestoppt, keine Ersatzboundary ausgewählt und keine warme vollständige Ersatzkette behauptet. GUI-Workflow-/Hardwareabnahme beider Zellen NICHT AUSGEFÜHRT:0/2 Starts,0/2 SSH, je1/4 Kandidaten,0 HEF/DXNN/TRT/Firmware/Wrapper, Energie AUS. Keine Rohbildaufnahme, kein Eval-/Final-/Nacht-/Fallbackstart.

Automatische technische Auswahl ist im lokalen Produktpfad implementiert; eine erfolgreiche normale Auswahl samt nativer Ersatzsplitinferenz ist noch NICHT abgenommen. Die alten Grenzen b364/b398 bleiben unrepariert. Der R9E-Allocatorfehler belegt allein den konkreten anderen a16-Kandidaten, keine universelle Compilergrenze. Kleinster nächster Schritt: vorhandene exakte native Outputmetadaten am normalen Controller-Resolver verfügbar machen, ohne neue Modell-/Wrapperbuilds; dann zunächst dieselbe deterministische Scope-/Warmkettenprüfung. Kein automatischer Folgelauf. Finale neue Testzahlen und Source-/Installed-/Mirrorbelege stehen im externen ABSCHLUSSBERICHT_R9F.md/TEST_RESULTS_R9F.json unter /home/kmika/.local/share/onnx-splitpoint-codex/v283_R9F_20260918_093308_gVtTP0; frühere1012/29 und R9E70/229 sind keine neue R9F-Ausführung.

Lokale R9F-Endabnahme: **679 PASS /0 FAIL /0 SKIP** in einer gemeinsamen relevanten Regression einschließlich echter Tk-Tests, keine Hardwareausführung. Der erste gemeinsame Versuch hatte677 PASS/2 FAIL: zwei bestehende Tests erwarteten noch erlaubtes Force-Rebuild bzw. den alten DeepX-Schleifennamen. Eng korrigiert; Force-Verbot und anschließender Missing-Binding-Stopp werden beide geprüft, die Versiegelungsreihenfolge bleibt geprüft. Gezielt2 PASS, danach gemeinsame Endabnahme; Fehlversuche bleiben im Bericht, Zahlen nicht addiert. Keine Änderung dieser beiden Produktpfade. Source/Installed und Script-Mirrors am Endstand werden mit vorhandenen Werkzeugen geprüft.

### R9F-Fortsetzung – normale frühe Metadatenbeschaffung (18.09.2026)

H10-02: Der normale Generator übergibt für aktive technische H10-Part1-Policy die aufgelösten Hardwareziele an den vorhandenen Resolver. Der bestehende gerätefreie HEF-Reader ist für konfigurierte separate lokale Interpreter und den vorhandenen lesenden SSH-Transport wiederverwendbar. Native/Host-Dtype, Form/Ordnung, QuantInfo und explizite Rundung werden getrennt erhalten; exakte HEFbytes/Größe und P1/P2 bleiben geprüft. Bereits gebundene intrinsische Boundarymetadaten benötigen keinen erfolgreichen Qualitylauf und keine P2-Engine. Alte vollständige Bindings bleiben wiederverwendbar; frühe unvollständige Bindings dürfen nur als exakte Datei-/Setupreferenz dienen und müssen neu gelesen werden. Keine Registry-/Cachearchitektur, keine neuen Produktidentitäten. Wiederholungen innerhalb derselben Generierung nutzen die unveränderten dateigebundenen Beobachtungen; Receipt-/Metadatenänderungen invalidieren diese Wiederverwendung.

Normale echte Tk-Editor-/Summary-/Startsnapshotauflösung und bestehender Generator jetzt für beide Modelle auf normalen Modellpfaden und Registry ausgeführt, ohne Archivbinding oder forced_cases. Ursprüngliche s/b364 und m/b398 jeweils als tatsächlicher HEF-Cache-HIT bestätigt; Full-H10 ebenfalls warm wiederverwendet. Beide frühen H10-Verträge bleiben required UNKNOWN: Im konfigurierten lokalen Reader fehlt hailo_platform und der normal aufgelöste Artefaktpfad enthält keine exakte Remote-HEF-Referenz beziehungsweise benachbarte gebundene Metadaten. Kein Remote-Pfad aus Diagnosearchiven geraten. Unabhängige generische Casezuordnungen bleiben erhalten; sie sind kein akzeptierter H10-Split. Historische UNKNOWN-Ereignisse im START_JOURNAL erhalten, lokale Neubewertungen ergänzt. Keine neue Boundary betrachtet.

Normale native GUI-/Hardwareabnahme daher weiterhin NOT_RUN: kumulativ GUI0/2, SSH0/2, je1/4 unterschiedliche Kandidaten, HEF/DXNN/TRT/Firmware/native Messwrapper je0, Energie AUS. Keine vollständige warme Ersatzkette bewiesen. Beide Original-HEFs bleiben unverändert und unrepariert; automatisches technisches Nachrücken ist lokal geprüft, seine reale Ersatzsplitinferenz bleibt offen. Positive SDK-Konfiguration und der negative alternative R9E-Allocatorversuch werden nicht erneut untersucht oder ausgeführt. DeepX-XYXY bleibt getrennt; keine Top-k-/Latenz-/Collector-Runde.

Abschließende gemeinsame lokale Auswahl: **730 PASS /0 FAIL /0 SKIP**,241,53s Pytest, Exit0. Alle679 bisherigen Endknoten erhalten, dazu34 neue Metadatenfälle und17 bestehende Generatorfälle. Echte Tk-Prüfungen und vollständiger Script-Mirror-Test enthalten. Der reale Generator/Resolver/Receipt-/Backfillpfad bleibt in den neuen Tests ungemockt; kontrollierte äußere Readerprozesse und vorhandene Testartefaktfixtures ersetzen Hardware. Keine neue Hardwareabnahme daraus abgeleitet. Vorprüfungen und deren konkrete Setup-/Fixturefehler bleiben separat dokumentiert; Zahlen nicht addiert.

Vorhandene ArtifactStore-Referenzen werden ausschließlich über bestehenden read_only-Zugriff und exakten HEF-Hash/Größe verfolgt. Fehlende beziehungsweise fremde Quellen erzeugen keinen Ausschluss. Bekannte Metadatendateien sowie Registry-DB/WAL invalidieren bei Änderung die lauflokale Beobachtung; eine defekte optionale Registry blockiert unabhängige lokale Quellen nicht. Die reale endliche Zuordnung enthielt keine nutzbare Metadatendatei und keine exakte Remote-Abbildung; deshalb weiterhin UNKNOWN an den ursprünglichen Rängen. Keine neue Boundary und kein Geräte-/Buildstart.

Source-/Installed-Verifikation mit vorhandenem Werkzeug: Exit0,2018 Dateien; keine entfernten Dateien. Bestehende Config-/Profil-/Registrydateien gegenüber Fortsetzungsbeginn unverändert. Vollständiger Fortsetzungsdiff einschließlich neuer Tests extern rekonstruierbar; vorherige Lieferung bleibt erhalten. Berichte, JUnit, Quellenzuordnung, Journale und ERGEBNISSE_R9F_FORTSETZUNG.zip liegen unter fortsetzung_metadaten/ im bestehenden R9F-Auftragsordner. Kein automatischer Folgelauf.


### R9F-Fortsetzung – HEF-Readertransport und normale Auswahl (18.09.2026)

H10-02: Fehlende frühe Dateibereitstellung im normalen Reader geschlossen. Ohne lokalen HailoRT-Import, ohne Qualitybinding und ohne registrierten Remote-Pfad wird das konkrete vorhandene Receipt-HEF über SSHTransport/SCP privat auf dem eindeutigen aktiven H10-Ziel bereitgestellt. Der bestehende HEF-Reader prüft Empfängerhash/Größe, native Streamzuordnung und QuantInfo; er öffnet kein Device/VDevice/InferModel. Vorhandene Workflowleases werden übernommen, kein zusätzlicher Plattformlock. Timeout/Cancel/Fehler schließen eigenes Staging; Primär- und Cleanupfehler bleiben getrennt, unbestätigtes Cleanup sperrt weitere Arbeit über die vorhandene Lease-Registry. Quellenwechsel invalidiert Beobachtungen; gelöschtes erfolgreiches Staging invalidiert sie nicht. Keine neue Auswahl-/Hash-/Registryarchitektur.

Realer normaler Generator auf Smartmirror2 mit konfiguriertem orin_nx_hailo10_01: s/b364 und m/b398 als vorhandene HEFs tatsächlich übertragen und in der H10-Runtime gerätefrei gelesen. 2 Runden,6 SSH+2 SCP,alle Exit0,33.611.776 Bytes,2 Reader,0 Deviceöffnungen/0 Inferenz; privates Staging jeweils erfolgreich entfernt. Beide nativen UINT8-Verträge beweisen den vorhandenen allgemeinen Scorebereichskollaps und bleiben BUILT/HIT, technisch INCOMPATIBLE. Danach reguläre Reihenfolge: b365 bzw.b399 mit bereits vorhandener exakter negativer Compile-Evidenz ausgeschlossen; H10 wählt s/b021 bzw.m/b038. An den frühen Featuregrenzen ist die enge Scorekollapsprüfung nicht anwendbar (required=false), daraus folgt kein numerischer/Quality-PASS. Alte HEFs bleiben unverändert und unrepariert.

Dabei sichtbar gewordener enger Integrationsfehler behoben: Der Planabgleich prüfte die Vereinigung verschiedener Backendfälle gegen eine Einzelquote. Er prüft nun den bestehenden Backend-Auswahlzustand, jede Vertragsquote und exakte Fallmitgliedschaft. Frozen-/historische Fälle behalten ihre bisherige Grenze. m-Generierung vollständig normal abgeschlossen; s aus demselben Zustand lokal fortgesetzt, ohne erneuten Remoteaufruf. Kumulativ je3/4 Kandidaten,keine Modell-/Engine-/Firmware-/Messwrapperbuilds.

Echte normale GUI-/Inferenzabnahme weiterhin NOT_RUN (GUI0/2): Für s/b021 und m/b038 fehlt der aktuelle Nachweis der vollständigen passenden nativen uint8_dequant_fp16-P2-/Full-Enginekette. UNKNOWN ist hier kein belegter Engine-MISS. Der bestehende normale TensorRT-Cachepreflight benötigt eigene aktuelle Remote-/ABI-/Receiptprüfung; er wurde nicht als weiteres HEF-Lesen ausgegeben. §5 des aktuellen Auftrags verlangt bestätigte Warmkette vor GUIstart. Native Outputs,Capture/Consumer,CompletedTask/P2/Einbildlatenz und normale Ergebnis-CSV/GUI bleiben daher offen. Vorhandener Python-H10-Runner erfordert keinen neuen Messwrapper. Keine Energie,kein Nacht-/Final-/Fallbackstart.

Gemeinsame lokale Endabnahme: **805 PASS /0 FAIL /0 SKIP**,272,97s Wandzeit,Exit0. Alle730 bisherigen Endknoten enthalten,29 neue Transport-/Fehler-/Planabgleichfälle und46 bestehende Frozen-Auditfälle. Echte kontrollierte SSH-/SCP-/SDK-Kindprozesse, Identität, Timeout/Cancel/Cleanup, Wiederverwendung, vorhandene Leases und normale Generatorkette geprüft; echte Tk-Tests und vollständiger Script-Mirror-Test enthalten. Spawn-sicherer GUI-Treiber separat: Worker Exit0. Erster gemeinsamer803-PASS-Lauf bleibt erhalten; danach zwei konkrete Belegverlustfehler bei Quelllöschung und falscher Ausgabeform mit RED2/GREEN2 korrigiert und gemeinsam abschließend geprüft. Produkt-/Testsource während des finalen Laufs unverändert. Source-/Installed-Nachweise und vollständiger Fortsetzungsdiff werden getrennt unter fortsetzung_reader_transport im bestehenden R9F-Aufgabenordner geliefert. Keine Hardware-PASS-Ableitung aus lokalen Tests.

## R9G – normale HOST-Ausführung vorbereitet (18.09.2026)

Lokale Vorbereitung gemäß neuem Auftrag unter `v283_R9G_20260918_133810_7vb48dms`: neue `H10.yaml` (yolo26s/yolo26m, nur H10-Setup, drei normale Runprofile) und `EVAL.yaml` (sieben Modelle, drei Setups, sieben Runprofile) aus unverändertem `CompleteSetDev.yaml` durch `apply_run_mode` und normalen Profilspeicher/Resolver erstellt. Aktueller Standardmodus ist mit dem vorhandenen Profilsnapshot-Vertrag gebunden; keine Privat-Registry und kein historischer Resume. Native 100/10/1, Quality 32/100 beziehungsweise 500/500; H10 ohne Energie, EVAL Native FS/command 1 s × 3 mit normalem r6-reviewed-Collector und bestehendem Quellen-/Retryschutz. Optionale Fensterprobe und generische Energie aus. Die kurze Energiemessung bleibt Screening.

Normale Buildfreigabe: `build_missing_engines=true`, Wiederverwendung vor fehlenden Standardrezept-Artefakten, Force aus. Je Backend/Modell höchstens vier Kandidaten, Nachrücksuche je Modell 12 Starts (Hailo-Part1 höchstens 6, TRT-Part2 höchstens 6); keine feste Ersatzboundary oder Warmpräferenz. Hailo-/Cold-/Nachrücktimeout 5400 s, TRT-Nachrücktimeout 1800 s; äußere Remote-/Native-Supervision endlich auf 21600 s gesetzt. B500, balanced/Opt1, Seeds, Margen, Queue/Inflight und Energiequellen unverändert. Produktpreflight bleibt aktiv ohne Warm-only-Startgate.

Software: 16 direkte Prüfungen der tatsächlichen Profile (7,64 s) und 25 relevante bestehende Regressionen einschließlich Script-Mirrors (3,10 s, zwei bestehende DeprecationWarnings) PASS, keine Skips. Unverändertes Host-`validate_scope`, Schema, sichtbare Summary und tatsächlicher Laufzeit-Startsnapshot geprüft, einschließlich negativer Scopefälle und Erhalt der Budgets beim Reload. Der erste direkte Testlauf hatte 14 PASS/2 FAIL wegen `setup_id` statt des tatsächlichen Targetfeldes `id` im neuen lokalen Test; ausschließlich dieser Testfehler korrigiert. Kein belegter zusätzlicher Produktdefekt, keine Produktcode-/GUI-Helferänderung. Logs/JUnit liegen außerhalb des Repos in `local_o_02zenp`.

Echte GUI-/Hardwareabnahme folgt durch den HOST nach Freigabe seiner Vorbereitungssperre; lokal keine GUI-, SSH-, GPU-, Compiler- oder Energieprozesse gestartet. Kein Hardware-PASS vorweggenommen. Nächster Schritt: normaler H10-GUI-Workflow einschließlich erforderlicher fehlender Builds, danach der freigegebene EVAL; Quality-PASS ist kein separates Startgate. Kein ZIP, keine neue Version, kein Commit/Push.

### R9G – belegter H10-Ablaufdefekt nach echtem GUI-Lauf (18.09.2026)

Der HOST-Lauf `r9g_h10_20260918_135548` erreichte den geordneten GUI-Abschluss (Operator Exit 0, Cleanup bestätigt). Zentrale Qualität: zehn technisch abgeschlossene Auswertungen, vier PASS, drei FAIL, drei INCONCLUSIVE, keine technischen Qualityfehler. Native Performance startete dagegen null Zeilen: `global remote infrastructure blocked: deepx=remote_ssh_missing; hailo8=remote_ssh_missing`. Der H10-Scope enthielt durch eine veraltete materialisierte Full-Zuordnung zusätzliche TRT-Full-Zeilen auf nicht ausgewählten H8-/DeepX-Systemen. Dies ist ein Ablaufdefekt und kein Geräte- oder Qualitätsverlustbefund.

`native_full_quality.resolve_native_full_plan` verwendet die bestehende Full-Zuordnung jetzt nur als Fallback, wenn keine aktuelle Produzentenauswahl erkannt wird. Erkannte logische Auswahlen werden nicht mehr um alte Produzenten erweitert. Die neuen Regressionen prüfen Entfernen und Deaktivieren für alle drei Produzenten, wiederholte Modusauflösung, die tatsächliche Stage-Matrix und Erhalt des Legacy-Fallbacks. H10/EVAL wurden erneut normal aus CompleteSetDev erzeugt; H10 hat vier Full-Zeilen und zwei ausgewählte Splits ausschließlich auf H10, EVAL behält 42 Full-Zeilen und den vollständigen beauftragten Scope. Benutzerprofil, Hoststarter, GUI-Helfer, wissenschaftliche Kriterien und Buildbudgets unverändert.

Software: Vor Korrektur sechs erwartete Regression-FAILs/ein PASS (1,16 s), danach sieben PASS (1,20 s). Direkte Profil-/Snapshot-/Runnerkonfigurationsprüfungen samt unverändertem Host-`validate_scope`: 18 PASS (92,98 s). Bestehende gezielte Regressionen einschließlich No-SSH-Fehlergate und Script-Mirrors: 17 PASS (2,48 s); keine Skips. Logs/JUnit und Befehle liegen im Sitzungsordner unter `local_xqk8yrh2`. Sourceindizes werden für die tatsächlich geänderten Quellen mit dem vorhandenen Installed-Inventar aktualisiert und anschließend unverändert verifiziert; das endgültige Prüfergebnis steht im lokalen Kontext.

Keine eigene GUI-/SSH-/GPU-/Compiler-/Energieausführung. Die korrigierte reale Native-Ausführung bleibt durch den HOST zu prüfen: ein relevanter H10-Nachtest, anschließend der autorisierte 7-Modell-Eval auch bei bekannten technischen Negativfällen oder Quality-FAIL. Kein weiterer Lauf zur Qualitätsverbesserung, kein ZIP, keine neue Version, kein Commit/Push; Vorbereitungssperre und STATE bleiben HOST-eigen.

### R9G – zwei belegte Dispatchfehler nach finalem HOST-Eval (18.09.2026)

Der HOST-Eval `r9g_eval_20260918_144510` endete mit bestätigtem Cleanup und Operator Exit 0, fachlich jedoch technisch fehlgeschlagen: 32/63 Native-Zeilen erfolgreich, 31 fehlgeschlagen/blockiert, 14 zusätzliche Validierungsfehler auf gemessenen Zeilen. Zentrale Qualität: 30 abgeschlossen, 19 PASS/7 FAIL/4 INCONCLUSIVE; fehlende Qualitätsbindungen bleiben technische Fehler. Energie: 32/32 gestartete Zeilen und 96/96 logische Wiederholungen verifiziert, kein Collectorfehler. Diese 1-s-Serie bleibt Screening, keine Langzeitabnahme.

Alle 14 H8/H10-Dispatches brachen vor den Providern mit `ort_tensorrt selected no deterministic case container` ab. Der Full-TRT-Qualitätsbegleiter verwendete die setupgebundene Split-Auswahl des generischen Performance-Eigentümers (hier DeepX), sodass auf anderen Setups kein Container übrig blieb. Im exportierten Suite-Template verwendet ausschließlich dieser bereits deklarierte Full-Begleiter jetzt den normalen ersten Modellcontainer innerhalb der expliziten Benutzerauswahl. Setup-/Endpointprüfung, genaue Qualitätsbindung, Split-Auswahl und Performance-Eigentümer bleiben erhalten. Keine feste Boundary, keine neue Auswahlpolicy.

Der separate YOLOv7-DeepX-Dispatch scheiterte vor Upload an `resolve_path_read_only rc=70`: lokale SSH-PID Exit 0, Ausgabe vollständig, Reader am Abschluss beendet, keine überlebenden Prozesse; Diagnosephase dennoch `pipe_drain_incomplete`. Der Transport wartete nur 250 ms auf den Reader und klassifizierte den Fehler vor seinem abschließenden Join. Er wartet nun vor der Klassifikation mit dem vorhandenen begrenzten 1-s-Readerbudget. Tatsächlich gehaltene Pipes bleiben rc70 und werden über die bestehende Prozesseigentümerschaft beendet; Exitcodes, Timeout-, Abbruch- und Cleanupgates bleiben wirksam. Kein automatischer Retry.

Software: elf neue Regressionen nach Fix PASS (3,98 s), zuvor sieben gezielte FAIL/vier PASS (3,33 s); ein erster Testentwurf hatte zusätzlich falsche CLI-Testargumente (neun FAIL/zwei PASS). Relevante bestehende Regressionen: 50 PASS/ein veralteter Fixturefehler (5,55 s); ausschließlich jener Test verwendet nun den regulären Runner-Konstruktor für die bestehenden Abschlusslocks, sämtliche Assertions erhalten; direkter Nachtest ein PASS (1,93 s). Tatsächliche Profile/Snapshots einschließlich unverändertem Host-`validate_scope`: 18 PASS (78,78 s). Damit 80 verschiedene abschließend bestandene Fälle ohne Skips, kein gemeinsamer 80er-Lauf. Die Runtimegrenzen der Suite werden in lokalen Regressionen vor jedem echten Engineaufruf abgefangen, keine Hardware-Success-Mocks. Script-Mirrors geprüft; bestehendes Installed-Inventar wird für vier geänderte Einträge und den neuen Test fortgeschrieben und anschließend verifiziert. Endergebnis und genaue Befehle im Sitzungsordner `local_52o_a8ia`.

H10.yaml/EVAL.yaml erneut normal aus CompleteSetDev erzeugt, nur Auflösungszeitstempel verändert. Benutzerprofil, GUI-Helfer/Hoststarter, Collector, wissenschaftliche Kriterien, Standardrezepte, normale fehlende Builds und freigegebene Budgets bleiben erhalten. Keine eigenen GUI-/SSH-/GPU-/Compiler-/Energieprozesse, kein zusätzlich erworbener Vorbereitungslock. Reale Abnahme der beiden Fixes steht beim HOST aus: genau ein begründeter finaler Eval-Folgestart, kein weiterer H10-Test und keine Wiederholung für Quality-PASS. CONTEXT/HANDOFF benennen diese Grenze; kein ZIP, keine neue Version, kein Commit/Push.


### R9H – lokale Vorbereitung der Drei-Split-Abnahme (19.09.2026)

Auftrag `v283_R9H_20260918_233209_748qhzjo`, bestehender Hauptbranch/Version unverändert. HOST hält Vorbereitungssperre; Codex startet keine GUI, SSH, GPU, Compiler oder Energieaufnahme. Collector/FS-Commandgrenze/1s×3, Qualitypolitik, Seeds und negative Compile-Evidenz bleiben erhalten.

AP1: TensorRT-Full-Classification bindet im erfolgreichen Commandvertrag den bestehenden Prepared-Input-/Top-k-Hotloop auch für Energie; vorhandene Preflight-, Engine- und Quellenbindung bleibt aktiv. Die konkrete Gegenprobe zeigte zusätzlich fehlendes Top-k im DeepX-Full-Energie-Hotloop; vorhandener ClassificationCompletion-Hook ergänzt. Hailo Full und Split-FIFOs besitzen den Hook bereits. Energie-Work-Units werden nur bei passenden Postprozesscompletions veröffentlicht; kompakte tatsächliche Task-/Quellenbelege werden mit ausgegeben. Keine Umdeutung alter trtexec-Aufnahmen.

AP2: Vor Full/Split-Fan-out dieselbe deterministische modellweite Vergleichsabbildung über den vorhandenen Image-Map-/Prepared-Feedpfad; backendgerechte Encodings bleiben getrennt. Native-Energieprojektion akzeptiert bestehende Detection-Execution-Attestationen nur mit Quellenabgleich, H8-Fast zusätzlich mit allen frischen gebundenen Energiereplikaten. Lesende R9G-Gegenprobe: zwölf verlorene Vergleichsendpunkte vollständig herleitbar; alle21 TRT-Full-Normierungen auf derselben Replikatbasis korrekt, davon20 vorher durch Verhältnis-von-Mittelwerten abgewiesen. Neue Prüfung validiert Einzelgleichungen und Mittelwerte mit unveränderter Toleranz. Abweichende Commanddauern bleiben Vergleichsgrenzen.

AP3: Aktuelle Authorityfehler bleiben bei gültiger leerer Fehlerliste leer; vorherige Fehler getrennt diagnostisch erhalten. Ausgabevertrag und Taskqualität im Statustext getrennt. Generic-Energie AUS wird not_requested, Native-AUS entsprechend angezeigt; keine Nativewerte in Genericzeilen. Nachweislich verlorene Producer-Vertragsfelder werden pro gleicher Variante im normalen JSON-Merge erhalten; unbekannte Endpoints werden nicht als vollständig erklärt. R9G-Qualitäts-/Numeriknegative bleiben Ergebnisse, kein Collectordefekt.

AP4: Vier vorhandene DeepX-Rohantworten erneut offline mit aktueller Semantikprüfung/Decoder geprüft:052891 fünf invertierte Zeilen,395801 eine;139/285 gültig (15/1 decodierte Kontrolldetektionen). Kein allgemeiner eigener Decoderfehler belegt, kein Fix-/500er-/Hardware-PASS behauptet; SDK-/Artefaktursache bleibt offen und kein Vorabgate für HOST-Kampagne.

Profile SMOKE/EVAL normal aus CompleteSetDev mit eingefrorenem Standardmodus erzeugt und unverändertes operator/scope_contract.py:validate_scope bestanden. EVAL sieben Modelle/drei Setups/sieben Profile, Zielquote3, Shortlist5, Reserve12, Buildbudgets24/12/12, Hailo5400s/TRT1800s; normale fehlende Builds erlaubt, Force AUS. Realer lokaler Generator belegt drei verschiedene Backendfälle, geordnete Nachrückung und Full-Deduplizierung. Nominal105 Native/315 Energiereplikate; tatsächliche Nenner aus Produktplan. Lokaler operator/postcheck.py trennt belegte Defekte und UNKNOWN und liest ausschließlich Reports.

Software-Endabnahme140 PASS/0 FAIL/0 SKIP (9,11s), darunter45 neue R9H-Fälle; Zwischenstände14 PASS/1 korrigierter Fixturefehler,30 PASS und133 PASS separat erhalten, nicht addiert. Exakte lokale Befehle/JUnit im auftragslokalen CONTEXT.md und local_5t0bebll. Der Checker erkennt im alten R9G zusätzlich neun belegte DeepX-Full-Energieaufnahmen ohne Classification-Postprozess; fehlender Logbeleg anderer Pfade bleibt UNKNOWN. Der alte Ein-Split-Scope erfüllt erwartungsgemäß nicht den neuen R9H-Drei-Split-Vertrag. Reale normale GUI-/Hardwareabnahme separat: HOST startet anschließend SMOKE und automatisch EVAL einschließlich normal fehlender Builds. Keine Warm-only-Hürde, keine Ersatzboundary, keine neuen Messbehauptungen, kein Commit/Push/ZIP durch Codex.

### R9H – belegte Korrekturen nach HOST-Smoke (19.09.2026)

HOST-Run `r9h_smoke_20260919_001205`: GUI/Worker terminal, Native9/9, zentrale Qualität11 abgeschlossen (vier FAIL/ein INCONCLUSIVE), Energie24/27 gültige logische Replikate bei25 Collectorstarts. Die H10-TRT-Full-Aufnahme endet mit `urecs_transport_unavailable: fast-firmware missing end packet at absolute deadline`; vorhandener Schutz stoppt die Quelle `192.168.0.176` mit `campaign_source_completion_unresolved`. Kein belegter Collector-Codefehler, keine Reparatur/Neumessung durch Codex. Ihr Workload selbst liefert760 echte Top-k-Completions. Kurze1s×3-Aufnahmen bleiben Screening.

Belegter Produktdefekt: DeepX Full überschreibt das vor Fan-out vereinbarte Modellbild durch das Bild des früheren Generic-Prepared-Inputs. Der bestehende Fullrunner bereitet nun bei expliziter modellweiter Zuordnung das gewählte Bild einmal mit dem vorhandenen DeepX-Eingabevertrag/Sealer vor und reicht genau diese Bytes an Semantik, Performance und Energie weiter. Alte Generic-Belege und Backend-Encoding bleiben erhalten; fehlendes Bild/ungültiger Vertrag scheitern vor Laufstart. Legacyzuordnung ohne modellweite Bindung unverändert. Kanonisches Script und Ressourcenmirror identisch.

Externer Checker verfolgt auch ausdrücklich abgewiesene Aggregate, tatsächliche gerenderte Commands samt vorhandenen Preflight-/Artefaktbindungen, ausgeführte Work-Units und begründet ausgefallene Wiederholungen. Terminaler Ablauf, technische Negative und Quality/Screening getrennt; UNKNOWN nur bei fehlenden Belegen. Lesende Prüfung des unveränderten Smoke:25/25 tatsächliche Classification-Top-k-Commands verifiziert,24 gültige Energiereplikate, keine UNKNOWNs; `pass=false` wegen historischem Referenzbildkonflikt und ungeklärtem Quellenabschluss. Die freigegebene `operator/gui_eval.py`-Kopie berücksichtigt jetzt den Energiecheckpoint beim Cleanup; bisheriges `cleanup_proven=true` belegte nur Worker/Quarantäne, nicht den Quellenabschluss. Hoststarter und unverändertes `validate_scope` bleiben unangetastet.

Lokale Software:43 gezielte Fälle PASS/0 FAIL/0 SKIP (5,75s), separater vollständiger Script-Mirrorcheck1 PASS (1,19s). 28 neue Regressionen gesammelt in `tests/test_v283_r9h_energy_reporting.py`; umbenannte Modelle/Boundary, reale Prepared-Input-Dispatches mit simuliertem SDK, drei Wiederholungen und negative Bindungsfälle. Erster enger Lauf6 PASS/1 Fixtureassertion-FAIL; Fixture danach gezielt auf vorhandenes Ersatzbild erweitert. Abschließende reine Checkpoint-Ergänzung separat nachgetestet; genaue Ergebnisse/JUnit/Befehle unter `v283_R9H_20260918_233209_748qhzjo/local__piljshh`, Kontext enthält Endstand. Keine pauschale Modellpfad-Wiederholung oder eigenen GUI-/SSH-/GPU-/Compiler-/Energiestarts.

SMOKE/EVAL erneut normal aus CompleteSetDev erzeugt; beide unveränderten Scopeprüfungen PASS. EVAL weiterhin7 Modelle/3 Setups/7 Profile, drei akzeptierte Splits je Backend, normale fehlende Builds mit Force AUS. Kein zusätzlicher Smoke zur Quality-/Streuungsverbesserung vorgesehen. Nächster HOST-Schritt: offenen H10-Quellenabschluss über vorhandenen Produktpfad klären, danach autorisierter finaler Eval; Quality-FAIL und bekannte saubere technische Negative sind kein Vorabgate. Keine Freigabe, die Quellen-/Cleanupsperre zu umgehen. Kein Commit/Push, keine Version/kein ZIP.

### R9H – zwei belegte Toolfehler nach abschließendem HOST-Eval (19.09.2026)

HOST-Run `r9h_eval_20260919_005754`: GUI/Worker terminal, Cleanup bestätigt; Native99/105 erfolgreich, sechs H8-Splitzeilen fehlgeschlagen, keine fehlenden Matrixzeilen. Zentrale Qualität127 abgeschlossen,40 FAIL/20 INCONCLUSIVE; sieben zusätzliche technische DeepX-Full-Quality-Bindungsfehler. Energie81/99 gestartete Zeilen vollständig verifiziert,18 fehlgeschlagen/gesperrt;245 Collectorstarts, zwei ungültige Aufnahmen mit `marker_dropped_samples_nonzero`, beide mit bestätigtem Quellenabschluss. Der vorhandene Schutz sperrt danach192.168.0.185 mit `campaign_source_transport_failure_limit`. Kein belegter Collector-Codefehler; keine Änderung an Collector, Kalibrierung,1s×3 oder wissenschaftlichen Grenzen.

Ursache1: Der vorhandene Planverteiler reservierte je Modell alle12 TRT-Part2-Starts für das erste Setup und null für die beiden weiteren. H8 konnte deshalb die normalen fehlenden Engines des zweiten Splits bei YOLO11l/YOLO26m/YOLO26s nicht bauen; der dritte Split blieb ohne vorgelagerte Bindung. `bind_plan_cases` verteilt den unveränderten Gesamtetat jetzt gleichmäßig auf eindeutige Setups, Restplätze in bestehender Reihenfolge: hier4/4/4. Kein höheres Budget, keine Warmpräferenz, keine zusätzliche Registry; der vorhandene dauerhafte Startzähler begrenzt weiterhin tatsächliche Starts einschließlich Fehlschlägen.

Ursache2: Die spätere Native-Bildauswahl widersprach dem bereits exakt gebundenen DeepX-Quality-Prepared-Input. Alle sieben Originalzeilen belegen dieselben drei Konflikte: Bild-ID, Bildhash und Tensorhash. Der Workflow löst die gemeinsame Vergleichsabbildung jetzt nach Erzeugung der vorhandenen Vendor-Quality-Bindungen und vor Native-Fan-out auf. Ein gebundenes Prepared-Input-Bild wird lokal anhand seines bestehenden Hashes gefunden und für Full sowie sämtliche ausgewählten Splits übernommen. Fehlende/manipulierte Bilder, Symlinks und widersprüchliche Bindungen scheitern vor Runtime. Ohne solche Bindung bleibt der bisherige Referenzpfad erhalten; exakte Quality-, Tensor-, Modell- und Artefaktjoins unverändert. Alte Runs werden ausschließlich gelesen.

Software: **71 PASS/0 FAIL/0 SKIP**,9,18s;30 neue Fälle gesammelt in `tests/test_v283_r9h_energy_reporting.py`, gezielte bestehende Budget-/Quality-/Profil-/Reportverträge und Script-Mirrors. Reale lokale Startreservierung mit kontrollierten Python-Kindprozessen; echter Prepared-Input-/DeepX-Dispatch mit simulierter SDK-Grenze. Erste Läufe scheiterten an unvollständig umbenannten beziehungsweise aus Detection übernommenen Classification-Testdeklarationen; diese Fixturefehler und sämtliche Aufrufe/JUnit bleiben in `v283_R9H_20260918_233209_748qhzjo/local_sy4bztcu` dokumentiert. Keine zusätzliche Produktänderung zum Bestehen der Fixtures.

Der externe Postchecker liest Workflowcontrol/Jobabschluss getrennt von erfolgreichen Matrixzeilen und meldet ausdrücklich technische Quality-Bindungsfehler. Original-Eval: `pass=false`, Defekte `native_execution_failed`, `native_quality_binding_failed`, `native_validation_technical_failure`, keine UNKNOWNs;244 tatsächlich gebundene Task-Ausführungen,243 gültige Energiereplikate. Quality-FAIL/Screening allein bleiben gültige Ergebnisse. Normale SMOKE/EVAL-Profile erneut aus CompleteSetDev erzeugt und unverändertes `validate_scope` bestanden. Ein begründeter EVAL-Folgestart durch HOST vorbereitet; kein weiterer Smoke. Native-Hardwareabnahme der Fixes offen. Kein eigener GUI-/SSH-/GPU-/Compiler-/Energiestart, keine Änderung an GUI-Helfer/Hoststarter/STATE, keine neue Version, kein Commit/Push/ZIP. Quellenintegritätsnachweise und Git-Endstand stehen im auftragslokalen CONTEXT.md.

### R9I – lokale Reparatur der R9H-Restfehler, HOST-Zellen A/B/C vorbereitet (19.09.2026)

Verbindliche Quelle: unveränderter Originalrun `r9h_eval_20260919_064323`, keine erneute Architekturprüfung. Alle 302 Empfangsbelege offline geprüft: zwei fensterrelevante 64-Sample-Lücken (DeepX Full RegNet, TRT Full ResNet), ein weiterer 64-Sample-Verlust außerhalb des YOLO26m-Splitfensters; sämtliche Quellenabschlüsse vorhanden. Kein belegter eigener Collector-/Top-k-Verursacher. Sender/Link/Host-NIC bleiben ursächlich ungetrennt; `socket_drops=0` entlastet diese nicht. Rust-Countertests aus bestehendem R6-Testbinary bestehen; Collectorquellen/Binary, Fenster, Kalibrierungen und R9H-Journal unverändert.

Produktkorrekturen: Der Energiepaarvergleich trennt technisch verifizierte TRT-Normalisierung von Quality-/Screeningfreigabe; 16 verifizierte Zeilen/48 Paare erhalten keinen falschen Unverfügbarkeitsgrund. Alle wissenschaftlichen Gates und Dauertoleranzen bleiben bestehen, null neue vergleichbare Paare/Quoten. Der Energieplan berücksichtigt die vorhandene `letterbox_pad_value`-Aliasquelle. Historische fehlende Padprojektionen werden ausschließlich aus dem exakt gebundenen, hashgeprüften Preprocessingvertrag ergänzt: vier DeepX-Detection-Fullzeilen/zwölf Paare; tatsächliche 0/114-Konflikte und fremde Quellen bleiben gesperrt. Keine Pixeländerung. Die Summary beschriftet bestehende Replikatzähler und ergänzt die vorhandenen Journalzahlen: 302 physische Versuche, davon zwei ungültig; 301 ausgewählte Datensätze, 300 gültige logische Replikate; eine gestartete Fehlzeile/vier nicht gestartete Sperrzeilen. Keine geänderte Replikatauswahl.

Generic: Der vorhandene exakt gebundene Request wird vor der Strukturentscheidung projiziert; 27 der 34 fehlenden Endpunktprojektionen sind dadurch auf derselben Generic-Quelle verfügbar. Numerische Abweichung wird nicht als Strukturfehler abgeleitet, und Full erbt keine abgeleiteten Split-Strukturfelder (zwölf Split-/vier Fullfälle). Originalgruppen: 57 Vertragsnegative, 42 numerische Negative, zwölf Screeningzeilen, ein falsch als `not_buildable` geführter SIGSEGV. Verbleibend sieben fehlende Endpunktnachweise und sieben fehlende Raw-Head-Host-Tails; keine Native-Evidenz übernommen. Source-only-Neuprojektion: 112 Zeilen, 70 numerische Negative, 27 vor zentraler Quality noch offene Entscheidungen, 14 Vertragsnegative, ein Runtimefehler; dies ist keine neue zentrale Qualitätsabnahme. Expliziter Buildnachweis bleibt trotz negativem Prozessende erhalten. H8-YOLO11l bleibt historisch SIGSEGV/-11. Logs belegen fertige Reports/Plots und abschließende Soft-fail-Meldung, keinen Stack; Faulthandler und Phasenmarker im normalen Runnerpfad ermöglichen die gezielte B-Diagnose. Keine unbelegte SDK-Reparatur behauptet.

Quellenbudget: Der Produktpfad akzeptiert jetzt die im R9I-Auftrag verlangte Verschärfung auf eins sowie unverändert zwei; Weitergabe bis Snapshot/Collectorvertrag getestet. Null, größere/boolesche Werte und Wechsel bestehender Checkpointbudgets bleiben abgewiesen. Dies setzt keinen alten Etat zurück. Normale CompleteSetDev-Profile A/B/C extern erzeugt, eingefrorener Standardmodus, Native100/10/1, Quality32/Bootstrap100, Energie nur A/C1s×3, generische Energie/Windowprobe AUS, keine HEF-/DXNN-Neubauten. Unveränderliche Scopeprüfung und Profilroundtrip bestehen. Finale Profilprüfung: acht Fälle PASS; das pro Modell geltende TRT-/Cold-Budget ist für A4+4, B8 und C8 (höchstens acht je Zelle). HEF/DXNN ausdrücklich `reuse_only`, fehlende TRT-/Wrapperbereitstellung bleibt normal aktiv.

Temporäre Operatoränderung ausschließlich `operator/postcheck.py`: Pflichtenergie/Runtimeabsturz verhindern technischen PASS; nie gestartete Sperrzeilen werden nicht als beschädigte Vollaggregate gezählt, primäre Counterlücken stehen vor Folgesperren. Abgeschlossener negativer Bericht bleibt terminal, Quality/AP/Screening allein ist kein technischer Fehler. Originalpostcheck jetzt false mit sieben Defektklassen/keinen UNKNOWNs. `gui_eval.py`, `scope_contract.py`, `strict_postcheck.py`, START/Hoststeuerung unverändert. Tatsächlicher HOST-Sammlerimport unter `-I` und kleine Textpacks geprüft: Symlink-Ausschlüsse getrennt von fehlenden Pflichtdateien, fremde Ziele nicht verfolgt. Altes Textpack nur gelesen: 24.878 Nutzdateien, 696 Symlinkausschlüsse, keine fehlenden Pflichtdateien.

Softwareabschluss: **162 PASS/0 FAIL/0 SKIP**,44,37s, eine bestehende pyparsing-DeprecationWarning; enthält alle drei vom HOST verlangten Dateien, zusätzliche direkte Prozess-/Identitätsregressionen und Script-Mirrors. Vorläufe55 PASS und119 PASS; ergänzender Identitätslauf109 PASS/3 FAIL. Die drei alten Cross-Runner-Fixtures scheitern identisch mit den gesicherten Anfangsmodulen (3 FAIL,0,32s); unabhängig vom R9I-Diff und unverändert dokumentiert. Keine volle historische Suite. Source-/Installed-Verifikation und Originalerhaltbelege im externen `local_l0byk9x8`; Auftragsdiff gegen `before`, keine kumulative Altänderung als R9I.

Reale normale GUI-/Hardwareabnahme **offen**: ausschließlich HOST-Zellen A/B/C anschließend, kein eigener Geräte-/SSH-/GUI-/Inferenz-/Compiler-/Energiestart. Transportverlustort, SIGSEGV-Verursacher und historische DeepX-YOLO26s-Zweibildsemantik bleiben offen; kein Retry für AP/Streuung. Version/Build-ID unverändert, kein Commit/Push. Übergabe und genaue Befehle/JUnit unter `v283_R9I_20260919_140340_uirbnhi_/local_l0byk9x8`; knapper Gesamtstand in auftragslokalem `CONTEXT.md`.

### R9I – belegter Cacheblocker nach HOST-Zelle B (19.09.2026)

Reale normale GUI-Abnahme getrennt von Software: HOST-Zelle A (`r9i_a_20260919_144128`) technisch PASS; B (`r9i_b_20260919_150240`) technisch FAIL. B hat den historischen SIGSEGV nicht reproduziert: generisches H8 Full wurde wegen fehlender Suite-HEFs vollständig übersprungen; H8→TRT führte lediglich Part2 aus (rc0). Native Full meldet primär `missing_hailo_full_hef`; fehlende Split-Qualitybindung, zentrale Qualität und Requestpaare sind Folgefehler. Historische Reports bleiben unverändert negativ.

Belegte Ursache: `legacy_benchmarkset_binding.py` setzte bei `reuse_only` den vorhandenen Hailo-Builder auf unavailable und verhinderte damit auch dessen Cacheabfrage. Jetzt bleibt der normale Receipt-geprüfte Resolver aktiv, Full und Split erhalten `cache_only=True`, und der Parser-Vorlauf wird übersprungen. Smoke/Native-Full kann diese Sperre nicht übersteuern; `disabled`/`skip` bleiben deaktiviert. Zweiter enger Negativfall: Ein decoded Full-Cache-Miss unterdrückte die bereits vorhandene Raw-Head-Fallbackabfrage. `benchmark/services.py` prüft diese Endpunkte nun ebenfalls ausschließlich im Cache und klassifiziert das endgültige Fallbackresultat. Keine Cold-Build-, Force-, Auswahl-, Modell- oder Qualitätsänderung.

Lokaler B-Cache-Replay über den normalen Builder mit ausdrücklich gesperrten Prozess-/Compilerdispatchern: vorhandenes negatives decoded Full-Evidence wiederverwendet, Raw-Full und Part1 gefunden, beide HEFs bytegleich zu R9H; null Kindprozessstarts. Ausschließlich lokale Ausgaben unter `local_frg546z9/cache_replay`, keine Geräteabnahme. Regressionen vor Fix: vier erwartete Produktfehler/drei PASS; danach sieben PASS. Zwei vorausgehende Vorläufe enthielten unvollständige Testfixtures, im Log getrennt erhalten. Fokussierter Abschluss: **128 PASS/0 FAIL/0 SKIP**,44,63s, einschließlich aller drei HOST-Dateien, Deferred-Build-Regressionspfad und sämtlicher Script-Mirrors. Anschließende Fixierung der historischen A/B-Testpfade: separat **2 PASS/0 FAIL/0 SKIP**,1,08s. Abschließende Defaultabsicherung optionaler Cacheflags: **8 PASS/0 FAIL/0 SKIP**,1,79s.

Profile A/B/C und unveränderliche Scopeguards bleiben verbindlich; Operatoren, HOST/START, alte Quellenbudgets, Collector und wissenschaftliche Gates unverändert. Nächster Schritt: ausschließlich HOST-Nachtest B und noch ausstehende Zelle C innerhalb der bestehenden Startgrenzen. Kein erneutes A, kein AP-/Streuungsretry, kein 7-Modell-/Abendlauf. SIGSEGV-Ursache bleibt offen, weil B1 H8 Full nicht ausführte; vorhandene Faulthandler-/Phasenbelege bleiben für B2 aktiv. Source-/Installed-Nachweis, inkrementeller Gesamtdiff gegen R9I-Anfang und genaue Befehle/JUnit im aktuellen lokalen Sitzungsordner; keine Version, kein Commit/Push.

### R9L – P0/P1/P2: letzter R9K-Stand und gebundene Offline-Replays (20.09.2026)

Letzte R9K-Sitzung `local_o8z2wufc` war eine Nullrunde; ihre korrekten Producer-/Consumeränderungen bleiben erhalten. Einstieg HEAD `c5eb66eaa562728e9397a550570740d6a03e734d`, vorhandener Hauptarbeitsbaum. Die 42 aktuellen R9K-Abschlussregressionen bestanden vor der Änderung. Der ursprüngliche YOLOv7-EVAL1-Konflikt betrifft den TRT-Full-Deploymentdefault `host_tail_available=False` trotz fünf gemessener Completionframes. Aktuelle Producer-Metadaten schließen ihn im separaten Offline-Replay; archivierte Reports und fehlende zentrale Qualität bleiben unverändert.

Belegter Restdefekt: Ein einzelner Completion-Importfehler brach die gesamte Normalisierung ab. Im normalen Workflow wird die betroffene Variante jetzt als technischer Fehler mit Originalursache und Quellenbindung erhalten; sieben unabhängige YOLOv7-Zeilen überleben den historischen Gegenfall. Strenger Import und echte negative Timer-/Frame-/Alias-/Sourcebindungen bleiben negativ. Weder Inferenz noch Genauigkeitskorrektur.

Der exakte alte vollständige Postchecker lehnt A7 mit fehlendem/falschem Restbudget2 ab und besteht mit dem gebundenen Restbudget1. Neuer `report_scope` besteht A7 nur als historische Version2.83 mit Budget1; keine rückwirkende Gerätefreigabe für2.90. Modelle/Setups/Profile ohne Duplikate und sämtliche fachlichen Mess-/Qualitäts-/Energie-/Builddimensionen explizit negativ geprüft. Erste Scope-Testmutation traf versehentlich deaktivierte Modelle; Fixture auf tatsächlich aktive Mitglieder korrigiert, Guards unverändert. Originalzustände, Operatoren und kumulativer Quellenfehler bleiben erhalten.

### R9L – P3/P4: Release und Exporte vor HOST-Freeze (20.09.2026)

Zentrale Version2.90.0/Build-IDv2.90.0-r9l, pyproject, ausschließlich Rootpaket in uv.lock, aktueller README-Kopf und bestehender Updater konsistent gesetzt. Historische Releasebelege und Artefakt-/Compileridentitäten unverändert. Source-/Installed-Verifikation und dependencyfreie Projektregistrierung sind im abschließenden Nachweis unten dokumentiert.

Konkrete Darstellungsreferenz: A7 `reports/scientific/figures/screening_energy_observations.png` sowie `thesis_tables/native_energy_observations.tex` und deren aktueller Scientific-Generator. Referenz-PNG tatsächlich angesehen; DejaVu-Sans/Matplotlib-Standardblau/Gitter und booktabs-Regeln erhalten. Exporte laufen über eigene FigureCanvasAgg-Instanzen ohne globalen Backendwechsel. TRT-normalisierte Energie ist mit Schraffur/Legende von gemessener FS-Eingangsenergie unterscheidbar; gespeicherte Replikat-CI, n und tatsächliche2,24–2,77s-Fenster bleiben getrennt vom1s-Ziel. Native/Generic beschriftet, fehlende Werte N/A, Integerzähler ohne Scheinkommaziffern. Tabellenkopf-Überlappung im ersten Render durch tatsächliche Glyphenbreiten plus Abstand korrigiert.

Neue Offline-PNG/PDF/LaTeX-Beispiele ausschließlich unter R9L `docs_preview`; keine alten Reports neu geschrieben. Diagramm und Vergleichstabelle verwenden dieselben neun Energiezeilen und Nenner; rohe FS-W/J-pro-Bild-Werte separat. Gemessene Request-Mean/P50/P95 aus originalen ns-Paaren nachgerechnet; kein1000/FPS-Latenzersatz. Accuracyverlust und Unsicherheit getrennt. Single-Repetition erzeugt kein Streumaß. Vollständige normale GUI-/Hardwareabnahme bleibt HOST-Aufgabe auf diesem Release; keine eigene Geräte-, Compiler-, Inferenz- oder Energiemessung.

### R9L – fokussierter Softwareabschluss vor erster aktueller HOST-Zelle

**408 PASS/0 FAIL/0 SKIP**,123,85s, vier vorhandene Warnungen; drei gezielt deselektierte Fälle: zwei echte Tk-Tests für HOST sowie ein anschließend separat ausgeführter Release-Smoke. Releaseidentität/Script-Mirrors/aktueller lokaler Smoke: **4 PASS**,1,21s. Alle sechs HOST-Integrationsdateien, drei aktuelle R9K-Abschlussdateien, neue R9L-Completion-/Scope-/Exportregressionen und die gezielten geerbten Dispatcher-/Endpointfälle enthalten. Artefakte und Befehle: R9L `local_p839jjlj/{final.xml,final.log,test_command.json,release.xml,replay/}`.

Die Vorläufe dokumentieren ausschließlich korrigierte neue Testannahmen (aktive statt deaktivierter Modelle, exakt gebundener Decodername, reale Feldnamen für Requestpaare und Replikatstreuung). Unzulässiges Umbenennen allein des Reports bleibt negativ; gültiger anderer Detectorname mit drei/sieben Klassen und anderer Boundary besteht den echten CPU-Completion-/Normalisierungs-/Scientificpfad. Kein AP-Tuning, kein Hardware-Neustart.

Echte normale GUI-/Hardwareabnahme von2.90.0 noch **NICHT AUSGEFÜHRT**. HOST führt zusätzlich den Tk-Canvas-/Worker-Gegenfall und anschließend Y→B→A→EVAL aus. Quelle192.168.0.185 bleibt mit einer Störung belastet; Budget1, kein Reset. Vor Freigabe keine unbelegte Hardware-/wissenschaftliche PASS-Aussage.


### R9L – Installed-/Tk-Nachweis und Übergabe an HOST

Beide zunächst separat gehaltenen echten Tk-Tests nun lokal ohne Workflow/Inferenz bestanden: **2 PASS**,5,01s. Normale Ergebnis-Widgets zeigen Accuracyverlust/Unsicherheit; ein vorhandener Tk-Canvas zeichnet auch nach dem Workerexport weiter. Keine Matplotlib-GUI-Workerwarnung und kein globaler Backendwechsel. Das sind Widgettests, keine Geräteabnahme.

Installierte Distribution war beim Einstieg noch2.82. Der vorhandene stdlib-only-PEP376-Refresh aus dem Releaseupdateweg hat ausschließlich die Projektregistrierung auf2.90.0 aktualisiert; alle übrigen67 Distributionsversionen unverändert. Aktueller CLI-Smoke besteht. Der Sourcegenerator verlangt einen reinen Releasebaum: vorhandene Runtime-Symlinks wurden deshalb nicht entfernt, sondern die2036 Release-Dateien extern gestaged und nur die erzeugten bestehenden Sourceindizes übernommen. Installed- und unabhängige Runtimeprüfung bestehen, Script-Mirrors bytegleich. Kein Dependencyupdate, keine Cache-/Receiptänderung.

Abschlusslieferung: `PATCH_R9L.diff` gegen gesicherten Einstieg einschließlich neuer Dateien, `REGRESSION_REPLAY.json`, `ARTIFACT_REVIEW.json`, `ADDITIONAL_TESTS.json`, knapper `CONTEXT.md` und aktuelle lokale HANDOFF. HOST verwaltet Sourcefreeze und neue Y/B/A/EVAL-Abnahmen. Nach diesen Abnahmen keine Kosmetik; nur belegte ursächliche Reparaturen mit relevanter Wiederholung.

### R9L – begrenzter Messquellenabschluss .176 (21.09.2026)

Neuer enger Folgeauftrag; der abgeschlossene Starter und seine alten Freigaben bleiben geschlossen. Originalreferenz: `v290_R9L_Abschluss_20260920_220613_nrl9ktbx/eval_02_5huz8obo/r9j_eval_20260921_091538`. Laut aktuellem Originalabschluss Y/B/A bestanden, EVAL mit offener Gesamtenergie; hier keine neue GUI-Abnahme. Historische 61 vollständige Energiezeilen/183 gültige Replikate/185 physische Versuche bleiben historische Evidenz. Ausschließlich die beiden offenen TRT-Full-Zeilen für `yolo26s` und `yolov7_paper`, Setup `orin_nx_hailo10_01`, sind Gegenstand des neuen Read-only-Plans.

Betroffener YOLO26s-Versuch: 467 zusammenhängende 256-Byte-Pakete/29.888 Samples, letzter Empfang 15,010437646 s, Receivefehler 20,105060816 s; GO fordert 27 s. Kein Endpaket, keine protokollierten Socketdrops, `WouldBlock ... silence timeout; device completion unverified`, Collector rc1. Writerabschluss danach erfolgreich; Workload rc0/152 Completed Tasks. Der Verlustort zwischen Quelle, Netzwerk und Empfang bleibt offen. Kein belegter Hailo-Compiler-/Firmwarefehler und kein Streuungsbefund. Der Hailo10H-YOLO26m-Markernegativversuch (256 Samples/128 ms) hatte verifiziertes Ende und wurde regulär erfolgreich ersetzt; keine neue Messung erforderlich. TRT-YOLOv7 hat null gestartete Ketten.

Belegter eigener Folgefehler minimal repariert: `EnergyTaskBudget.__init__` überschrieb beim Eintritt der nächsten Zeile `campaign_source_completion_unresolved` mit dem Unterbrechungsfallback. Ein bereits gespeicherter erster Sperrgrund bleibt nun erhalten; der Fallback bei fehlendem Grund, alle Sperren und Quellenzähler bleiben bestehen. Drei neue Regressionen scheiterten vor dem Fix genau daran. Originaljournale werden weder migriert noch rückwirkend geändert.

Software: **21 PASS**, 10,63 s (7 neue Budget-/Markerfälle plus 14 bestehende Campaignfälle; zwei bekannte JUnit-Warnungen). Exakte unveränderte R6-Collector-Binary gegen lokale UDP-/Fake-Prozesse: **5 PASS**, 17,92 s. Normales, rechtzeitiges/spätes Ende innerhalb vorhandener Grenzen, vorzeitiges Ende, fehlendes Ende/Stille, absolute Deadline, Writerabschluss, Originalfehler, blockierter Folgestart, verifizierter Retry und Markergrenze geprüft. Sandbox-EPERM vor Socketerzeugung separat dokumentiert; erfolgreicher Wiederholungslauf nur über Loopback. Keine physischen Messungen aus diesen Tests zählen.

Aktueller lesender HOST-/SSH-Nachweis 13:41:03–13:41:04 MESZ: keine passenden Collector-/Workloadprozesse, relevanten UDP-Sockets oder Original-Leases auf Smartmirror2/H10-Jetson; sichtbare kanonische Locks frei. Fremde Prozess-FDs sind teilweise zugriffsbeschränkt; Prozessargumente und relevante Sockettabellen lesbar. Kein Kill, Lockentfernen, Powerbefehl oder GO. Das ist kein Geräte-Idle-ACK. Die gebundene Quelle kennt laut vorhandener Source keinen Status-/STOP-/Reset-/Idle-ACK-Weg; lokales `source_closed` bestätigt nur lokalen Abschluss.

**Wiederanlauf BLOCKIERT; 0 neue physische Starts.** Betreiber muss ausschließlich Aufnahmequelle `192.168.0.176` über ihren dokumentierten Quellencontroller-Reset/-Neustartweg in definierten Mess-Idle bringen und Quelle/Geräteidentität, genaue Zeit mit Zeitzone, Handlung und Boot-/Idle-Nachweis dokumentieren. Der sichere konkrete Bedienweg ist hier nicht dokumentiert und muss vom Quellenbetreiber geklärt werden; kein vorsorglicher Board-/Jetson-/Hostneustart. Kein Quellenneustart wurde behauptet oder ausgeführt.

Der passive Zweizeilenplan bindet originale Task-/Artefakt-/Eingabe-/Collectoridentitäten und unveränderte FS-/Command-/Completed-Task-Rezepte, 3 Wiederholungen je Zeile, 6 Starts plus höchstens einen zulässigen Ersatz. Bestehendes GUI-/Managed-Resume übernimmt terminale Ergebnisse; historischer Einzelzeilen-Resume verweigert das Managed-Journal und würde Originalresultate zusammenführen. Kleinster vorhandener Messpfad sind die gebundenen nativen `energy_measurement_cli.py measure`-Rezepte mit separater Ausgabe. Das vorhandene Campaignbudget erzwingt keine gemeinsame Obergrenze 7 über beide Zeilen; Startsteuerung ist daher zusätzlich offen, keine Ersatzpipeline erstellt. Originalrun bleibt rot/unverändert.

Collector/Methodik/Kalibrierung/Power unverändert, keine neue numerische Zusammenführung. Version 2.90.0 / v2.90.0-r9l bleibt bestehen; vorhandene Sourceindizes werden über den bestehenden externen Stagingweg aktualisiert. Lokale Belege und fokussierter Diff: R9L-Auftragsordner `messquellenabschluss_20260921_y4bngk3f`. Keine neue GUI-/Hardwareenergieabnahme, keine Dauerzuverlässigkeits- oder wissenschaftliche Freigabe; größere Modell-/Final-/Nachtläufe bewusst nicht ausgeführt.

### u.RECS .176 Nachabnahme – lokale Vorbedingungen (21.09.2026)
GUI-Präzisionsverlust durch geteilte `.6g`-Tk-Variable reproduziert und behoben; echte Wertänderung entfernt alte Kalibrierbindung. Live-H10-Wert mit Backup aus unveränderter passender Originalevidence wiederhergestellt, andere Einstellungen unverändert. CPU-Referenz erhält fehlenden CPU-Koordinatenvertrag nur bei identischem ONNX-Graphnachweis; Originalsuite unverändert, Sidecar in Referenzidentität aufgenommen. Explizite Nachtest-CLI akzeptiert Quellenfehlergrenze3, globale Profile bleiben bei bisherigen Grenzen; Workloadfehler sind keine Transport-Retries.
Software:107 PASS/0 FAIL/0 SKIP,62.55s, einschließlich verborgenem echtem Tk-Roundtrip und lokalem UDP-Collector. Kein physischer Energie-/normaler GUI-Abnahme-PASS daraus. Auftrag/Logs: `/home/kmika/.local/share/onnx-splitpoint-codex/energie176_nachabnahme_20260921_183206_6v4rubu_`. Zwei selektive TRT-Full-Energiezeilen noch ausstehend; keine Qualitätskampagne, kein Commit/Push.

### u.RECS .176 Nachabnahme – selektive reale CLI-Abnahme abgeschlossen (21.09.2026)
Regulärer lokaler Management-CPU-Referenzpfad auf separater Suitekopie:16 Referenzdatensätze,rc0/completed,8.73s;100 Bootstrap konfiguriert, keine separate Bootstrapauswertung im Referenzgenerator. Die alten CPU-Ausgabedateien waren bereits bereinigt; ein archivierter Prepared-Input wurde in der direkten Regression per echtem ONNX/CPU neu verarbeitet. Keine neue Hardware-Qualitykampagne.
Remote-Präflight entdeckte sechs fehlende Dateien (zwei Manifeste,zwei Tensoren,zwei Originalbilder). Exakte lokal archivierte Bytes via bestehendem SSH/SCP/Leasepfad wieder bereitgestellt, kein vorhandenes Artefakt ersetzt und kein Build. Negative Präflights erhalten, danach beide vollständigen Präflights verified/rc0; Laufzeit-/Input-/Completionverträge unverändert.
Reale autorisierte Energie-CLI: `native_full_tensorrt|yolo26s|full|orin_nx_hailo10_01` und `native_full_tensorrt|yolov7_paper|full|orin_nx_hailo10_01` seriell auf .145, Collector Smartmirror2/.1, Quelle .176:3000/2000Sps/Kanal0. Beide Aggregate ok/rc0,6/6 gültige logische Replikate bei6 physikalischen Starts,0 Retries,6 bestätigte Protokollenden/Quellenabschlüsse,keine verbleibende eigene Remotelease. Je27s Capture mit unverändertem1s-Lastsoll,5s Vor-/Nachlauf und Commandfenster. TRT-Idle-Normierung mit exaktem Evidencewert in allen sechs Replikaten verifiziert. Mittelwerte FS/hostnormiert:YOLO26s56.746764/53.756649J;YOLOv7 49.641693/46.610328J. Kurze Screening-Evidenz,keine Langzeit- oder63/63-Gesamtabnahme; alte Resultate unverändert.
Kein Paketmitschnitt: optionaler freigegebener Hosthelfer scheiterte vor Capture an fehlender sudo-Authentifizierung. Historische Quellen-/Netzursache bleibt offen,kein ESP-Defekt oder Resetnutzen abgeleitet. Verborgener echter Tk-Konfigtest ist Softwareabnahme; kein zusätzlicher normaler GUI-Eval gestartet. Ergebnisse,Auftragsdiff und Archiv im Auftragsordner;kein Commit/Push. Nächste Aktion: diese beiden kompatiblen Energieergänzungen verwenden,kein weiterer automatischer Lauf.

### Release 2.90.1 – Quellenabschluss und private Finalkonfiguration (21.09.2026)

Version 2.90.1 / Build-ID v2.90.1 konsolidiert den vorhandenen geprüften Stand.
Nur Releaseidentität, aktuelle Assertions, Releaseanzeigen/-notizen und bestehende
Sourceindizes werden fortgeschrieben; die kumulierten Produkt-/Testkorrekturen
bleiben erhalten. Dependencyfreier Projektrefresh: alle 67 anderen
Distributionsversionen unverändert. Kein neuer Compiler-/Cachevertrag, keine
neue Messdauergrenze, keine automatische Verkürzung und keine neue Retrypolicy.

Softwareabnahme: der dokumentierte Energie176-Umfang besteht erneut mit
**107 PASS / 0 FAIL / 0 SKIP**, 104,65 s, 19 bekannten Warnungen. Enthalten sind
die drei Energie176-Testdateien, echte lokale Tk-Konfiguration, CPU-Replay und
ausschließlich lokale UDP-/Fake-Prozesse. Gezielte Release-/Import-/Mirror- und
Profil-Ausnahmeprüfungen: **15 PASS / 0 FAIL / 0 SKIP**, 3,61 s. Finalprofil über
normalen Loader, GUI-Vorschau und Runtime-Startsnapshot lokal konsistent geprüft;
keine Runanlage. Private Belege liegen im Releaseauftrag `release_v2901_aqbdav68`.

Die private Kopie `profiles/Thesis_Final_v2.90.1.yaml` stammt aus
`profiles/CompleteSetDev.yaml`. Final Quality (Standard+) behält sieben Modelle,
drei Setups, einen akzeptierten Split pro Modell, `stratified_windows`, Shortlist 2
und Suchpool `auto`. Wirksam: Validierung 5000/5000, Bootstrap 5000, B500,
Nativeframes/Warmup/Wiederholungen 1000/100/3; Native Full/Split und Native-Energie
an, Generic-Energie/Force aus. Originalprofil und globale Konfiguration bleiben
bytegleich; die Kopie und ihre Livepfade werden nicht veröffentlicht.

Explizites Energie-Soll des normalen Benutzerprofils: **1 s**, vor dem
Hardwarestandard von 60 s. Die unveränderte Produktableitung ergibt nominell
17 s Aufnahme plus 5 s Vor-/5 s Nachlauf. Das wird nicht aus dem selektiven
Energie176-CLI-Rezept kopiert und bleibt kurze Screeningenergie. Fehler-Timeouts
sind davon getrennt. Reguläre Policy: drei logische Energiereplikate, ein Retry
je Replikat, 5 s Backoff, Quellenfehlergrenze 2, sofortige Sperre bei ungeklärtem
Quellenabschluss; erster gültiger Versuch zählt. Die separate CLI-Freigabe 2/3
gilt nicht für dieses Profil. Ein späterer Start löst die Tool Config erneut auf.

Reale normale GUI-/Hardwareabnahme in diesem Releaseauftrag: **NICHT AUSGEFÜHRT**.
Bestehende Evidenz: R9L-Standard 63 Native-/77 zentrale Qualitätszeilen technisch
vollständig und 61 vollständige Energiezeilen; separate Energie176-Ergänzung
zwei Zeilen, 6/6 gültige Replikate, kein Retry. Zusammengesetzte Evidenz, kein
rückwirkend ununterbrochener 63/63-Lauf. Historische Transportursache/fehlendes
PCAP, normale Qualitätsverluste, 95%-Unsicherheit und Screening-/Stationaritätsgrenze
bleiben getrennt offen. Qualitätsforschung und neue Final-/Nachtläufe sind
zurückgestellt. Commit-/Tag-/Pushnachweis wird nach Ausführung ausschließlich im
privaten FINAL.json ergänzt, ohne zirkuläre Commitidentität in Sourceindizes.


## 2026-09-22 – Quality-Speed AP00–AP05, isolierter Kandidat

Präzisierter Auftrag und Reviews vollständig gelesen; Basis f36dd9ad/v2.90.1.
Originalworkflow regulär abgebrochen/terminal, Produktlocks freigegeben; 34 fertige
von 77 zentralen Zeilen erhalten, übrige cancelled. Lokale Cleanupbelege geprüft,
kein neuer SSH-/Hardwarestart. Hauptbaum/Installation unverändert. Sitzung
nachweisbar gpt-6-astra, lokale Einstellung xhigh; Ultra nicht attestiert.
Externe Belege: `/home/kmika/quality_speed_ap00_ap05_20260922_quoqh0fv/evidence`.

**AP00:** Produktloader und tatsächliche Importbindung geprüft. 20 geschützte
Originaldateien unverändert; n=B=5000, Seed20260710, Python3.12.3, NumPy1.26.4,
pycocotools2.0.11. YOLOv7 besitzt vollständige Producerrecords, keine fertige
historische zentrale CI. Klassifikation/YOLO26/YOLOv7 zusätzlich an Punkten und
sechs expliziten Originalziehungen geprüft; diese Diagnose ist kein vollständiges CI.

**AP01/AP02/AP03:** Normale Profile, Runmodes, Schema, beide GUIeditoren und
Service führen dieselben Ausführungsflags. Referenzthreads separat einstellbar;
Legacy-default und bisherige Profile erhalten. Echter Worker-/Phasen-/Drawstatus,
Wartezustände, gedrosselte Heartbeats und getrennte Laufzeiten. Kompakter
All-area/maxDets100-COCO-Kern, unveränderte Raster/Ties/Multiplizitäten/Dtypes;
offizieller Vollreport separat. Runlokale absolute Referenzvektoren binden reale
Payloads, geordnete IDs, sämtliche GT-Felder, Plan und numerische Versionen.
Identität nur bei echter Payloadgleichheit; Nullreferenzpositionen bleiben erhalten.
M1-Voraussetzung vor Schedulerarbeit: vollständige exakte Originalparität,
Legacy4997,068s versus kompakt561,270s, jeweils vier Worker und eine kalte Messung.

**AP04:** Ein aktiver schwerer Request, begrenzter Loader-Koordinator, Queue nur
mit gebundenen JSONdescriptoren. CPUbudget aus Affinität/Quota, RAMvorprüfung
inklusive Payloads, privater Daten, Plan und Arbeitsreserve. PCG64/int64-Plan einmal
200000000Byte, read-only mmap und Wiederverwendung. Pro Worker/Phase einmalige
Vorbereitung, Folgeblöcke mit nachgewiesenen Cachehits. Referenzphase vor Kandidat,
keine wartenden Kandidaten in belegten Referenzslots. Atomare vollständige,
identitätsgebundene Checkpoints; Merge prüft Plan/Punkte/Metriksets und [0,B)
exakt einmal. Kein fertiger Paarcache vor Gesamtmerge. Crash/Cancel/Resume,
Teilwrites, korrupte Pläne, geänderte Blockgröße und Slotfreigabe lokal geprüft.
Nur eigene terminale Payloadkopien werden entfernt, gültige Ergebnisse bleiben.
Delta-only-Customfactories behalten ihre Auswertung, melden fehlende absolute
Checkpointfähigkeit ausdrücklich; normale Classification/Detection vollständig.

**AP05:** Normaler Replay-CLI mit Profil/Snapshot, expliziten zentralen Zeilen und
sequenziellem Laden; gespeicherte Wissenschaftsbudgets unverändert. Historische
Completed- und explizite Producer-only-Admission getrennt. Moderne Accuracyklassen,
Ratiointervalle und Warnungen durch normale JSON/CSV/Markdown/LaTeX-Projektion bis
zur normalen GUIanzeige. Wissenschaftlicher Hash ohne Laufzeit-/Cacheidentität.
Softwarematrix443 eindeutige Fälle abgedeckt: breiter Lauf440PASS/3FAIL, betroffene
94 nach Korrektur PASS; abschließende Beobachtbarkeitsänderung62PASS +Timing1PASS.
Replay69PASS einschließlich normaler Profil→Spawn→Writer→GUI-Kette und negativer
Startgrenzen. Tatsächliche neue Originaladmission70/3/33 ohne Numerikstart bestanden.

**Finaler Originalnachweis:** eingefrorene Source `m2_final_frozen_source_v2`,
vier Worker, Block256, Checkpoints an, n=B=5000/Seed20260710 unverändert.
YOLO11l/Hailo8 Zeile30: kalt604,977s, exakt alle absoluten Komponenten/Deltas/
Ratios/Undefinedmasken/Punkte/Quantile/wissenschaftlichen Felder wie Legacy.
Beobachtet8,260-fach, eine Messung pro Bedingung, keine generelle Faktor-Zusage.
Paarwarm3,860s mit null neuen Draws. Anderer Originalkandidat Zeile25 mit derselben
Referenz315,219s; keine neue Referenzberechnung, alle5000 Referenzwerte identisch.
Historische Berichte beider Zeilen exakt erhalten: Accuracy-Loss/Reference-Close.
Prozesssummen alle5s: kalter RSS4482964→3446688KiB, PSS4402315→3323296KiB.
Nur Stichprobenpeaks. Normaler Loader/Service/Writer, unveränderte Inputs/Source
und null Aufrufe der gesperrten Parent-Startgrenzen belegt; Spawn-Kinder separat
an reinen CPU-Statistikcode gebunden, keine behauptete allgemeine Prozesssandbox.

**Abnahme getrennt:** Software/Offline-Statistik PASS. Neue reale normale GUI-/
Hardware-/Quellen-/60-s-Energieabnahme NICHT AUSGEFÜHRT. Originalrun bleibt
abgebrochen. AP06–AP11, Qualitätsforschung und Nacht-/Finalkampagnen zurückgestellt.
Keine Installation, Builds, Budgetkürzung, Commits, Tags oder Pushes.
Nach separater Freigabe normal aktivierbar: statistics.engine=optimized_coco_v1,
workers=4, block_repetitions=256, checkpoint_blocks=true,
prepared_cache_limit_mib=512; management_reference.intra_op_threads=4 explizit.
Rollback engine=legacy; gültige Altresultate bleiben, alte Teilresultate werden
nicht als Checkpoints interpretiert. Vollständiger Patch und Abschlussbericht
liegen im externen Arbeitsordner; keine automatische Produktionsaktivierung.

### AP06–AP08: Cancel-Übergang, lokale Teilabnahme (2026-09-22)

Arbeitsbasis ist der geprüfte AP00–AP05-Kandidat, nicht der unveränderte HEAD.
Der Originalrun enthält 34 Completed und 43 tatsächliche Cancels (31 primäre,
12 TRT-Full-Begleiter). Der normale Join behandelte fehlende Erfolgsattestierungen
dieser Cancels als technische Fehler. Der enge Übergang prüft jetzt tatsächlichen
Run-Cancel, Requestidentität, Manifestbindung und Reihenfolge der Cancelzeiten.
Frühere technische Fehler und ServiceClosed ohne Run-Cancel bleiben Fehler.
Erhaltene Completed-Identitäten können erneut strikt zusammengeführt werden.
Cancel zeigt Abwicklung statt Bootstrap-ETA; keine Draws werden ergänzt.

Software: 27 neue Cancel-Regressionen PASS (33,29 s), einschließlich Original-
Replay, normalem Writer, GUIprojektion und negativen Bindungsfällen. Original-
dateien unverändert; sämtliche Ausgaben liegen im externen AP06–AP08-Arbeitsroot.
Reale normale GUI-/Hardwareabnahme: noch NICHT ausgeführt. AP06-Captureanbindung,
AP07-Fallfreigabe und AP08-Integration sind noch in Arbeit; keine Übernahme.


## AP06–AP08 – Softwareintegration, GUI-Abnahme noch offen (22.09.2026)

Auf dem geprüften AP00–AP05-Kandidaten: gemeinsamer Statistikpool mit bis zu zwei aktiven Kontexten; CPU/RAM-Zulassung für Referenz, Loader, Draws und Flush. Der einmalige Vergleich mit n=B=5000 für 6 Worker/2 Kontexte (768 MiB private Vorbereitung) ist für beide Originalfälle bitgenau zu Draws und wissenschaftlichen Ergebnisfeldern einschließlich CI; Batch872,35s, kein Swap. `accuracy_loss` und `reference_close` bleiben unverändert. Die kleinen Skalierungsvergleiche liefern keinen stabilen Vorteil für6/2; neue normale Abnahmekopien behalten4/1. Kein universeller Speedup behauptet.

AP06 erweitert vorhandene Ressourcen-, Prozessbaum- und Leasemechanismen: zuerst DUT-Neuzulassung sperren und laufende Transfers fertigstellen, dann Controller-/Empfangspfad drainen. GO/END/CLOSE und tatsächliches Cleanup binden die Freigabe; verlorene Besitzer bzw. unbekanntes Quellenende bleiben gesperrt. Energie-Checkpointlocks schützen kurze Transaktionen; geteilte Transfer-/Postcalcslots gelten auch für Kindprozesse. Lokale Regressionen decken Aliase, Cancel, Ownerverlust, verschachtelte Arbeit, Fairness und Wiederverwendung derselben Poolprozesse ab.

AP07 publiziert vollständige gebundene Fälle vor dem globalen Quality-Abschluss. Strikte Vendor-/TRT-/Splitbindungen, Full-Owner, unveränderliche Leaf-Snapshots und laufende Checkpoints vor Dispatch bleiben verbindlich. AP08 verwendet dieselbe physische Setupqueue für Generic und Native, auch über Modellgrenzen. Kontrollierte normale Stage-/Coordinatorprüfungen zeigen Modell2 auf SetupA während Modell1 aufSetupB und verhindern offene Starts nach Cancel. Die normale Profilauflösung, Summary und Startsnapshots tragen die neuen getrennten Schalter.

Softwarebelege und reale Abnahme bleiben getrennt: gemeinsame Schlussauswahl und echte G1/G2-Host-GUI noch ausstehend; bisher0 Workflow-/Collectorstarts, keine Übernahme in die Hauptinstallation. Die bestehende globale Single-Input-Filtergrenze ist weiterhin offen; die lokale20er-Auswahl/Native-Subset-Regression beweist keine End-to-End-Aufhebung. AP09–AP11 und Qualitätsforschung bleiben vertagt.

### AP06–AP08: Abschlussintegration und echte Profilbedienung (22.09.2026)

- Isolierter Kandidat auf geprüftem AP00–AP05-Stand; Hauptinstallation weiterhin unverändert. Keine Commit-/Push-/Versionsaktion.
- Echte Tk-Editorprüfung: Save/Reload → sichtbare Summary → normaler Startsnapshot bewahrt feste Fälle, begrenzten Native-Checkpoint und 60-s-Nativeenergie. Drei konkrete Editorverluste korrigiert; neue Regression 2 PASS.
- Gemeinsame erste Auswahl: 749 PASS, 10 FAIL, 11 Setupfehler. Decorator-Import, fehlender optionaler Runner-Optionszugriff und Cancel-Spool-Bereinigung korrigiert; gezielte Nachprüfungen 51 bzw. 95 PASS. Abschließende gemeinsame Auswahl folgt.
- Drei zusätzliche Cleanup-Fehler auf unveränderter AP00–AP05-Basis reproduziert. Bestehender Guardian-Endbeleg bleibt für konkurrierende exakte Cleanup-Leser erhalten; 6 Regressionen PASS. Eine veraltete Testannahme auf schon vorhandenen technischen Fehler `pipe_drain_incomplete` präzisiert; kein Erfolg aus abgebrochener Pipe.
- Strenge Warm-Policy gilt auch direkt an späterer Hailo-Continuation: ein zwischenzeitlich verlorener HIT darf keinen Kaltbau auslösen. Native-Engine-Builds bleiben im separaten Abnahmeprofil aus.
- Normale Profile G1/G2: per_case/per_setup, Statistik 4 Worker/1 Request, Upload1; funktionale Optionen 2 Requests/Upload2 lokal geprüft. Messvergleich rechtfertigt keinen stabilen größeren Default. Vollständige n=B=5000-Parität mit 6/2 und 768 MiB Vorbereitung exakt (872,35 s); kein universeller Speedup behauptet.
- Reale normale GUI-/Hardwareabnahme noch ausstehend; Startjournal weiterhin 0/3 Workflows und 0/36 Collector. Softwarebelege sind keine Hardwarefreigabe. Kein Final-/Nachtlauf, keine geänderte Messdauergrenze; Qualityforschung/AP09–AP11 und bestehende globale 20er-Filtergrenze bleiben getrennt.

### AP06–AP08: Softwareabschluss, Hardwareabnahme vor Start blockiert (22.09.2026)

- Gemeinsame finale lokale Auswahl: **793 PASS, 0 FAIL, 0 ERROR**, 242,33 s; 24 Warnungen (Deprecation/JUnit-Metadaten), keine übersprungenen Tests. Normale echte Host-GUI-Vorprüfung G1/G2 bis Summary/Startsnapshot/Startbutton erfolgreich; Button nicht betätigt, keine Worker gestartet.
- Lesende aktuelle Registry-/TRT-Receipt-/ABI-Prüfung auf H8/H10: keine bekannten konkurrierenden Produktprozesse, aktiven Produktleases oder gehaltenen Produktlocks. Vorhandene Full- und Native-Part2-Engines passen.
- Konkreter gemeinsamer G1/G2-Blocker: normale Generic-TRT-Performancepflicht auf H8 für MobileNetV3Large b135. `trt_p1`: MISS/not_found; `trt_p2 b135:generic`: MISS/source_onnx_mismatch. Der echte normale Generator-/Owner-/Requirement-Resolver bestätigt beide Pflichtbindungen; H10-Übermengen der lesenden Vorprobe separat ausgeschlossen. Exakte Hashes/Shapes/Receipt-Prüfung im Aufgabenroot `evidence/gui_preparation/blocked_artifact_bindings.json`.
- Gemäß Auftrag keine Kaltbauten, kein Scopeaustausch zum Umgehen fehlender Artefakte, kein G1/G2-Workflow-/Collectorstart. Teststartjournal **0/3 Workflows, 0/36 Collectors**. Damit keine reale Native-/Energieabnahme und **keine produktive Übernahme**; Hauptinstallation unverändert und eigene Quellintegritätsprüfung PASS.
- Implementierung und separate aktivierte Testprofile bleiben im isolierten Kandidaten. Erst nach Verfügbarkeit der exakt benötigten warmen Artefakte kann die genehmigte kleine normale GUI-Abnahme unter demselben Root/Zähler fortgesetzt werden. Kein automatischer Build oder Final-/Nachtlauf. AP09–AP11 und die bestehende globale 20er-Auswahlfiltergrenze bleiben offen.

- Abschlussnachträge: 5 instrumentierte lokale Schedulerfälle PASS (4,84 s) mit monotonic-Zeitbelegen; zusätzlicher explizit deaktivierter Warm-Policy-Kompatibilitätsfall, 13 fokussierte Continuation-Tests PASS (1,69 s). Deaktivierte Altpolicy bleibt unverändert. Keine Hardwarestarts.


## AP00–AP08 Übernahme v2 – lokale Ergänzungen (23.09.2026)

Der vollständig ersetzende Übernahmeauftrag v2 wird im bisherigen Aufgabenroot
fortgeführt. Ursprünglicher AP00–AP08-Abschlussstand einschließlich Cancel und
WarmPolicy erhalten; die abgebrochene zusätzliche supported_subset-Architektur
wird nicht übernommen. Kein neuer Hash-/Journal-/Buildermechanismus.
Der alte Thesisrun vom 23.09.2026 wurde durch GUI-Cancel beendet:
Artefaktabschluss PASS, Run-Lock released, keine aktiven lokalen/Remotejobs.
Kein Resume dieses Runs und kein neuer Thesisstart.

R2: Direkter Native-Workflow reicht --resume-checkpoint genau bei explizitem
Resume und vorhandenem Energiephasencheckpoint weiter. Coordinator verwendet
denselben Übergang; erstmals erreichte Energie startet frisch. Software:
8 echte Eltern-Argumentprüfungen PASS; 19 vorhandene Checkpoint-/Recoverytests
PASS (zwei alte P02-Runnerfixtures an normalen Konstruktor angepasst); sechs
Retry-/Quellen-/Budgetgegenproben PASS. Keine Hardwaremessungen daraus.
Fertige Zeilen und vollständig verifizierte Kinder vor Parentimport bleiben
erhalten. Ohne fertigen Zeilencheckpoint kann die unterbrochene Energiezeile
innerhalb derselben persistenten Versuchs-/Quellenbudgets wiederholt werden;
der bestehende Collector hat keinen Resume-Einstieg für einzelne Replikate.
Keine neue Replikatcheckpointarchitektur ergänzt.

G-N1–G-N3 und begrenztes R1 sind in lokaler Prüfung. R1 kann nur vorhandene
dauerhafte lokale Cleanupbelege mit nachfolgender exakter Remote-Leaseklärung
unter exklusivem Run-Lock verbinden. Ein harter Controllerabbruch ohne solche
lokalen Besitzbelege bleibt gesperrt; die bestehende lokale Prozessbaumregistry
ist flüchtig. Fehlende PID oder Zeitablauf geben keine Wiederaufnahme frei.

Private normale Ein-Split- und ungestartete 20er-Profilkopien aus dem
Original-Thesisprofil normal aufgelöst: 4 Statistikworker/1 Request/1 Upload,
per_case/per_setup; wissenschaftliche Werte unverändert. 20er: stratified_windows,
Single-Tensor aus, kein Native-Backfill. Hauptinstallation vor Integration noch
f36dd9ad; G1/G2 noch offen, Zähler 0/3 Workflows und 0/36 Collectorstarts.
Bestehende 793/13/5-Softwarebelege und 5000/5000-Parität bleiben historische,
überlappende Evidenz; keine erneute Skalierungs- oder Legacybaseline.

Abgeschlossener lokaler Resume-Arbeitsteil: R1 bindet die explizite GUI-Aktion
an Run-ID und gespeicherten Auftrag. Die enge Recovery verwendet ausschließlich
bestehende lokale Cleanupbelege und Remote-Leasejournale unter exklusivem Lock;
Abschluss im vorhandenen Ausführungsjournal. 92 PASS / 4 abgewählt / 2 Warnungen
(23,76 s). Die vier vorbestehenden Report-Fixturefehler wurden separat auf dem
unveränderten Abschlussstand reproduziert; wissenschaftliche Produktlogik nicht
angepasst. Kontrollierter SIGKILL betrifft ausschließlich einen eigenen lokalen
Testcontroller nach belegtem lokalem Cleanup; Remotezustände sind Testdoubles,
die nachfolgende Arbeitsstufe ist eine kontrollierte Fixture. Das belegt keine
universelle Crash-Recovery oder Hardwarefortsetzung. Reguläres Cancel/gleiche
Run-ID, aktive Writer, Auftragsdrift und Quellsperrarten eingeschlossen.

Zusätzliche betroffene AP07/AP08-, Cancel- und WarmPolicy-Regressionen:
88 PASS / 2 Warnungen (68,46 s). Kein Ersatz für G1/G2-Hardwareabnahme.

G-N1–G-N3 abgeschlossen: Ausschließlich der bestehende Single-Tensor-Haken
begrenzt Generic. Der normale Resolver/Generator erhält IDs und Reihenfolge aus
der vorhandenen Auswahl; kein Native-Backfill. Hailo-/Native-Ausschlüsse bleiben
pfadbezogen, gemeinsame ONNX-Exporte erhalten. Im normalen Workflow keine
globale Part2-Vorfilterung oder YOLO26-Promotion nach der Auswahl. Native läuft
nur für die unterstützte Teilmenge, einschließlich leerer Teilmenge; Full bleibt
einmal je Identität. Ohne Split und ohne Full ist Native nachvollziehbar skipped.
Vorhandene Forced-Scope-/Duplikat-/Reihenfolgeprüfungen erhalten.
Software: 118 PASS / 2 Warnungen (70,19 s), zusätzliche gezielte Randfälle
15 PASS / 23 abgewählt (4,30 s), nach Review tatsächliche Orchestrierung plus
leere Teilmenge und Workflowkonfiguration 15 PASS / 2 Warnungen (6,54 s).
Die Mengen überlappen. Echte ONNX-Single-/Multi-Fixture und Tk Save/Reload;
Backend-Compiler lokal gedoppelt, keine Hardwareinferenz oder Artefaktbuilds.
Zwei konkrete Vorfilter-/Umordnungsfehler aus unabhängigem Review behoben.

Übernahme in Hauptinstallation erfolgt: 88 gezielte Dateien mit externer
Sicherung unter bestehendem Plattform-/Verzeichnislock, GUI-Guard vor Ersatz.
11 relevante Zielmodule aus der Hauptinstallation; vier normale Profilresolver
PASS; vorhandene Quellintegrität verified. Dortige fokussierte Auswahl-/Resume-
Abnahme: 39 PASS / 2 Warnungen (82,41 s), git diff --check ohne Befund.

Hardwarevorbereitung: exakt freigegebene H8/MobileNet-b135 Generic-TRT-P1/P2
über unveränderten Produktbuilder gebaut (138,09 s / 18,32 s); anschließender
normaler Cachelookup alle vier Rollen HIT, keine eigenen aktiven Remoteleases.
Ein erster Vorbereitungsaufruf scheiterte vor Builderbeginn an falscher Python-
Auswahl des externen Aufrufs; mit vorhandener normaler Runtimeauswahl behoben,
keine Paketänderung und kein wiederholter Enginebau.

Erster normaler G1-Start (ap06_ap08_g1_20260923_154356) technisch FAILED vor
Runtime-Dispatch, Quality nicht ausgewertet; GUI/Worker beendet, Quellintegrität
vor/nach PASS, Abschlussindex PASS, keine offenen Remoteleasejournale. Grund:
Cachevorprüfung behandelt H10-ort_tensorrt trotz Full-only-Qualitätsrolle als
Generic-Split und fordert drei dort nicht geplante P1/Generic-P2-Bindungen.
Benötigte H10-Vendor-Part2-Bindungen HIT. Enge Korrektur gleicht die Vorprüfung
an den bestehenden Setup-Dispatch und dessen Quality-only-Argumente an; keine
zusätzlichen Builds. Journal jetzt 1/3 Workflows, 0/36 Collectorstarts. G1/G2
noch nicht abgenommen; ein begründeter G1-Nachtest bleibt im vorhandenen Budget.

Vorprüfungsfehler lokal korrigiert: derselbe vorhandene Setup-Dispatch wie beim
Runtime-Start liefert jetzt die Quality-only-Run-IDs. Der bestehende Cacheprobe
interpretiert diese vorhandenen Argumente bei Generic-TRT-Rollen als Full-only;
Vendor-Part2 und Generic-Splits des Performance-Owners bleiben erforderlich.
Keine neue Identität, kein zusätzlicher Build. 60 fokussierte Tests PASS
(8,53 s), unabhängiges Quellreview ohne offenen Befund. Zwei alte Minimalfixtures
an vorhandene Profil-/Prediction-Verträge angepasst. Hardware-Nachtest offen.

G1-Nachtest ap06_ap08_g1_20260923_160011: Cachekorrektur im normalen GUI-Pfad
bestätigt (20 HIT, keine MISS/UNKNOWN, keine Compilerstarts). Neuer technischer
Abbruch vor gültiger Quality-/Native-Ausführung: lokal vollständige gefrorene
32er-Validierung fehlt vollständig im übertragenen Minimalbundle (0 Dateien).
Bei validation_budget_authoritative bleibt die globale CLI-Angabe absichtlich
leer; per_setup überspringt die bereits abgeschlossene Suiteaktualisierung.
Die Bundleauswahl übersieht dadurch die vorhandenen per-run Planpfade. Kein
Datensatz-/Erkennungs-/Compilerqualitätsfehler. Beide unveränderten lokalen
32er-Quellen bestehen die tatsächlich verwendete Templateerkennung.

Alle 12 Nativevarianten not_started/physical_dispatch_started=false wegen
fehlender vorgelagerter zentraler Quality; keine unabhängige Native-Runtime-
Fehlfunktion belegt. GUI/Worker/Jobs beendet, Closure und Quellprüfung PASS,
keine offenen Remoteleases oder Quarantäne. Journal 2/3 Workflows, 0/36
Collectorstarts. G1 nicht hardwareseitig abgenommen; Commit/Push-Bedingung
nicht erfüllt. Der verbleibende Start ist für das beauftragte G2 vorgesehen,
sofern die enge Bundlekorrektur lokal geprüft und übernommen werden kann.

Bundlekorrektur abgeschlossen: verbindliche Planpfade bestimmen jetzt die
exakten, deduplizierten suite-lokalen Transportwurzeln des gemeinsamen Bundles.
Globale Validierungs-CLI bleibt leer, Plan/Daten/Seed/Budgets unverändert;
bisherige Grenze budget>0 bleibt auch für Final unverändert. 15 fokussierte
Tests PASS (1,87 s), darunter zwei echte Remote-Einstiegs-/Archivtests mit
lokalem Transportdouble, zwei Setupselektionen, Bytevergleich aller Bilder und
Labels/Annotationen, Nachbarbestand ausgeschlossen, keine erneute Suiteauswahl.
Ein zusätzlicher älterer Generated-Suite-Test scheitert an einem Fake-SDK-
HEF-Dtype-Mismatch; auf unverändertem final_ap00_ap08 ebenfalls reproduziert.
Keine wissenschaftliche Produktlogik dafür angepasst.

Bundlefix in Hauptinstallation bestätigt: zwei neue Regressionstests PASS
(1,54 s), 11 Zielimporte/vier Resolver/Quellintegrität verified. Danach G2 als
dritter und letzter Workflowstart: ap06_ap08_g2_20260923_162251. Warmvorprüfung
ohne Kaltbuild, echtes übertragenes Bundle mit 32 Bildern plus Manifest.
Alle acht zentralen Qualityanfragen technisch vollständig: vier referenznah,
vier Genauigkeitsverlust, davon zwei statistisch unsicher. Keine Änderung
wissenschaftlicher Kriterien und keine Umwertung negativer Ergebnisse.

G2 reale normale GUI-Abnahme technisch FAILED. Beide H8/H10-Native-Splits
mit 100 Frames/10 Warmup/einer Wiederholung erfolgreich. Beide Hailo-Full-
Blätter scheitern vor Transfer an _read_json(path, {}) bei Einargumentsignatur.
Beide TensorRT-Full-Blätter scheitern vor Engineausführung an falscher
Producer-Set-Pfadbindung (.quality_first statt kanonisch quality_first).
Beide Implementierungsfehler bereits im ursprünglichen AP00–AP08-Export;
enge Aufrufkorrekturen und reale lokale Übergangsregressionen folgen, ohne
Validatorlockerung oder neue Hardwarestarts.

Energie: erster H10-Split-Collector empfängt Daten, dann WouldBlock/EAGAIN
und Empfangsstille ohne bestätigtes Protocol-END. Nach etwa 51 s Quellenende
ausdrücklich unbestätigt; ControllerResourceBroker stoppt bestimmungsgemäß
mit campaign_source_completion_unresolved. Null gültige Energiereplikate,
keine 60-s-mal-drei-Abnahme. Bestehende Dauer und Capturehülle unverändert.
Journal endgültig 3/3 Workflowstarts und 1/36 Collectorstarts; keine weitere
Ausführung innerhalb dieses Auftrags. Keine neue Thesis/kein alter Run-Resume.

GUI/Worker/Collector beendet, Artefaktabschluss PASS, Quellintegrität vor/nach
PASS, Originalsettings unverändert. Lesende Produkt-SSH-Nachprüfung zeigt auf
Controller/H8/H10 keine bekannten aktiven Produktprozesse/Leases/gehaltenen
Locks. Vier dauerhafte Ressourcenquarantänen (Controller capture/NIC, H10-DUT,
Quelle 192.168.0.176) bleiben erhalten: Prozessende ist keine Quellenfreigabe.
G1/G2-Gesamtabnahme und Commit/Push-Bedingung offen. Qualitätsforschung bleibt
vertagt. Details im bestehenden Aufgabenroot evidence/uebernahme_v2/G2_REVIEW.md
und selection/G2_SOURCE_STOP_FINDING.md.

Nach G2 beide Full-Aufruffehler eng korrigiert: je zwei Produktzeilen in
Updater und identischem Paketspiegel. _read_json erhält ein Argument; nur
TRT-Full-Staging verwendet quality_first, Native-Split behält .quality_first.
Consumer-Validatoren, Identitäten und Buildschutz unverändert. Software:
fünf Vendor-Full-Übergangs-/Transporttests PASS, acht abgewählt, eine Warnung
(2,34 s); 15 TRT-Stager-/Runtime-Bindungstests PASS (2,04 s). Echte lokale
Parent/CLI/Child- und Stager/Validator-Übergänge, Transport gedoppelt; geänderte
und unvollständige Bindungen sowie falsche Rollenpfade bleiben blockiert.
Unabhängiges Review ohne offenen Befund. Gezielte Übernahme mit externer
Sicherung und bestehender Zielimport-/Quellprüfung; endgültiger Gitstand und
Zieltests im externen ABSCHLUSS_UEBERNAHME_v2.md. Kein Hardware-Nachtest,
keine Quellsperre entfernt, kein Commit/Push und keine wissenschaftliche
Freigabe aus diesen Softwaretests.

Fortsetzung nach korrigiertem Folgeauftrag v3.1 (23.09.2026): Der vorhandene
Entwicklungsstand ist als WIP auf codex/review-ap00-ap08-v3-1-20260923 gesichert
und erfolgreich ins vorhandene Source-Remote gepusht. Sicherungscommit
f52cf5ccb0d2c4172a0c3536eac8b67cc55e3d0c umfasst die vorgesehenen 92 Dateien,
einschließlich aller 33 neuen Source-/Testdateien. Die ausdrückliche Freigabe
dieses Reviewpushs ersetzt nur die frühere Commit-/Push-Bedingung; Hardware-,
Release- und Thesisabnahme bleiben offen. Private Runtimebelege bleiben lokal.

Die frühere gezielte Energie176-Nachabnahme ist anhand der Originale erhalten:
YOLO26s und YOLOv7-Paper TRT Full auf H10 je 3/3 gültig, sechs Collectorstarts,
null Retries, sechs bestätigte Quellenenden. 27 s angeforderte Geräteaufnahme
bei innerem Lastsoll 1 s und Commandfenstern etwa 3,07–3,17 s. Das ist kein
60-s-Nachweis. Keine Wiederholung, allgemeine Hardwarediagnose oder Resetpflicht.

Enger lesender G2-Vergleich: gleicher gespeicherter Collectorpfad, relevante
Konfiguration und GO-/Vorlaufreihenfolge; anderer Workload und 86 statt 27 s
angeforderte Geräteaufnahme. Empfangsabbruch nach etwa 5,099 s Stille, kein
Ablauf des äußeren 900-s-CLI-Limits. Die 233 anderen registrierten AP06-Aktivitäten
waren vor dem einzigen Capture freigegeben. Kein konkreter reproduzierter
Aufruf-/AP06-Übergangsfehler und keine daraus begründete Produktkorrektur;
unregistrierte Fremdlast und die Empfangsursache bleiben durch diese Belege offen.

Aktuelle lesende Belegungsprüfung über normalen Produkttransport: auf Controller
und H10 keine bekannten aktiven Produktprozesse, Leases oder gehaltenen Locks.
Die vier dauerhaften G2-Quarantänen bestehen unverändert. Installiert fehlt der
dokumentierte, auf genau diese physischen Ressourcen begrenzte Freigabeübergang
unter Lock/Ownership nach campaign_source_completion_unresolved. Der vorhandene
Workflow-Resume-Callback schließt physische Collector-/Source-Locks aus; beide
physischen Zulassungspfade verwenden acquire ohne Recoverycallback. Prozessende
und historische Erfolge geben die Ressourcen deshalb nicht frei. Dies ist eine
Software-Freigabelücke, kein nachgewiesener Hardwaredefekt oder Resetbedarf.

G1-/G2-Restabnahme vor Start blockiert; keine neuen Run-IDs oder Messungen.
Bestehendes Abnahmejournal ausschließlich von drei auf fünf erlaubte Workflows
erweitert, Einträge unverändert; vorhandener externer GUI-Operator auf dieselbe
Grenze angepasst und nicht gestartet. Verbrauch 3/5 Workflows und 1/36
Collectorstarts, verbleibend zwei bzw. 35. Keine Ressourcenfreigabe oder
Sperrenänderung, keine Umdeklaration des alten finished=false-Versuchs.

In v3.1 keine neuen Produkt-/Testimplementierungen oder Testläufe; vorhandene
lokale Full-Fix-Belege gelten weiter, reale Full-/Detectionabnahme bleibt offen,
G2-Energie unvollständig. R1 bleibt partiell: harter Controllerabbruch vor
dauerhaft belegtem lokalem Cleanup weiterhin nicht abgedeckt. Kein Thesisstart
oder Resume alter Runs. Details und Originalbezüge im bestehenden Aufgabenroot
unter evidence/restabnahme_v3_1; nächster notwendiger Schritt ist die begrenzte
Klärung des fehlenden physischen Ressourcen-Freigabeübergangs.

Abendfortsetzung v3.2 (23.09.2026): Der konkret fehlende physische
Freigabehandler ist jetzt im normalen Produktpfad implementiert. Ausdrücklicher
Einstieg: python -m onnx_splitpoint_tool.remote.process_lease_cli recover-capture.
Er verbindet bestehende Ressourcenlocks und die ursprüngliche Captureoperation;
keine automatische Recovery bei normalem acquire, kein Collector-/Firmwareumbau.
Alle vier EX-Locks werden vor einer Änderung gehalten. Exakte Originalowner,
Request/STOP, Registrybindung, dauerhafter Collector-Cleanup, terminaler Parent,
offene Remote-Leasedeskriptoren und aktuelle Prozesssicht werden geprüft.
Ein tatsächlich ausgeführter, zeitlich gebundener Betreibereingriff mit Quelle,
Versorgungswirkung und beobachteter Bereitschaft ist ausdrücklich erforderlich.

Originalquarantänen, Besitzer, Betreibereingriff und Beobachtung bleiben im
recovery-Feld der vorhandenen Captureantwort erhalten; ursprünglicher STOP und
alte Mess-/Retry-/Quellenbudgets werden nicht umgeschrieben. Strikte persistente
Ownerwrites erfolgen noch unter allen Originalquarantänen; controller:capture
wird zuletzt freigegeben. Fehler rollen auf die ursprünglichen Fences zurück.
Wiederholung derselben erfolgreichen Recovery ist ohne Messung idempotent;
spätere fremde Quarantänen bleiben gesperrt. Dokumentierter Bedienaufruf und
Grenzen in docs/PLATFORM_POWER_CONTROL.md. Kein neuer Hash-/Journalmechanismus.

Fokussierte lokale Softwareprüfung: 34 Recoverytests PASS (3,46 s),
23 Besitzer-/Transporttests PASS (0,48 s), dazu 44 bestehende AP06-/Remotelease-
und Lock-Fallback-Regressionen PASS. Die Tests enthalten echte temporäre flocks,
kontrollierte lokale Prozesse, konkurrierende Recovery, fehlende/falsche Belege,
Schreib-/fsync-Fehler und Prozessabbruch während des letzten Owner-Schreibens.
Keine Produktivhardware aus diesen Tests. Im kombinierten Zwischenlauf waren
zwei neue parametrisierte Test-Erwartungen noch auf einen festen Programmnamen
gesetzt (98 PASS/2 FAIL); Test-Erwartung korrigiert und ganze betroffene
Besitzerprüfungsdatei danach erfolgreich. Eine vorhandene Protobufwarnung.

Aktuelle lesende Produkttransportprüfung auf Controller/H10: bekannte exakte
G2-Vorgänger abwesend, keine bekannten konkurrierenden Produktprozesse oder
aktiven Remoteleases, keine Sichtfehler. Das bestätigt keine Quellenbereitschaft.
Die Betreiberklärung wurde gleich zu Beginn angefordert: lokal ist für den
Quellencontroller kein dokumentierter Status-/STOP-/Resetweg mit geklärter
Versorgungswirkung vorhanden. Die Jetson-/M.2-Railtoggles sind kein belegter
Quellenreset. Eine konkrete tatsächliche Bedienhandlung ist bislang nicht
belegt; kein pauschaler Resetbedarf wird daraus abgeleitet.

Daher produktive Recovery noch nicht angewendet, alle vier G2-Quarantänen
erhalten. G1/G2 und der ausdrücklich autorisierte anschließende Final-7x1-Lauf
noch nicht gestartet. Abnahmejournal unverändert 3/5 Workflows, 1/36
Collectorstarts. Statische Prüfung der vorhandenen Ein-Split-Kopie bestätigt
den vereinbarten Finalumfang; der tatsächliche Startsnapshot/Preflight folgt
erst innerhalb der freigegebenen Abfolge. Final erhält einen eigenen normalen
Kampagnenscope, keinen erweiterten 5/36-Testetat. R1 bleibt partiell; keine
allgemeine Crash-Recovery oder wissenschaftliche Freigabe. Belege im bestehenden
Aufgabenroot unter evidence/abendstart_v3_2.

### 2026-09-24 – gezielte Quality-/Transfer-Zulassungsregression

Auftragsbasis ist der saubere Reviewstand 0bba448. Vor Produktänderungen den
bereits durch den Nutzer angeforderten GUI-Cancel geprüft: keine lokalen
GUI-/Qualityworker, keine offenen Remoteoperationen im Run-Leasejournal und
keine bekannten Produktprozesse oder aktiven Leases auf allen drei DUTs bei
fehlerfreier lesender Produktsicht. Der alte Jobbericht bleibt unvollständig
(fünf gespeicherte running-Zeilen); kein nachträgliches Umschreiben und kein
Resume. Originalruns, Profile und Caches bleiben erhalten.

ResourcePauseGate.poll_activity berücksichtigt den Vorrang eines älteren
Tickets jetzt nur, wenn auch dessen CPU-Bedarf aktuell ausführbar ist. Die
isolierte Änderung behebt den Fall Kapazität 7, aktive Quality 4, älterer
Wartender 4, späterer Transfer 1. Echte CPU-/RAM-Grenzen, Transferkapazität,
Capture-DRAINING-/QUIET- sowie DUT-/Quellenfences bleiben unverändert. Zwei
Uploadmeldungen kennzeichnen nun mögliche Admissionwartezeit ausdrücklich.

Softwarebeleg: derselbe neue reale Service-/Gate-/SCP-Wrappertest vor Änderung
für Legacy und optimized_coco_v1 jeweils gezielt FAIL, danach 11 neue Tests
PASS (0,49 s). Vier kontrollierte Statistikworker bleiben während des
Transporteintritts aktiv; nur die externe SCP-Prozessausführung ist gedoppelt.
Ausführbarer älterer Auftrag, Weiterlauf des großen Wartenden, sämtliche
Ressourcenfences und wartender Cancel ohne Ticket-/Slotverlust sind abgedeckt.
Weitere 51 vorhandene Gate-/Ressourcen-/Canceltests PASS (13,26 s), 72
Statistik-/Checkpoint-/Canceltests PASS (20,82 s; zehn Warnungen einschließlich
ResourceTracker-Warnungen). Das ist keine Hardwareabnahme.

Separate private Finalprofilkopie über den echten Tk-Editor gespeichert und
neu geladen; normaler Startresolver und sichtbare GUI-Zusammenfassung stimmen
überein. Effektiv optimized_coco_v1, vier Worker, ein aktiver Request,
Block256, Checkpoints an, 512 MiB Vorbereitungscache; global_barrier und
model_barrier ausdrücklich erhalten. Die wissenschaftlichen Felder bleiben
gleich: sieben Modelle, drei Setups, ein Single-Tensor-Split, stratified_windows,
5000 Bilder/5000 Bootstrap, Native1000/100/3, Native-Energie60s mal drei,
FS/command, ein Uploadslot, Force/Generic-Energie/Windowprobe aus. Native bleibt
auf die unterstützte Teilmenge ohne Nachrücken begrenzt. 47 fokussierte
Profil-/Resolver-/GUItests PASS (55,72 s). 48 geschützte Originaldateien
bytegleich, keine globalen Runmode- oder anderen Profiländerungen.

Repräsentativer Offlinebeleg auf vorhandenen technisch validierten YOLO11l-
Detectioninputs: 5000 Bilder, Referenz31931 und Kandidat28238 Detektionen,
4949 unterschiedliche Bildrecords. Die ersten 16 identischen Ziehungen liefern
zwischen Legacy und optimiert bitgleiche absolute Komponenten, Deltas, Ratios,
Undefinedmasken, Punkte und wissenschaftliche Ergebnisfelder. Legacy62,909s,
optimiert41,517s mit dominanter Vorbereitung; daraus kein allgemeiner Faktor.
Während tatsächlich rechnender optimierter Statistik gelang der kleine
Transfer durch reales Gate und SCP-Wrapper vor Ende der Qualityanfrage, ohne
Hardwaretransport oder Messphase. Frühere vollständige 5000/5000-Paritätsbelege
der unveränderten Statistik bleiben erhalten; kein erneuter Legacyvollvergleich.

Danach genau ein vollständiger optimierter 5000/5000-Request mit vier echten
Prozessworkern: 673,149s von Submit bis Ergebnis, Referenzphase351,613s,
Kandidatenphase310,789s, Plan0,644s, Merge0,014s. Je20 vollständige Blöcke
decken exakt alle5000 Ziehungen ab; die ersten16 bleiben bitgleich zum direkten
Vergleich. Kalt: kein fertiger Ergebnis-, Referenz-, Plan- oder Checkpointhit.
Warm innerhalb dieses Requests: je16/20 Blöcke mit Prepared-Cachehit, jevier
Vorbereitungen pro Seite. Kein neuer Zeitfaktor gegenüber einem ungemessenen
Legacyvollrequest. Gespeicherte Originalinputs unverändert, alle vier Worker
mit Exit0 beendet, Dispatcher/Manager beendet, keine Cleanupfehler.

Technisch completed, wissenschaftlich accuracy_loss/Legacy-FAIL: AP50:95
Referenz0,466826742, Kandidat0,441505441, Delta-0,025321301 mit 95%-Intervall
[-0,029735571;-0,021772647]. Relative Verlustschätzung5,424% mit Intervall
[4,658%;6,288%], Berichtsunsicherheit borderline. Keine Qualitätsumwertung,
keine neue Inferenz und keine Hardware-/Energie-/Releaseabnahme daraus.

Die vier dauerhaften G2-Quarantänen bestehen weiterhin. Der normale physische
Leasepfad verwirft quarantänisierte Ressourcen, wird im Runner jedoch nur bei
per_case oder per_setup aktiviert. Bei den beauftragten beiden Barrieren ist
dies deshalb kein belegter automatischer Startschutz. Der neue Finalstart wird
vor Hardwarearbeit zurückgehalten, um die erhaltenen Sicherheitsquarantänen
nicht zu umgehen. Keine automatische Recovery, Bedienbestätigung oder
Quellenfreigabe; kein zusätzlicher Architekturfix in diesem engen Auftrag.
Keine zusätzliche G1/G2-Runde und keine Neustartschleife. Private Detailbelege
liegen im bestehenden Aufgabenroot unter evidence/quality_fix_20260924.

### 2026-09-24 – ausdrückliche Neuversuchsfreigabe für Final 7×1

Neuer Auftrag auf sauberem Stand 18ba7f3: frühere G1/G2-Startbedingungen sind
abgelöst; alte Berichte und Zähler bleiben Historie. Die Quality-/Transferkorrektur
wird nicht erneut implementiert und der vollständige Offline-Qualitytest nicht
wiederholt. Beauftragt ist genau ein neuer normaler Workflow mit der bereits
geprüften privaten Kopie Thesis_Final_7x1_QualityFix_20260924.yaml.

Der bestehende recover-capture-Einstieg akzeptierte bislang nur eine tatsächlich
ausgeführte physische Betreiberhandlung. Er erhält die enge Alternative
authorize-new-attempt: ausdrückliche, zeitlich und an Quelle/Profil gebundene
Freigabe eines neuen Workflows trotz unbestätigtem altem Quellenende. Kein Reset,
Idle-ACK, Bereitschaftsnachweis oder Erfolg des alten Versuchs wird behauptet.
Beide Freigabegründe sind exklusiv. Die vorhandenen vier flocks, Originalowner-,
STOP-, Registry-, Cleanup- und aktuellen Ownershipprüfungen, Wiederholschutz
gegen fremde Quarantänen sowie persistenter Rollback bleiben erhalten. Nur das
recovery-Feld der alten Antwort ergänzt die Belege; alte Messung, STOP, Budget
und Historie bleiben erhalten. Keine neue Recovery-, Registry- oder Hasharchitektur.

Softwareprüfung: 81 Recoverytests PASS (7,12 s), einschließlich beider
Freigabegründe bei Konkurrenz, unklarer Ownership, falschem Cleanup, Replay,
Schreib-/fsync-Fehlern und kontrolliertem Prozessabbruch. Weitere 23 vorhandene
Ownership-/Transporttests PASS (0,46 s). Read-only-Sicht auf Controller und
H8/H10/DeepX: keine bekannten konkurrierenden Produktprozesse, aktiven Leases
oder Sichtfehler. Alter G2-Parent failed, exakter dauerhafter Collector-Cleanup
vorhanden, keine offenen Remote-Leasedeskriptoren. Dies ist keine Quellen- oder
Hardware-PASS-Aussage.

Der normale Resolver bestätigt optimized_coco_v1/Worker4/Request1,
global_barrier/model_barrier, sieben Modelle, einen Single-Tensor-Split,
Quality5000/5000, Native1000/100/3, Native-Energie60s×3 FS/command und die
angeforderten Full-Baselines. Force bleibt aus, gültige Artefakte werden
wiederverwendet, notwendige fehlende über die bestehenden Regeln gebaut.
Neue Aufnahmen erhalten eigene Run-/Session-/Versuchsverzeichnisse, frische
Collectorprozesse und Socket-/GO-Belege sowie Run-/Fensterbindung; kein
rückwirkender Quellenabschluss wird daraus abgeleitet. Die bekannte fehlende
AP06-Brokeraktivierung bei den Barrieremodi bleibt eine Grenze. Deshalb erfolgt
der Start erst nach erfolgreicher expliziter Freigabe der konkreten alten Fences.
Bei erneutem Quellenfehler gelten normaler Kampagnen-STOP und Cleanup, keine
automatische Workflow-Neustartschleife. Reale normale GUI-Abnahme ist separat
von den lokalen Tests zu bewerten, ohne vorweggenommenes Gesamtergebnis.

Private Belege und tatsächlicher Startstatus:
evidence/final_retry_20260924_axub7jja im bestehenden Aufgabenroot.

Produktive Recovery erfolgreich: released für genau die vier alten G2-Ressourcen,
0 Messungen durch Recovery. Acht geschützte Dateien bytegleich, die alte
Captureantwort ausschließlich um recovery ergänzt; alle vier Originalquarantänen
darin vollständig erhalten. Unmittelbar vor Start erneut fehlerfreie, freie
Ownershipsicht auf allen drei DUTs. Installierte Source-Integrität verified
(2080 Dateien), git diff --check ohne Befund.

Danach den echten normalen GUI-Startbutton genau einmal betätigt. Tatsächliche
neue Run-ID: thesis_final_7x1_qualityfix_20260924_20260924_105448.
Start im Produktlog: 2026-09-24T10:54:48+02:00; Runverzeichnis unter
/home/kmika/Models/EvaluationRuns, permanentes Log darin evaluation_workflow.log.
Der mehrstündige Workflow bleibt in der GUI aktiv; Startbeleg ist keine
abgeschlossene Hardwareabnahme oder wissenschaftliche Gesamtfreigabe.
Keine zusätzlichen G1/G2- oder Offline-Qualityvolltests, kein Commit/Push.

### 2026-09-25 – begrenzter Main-Release 2.91.0: Retries und zukünftige Claims

Der neue Abschlussauftrag ersetzt frühere Abnahme-/Recovery-Startbedingungen.
Der oben dokumentierte 7×1-Lauf ist inzwischen terminal partial; die damalige
GUI arbeitet nicht mehr. Vor Produktänderungen waren keine aktiven alten
Produktjobs sichtbar. Der komplette installierte Source samt zuvor lokalen
Recoveryänderungen, Installationsmetadaten und privaten Konfigurationen wurde
extern gesichert. Ausgangspunkt: Reviewbranch 18ba7f3; tatsächlich gelesenes
origin/main ef44c9446bb90a837637746ef8651bae552d126f ist dessen Vorfahr.
Version/Tag 2.91.0 waren lokal und remote frei. Normales Main-Merge/Push und
Installation sind ausdrücklich beauftragt; keine Hardwarekampagne.

Energie: Ein gemeinsamer positiver Ganzzahlvalidator ersetzt die voneinander
abweichenden Wertemengen bei Profil, CLI-Weitergabe und Taskbudget. Null,
Booleans, nichtpositive und nichtganzzahlige Werte bleiben Fehler. Explizite
Kampagnen-Retries erreichen den vorhandenen äußeren Invalid-Repeat-Pfad auch
bei direktem Mess-CLI-Einstieg; ausdrücklich interne Null-Retries bleiben null.
Keine neue Retryschleife, unveränderte Standardwerte anderer Profile. Die neue
Konfiguration 2/20 bedeutet drei logische Replikate, jeweils höchstens drei
Versuche und höchstens neun reservierte Ketten je Zeile. Quellenfehler zählen
kumulativ über Erfolge, Modellwechsel und Resume; Fehler20 stoppt die Quelle.
Sauberer Quellenabschluss bleibt Voraussetzung für einen Retry. Cancel,
aktive Owner und unbestätigtes END werden nicht durch verfügbares Budget
überstimmt; keine erneute Freigabe alter Quarantänen in diesem Auftrag.

Qualityzuordnung: Der Nativevalidator rief das abschließende Claim-Gate im
Erfolgszweig zusätzlich vor der zentralen Bindung auf. Dieser erste Aufruf
löschte einen gültigen vorläufigen semantischen Claim; ein späterer exakter
Join konnte ihn nicht wiederherstellen. Entfernt wurde ausschließlich dieser
vorzeitige Aufruf, identisch im Source und gepackten Remoteskript. Der echte
Validator-, Energieplan- und Reporterpfad wurde mit unveränderten gespeicherten
ResNet/H8-Full-Eingaben und kontrolliert ersetzter Neuberechnung verfolgt:
quality_claim_result_verified ist künftig true. Der wissenschaftliche
Energieclaim bleibt wegen screening_energy_policy_nonclaimable false.
H10 YOLO26s/b364 bleibt semantisch ungültig; eine wirklich fehlende zentrale
Bindung bleibt no_exact_identity_match. Keine pauschale Claimfreigabe, keine
Reporteränderung, keine Entfernung von Matrix-/Rollen-/Strukturgrenzen.
14 konsumierte Originaldateien sind nach der separaten Wiedergabe bytegleich.
Der neue Positivtest scheitert mit ursprünglichem Code (1 FAIL/3 PASS); mit
Korrektur bestehen alle vier Kontrollen und insgesamt46 fokussierte
Claim-/Admission-/Provenienzregressionen (3,18s).

### 2026-09-25 – private 20er-Profilkopie und lokale GUI-Abnahme

Der echte Tk-Profil-Editor lädt die gelieferte Vorlage, speichert sie normal
und lädt die endgültige private Kopie erneut. Datei-/Hinweisdialoge sind im
lokalen Test kontrolliert beantwortet; Produktwidgets und Resolver bleiben
unverändert. Der Startresolver materialisiert dasselbe Profil. Wechsel
standard→smoke→final halten sämtliche Sollwerte. follow_tool_config=false
bindet den normalen Snapshot; nur normale neue Resolvermetadaten entstehen.
Die GUI-Normalisierung ersetzt die Native-Budgetherkunft durch frozen_resume
und entfernt redundante Native-Performance-Overrides, nicht die Werte.
Die Energie-Budgetherkunft bleibt profile_override. Alle46 anderen geschützten
Profile/Konfigurationen einschließlich der Downloads-Vorlage sind bytegleich.

Endkopie: profiles/Thesis_20Splits_N5000_B1000_20260925.yaml (maschinenlokal,
nicht Teil der Veröffentlichung). Sieben Modelle/drei Setups/sieben Runprofile;
20 stratified_windows-Grenzen, shortlist20, Single-Tensorfilter aus, kein
zusätzlicher Auditsatz, keine Nativequote/kein Backfill. N5000/B1000/Seed20260710,
optimized_coco_v1/Worker4/Request1/Block256/Checkpoints; Native1000/100/3,
Queue3/Inflight8 und unveränderter Precisionvertrag. Energie60s×3 FS/command,
Split und Full an, Generic-/CPU-/ORT-Energie und Windowprobe aus. Beide
Barrieren/ein Uploadslot, Hailo balanced/Opt1/B500/Batch8, DeepX EMA/Opt0/B500,
Force aus/keep_artifacts an, Rollen development/Claimscope evaluated_matrix.

Die tatsächliche stratifizierte Produktselektion auf gespeicherten realen
Graphanalysen erreicht für jedes Modell20 legale Grenzen, insgesamt140 ohne
Shortfall. Die Single-Tensor-Nativekandidaten pro Setup sind MobileNet17,
ResNet19, YOLO11l2, YOLO26m3, YOLO26s2, RegNet20 und YOLOv7 1 (zusammen64).
Nur explizite alte Checkboxausschlüsse werden für die jetzt deaktivierte
Checkbox rückgängig gemacht; kein Score-/Backendfilter ersetzt Kandidaten.
Vier Kohortenfälle haben passende historische erfolgreiche Nativebelege auf
allen drei Setups. Für136 Fälle fehlt im betrachteten 7×1-Cachebericht die
genaue Splitbindung; aktuelle Vorbereitung/Cacheabdeckung bleibt unbekannt,
keine136 behaupteten Cold Builds. Die konkreten H10-/H8-Negativgrenzen b398/b364
liegen ohne Sonderfilter außerhalb der neu selektierten Kohorte. Keine
globale H10-Sperre, kein Neubau zum Erzwingen von PASS, keine Laufbarkeits-,
Builddauer- oder Energiezeilengarantie. Angeforderte Full-Baselines bleiben
unabhängig von der Native-Split-Teilmenge erhalten.

Gezielte G-N1/G-N2/G-N3-, Profilroundtrip- und Quality-Speed-Regressionsauswahl:
69 PASS/0 FAIL/0 SKIP in109,30s (zwei Deprecationwarnungen). Recovery-,
Ownershipbeobachtungs- und Transfer-Admissiontests:115 PASS in7,87s.
Die bestehende Installation wurde mit dem vorhandenen stdlib-only Editable-
Refresh auf2.91.0 aktualisiert;209 Entry-Points, Abhängigkeiten/Venv erhalten.
CLI meldet2.91.0/Buildv2.91.0. Die tatsächlich gestartete vollständige normale
GUI zeigt ONNX Split-Point Analyser v2.91.0 (core v2.91.0), löst die Profilkopie
im Workflowpanel auf und schließt normal:0 Jobs,0 Starts,0 Hardwareaktionen.
Logs/Save-on-close-Ausgabe bleiben in separater Testablage; alte Runmetadaten,
Profile und globale Settings unverändert. Dies ist eine lokale GUI-Abnahme,
keine normale GUI-Hardwareabnahme.

Privater Aufgabenroot: retry_main_release_20260925_b6cgpl9a unter
~/.local/share/onnx-splitpoint-codex. Er enthält vollständigen Startbackup,
genaue Testaufrufe/-logs, Claimtrace, Kandidatenlisten und Profilbericht.
Keine alte Messung, STOP-/Budgethistorie oder Ergebnisdatei wurde repariert
oder nachgemessen. Der alte Lauf bleibt48/61 erfolgreiche Energiezeilen mit
zwei ungültigen physischen Versuchen; R1-Hardcrash-Recovery bleibt partiell.
Keine neue H10-Numerikdiagnose, G1/G2-Runde, Offline-Qualityvollprüfung oder
7×1-/20er-Kampagne. Die historischen Hardwaregrenzen stehen getrennt in den
Release Notes2.91.0 und sind keine zusätzliche Software-Releasebedingung.

### 2026-09-25 – abschließende Retry-/Releaseprüfung

Ein ergänzender enger Reentrytest belegt, dass ein direkter neuer Mess-CLI-
Einstieg mit frischem Ausgabeverzeichnis bislang ein bereits gültiges
Kampagnenreplikat erneut reservieren konnte. Die vorhandene Budget-Admission
sperrt dies jetzt mit energy_repeat_already_valid, ohne alte Ergebnisse zu
laden oder Source-/STOP-Zustände zu verändern. Normales Resume überspringt
bereits vollständig verifizierte Zeilen weiterhin; partieller direkter
Wiedereintritt ist fail-closed, keine neu implementierte Teilresume-Recovery.
Andere Zeilen derselben Quelle und andere Quellen bleiben unabhängig.

Finaler fokussierter Retry-/Lifecyclelauf:84 PASS in78,71s; zusätzliche
Reentry-Isolationsprüfung:2 PASS in6,27s. Vorheriger Code scheitert nachweislich
an beiden neuen Reentrykontrollen. Kontrollierte echte lokale Prozessketten
prüfen zwei ungültige sauber abgeschlossene Versuche→dritter gültig, keine
vierte Aufnahme bei drei Fehlversuchen, höchstens neun reservierte Ketten,
kumulative20.Grenze, getrennte Quellen, Cancel, Owner, fehlendes END und
Vertragsabweichung bei Reentry. Normaler Plan→Mess-CLI→Collectoraufbau übernimmt
2/20 und invalid-repeat-max-retries2, drei Replikate/60s Last/FS/command.
Das erzeugte76s-Collectorfenster mit5s Vor-/Nachlauf,2000Hz/Kanal0 bleibt
unverändert. Physische Collectorstarts sind kontrolliert ersetzt. Die
Softwareprüfung behauptet keine Reparatur des physikalischen Sampleverlusts.

Release-/Versions-/Packagingregressionen:83 PASS in10,84s. Der erste Aufruf
hatte75 PASS und vier Fehler einer veralteten Testerwartung: vier historische
Diagnosewrapper verlangen weiterhin ausdrücklich Installation2.83, auch vor
--help. Diese Produktguards bleiben unverändert. Der Test prüft jetzt sowohl
den isolierten2.83-Hilfezweig als auch die exakte STOP-Ablehnung unter2.91.0;
der unterstützte DeepX-Wrapper bleibt separat geprüft. Keine Diagnose wurde
ausgeführt und kein historischer Hardwarepfad für2.91 geöffnet.

Die vom Nutzer nachbenannte kanonische KB REV5 vom19.09. wurde in den
maßgeblichen §§1.6–1.9,2.1, H10-Befunden und deren R9G-Fortschreibung abgeglichen.
Repariertes Diagnosebinding bedeutet weiterhin keine reparierten späten HEFs;
eine frühere Ersatzboundary ist ein anderer Fall. Der neue explizite
B1000-/20er-Auftrag ersetzt keine historische B5000-/Hold-out-Abnahme.
Die KB verbleibt privat; ihr historischer Arbeitsauftrag löst keine neue
Diagnose-/Abnahme-/Hardwarearbeit aus.
