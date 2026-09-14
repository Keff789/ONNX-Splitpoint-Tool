# ONNX Splitpoint Tool – kanonische, konsolidierte Knowledge Base

## Stand v2.80.4 · Produktivkontext, Cancel und Debugexport

| Feld | Arbeitsstand dieser Fassung |
|---|---|
| Stichtag | 12. September 2026 |
| Releasequelle | v2.80.4 auf der vollständigen v2.80.3-FIX5-Quelle; Softwareabnahme und Quellmanifest stehen im zugehörigen Lieferpaket |
| Bereits auf Smartmirror2 beobachteter Stand | v2.80.3 mit den gelieferten Korrekturen; das neue .4-Target ist dadurch noch nicht installiert oder hardwareseitig abgenommen |
| Zweck von v2.80.4 | Hailo8-Zusatzbestand normal konfigurieren, Quality-Abbrüche richtig zählen, große gültige Debugdaten vollständig exportieren |
| YOLO11l/Hailo10H, 5.000 Bilder | Untersuchung abgeschlossen: TensorRT Full PASS; Hailo Full und b062-Split FAIL unter der unveränderten 1-pp-Marge |
| Hailo8/MobileNet GPU | Vorhandene isolierte Compute-/Build-/Fixed16-Evidenz; keine neue B5000-Freigabe des privaten GPU-HEFs und kein nachgewiesener Bauzeitgewinn |
| Hailo-Rezept | balanced / Opt1 / B500 / Batch8; kein zusätzlicher Opt2-/Opt3- oder B1024-Sweep |
| Dokumentrolle | Aktuelle Arbeits-, Methoden- und Evidenzreferenz; historische Ergebnisse behalten ihre ursprüngliche Identität |

**Entscheidung:** Die neue Version repariert konkrete Softwareübergänge. Ein korrekt ausgewerteter Qualitäts-FAIL bleibt ein abgeschlossenes negatives Ergebnis. Er ist weder Anlass für fortlaufende unveränderte Mini-Tests noch für eine nachträgliche Änderung von Recipe, Grenzwerten, Referenz oder Fallauswahl. Passende Artefakte weiterverwenden, fehlende benötigte Artefakte normal bauen; produktives Force bleibt AUS. [E-2804-PLAN] [E-2804-ABGLEICH]

## Navigation

[0. Dokumentführung](#dokumentfuehrung) · [1. Methode](#methode) · [2. Fragenkatalog](#fragen) · [3. Setups](#endpunkte) · [4. Releasefortschritt](#v30) · [5. DeepX-Historie](#deepx-full) · [6. Energie](#energie) · [7. Modellqualität](#deepx-r1-r2) · [8. Cache und Abschluss](#abschluss) · [9. Historisches Complete Set](#complete-set) · [10. v2.80.4](#v31-plan) · [11. Abnahme](#gates) · [12. Aktuelle Aufgaben](#todo) · [13. Betrieb](#betrieb) · [14. Evidenz](#ablage) · [15. Klärungen](#grenzen) · [16. Übergabe](#uebergabe) · [17. Änderungen](#aenderungen) · [Quellen](#quellen)

<a id="dokumentfuehrung"></a>
## 0. Dokumentführung und Quellenrang

Diese Fassung löst die kanonische v2.79.30-KB vom 8. September 2026 als aktuelle Arbeitsübersicht ab. Methodenrahmen, Fragenkatalog, Energievertrag und historische Quellen bleiben erhalten. Frühere v31-Implementierungsaufträge werden nicht nochmals als aktuelle Aufgaben ausgegeben. Abschnitt 12 ist die einzige aktuelle Aufgabenliste.

Originale Ergebnisdateien haben Vorrang vor Auswertungen und Chat-Zusammenfassungen. Ein Implementierungsplan belegt einen Auftrag und die dort benannten Befunde; seine Beschreibung ersetzt keine unabhängige Prüfung einer fehlenden Originaldatei. `SOURCE_INDEX` und die Verifikation im .4-Bundle halten die tatsächlich verfügbaren Quellen und Testausführungen fest. Nachgereichte Archive oder neue Reporter verändern weder das Alter noch den ausgeführten Scope einer Messung.

Software-PASS, technische Hardwareausführung, Qualitätsentscheid und wissenschaftliche Vergleichsfreigabe sind verschiedene Aussagen. Ein fertiges negatives Ergebnis bleibt auswertbar. Ein abgebrochener Auftrag ohne Qualitätsresultat bleibt nicht ausgewertet. Ein neuer Downloadname, ein `plan.json` oder eine GUI-Erfolgsmeldung ersetzt keinen terminalen Ergebnisbeleg.

Das .4-Dokumentupdate selbst führt keine GPU-, Compiler-, Inferenz- oder Energiemessung aus. Zielinstallation und normaler H8-Kaltbuild dürfen nur mit ihren tatsächlichen neuen Nachweisen als .4 bestanden bezeichnet werden. Frühere H8-/H10-Smokes werden als historische Evidenz erhalten und bei unverändertem Pfad nicht erneut verlangt.

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

Quellen: [E-KB] [E-V30] [E-R1] [E-R2] [E-PLAN31].

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

<a id="v30"></a>
## 4. Releasefortschritt bis v2.80.4

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

Diese Zeilen sind ein Fortschrittsindex. Alte Testzahlen werden nicht als neue .4-Tests ausgegeben. Die auf Smartmirror2 tatsächlich gestartete Version ist aus ihrem Modul-/Source-Manifest festzustellen; gleiche sichtbare Versionsnummern können unterschiedliche ausdrücklich benannte FIX-Stände haben. [E-2804-CHAT] [E-2804-PLAN]

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

Die Werte sind als Befunde aus dem Plan/Q5 benannt. Ob ihre Originalresultate im .4-Build zusätzlich unabhängig gelesen werden konnten, steht im Quellenindex; fehlende Originalarchive werden nicht durch synthetische Nachbauten als geprüft ausgegeben. Die früheren „MobileNet erst noch erstmals größer bewerten“-TODOs sind mit diesen benannten abgeschlossenen Befunden überholt. Die genaue interne Verlustursache und die B5000-Qualität eines anderen privaten GPU-HEFs sind andere Fragen. [E-2804-PLAN, §8.2]

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

Vorhandene parsed/quantized HARs können bei einem späteren konkret begründeten Diagnoseauftrag mit denselben logischen Inputs emuliert werden. Nicht ausgeführte oder fehlende Stufen bleiben `not_available`; aus einem freien `origin`-Text folgt keine nachgewiesene Quantisierungsursache. Das ist keine neue Pflichtschleife für den Softwarerelease.

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

<a id="abschluss"></a>
## 8. Cache, Prozessgrenzen und terminaler Abschluss

Eine GPU-Buildpräferenz, GPU-UUID oder ein Overlaypfad sind Buildprovenienz, keine zusätzliche Modellcache-Identität. Der normale Builder prüft erst Modell-/Recipe-/Artefaktvertrag, dann positiven Cache bzw. genaue negative Compileevidenz. Nur ein tatsächlich erforderlicher Neubau benötigt den GPU-/Overlaykontext. Ein inzwischen fehlendes Overlay darf einen gültigen CPU-HEF-HIT nicht verhindern.

HEF, Receipt und Cachemetadata behalten ihre bestehenden atomaren Generationen-/Publikationsregeln; alte HEF-only-Bestände bleiben gegebenenfalls `legacy_unsealed`. Duplicate-/Generationauswahl bleibt deterministisch. Kein „neueste Datei gewinnen lassen“, kein Auswählen nach besserer Accuracy und kein automatisches Publizieren privater GPU-HEFs. Ein anderes tatsächlich verwendetes Artefakt braucht korrekt gebundene Runtime-/Qualityrequests. Identische Vorhersagen dürfen vorhandene mathematische Statistik wiederverwenden; fremde Hardwareprovenienz darf dabei nicht erfunden werden.

Bekannte `PARSER_UNSUPPORTED`-/`COMPILE_INFEASIBLE`-Fälle werden bei gleichem Vertrag nicht jede Nacht neu gebaut. Fehlender GPU-Kontext ist Infrastruktur und erzeugt keine dauerhafte negative Modellevidenz. `ABORTED_UNKNOWN` bleibt unvollständige Evidenz. Geänderte Splitauswahl muss alte/aktuelle Boundaries und erwartete Cold Builds erklären; ein neuer tatsächlich benötigter Split ist kein pauschaler Cacheverlust.

Der frühere globale Hashcache-Nachlauf ist als behobener und fortgeltend regressionspflichtiger Produktpfad dokumentiert. Keine zusätzliche Hash-/Seal-/Signatur-/Registryebene, kein neues Vollverzeichnis-Hashen und keine Metadatenarbeit in Native-Performancefenstern. Fehler beim Schreiben von Ergebnis oder Debug-ZIP dürfen kein finales Teilarchiv mit altem PASS publizieren. Primärfehler und Cleanupfehler bleiben getrennt.

Workflowlocks schützen aktive Prozesse und bleiben erhalten. Regulärer Cancel räumt eigene Worker/Supervisoren geordnet auf. Keine alten Chat-PIDs killen, keine Lockdateien als Cachefix löschen. Ein abgeschlossener Build ist noch kein Nachweis von Runtime, Qualität oder Energie.

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
## 10. Aktueller v2.80.4-Auftrag und unveränderte Grenzen

| AP | Aktuelle Umsetzung / Abnahmevertrag |
|---|---|
| AP0 | Vollständige .3-FIX5-Basis erhalten; aktuelle Nachtmodulidentitäten und verfügbare Originalquellen separat dokumentieren |
| AP1 | Zentraldeskriptoren 512 MiB gesamt / 64 MiB pro Datei; freigegebene strukturierte Ergebnisse/Indizes 256 MiB pro Datei; keine spätere alte 2-/8-/32-MiB-Rückstufung |
| AP2 | Ausgewertet, abgebrochen und technisch fehlgeschlagen getrennt; fertige Resultate und beobachtete Qualitäts-FAILs erhalten |
| AP3 | Vorhandenes Hailo8-Manifest explizit im Run-Mode-Editor wählen; gleicher Resolver bis zur realen Compiler-Kindumgebung |
| AP4 | Rezept-Reuse von konkreter Artefakt-/Qualityidentität unterscheiden; bestehende Cacheverträge absichern |
| AP5 | Historische H8-/H10-Smokes, negative B5000-Resultate, fehlende HAR-Stufen und Energieclaims korrekt beschriften |
| AP6 | Eindeutige .4-Identität; finales Archiv prüfen, echtes isoliertes Upgrade und durchgehenden normalen Zielworkflow vorbereiten |

Die 56 IDs im Plan sind Prüfanforderungen, keine vorab bestandene Testzahl. Der tatsächliche Umfang einschließlich nicht verfügbarer Originalquellen steht in der Lieferung. Modelle, Bilder, Roharrays und CUDA-Wheels bleiben aus Debug-/Gitbelegen ausgeschlossen. Größere erlaubte JSON-Klassen ändern weder Pfadkonfinement noch Symlink-/Hash-/Atomaritätsregeln. Größenfehler sind `size_limit_exceeded` mit Ist- und Grenzgröße, nicht `json_invalid`.

### 10.1 Hailo8 im normalen Editor

Neben „Hailo8 compute for new builds“ stehen Manifestpfad, Dateiauswahl und lesende Prüfung. Ein fehlendes optionales Feld lässt die bisherige explizite Environment-Auswahl zu. „Explizit auswählen“ mit leerem Pfad bedeutet bewusst kein Overlay. Jobwert → Familienwert → bisherige explizite H8-Environment-Variable → kein Overlay. Ein Default enthält keinen maschinenspezifischen Pfad.

CPU verwendet den gespeicherten Pfad nicht und startet keine Overlay-/GPUprüfung. Lesende GUI-Prüfung ist keine GPU-/XLA-Messung. Sie prüft vorhandene Familie, ausgewählte Venv, Metadaten und Komponenten mit dem bestehenden Validator. Kein Paketdownload und keine Mutation von `os.environ` des GUI-Hauptprozesses. Hailo10, DeepX, Runtime und CPU-Referenz dürfen keine H8-Librarypfade erben. Benutzerprofile und Registrys werden beim Upgrade nicht still auf einen lokalen Overlaypfad umgeschrieben.

### 10.2 Abbruchzählung und ETA

Der dokumentierte Q5-Sollreplay enthält 123 terminale Requests: 63 fertig ausgewertet mit 43 PASS, 18 FAIL, 2 INCONCLUSIVE sowie 60 Abbruchfolgen (vier `CancelledError`, 56 `QualityServiceClosedError`). `completed_count` zählt weiterhin ausgewertete Ergebnisse. Terminal ist nicht gleich erfolgreich berechnet und Requestzahl ist nicht Hardwaremesszahl.

Eine Service-Closed-Ausnahme ist nur mit passendem vorangehendem Run-Cancel als Abbruchfolge einzuordnen; ohne diesen Kontext bleibt sie technischer Fehler. Früher entstandene technische Fehler werden durch späteren Cancel nicht gelöscht. Fertige Cache-/Berechnungsergebnisse bleiben beim Cancelrennen einmalig erhalten; unvollständige Shards veröffentlichen kein CI/PASS oder erfolgreichen Cacheeintrag. Der alte abgebrochene Run bleibt cancelled und unvollständig, seine 18 beobachteten Qualitäts-FAILs bleiben fail.

Abbruchterminals, Cachetreffer und frühe Punktentscheidungen gehören nicht in die mittlere Laufzeit eines vollständigen Detectionbootstraps. Hailo-GPU beschleunigt nicht automatisch zentrale CPU-Statistik. Workerzahl, Methode, Wiederholungen und Margen werden für eine angenehmere ETA nicht geändert. [E-2804-PLAN]

<a id="gates"></a>
## 11. Ein gemeinsamer Zielablauf statt erneuter Mini-Testkette

Nach der Softwarelieferung folgt eine zusammenhängende, begrenzte Abnahme mit einem Startablauf und gesammelt exportierten Ergebnissen. Bereits bestandene isolierte MobileNet-Compute-/Build-/Fixed16-Schritte werden nicht erneut vorgeschaltet.

| Gate | Ziel und korrekter Ausgang |
|---|---|
| G0 | .4 installieren und fokussierte/fortgeltende Softwaretests; Profile, Venv, Overlay, Registrys und Caches erhalten |
| G1 | Alten gecancelten Nachtlauf mit normalem .4-Exporter: 123 gültige Originalrequests / 35.157.419 Byte und 14 Referenzdiagnosen; keine zweite Nachsammlerpflicht |
| G2 | Kleine echte CPU-Qualityprozessfixture mit Cancel; disjunkte Zähler und beendete eigene Prozesse |
| G3/G4 | Gespeicherte H8-Auswahl in frischen Prozessen; bekannte CPU-HEFs trotz GPU-Präferenz wiederverwenden, null Compilerdispatch |
| G5 | Normaler begrenzter MobileNetV3- plus YOLO11l-Workflow mit je einem vorhandenen Split, Standard B500/Bootstrap500, Referenz → Consumer → Native → Bericht |
| G6 | Nur ein tatsächlich noch benötigter vorab bestätigter H8-MISS, z.B. YOLO11l b064: gespeicherter Kontext bis ins reale Compilerkind, danach Reuse |

G5 verwendet den bestehenden Erwartungs-/Admissionpfad; ein unerwarteter MISS wird sichtbar, statt die kurze Runde heimlich in einen langen Build zu verwandeln. Nicht global `cache_verify_only` setzen, weil dann der normale Referenzpfad fehlt. Wenn G6 bereits warm ist, bleibt „aktueller normaler Kaltbuild nicht beobachtet“ ein benannter Scope; es wird kein vorhandenes HEF gelöscht oder per Force neu gebaut. Weitere wissenschaftliche Final-/Energieaufträge sind keine versteckten Voraussetzungen für Software-PASS. [E-2804-PLAN, §9]

<a id="todo"></a>
## 12. Einzige aktuelle Aufgabenliste

### Abgeschlossen oder als dauerhafte Regression erhalten

- [x] Hailo10 isolierte Toolchain-/GPU-/Build-/Fixed16-Untersuchung im benannten v34-Scope.
- [x] Hailo8 isolierte Compute-/Build-/Fixed16-Untersuchung gemäß benannter Q1–Q4-Evidenz; unabhängige Quellenverfügbarkeit im .4-Bundle separat ausgewiesen.
- [x] YOLO11l/Hailo10H zentraler 5.000-Bilder-Vergleich mit vollständigen Vorhersagen: TensorRT PASS, Hailo Full/Split FAIL.
- [x] Historischen negativen Opt2-Befund berücksichtigt; zusätzlicher Opt2/B1024- und Opt3-Versuch verworfen.
- [x] Produktive Force-Politik AUS und Wiederverwendung passender Artefakte festgelegt; CPU/GPU-Präferenz ist kein Neubaugrund.
- [x] MobileNet-B5000-Befunde aus dem .4-Plan als bereits ausgewertete negative Resultate statt pauschal „noch nie bewertet“ eingeordnet.
- [x] Alte v31-Planlisten und Installationsaufforderungen aus der aktuellen Aufgabenliste entfernt; frühere Quellen bleiben historisch erhalten.

### Aus der neuen Softwarelieferung auf dem Zielsystem auszuführen

- [ ] Den gemeinsamen .4-Installations-/Normalworkflowablauf mit erhaltener Nutzerkonfiguration ausführen und dessen vollständigen Ergebnisexport prüfen.
- [ ] H8-Kaltbuild nur bei ohnehin benötigtem echten MISS beobachten; ein warmer Bestand bleibt warmer Bestand.

### Nur für die ausdrücklich beanspruchte wissenschaftliche Aussage

- [ ] Passende Power-/Clock-/TPC-, FS-/Idle- und Dauer-/Replikatbelege für den konkreten Energievergleich verwenden; 1-s-Screening nicht zu 60 s × 3 umetikettieren.
- [ ] Einen benötigten offiziellen COCO-Bericht aus vorhandenen Vorhersagen ergänzen; das erfordert keine erneute Hailo-Inferenz.
- [ ] Hailo10H/YOLO26m b398 und YOLO26s b364 getrennt nach P1 → QuantInfo/Layout → P2 diagnostizieren, falls diese Varianten beansprucht werden; keinen heuristischen Decoderfix erfinden.

**Keine aktuelle Pflicht:** weiteres identisches Metadata-Smoke, neuer pauschaler Hailo-Optimierungssweep, private GPU-HEFs wegen Fixed16 automatisch austauschen, komplette nächtliche Kampagne nach jedem Patch wiederholen oder ein abgeschlossenes Qualitäts-FAIL erneut erzeugen. Ein weiterer Reparaturlauf braucht einen konkret nachgewiesenen Produktfehler oder einen ausdrücklich neu geplanten methodischen Auftrag.

<a id="betrieb"></a>
## 13. Betriebsregeln

GUI und laufende Workflows vor dem Update geordnet beenden. Der Installer erhält Tool-/Vendor-Venvs, Benutzerprofile, Run-Mode-/Hardware-Registry, vorhandene Overlays, Modell-/Qualitycache und Originalruns. Das Manifestfeld wird erst durch eine explizite Nutzeraktion ausgewählt. Kein Treiber-, System-CUDA-, DFC-, TensorFlow-, Torch- oder DeepX-Upgrade in v2.80.4.

`relaxed` bleibt auch für Final der vereinbarte Repro-/Cachemodus. Hailo balanced/Opt1/B500/Batch8, DeepX EMA/Opt0/B500, ImageNet-Mean/Std, Qualitätsmargen, Seed, AP-/Top-k-Definitionen und Bootstrapmethode bleiben gleich. Der vorgeschaltete Standarddurchlauf und der separate Finalumfang dürfen beim Resume nicht als identischer Run mit verändertem Profil vermischt werden.

Generische Energie bleibt aus. Native-Energie ist ein eigener Scope; Dauer und Replikate werden nicht still geändert. Vorhandene alte Fehlermeldungen oder PIDs sind keine aktuellen Betriebszustände. Primärfehler, erwartete Nichtrealisierbarkeit, technische Ausführung, Qualitätsentscheid und Cleanup werden separat gelesen.

<a id="ablage"></a>
## 14. Evidenzablage und dauerhafte Referenzen

Git erhält Code, Dokumentation, kleine Profile, Logs und ausgewählte strukturierte Ergebnisse. Modelle, Bildcorpora, HEFs/DXNNs/Engines, CUDA-Bibliotheken und Roharrays bleiben im gesicherten Originalbestand. Ein exportierter Ergebnisbericht ersetzt nicht die tatsächlich verwendeten Modelle oder Annotationen.

Das .4-Lieferpaket enthält den Planabgleich, konkrete Prüfprotokolle und Quellenindex. Original-Q1/Q3/Q5 werden nur als Originalfixture bezeichnet, wenn ihre Bytes tatsächlich verfügbar und geprüft sind. Abgeleitete kleine Cancel-Fixtures sind entsprechend benannt und ersetzen nicht den angeforderten Original-123-Request-Export. Ein Q2-Terminalreport ist keine unabhängig nachgezählte Q1-Rohdatei. Downloadfehler bleiben als Quellenlücke sichtbar.

Ein Replay schreibt neue Darstellung separat; vergangene `run_manifest`, Requests, Fingerprints und Qualitätsentscheidungen bleiben unverändert. Die alte kanonische KB vom 8. September ist die historische Quelle dieses Updates und bleibt als vorherige Fassung nachvollziehbar. Gleiche Dateinamen von Standalone-R1-Paketen beweisen keine gleichen Bytes.

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

<a id="uebergabe"></a>
## 16. Kompakte Übergabe

Produktbasis .3-FIX5 erhalten; neue Softwareidentität .4. Die aktuellen Änderungen betreffen H8-Konfiguration bis zum Compilerkind, wahrheitsgemäße Quality-Cancelzählung und größere normale Debugexports. Keine Numerik-/Methodenänderung, kein Force und kein Opt2-Sweep. YOLO11l/H10 unter seinem zentralen 5.000-Bilder-Vertrag abgeschlossen mit Hailo-FAIL; dieser Befund bleibt im Audit.

Historische isolierte GPU-/Fixed16-Tests erneut zu verlangen ist kein nächster Schritt. Der konkrete verbleibende Softwareabnahmeschritt auf Smartmirror2 ist der gemeinsame .4-Installations-/Normalworkflowablauf; ein echter Cold Build wird nur genutzt, wenn benötigte Artefakte tatsächlich fehlen. Wissenschaftliche Zusatzclaims erhalten ihre eigenen Grenzen und werden nicht als angeblicher Softwarefehler oder immer neuer Mini-Test dargestellt.

<a id="aenderungen"></a>
## 17. Änderungsprotokoll

### 12. September 2026 – v2.80.4

Aktuelle Kopfidentität, Releasefortschritt und Aufgabenliste auf .3-FIX5/.4 fortgeschrieben; v31-Planaufträge historisiert. Hailo8/Hailo10-Fixed16 getrennt, MobileNet-B5000-FAILs aus dem neuen Plan eingeordnet, abgeschlossene YOLO11l-Qualität mit vollständigen Vorhersagen und historische Opt2-Entscheidung aufgenommen. Alte automatische Wiederholungs- und Optimierungsvorschläge gestrichen. Cancel-, Debugbudget-, Overlay-, Reuse- und Quellenregeln integriert. Originalergebniswerte und vorhandene wissenschaftliche Methodik unverändert.

### 8. September 2026 – konsolidierte Fortschreibung nach R2

Die bisherige v17-Kopfidentität wurde als historischer Stand abgelöst, nicht als aktuelle Installationsanweisung weitergeführt. Methodenrahmen, Fragen-/Boundarythemen, FS-Primärmessung, Generic-/Native-Scope, Rankinggrenzen und einfache Evidenzablage bleiben erhalten. [E-KB]

Neu integriert sind die v26–v30-DeepX-Full-Fehlerfolge, die unabhängige v30-Teilabnahme, der bestätigte Abschluss-Cachefehler und dessen enger Fix, die acht Complete-Set-Fehlergruppen beziehungsweise zusätzlichen Hailo8-Bindungslücken, korrekte Ergebnis-/Energie-/Qualityzähler und der aktuelle v31-Plan REV4 FINAL. [E-V30] [E-PLAN31]

R1/R2 ersetzen die allgemeine DeepX-Pre-/Postprocessing-Ursachensuche durch einen konkret bestätigten Build-/Profil-Normalisierungspfad. Die MobileNet-Klassifikationsqualität bleibt trotzdem offen: kleine Nenner, gleiche Bild-IDs, Logitähnlichkeit und Accuracy sowie historische B500-Werte sind jetzt ausdrücklich getrennt. Überzogene Aussagen zum bereits bewiesenen Anteil des B500-Gewinns sind eingegrenzt. [E-R1] [E-R2]

Zusätzlich wurden beim Dokumentabgleich die tatsächlich gespeicherten **1-s-Energieaufträge** und die spätere FS-Gain-/M.2-Idle-Anwendung aufgenommen. Nicht vorhandene ursprüngliche Kalibrierungsbelege werden nicht als hier geprüft ausgegeben; die Abweichung zwischen einer alten Gateuntergrenze und einem später verifizierten Faktor bleibt sichtbar. [E-CS] [E-DOC]

Diese Beschreibung betrifft das historische Dokumentupdate vom 8. September, nicht die neue .4-Softwarelieferung. Originale Messdaten und frühere Snapshots bleiben unverändert.

<a id="quellen"></a>
## Quellen- und Fundstellenverzeichnis

Die Kennungen verweisen auf vorhandene Dateien beziehungsweise klar benannte Chatbeobachtungen. Innerhalb von ZIPs sind die angegebenen Pfade relativ zur Archivwurzel. Das kompakte KB-Begleitpaket enthält Dokumente und die kleinen read-only Projektionen, **nicht** die großen Roharchive oder Modellbinaries.

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
