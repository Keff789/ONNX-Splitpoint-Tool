## 22.10.2026 — Diagnose der normalen Ergebnissammlung

Der normale Collector protokolliert jetzt seinen Beginn, den Status seines
EXIT-Traps und fehlgeschlagene Kopien mit ursprünglichem Kopier-Returncode
sowie Shell-escaped Quell- und Zielpfad. Der Trap setzt keinen Returncode
und installiert keinen Signalhandler. Sein `exit_trap_status` ist ausdrücklich
kein allgemeiner Prozess-Returncode: Bei Signalabbruch kann der zuletzt
beobachtete Shellstatus vom tatsächlichen Prozessstatus abweichen. Der echte
Transportstatus bleibt getrennt erhalten. Dateiauswahl, Kopierreihenfolge,
optionale Quellen, Fehleraggregation, Zeitbudgets und Aufbewahrung bleiben
unverändert; diagnostische Schreibfehler ersetzen keinen ursprünglichen Fehler.

Der normale SSH-Aufruf persistiert zusätzlich die bereits erhobene
Capturediagnose. Leser- und Wait-Fehlertypen werden ohne Payloadinhalte
aufgezeichnet; ein unbekannter Wait-Ausgang wird nicht mehr diagnostisch als
`completed` überschrieben. Rückgabewerte, Reader-Wartezeiten, Environment,
Leases, Bereinigung und deren bestehender Returncode-70-Vorrang bleiben gleich.
Es gibt keine neue Marker-basierte Workflow- oder wissenschaftliche Freigabe.

Die gezielte lokale Regression bestand 49 Tests ohne Fehler oder Skips,
einschließlich stiller Kopierfehler, früher `set -e`-Abbrüche, optional leerer
Quellen, SIGTERM/SIGKILL, Reader-/Waitfehler und bestehender Collection-/Lease-
und Umgebungsverträge. Der normale Writer liefert dieselben 42 Bundle-Dateien
einschließlich aller 37 Python-Dateien bytegleich zur Quality-Epoche v4.
Die geänderten Dateien gehören ausschließlich zum Controller. Folgende Läufe
benötigen eine ehrlich neu gebundene Controller-Quellenepoche; alte Freeze-,
Run- und Qualitybelege bleiben unverändert. Eine neue Hardwareprüfung dieser
Controllerfassung steht aus. Die erfolgreiche einzelne H8-Sammlung an einem
neuen Diagnoseziel erklärt die ursprünglichen drei Collection-Fehler nicht.

## 22.10.2026 — Lebensdauer des Generic-Quality-Eingabebeobachters

Der explizite Generic-Quality-Beobachter stellte die Klassenmethode `infer`
als gebundene Instanzvariable wieder her. Dadurch entstand eine Selbstreferenz,
die die Session nach der Beobachtung bis zur zyklischen Garbage Collection
festhalten konnte. Die Korrektur entfernt den vorübergehenden Instanzeintrag,
wenn ursprünglich die Klassenmethode verwendet wurde, oder stellt den exakt
vorher vorhandenen Instanzwert wieder her. Argumente, Ergebnisse, Reihenfolge,
Eingabesnapshots und die Prüfung auf genau einen Inferenzaufruf bleiben gleich.

Auslöser war ein tatsächlicher H8-Kindprozessabsturz mit Returncode -11 nach
einem 5.000-Bilder-Capture und dem Schreiben des Berichts. Die Selbstreferenz
ist mit den ursprünglichen Quellen reproduziert; ihre Entfernung ist mit
68 gezielten Quality-/H8-Regressionen und 29 unabhängigen Prüfungen belegt.
Diese sich überschneidenden Tests sind getrennte Prüfläufe. Sie beweisen
weder die native Absturzursache noch eine erfolgreiche Hardwarebehebung.
SDK-Teardown, Garbage-Collection-Policy und Prozessbeendigung werden durch
diese Korrektur nicht verändert.

Genau eine Datei der gemeinsamen Runtime-Closure ändert sich. Folgende Läufe
benötigen deshalb einen neuen echten Freeze und begrenzte Hardwarekontrollen.
Die numerische Generic-FPS-Ausführung, Slotzahlen, Pipeline und vollständige
Tasknachverarbeitung bleiben unverändert. Abgeschlossene Beobachtungen und
Qualitätsbelege behalten ihre ursprünglichen Quellenepochen und passenden
Verifier; sie werden weder umetikettiert noch automatisch wiederholt. Der
bereits laufende H8-Workflow bleibt bis zu seiner normalen Finalisierung
unverändert. Die Korrektur ist zunächst ausschließlich softwaregeprüft.

## 22.10.2026 — Controllerkorrekturen für Umgebung, Ergebnisaufbewahrung und Cacheproben

Die implementierten Controllerkorrekturen setzen ausdrücklich deklarierte
Umgebungswerte bei geleasten SSH-Aufrufen zusätzlich im eigentlichen Payload
nach dem Start seiner Login-Shell. Die äußere Launcherumgebung bleibt
erhalten. Der normale Runtime-Identitätsvergleich bleibt unverändert aktiv;
Aufruferbefehle, Dispatchhistorien, Wiederaufnahmebudgets und das
Lease-/Bereinigungsprotokoll bleiben erhalten. Die lokale Reproduktion mit
normalem Launcher und Guardian scheiterte zuvor in beiden geleasten Pfaden
an einer durch Shell-Startdateien veränderten Umgebung. Zehn gezielte
Umgebungsprüfungen und 27 bestehende Lease-/Fallbudgetregressionen bestanden
nach der separat unabhängig geprüften Korrektur.

Die zweite Korrektur schützt den gesamten Remote-Laufordner vor Löschung,
wenn die Ergebnissammlung fehlschlägt oder ihr Erfolg nicht aufgezeichnet
wurde. Ein später erfolgreicher Download und die ausdrückliche Erlaubnis
zur Bereinigung von Teilergebnissen heben diese Sperre nicht auf. Echte
Kopierfehler werden gesammelt und abschließend gemeldet; spätere Kopien
laufen weiter und fehlende optionale Quellen bleiben erlaubt. Ursprüngliche
Inferenz-Returncodes und frühere Hauptfehler bleiben erhalten. Vollständige
Collection mit Paket-Timeout und erfolgreichem rekursivem Download behält
den normalen Pfad. Die separat geprüfte Korrektur bestand 38 gezielte
Controller-/Transportregressionen; die ursprünglichen Fehlerreproduktionen
bleiben erhalten.

Die dritte Korrektur begrenzt das tatsächlich SSH-gequotete Argument großer
normaler TensorRT-Cacheproben einschließlich NUL auf 120 KiB. Disjunkte
Batches behalten die vollständigen unveränderten Validator-, Quellen-,
Owner- und ABI-Verträge und teilen das ursprüngliche einmalige
Verifierbudget. Kleine Befehle bleiben bytegleich. Nach einem Transport-
oder Antwortfehler startet kein weiterer Batch; vorherige gültige Ergebnisse
bleiben erhalten und übrige Anforderungen werden ausdrücklich als nicht
ausgeführtes UNKNOWN gemeldet. Rein lokale Größenfehler und normale
MISS-/UNKNOWN-Prüfantworten erlauben weiterhin unabhängige Anforderungen.

Die genannten Prüfzahlen stammen aus den getrennten Prüfständen. Die drei
unabhängig geprüften Änderungen sind hier gemeinsam integriert. Die gezielte
gemeinsame Regression bestand alle 121 Tests in 15,95 Sekunden, ohne
übersprungene oder fehlgeschlagene Tests. Der normale Produktwriter erzeugt
weiterhin genau dieselben 42 Bundle-Dateien einschließlich 37 Python-Dateien.
Es gibt keine Änderung der numerischen Runtime,
wissenschaftlichen Qualitätsentscheidungen, Dateiauswahl oder Zeitbudgets.
Collection-Erfolg bescheinigt erfolgreiche Kopierversuche und keine
wissenschaftliche N5000-Vollständigkeit.

Auslöser waren tatsächliche lokale Startfehler der H10-/DeepX-Preflightaufrufe
mit Controller 417ac3a. Deren gebundene Argumentgrößen reproduzierten lokal
`E2BIG`; der ursprüngliche Fehlercode wurde nicht persistiert und wird nicht
als unmittelbar gespeicherte Beobachtung ausgegeben. Dieser gemeinsame
Blocker ersetzte die frühere Planung, zunächst alle 177 Qualitätsfälle mit
417ac3a auszuführen. Die ursprünglichen Läufe und ihre Quellenbindungen
bleiben unverändert.

Die Übernahme der Controllerkorrekturen und der neue Quellen-Freeze setzen
die vollständige normale Finalisierung der drei begonnenen Abläufe voraus,
einschließlich zentraler Auswertung, Bereinigung und Freigabe ihrer eigenen
Prozesse und Leases. Ein noch laufender H8-Pilot wird dafür nicht unterbrochen;
größere Folgegruppen bleiben bis zu dieser Freigabe angehalten. Die neue
Controllerfassung ist lokal softwaregeprüft. Ihre tatsächliche normale
Hardwareprüfung steht noch aus; die bisherigen Ergebnisse sind kein
Hardware-PASS dieser Fassung.

Die 29 betroffenen H10-/DeepX-Fälle ohne Capture benötigen später ausdrücklich
neue Versuche mit dem neuen Controller-Freeze. Alte Startmarker, Budgets,
Fehlversuche und Freezes werden nicht verändert. Vorhandene Runtime- und
Konfigurationsbytes bleiben gebunden; neue Profil-, Action- und
Controllerindexversionen müssen ihre tatsächliche Herkunft belegen.
Historische abgeschlossene Qualitätscaptures werden nicht wiederholt.
Die beschriebenen Softwareprüfungen verwendeten keine Verbindungen zu
Beschleunigersystemen, keine Hardwareinferenz und keine Builds.

## 21.10.2026 — Parallele physische Setup-Warteschlangen für Generic-only

Der ausdrücklich angeforderte Einstieg `--execute-batch` verteilt vorbereitete
Generic-only-Pläne mit `parallel_remote_setups=true`, `max_parallel_setups=3`
und `setup_queue_mode=per_setup` über den vorhandenen
`PhysicalSetupDispatchManager`. Jeder Plan bleibt an genau ein physisches
Setup und einen normalen Ausführungs-/Ergebnisordner gebunden. Die drei
Wiederholungen eines Cuts bleiben seriell. Die innere Generic-Pipeline,
H10 mit acht Slots, DeepX mit vier Slots und H8 mit synchroner Stage1 bleiben
unverändert. Qualitätsinferenz und Builds gehören nicht zu diesem Batch.

Physische Aliasse teilen dieselbe Warteschlange. Normale Ressourcenlocks,
Ownershipprüfungen, Leasejournale und Prozessbereinigung bleiben wirksam.
Ungeklärte Freigaben erzeugen vor Rückgabe des Workers einen dauerhaften
gerätespezifischen Sperrbeleg, den auch eine spätere explizite Wiederaufnahme
beachtet. Ein vollständig bereinigter einzelner Messfehler sperrt dagegen
keine unabhängigen Zellen. Gemeinsame Quellenfehler beenden weitere betroffene
Dispatches an den normalen Fall-/Wiederholungsgrenzen.

Der Batch bindet genau einen veröffentlichten Source-Freeze. Importierte
Controllerdateien müssen dem verifizierten Quellenmanifest angehören;
die tatsächlichen Bundlemitglieder müssen dem normalen Produktwriter
entsprechen. Jeder Plan behält seinen ursprünglichen Startmarker, seine
Execution-ID und sein gebundenes Wiederholungsbudget. Gemeinsamer Status
wird ausschließlich vom Controllerhauptthread atomar geschrieben,
Einzelergebnisse bleiben in getrennten Planordnern. Gruppentimestamps
beschreiben Controller-/Prozessfenster und keine NPU-Auslastung.

61 gezielte Softwareprüfungen des Batchs und der bestehenden Dispatchpfade
bestanden. Ein unabhängiger Test mit echten lokalen Kindprozessen belegte
Überlappung verschiedener physischer Kennungen, Serialisierung von Aliasen
und den ausbleibenden Start eines vorher abgebrochenen Queueeintrags.
Der begrenzte Test auf tatsächlichen unabhängigen Beschleunigersystemen
bleibt vor dem vollständigen Parallelbetrieb erforderlich.

Die abschließende gemeinsame Regression bestand 1.233 Tests in 356,87 Sekunden
bei unveränderten geprüften Produkt-/Testquellen. Vier optionale Tests eines
explizit anzugebenden unveränderlichen Native-Archivs wurden mit dieser
Begründung übersprungen. Die tatsächlich vorhandenen vier Containerbindungen
sind separat geprüft. Der vorherige Integrationslauf mit acht Fehlern bleibt
als Diagnose erhalten: Drei Runnerfixtures benötigten den normalen kopierten
Bibliothekskontext, eine Bridgefixture den tatsächlichen Child-Importkontext.
Außerdem fehlten einer schon im Ausgangsstand vorhandenen, nur lesenden
Budgetprojektion zwei Kontextfelder; deren Ergänzung bestand 49 gezielte
Regressionen. Es wurden keine Budgets oder historischen Energie-Freezes
verändert und keine Energieaufnahme ausgeführt.

Der abschließende Quellenreview bindet zusätzlich den expliziten
`product_freeze` jedes Kindplans an denselben vollständigen Batch-Freeze.
Ein identisches Runtimearchiv allein darf keine frühere Controller-Epoche
umetikettieren. Die erste noch unveröffentlichte Paketkopie bleibt als
Vorbereitungsstand erhalten; die endgültigen Quellenindizes werden nach
dieser eng begrenzten Ergänzung erneut mit dem normalen Builder erzeugt.
Die gezielte abschließende Wiederholung bestand 61 Tests in 9,22 Sekunden.

## 21.10.2026 — Quellengebundene Eingaben und erweiterte Qualitätszulassung

15 späte YOLOv7-Zellen verwenden neben den P1-Grenztensoren die ursprüngliche
Bildeingabe für eine echte Part2-Faltung. Die vorherige Umwandlung des
SDK-UInt8-Inputs nach Float32 stellte deren kanonische Normierung nicht her.
Die Vorbereitung bindet nun den normalen vollständigen Bild-Feed und dessen
tatsächliche Bytes je Eingang; der Loader materialisiert ihn vor dem Timer.
Die gemischte Kontrolle verwendet unveränderliche Zuordnungen je Aufgabe und
prüft den tatsächlichen Part2-Passthrough. Sechs dieser Zellen haben nur einen
P1-Grenztensor und zusätzlich das Bild als zweiten Part2-Eingang; die Zulassung
bildet diese tatsächliche Struktur ab. Die neun zentralen Inferenz- und
Schedulingdefinitionen sind weiterhin bytegleich zum veröffentlichten Stand
`4a0be2c3464a873c348783cf20010a526f911a7e`.

Die unabhängige Softwareprüfung umfasst den normalen Qualitätswriter,
strikten Reader und zentralen Dienst mit ausdrücklich synthetischen N2/B7-Daten
sowie manipulierte Bild-, Skalierungs-, Quellen- und Bytebindungen. Sie
bestand 91 fokussierte und 21 zusätzliche Prüfungen. Die bestehende
Qualitäts-/Funktionsprüfung auf der Hardware bleibt erforderlich.

H8 verwendet für Qualitäts- und FPS-Eingaben den vorhandenen Bound-Tensorloader
auch bei echten BIN-Dateien. Die Beziehung zwischen den ursprünglichen
UInt8-Eingaben und dem tatsächlichen SDK-Float32-Adapter wird ausdrücklich
geprüft. Bestehende transformierte TensorRT-Engines werden an den tatsächlichen
ursprünglichen Generic-Request mit dessen versiegelter Bridge gebunden.
Der normale Validator prüft alle acht ursprünglichen Dateien vor und nach
Engineinitialisierung; ein anderer Native-Vergleichsbeleg ersetzt diesen
Request nicht. Die 64 zusätzlichen Detektorrezepte wurden einzeln geprüft:
40 direkte FP16-Engines, 16 UInt8-Dequant-Bridges und acht DeepX-Float32-Bridges
ohne zusätzliche Layouttransformation. 57 fokussierte und acht unabhängige
Vorprüfungsregressionen bestanden.

Die reine Qualitätsaufnahme protokolliert die monotone Dauer ihrer vollständigen
Bildschleife und deren Bildanzahl. Diese Beobachtung umfasst Bildvorbereitung,
serielle Inferenz, Completion und Belegerfassung je Bild; sie schließt
Engineinitialisierung, nachfolgenden Export und zentrale CPU-Auswertung samt
Bootstrap aus. Sie wird getrennt von Generic-FPS geführt. Der unabhängige
H8-Review bestand 19 zusätzliche und 26 fokussierte Softwareprüfungen,
einschließlich des tatsächlich generierten Programmeinstiegs und des normalen
Qualitäts-/Collectorwegs. Die verwendeten synthetischen Prüfdatensätze sind
keine N5000-Hardwarebelege.

## 21.10.2026 — Cache-Abgleich, Wiederverwendung und reale Vorprüfungsfehler

Der unabhängige 420er-Cache-Audit ist mit späteren vorhandenen Quellen- und
Receiptbelegen abgeglichen. Alle 36 offenen ResNet-Artefaktidentitäten sind
geschlossen. Für H8 und H10/b119 waren die zum richtigen P1 gehörenden,
bereits inventarisierten Enginealternativen erforderlich. 359 Artefaktpaare
sind damit bestätigt; 61 bekannte technische Negativfälle bleiben erhalten.
Der ursprüngliche Audit wird nicht umgeschrieben, und dieser Abgleich führt
keine zusätzlichen Builds oder Hardwareproben aus.

112 historische H10-/DeepX-Klassifikatorfälle wurden einschließlich ihrer
336 Wiederholungen mit dem normalen Parser und ihren ursprünglichen
Quellen-, Eingabe-, Artefakt- und Completionbindungen geprüft. Zusammen mit
15 zuvor zugelassenen Detektorfällen und 64 ebenfalls vollständig geprüften
H8-Fällen sind 191 vorhandene FPS-Datensätze mit 573 Wiederholungen
wiederverwendbar. H8 behält seine tatsächliche archivierte 2.92.0-Quellmischung
und die synchrone Stage1 mit zwei Pipelineworkern; daraus wird kein sauberer
Commitstand oder neuer Async-Lauf abgeleitet. 168 unterstützte Zellen benötigen
noch neue FPS; die 61 technischen Negativfälle bleiben getrennt erhalten.
Diese Wiederverwendung ist keine neue Messung des erweiterten Runners.

Alle 180 Klassifikator-Qualitätsbelege erfüllen den unveränderten Vertrag.
Die Zulassung prüft tatsächliche Top1-/Top5-Trefferzahlen und die normale
Qualitätsentscheidung anhand der vorhandenen Konfidenzintervalle erneut;
Datasetinferenz, Bootstrap und Qualitätsgrenzen werden nicht neu ausgeführt
oder verändert. 160 Fälle sind qualitätszulässig, 20 behalten ihren FAIL.
Für alle 179 unterstützten Detektorfälle ist dagegen eine Qualitätsaufnahme
mit dem vollständigen aktuellen Completionpfad nötig. Die 23 darunter bereits
FPS-gemessenen Fälle behalten ihre Durchsatzwerte, sind in der neuen
qualitätsgefilterten Ansicht aber bis dahin ausstehend. Historische A/B-Belege
und deren damalige Qualitätszulassungen werden nicht geändert.

Die erste gezielte Native-Funktionskontrolle auf DeepX scheiterte am fehlenden
vollständigen Output-Contract-Container im privaten Staging. Die Korrektur
bindet den tatsächlichen historischen Container einschließlich Dateihash,
überträgt ihn vor dem normalen Dispatch und prüft ihn vor und nach Ausführung.
Gleiche ausgewählte Vertragszeilen in einem anderen Container genügen nicht.
77 fokussierte und 23 unabhängige Softwareprüfungen bestanden. Der fehlgeschlagene
Hardwareversuch und sein nachgewiesener Prozessabschluss bleiben erhalten;
ein neuer funktionaler Hardware-PASS ist noch nicht erreicht.

Der erste normale H10/YOLO26s-b039-Qualitätslauf endete nach 348,758 Sekunden
vor der Inferenz. Der Generator hatte den vollständigen 20er-Katalog erzeugt,
während das ausdrücklich selektive Profil nur b039 ausführbar bindet.
Die Scopeprüfung verwendet für diese Ausnahme nun dieselbe vollständige
Profil-/Plan-Autorisierung wie der Qualitätsworkflow. Fehlende Autorisierung,
abweichende Quellen und nicht akzeptierte angeforderte Fälle bleiben sperrend;
andere angeforderte Messpfade behalten ihren vollständigen Umfang.
38 fokussierte Scope-/Accountingprüfungen sowie 30 unabhängige Prüfungen bestanden.
Der weitere Fehler der TensorRT-Vorprüfung beruhte auf der irrtümlichen Wahl
des Native-Präzisionsrezepts für die explizit gebundene Generic-Konfiguration.
Die Vorprüfung bindet nun genau deren Engine, Build-Receipt und Build-ONNX
einschließlich Pfad, Größe und Hash. Die Runtimekennung `float32_layout_fp16`
bleibt von der tatsächlichen FP16-Compilerrezeptur getrennt. Bestehende
Native-Bridge-, Owner-, ABI-, Befehls-, Form- und Byteprüfungen bleiben wirksam.
35 fokussierte Prüfungen und deren unabhängige Wiederholung bestanden;
beide tatsächlichen H10-/DeepX-b039-Konfigurationen bestanden die lokale
Rezeptprojektion. Eine Remote- oder Hardwareabnahme ist damit nicht belegt.
Es gab keine neue Kompilierung, keine Qualitätsbilder und keine FPS-Ergebnisse.
Die tatsächliche Capturegeschwindigkeit und eine darauf beruhende Restlaufzeit
sind daher noch nicht bestimmt. Alle Belege liegen im externen Auftragsordner
`/home/kmika/Reports/generic_detectors20_multitensor_20261021T163124Z/`.

## 21.10.2026 — Begrenzte Wiederaufnahme unabhängiger Generic-Zellen

Der ausdrücklich gebundene Planmodus `independent_cells_v1` erlaubt höchstens
zwei Controllerstarts je fehlender Wiederholung. Das Budget wird vor dem
Dispatch dauerhaft reserviert. Gültige Wiederholungen werden unverändert
erhalten. Ein explizites `--execute PLAN --resume` prüft ursprünglichen Plan,
Execution-ID, Konfiguration und Bundle; der exklusive ursprüngliche Startmarker
bleibt bestehen. Ohne diesen Planmodus gilt weiterhin die bisherige einmalige
Ausführung. Jeder weitere Dispatch benötigt seine vorher dauerhaft gebuchte
Versuchsnummer und startet über den normalen Produktpfad einen frischen
Prozess; unbeschränkte Wiederholungen sind ausgeschlossen.

Vorprüfung, physischer Lock, Prozessregistry, Leasejournal und Quarantäne sind
im neuen Modus je Backend gebunden. Ein Fehler sperrt den betroffenen Zweig;
unabhängige saubere Zweige laufen weiter. Vor jedem weiteren Versuch sind die
tatsächlichen Besitz- und Freigabeprüfungen erforderlich. Parsergültige
Rohdaten eines erst danach fehlgeschlagenen Prozesses werden erhalten, aber
ohne erfolgreichen technischen Abschluss nicht übernommen. Eine erstmalige
Ausführung darf keine zufällig vorhandenen Ergebnisdateien übernehmen.

121 Controllerregressionen bestanden; nach der abschließenden Versionierung
der Fehlerbelege bestanden 29 fokussierte Tests. Der unabhängige Review bestand
29 Tests und prüfte zusätzlich den normalen Generic-only-Wrapper mit drei
frischen CPU-Testprozessen: Ein kontrollierter fehlender Repeat wurde begrenzt
ergänzt, der erste gültige Repeat blieb unverändert, und ein vollständiger
Reentry startete keinen Prozess. Simulierte SDKs und Testtransporte belegen
keine Acceleratorabnahme. Quellen und Rohbelege stehen im externen Auftrag.

## 21.10.2026 — Laufende Erweiterung auf 420 Generic-Zielkonfigurationen

Die Ergänzung `CODEX_Generic_420_Live_Ergaenzung_20261021.txt` hebt den bisherigen H8-Ausschluss auf und erweitert den laufenden Auftrag auf sieben Modelle mit jeweils 20 historischen stratified_windows-Cuts auf drei Backends. Der Auftrag und seine bisherigen H10-/DeepX-Belege werden fortgeführt. MobileNet erhält die ursprünglichen zusätzlichen Cuts b029, b043 und b119, ResNet b028; die bisherige Auswahl wird nicht anhand neuer FPS verändert. Neue 420er-Inventare und Statusstände werden getrennt von den bereits eingefrorenen 160er-Belegen geführt.

Die installierte H8-Stage1 arbeitet synchron, während die bestehende Pipeline zwei Worker und eine begrenzte FIFO mit Tiefe zwei verwendet. Stage1 kann dadurch mit TensorRT und Taskabschluss des vorigen Bildes überlappen. Die geprüften Inferenz- und Schedulingfunktionen sind identisch zum zuvor tatsächlich gemessenen H8-Freeze. Eine H8-Async-Umstellung ist nicht vorgesehen; zusätzliche Zulassungs- und Mehrtensorprüfungen für H8 und Klassifikatoren sind noch ausstehende Arbeiten. H10 und DeepX behalten acht beziehungsweise vier Async-Slots.

Einzelne technisch nicht ausführbare Zellen bleiben mit Ursache und Beleg erhalten. Qualitäts-FAILs werden bei funktionaler Korrektheit auf FPS gemessen. Vorhandene gültige Wiederholungen werden erhalten; der normale Fallrunner kann fehlende Wiederholungen gezielt ergänzen. Ein ausdrücklich begrenzter kampagnenweiter Fortsetzungsweg mit unveränderten Quellenbindungen und erhaltenen Startmarkern ist noch zu integrieren. Die vier offenen gepaarten Fälle sperren keine unabhängigen Generic-only-Fälle. Es entstehen keine neuen wissenschaftlichen Native-, Full- oder Energiewerte.

## 21.10.2026 — Generic-only-Detektoren: feste Auswahl und getrennte Zulassung

Der Folgeauftrag `CODEX_Generic_Detektoren_20_MultiTensor_20261021.txt` erweitert den Generic-Vergleich um die ursprünglichen 20 stratified_windows-Cuts für YOLO11l, YOLOv7, YOLO26m und YOLO26s auf H10 und DeepX. Die historischen vier Auswahlpläne wurden anhand ihrer tatsächlichen Modellhashes und des installierten Selectors reproduziert und vor Hardwarearbeit eingefroren: 160 Backendfälle, davon 144 mit mehreren Grenztensoren. Die Auswahl bevorzugt innerhalb gleichbreiter Graphfenster kleine Grenzdatenmengen; sie ist keine Zufallsstichprobe. Backfill nach Backendfehlern bleibt entsprechend dem ursprünglichen Profil aus.

Die Cacheprüfung weist 131 vorhandene, durch Receipts gebundene Part1-Artefakte mit passenden vorhandenen TensorRT-Engines nach. Die weiteren 29 Fälle bleiben sichtbar: 17 belegte deterministische Compilefehler und zwölf reguläre statische Hailo-YOLO26-Ausschlüsse. Es wurde kein tatsächlich fehlendes erforderliches Buildartefakt gefunden. Diese negativen Befunde lösen keinen erneuten Compileversuch aus. Die 132 erfolgreichen Messungen des vorangegangenen Auftrags werden erhalten. Das Originalpaket und eine nur in veralteter Übergabemetadaten korrigierte neue Paketversion sind auf dem tatsächlich gemounteten Sprite-Ziel mit vollständigem Größen-/SHA-256-Vergleich versioniert gesichert.

Die neue ausdrücklich getrennte Generic-only-Zulassung bindet vorhandene kompilierte Teile, Eingaben, Quellen und CPU-ONNX-Qualitätsbelege. Ihre Funktionsreferenz entsteht seriell aus denselben Runtimefunktionen außerhalb des Durchsatztimers. Sie liefert keine Native-Leistung. Der normale gepaarte Pfad behält seine bisherigen Anforderungen; ausschließlich die vier beauftragten offenen Single-Input-Fälle erhalten einen gesonderten, streng gebundenen Einstieg für eine frische funktionale Native-Referenz mit einer Aufgabe. Wissenschaftliche Native-FPS bleiben unverändert.

Die erweiterte gemischte Kontrolle beobachtet die tatsächlich übergebenen Grenztensoren einschließlich Namen, Form, Datentyp, Layout, Mapping und Pufferzustand. Sie verlangt mindestens zwei Grenztensoren und 32 Aufgaben aus zwei unterscheidbaren Eingaben mit Wiederverwendung aller acht H10- beziehungsweise vier DeepX-Slots. Die neun untersuchten Inferenz-/Schedulingfunktionen sind gegenüber Commit `599a1fe924d74f9eb96073ea283a34c4feeb1795` bytegleich. Die Zulassung prüft zusätzlich drei verschiedene Prozesse und die zugehörige Execution-ID. Serielle Referenzen, tatsächliche Grenztensorbeobachtung und Completionbelege entstehen außerhalb des Durchsatztimers. Prüfungen am tatsächlich erzeugten Runner nutzen reale Produktadapter, Completion und Messschleifen mit simulierten SDKs; das ist kein Hardware-PASS.

Eine gesonderte Quellenprüfung fand unterschiedliche vollständige Qualitätsendpunkte: Historisches YOLO26-Composed-Quality führt eine zweite NMS aus, der gemessene Completionpfad ausdrücklich nicht. Für YOLO11l und YOLOv7 ist die Gleichwertigkeit der gesamten historischen Geometrie-/Nachverarbeitung ebenfalls nicht belegt. Die 115 vorhandenen Multi-Tensor-Qualitätsresultate werden deshalb nicht als gleichwertige Belege umdeklariert. Der neue explizite Generic-Quality-Einstieg verwendet pro Bild die normale Completionfactory, erhält deren numerische Detektionen unverändert und ergänzt nur die gebundenen Klassennamen für den normalen Qualitätsloader. Er bindet tatsächliche Vorverarbeitung, Bildgeometrie, vorbereitete Eingaben, Artefakte und Runnerquellen. Full-Modell- und Part2-Hashes bleiben getrennte, tatsächlich geprüfte Rollen; die ursprüngliche Endpunktquelle bleibt erhalten.

Der normale Profil-/Suitepfad transportiert die expliziten Konfigurationen und vorhandenen Artefakte. `quality_gate.management_reference.existing_reference` ermöglicht ausschließlich für diesen Generic-Quality-Umfang die geprüfte Wiederverwendung bestehender CPU-Referenzbytes über die normale unveränderliche Veröffentlichung im neuen Run. Es wird kein CPU-Inferenzauftrag eingereiht. Dataset, CPU-Referenz, Qualitätsgrenze und N5000/B1000-Vertrag bleiben unverändert. Die reale Offlinevorbereitung von H10 und DeepX/YOLO26s b039 bestand einschließlich unverändertem Datasetmanifest, 5.000 vorhandenen CPU-Referenzrecords, normalem Runtimewriter, Suitefinalisierung und Bundle. Unabhängige Tests durchlaufen den normalen Export, Loader, zentralen Qualitätsdienst und Collector mit ausdrücklich synthetischen N2/B7-Daten; sie ersetzen die erforderliche N5000-Hardwareprüfung nicht.

Die reine Generic-Qualitätserfassung erhält eine eigene Ergebnisführung im normalen Workflow. Nur ausdrücklich durch das Profil gebundene Fälle entfallen aus der Leistungsmatrix. Ihr Erfassungsabschluss erfordert die tatsächlichen Request-, Kandidaten-, Completion- und Child-Report-Dateien mit unveränderter Konfiguration und Quellenbindung; widersprüchliche FPS-/Timingangaben werden zurückgewiesen. Die zentrale Qualitätsentscheidung bleibt ein gesonderter erforderlicher Schritt. Es entstehen weder künstliche Leistungszeilen noch Native-/Full-Identitäten. Bestehende Full-only- und gewöhnliche Benchmarkprüfungen bleiben erhalten.

Der zentrale Abschluss bindet die tatsächlich ausgewertete CPU-Referenz, Datasetmenge, aktuelle normale Qualitypolicy und vollständige Completionbelege. Er erhält die getrennte Generic-Qualitätserfassung auch bei Wiederaufnahme, Ergebnisübernahme und Validierungsrefresh. Eine abgeschlossene Qualitätsprüfung ohne Leistungsmessung erzeugt keine fehlenden FPS-Zeilen; ein fehlgeschlagener Hardware-Smoke bleibt sperrend. Der vollständige normale Suite-/Collector- und Quality-/Abschlussweg ist mit synthetischen N2/B7-Daten geprüft. 798 kombinierte Regressionen bestanden; nach den anschließenden Collector- und Abschlusskorrekturen bestanden 251 beziehungsweise 235 gezielte Regressionen. Nach der finalen Normalpfadkorrektur und dem Laufzeitnachweis bestanden weitere 248 gezielte Tests; nach Ergänzung der tatsächlichen Grenzarray-Spezifikationen bestanden 121 fokussierte Tests. Diese Testmengen überschneiden sich. Hinzu kommen 18 Release-/Packagingtests und getrennte unabhängige Prüfbelege im externen Auftrag. Kein Softwaretest ersetzt die ausstehende reale Hardwareprüfung.

Ein Fehler der neuen Auswahlinventarisierung betraf ausschließlich die Präzisionsbeschriftung von 43 H10-Multi-Tensor-Zeilen: Deren tatsächliche bestehende Produzenten und Engine-Receipts binden `float32_layout_fp16`. Eine getrennte, belegte Ausführungsprojektion korrigiert diese Labels; die eingefrorene Auswahl und CSV bleiben unverändert. Es entstehen weder neue UInt8-Bridges noch geänderte Enginepräzision oder Buildartefakte.

Die ausdrückliche Generic-Qualitätsaufnahme protokolliert außerdem die tatsächlichen P1-/P2-IO-Beschreibungen, die nach dem Transfer aufgelöste Tensorzuordnung und die verwendete Python-Datei einschließlich Hash und ausgewählter Laufzeitvariablen. Die Zuordnung wird nach jedem Qualitätsbild auf Stabilität geprüft und mit dem normalen Qualitätsnachweis versiegelt. Das geschieht außerhalb der FPS-Messung ohne zusätzliche Inferenz; diese Metadaten ersetzen nicht die spätere gemischte Funktionskontrolle.

Hardwarekontrollen und die neue Durchsatzserie stehen zum Zeitpunkt dieses Publikationseintrags noch aus. Der externe Abschlussbericht dokumentiert den veröffentlichten Commit, die endgültigen Softwareprüfungen, tatsächliche Hardwareergebnisse und verbleibende Fälle. Neue Quellen-, Plan-, Test- und Laufbelege liegen ausschließlich unter `/home/kmika/Reports/generic_detectors20_multitensor_20261021T163124Z/`; historische Messungen, Source-Freezes, Results-Repo und Paper werden nicht überschrieben.

## 21.10.2026 — Veröffentlichung des geprüften Controllerstands und neuer Generic-Umfang

Der Folgeauftrag `CODEX_Freeze_Main_Generic_H10_DeepX_20261021.txt` autorisiert ausdrücklich die Sicherung des vollständigen geprüften Controllerstands auf `origin/main` sowie eine neue Generic-Durchsatzserie mit 136 bestehenden Cuts auf H10 und DeepX. H8, Native-/Full-Neumessungen, Energie, Dataset-Evaluation und Builds sind ausgeschlossen. Der vorherige 168er-Auftrag bleibt mit seinem dokumentierten Teilstand und BLOCKED-Pilotgate erhalten.

Vor Git-Integration wurde der vollständige nicht ignorierte Quellbestand außerhalb des Produktbaums gesichert. Die 132 geänderten beziehungsweise neuen Dateien sind dem vorhandenen manifestierten 2.95.5-Stand und dem tatsächlichen 15-Dateien-Async-Handoff zugeordnet. Die sechs veränderten und neun neuen Handoffdateien werden über den bestehenden Manifestgenerator in neue aktuelle Quellenindizes aufgenommen; alte Messfreezes werden nicht verändert. Version und Build-ID bleiben 2.95.5/v2.95.5. Alle 41 erzeugten Runtime-Dateien sind vor Veröffentlichung bytegleich zum geprüften Pilotbundle; ein neuer endgültiger Freeze erhält zusätzlich die veröffentlichte Commitbindung. Lokale Profile, Umgebungen, Artefakte und Messarchive gehören nicht zum Commit.

Im getrennten Releasekandidaten bestanden 18 normale Release-/Packagingtests sowie 807 Regressionen für die mitveröffentlichten 2.95.5-Änderungen. 29 optionale Prüfungen erfordern ausdrücklich angegebene abgeschlossene Klassifikator-/BN6-Kohorten und wurden mit dieser Skipbegründung nicht ausgeführt. Die gezielten Generic-Tests und Regressionen werden zusätzlich im sauberen Checkout des neuen Commits geprüft. Softwareergebnisse sind keine neue Hardwareabnahme.

Der neue Umfang besteht aus MobileNetV3-L, RegNet-X, ResNet-50, YOLO11l, YOLOv7, YOLO26m und YOLO26s mit genau 68 Cuts je Backend; 128 bisher qualitätszulässige und acht H10-MobileNet-Qualitäts-FAILs bleiben erhalten. H10 bindet fest acht Async-Slots, DeepX vier; Standard bleibt synchron. Neue selektive Teilpläne erhalten eigene Ausgabepfade, normale Quellen-/Endpunktprüfungen, Locks und Prozess-Leases. Ein belegter lokaler Referenzfehler sperrt nur abhängige Fälle; ein gemeinsamer Runtime-/Korrektheits-/Ressourcenfehler sperrt alle betroffenen Fälle. Es gibt keine Prüfungsumgehung oder nachträgliche Umdeklaration des alten Pilotgates.

Neue Quellen-, Test-, Plan- und Ergebnisbelege liegen ausschließlich unter `/home/kmika/Reports/generic_main_freeze_h10_deepx_20261021T123251Z/`. Zum Zeitpunkt dieses Publikationseintrags hat die neue Messserie noch nicht begonnen. Der Abschlussbericht dort dokumentiert veröffentlichte Commits, tatsächliche Hardwareergebnisse und offene Referenzfälle.

## 21.10.2026 — Generic-Async: selektive Integration in Controller 2.95.5

Der gelieferte Code- und Testpatch ist nach separatem `git apply --check --exclude=docs/ARBEITSSTAND.md` übernommen. Der vollständige Patchcheck scheiterte ausschließlich am erweiterten Dokumentanfang; der folgende gelieferte Eintrag wird davor eingefügt. Die neuere Hailo-Interface-Diagnose im Template und sämtliche übrige lokale Arbeit bleiben erhalten. Keine Versionsänderung, kein Commit und kein Push. Die unten genannten 129 Tests beschreiben die mitgelieferte Vorprüfung. Auf dem Controller bestanden anschließend alle 44 fokussierten und 86 bestehenden Completion-/Dispatch-/Energievertragsregressionen ohne Deselektion. Externer Auftragsbestand: `/home/kmika/Reports/generic_async_remeasurement_20261021T101830Z`.

Die vorhandene Acceptance-Vorbereitung bindet synchrone Konfigurationen und prüft keine gemischte Aufgabenfolge. Für den ausdrücklich beauftragten Funktionsnachweis ergänzen `generic_mixed_control.py` und `generic_mixed_control_binding.py` daher eine optionale, quellengebundene Kontrolle im normalen Generic-Prozess vor dessen Durchsatztimer. Sie verwendet dieselben initialisierten kompilierten P1/P2-Funktionen, denselben Async-Adapter, dieselbe begrenzte FIFO und die normalen Completion-Prozessoren. 32 serielle Referenzen werden mit 32 Pipelineaufgaben aus zwei bereits vorhandenen Eingaben verglichen; eigene Diagnose-FPS entstehen nicht. Die zweite Hälfte der festen Eingabefolge kehrt die erste um, damit jeder der vier bzw. acht Slots beide Eingaben verarbeitet und veraltete Slot-Ausgaben erkennbar bleiben. Die Prüfung verlangt exakte Logits/Top1/Top5 bzw. vollständige Detektionsresultate, eindeutige Reihenfolge, vollständigen Drain und unveränderte Quellen. Die Produktparser prüfen den gespeicherten Kontrollnachweis mit.

Die Detektorkontrolle bindet die tatsächliche Bildgeometrie je vorhandener Eingabe über den bestehenden Completionvertrag-Builder. Der Durchsatzlauf behält seine ursprüngliche Eingabe und seinen ursprünglichen Vertrag. Die kanonischen Klassifikator-Scopepläne umfassen jeweils exakt 56 bestehende Identitäten; die 40 Pilotvarianten sind separat vorab festgelegt. Vorbereitung und Softwaretests sind keine Hardwarefreigabe. Hardwareprüfung und anschließende Vollkohorte stehen zum Zeitpunkt dieses Eintrags noch aus. Historische Source-Freezes und Ergebnisse bleiben unverändert; neue Energieaufnahmen sind nicht Bestandteil dieses Auftrags.

Abschluss des Hardware-Teilstands am 21.10.2026: 247 integrierte Controller-Softwaretests bestanden. Das eingefrorene Runtimebundle hat SHA-256 `5a97142b78c79575be49e19982a030d277bc7101ba864e76116704c24950d6b9`. Alle zehn H10-A/B-Paare und die sechs DeepX-Klassifikatorpaare bestanden die gemischte 32-Aufgaben-Kontrolle und je drei frische 100/1000-Prozesse pro Variante. Tatsächlich beobachtete Hostspitzen waren H10=8 und DeepX=4; direkte SDK-Wartezeiten und NPU-Auslastung wurden nicht gemessen. DeepX RegNet b001 war im Median 1,54 Prozent langsamer, ResNet b002 praktisch unverändert; langsamere Ergebnisse bleiben erhalten. H8 MobileNet b001 und YOLO26m b038 bestanden separat je drei synchrone Prozesse einschließlich normalem Ressourcenabschluss.

Die zusätzliche DeepX-YOLOv7-b044-Syncvariante besitzt drei normale gültige Messungen und bestandene Funktionskontrollen, wurde danach aber von der bestehenden Native/Generic-Startpaarprüfung angehalten. Die gebundene historische Native-Completionimplementation und die aktuelle Generic-Konfiguration besitzen unterschiedliche vollständige Verträge. Der Verifiergrund `yolov7_sigmoid_arithmetic_implementation_mismatch` ist kein gemessener Nachweis einer numerischen Abweichung. Sechs vorhandene Oct8-Native-Alternativen haben passende Artefakt-/Tensorbytes und denselben aktuellen Vertrag, bestehen jedoch die unveränderten Originalrun-, Prepared-Input- und signierten Manifestbindungen nicht. Kein Flag, historischer Beleg, Source-Freeze oder Startmarker wurde zur Umgehung verändert.

Gesichert sind 99 gültige Pilotprozesse (32 zugelassene Varianten plus die drei Messungen der angehaltenen 33. Variante) und sechs H8-Kontrollprozesse. Sieben Pilotvarianten und sämtliche 168 Klassifikator-Neuaufnahmen bleiben ungestartet; die 151/17 Qualitätszulassungen bleiben unverändert. Die Vollkohorte wurde wegen dieses konkreten Vertragsblockers nicht gestartet, und die neue Native-Auswahlgüte ist nicht bestimmt. Empfehlung bis zur regulären Auflösung und vollständigen Neuaufnahme: Opt-in beibehalten, keine Standardumschaltung. Teilresultate, Quellvergleich, tatsächlicher Diff, Cleanup und Blockerbelege liegen im oben genannten externen Auftragsbestand. Keine Energie-/Native-Neuaufnahme, kein Build, Commit oder Push.

## Optionale asynchrone H10- und DeepX-Producer im Generic-Runner (21.10.2026)

Die Codeprüfung fand keinen besonderen parallelen Generic-Pfad für YOLO gegenüber
RegNet, ResNet oder MobileNet. Alle verwenden denselben Producer, denselben
TensorRT-Consumer und die bestehende begrenzte FIFO. Der bisherige H10-Producer
wartet jedoch unmittelbar nach jeder asynchronen Hailo-Einreichung auf deren
Abschluss. Der Native-H10-Producer kann mehrere Accelerator-Aufträge gleichzeitig
offen halten. Dasselbe gilt für DeepX: Generic verwendet `engine.run`, während
der archivierte Native-A4-Stand von 2.95.5 `run_async` und `wait` mit vier Slots
verwendet. Für H8 wurde kein entsprechender Unterschied gefunden. Sein Generic-
VStreams-Pfad blockiert pro Aufruf, und der Native-Pfad schreibt und liest jeweils
ein Bild. Ob die Unterschiede auf H10 und DeepX die beobachteten Rankings erklären,
bleibt durch neue Hardwaremessungen zu prüfen.

Der normale Generic-Completed-Task-Pfad bietet deshalb jetzt die ausdrücklich
gewählte Policy `hailo_async_slots_v1`. Die Vorbereitung mit
`--hailo-async-inflight 8` bindet für H10 die Konfiguration
`stage1_execution={"policy":"hailo_async_slots_v1","inflight":8}`.
Für DeepX setzt `--deepx-async-inflight 4` die zusätzliche Policy
`stage1_execution={"policy":"deepx_async_slots_v1","inflight":4}`. Beide
Vorbereitungsoptionen können gemeinsam angegeben werden und gelten unabhängig
vom Modelltyp. H8 bleibt unverändert. Ohne Option bleiben H10 und DeepX synchron.
Die bestehende Queue-Tiefe,
Übergabe, TensorRT-Ausführung und abschließende Top1/Top5- bzw. Detektorauswertung
werden nicht verändert. Ein Slot ist als explizite Kontrollbedingung zulässig.

`runners/generic_async.py` besitzt einen begrenzten Pool separater Hailo-Bindings
bzw. DeepX-Eingabe- und Ausgabepuffer.
Der Producer sammelt Ergebnisse in Einreichungsreihenfolge und kopiert fertige
Outputs, bevor er ein Binding erneut verwendet. Der Warmup wird vollständig
abgeschlossen, bevor das gemessene Intervall beginnt. Einreichung, Pipelinefüllung
und abschließender Drain liegen innerhalb dieses Intervalls. Daueraufnahmen behalten
ihren bisherigen Vertrag ohne Warmup und zählen alle zugelassenen, tatsächlich
abgeschlossenen Aufgaben. Der Report nennt außerdem die höchste Zahl eingereichter,
noch nicht eingesammelter Aufträge im Messfenster. Dies beschreibt die offenen
Host-Aufträge, nicht die gleichzeitige Ausführung im Accelerator.

Die DeepX-Implementierung benutzt die im archivierten A4-Quellstand belegte SDK-API
`run_async(inputs, user_arg=..., output_buffer=...)` und `wait(job_id)`.
Jeder Slot besitzt einen zusammenhängenden Ausgabepuffer. Die Offsets mehrerer
Ausgaben folgen den kumulierten Bytegrößen in SDK-Reihenfolge. Exakte SDK-Metadaten,
Pufferadressen, Form und Datentyp werden geprüft. Eingaben werden pro Slot kopiert.
Nur eine ausdrücklich fehlende führende Batch-eins-Achse darf ergänzt werden.
Normalisierung, Layout und die bestehende kanonische Ausgabezuordnung ändern sich
nicht. Fehlende Async-API oder widersprüchliche Metadaten führen zum Fehler, nicht
zu einem synchronen Ersatzpfad.

Fehler oder Timeout stoppen die Einreichung. Hailo besitzt begrenzte SDK-Waits.
DeepX `wait` bietet in der belegten API weder Timeout- noch Abbruchparameter.
Unbekannte Abschlüsse behalten deshalb ihre Engine und Puffer bis zum Prozessende.
Ein blockierter DeepX-Aufruf kann erst über die bestehende Besitzbindung des
Messprozesses beendet werden. Ein unvollständiger Drain oder überschrittenes
Gesamtlimit liefert keinen gültigen Durchsatzwert.

Policy und Slotzahl stehen ausdrücklich in Ergebnisheader und Wiederholungen.
Sie sind außerdem Teil des Energie-Aufrufvertrags. Fehlende Metadaten bezeichnen
ausschließlich den bisherigen synchronen Pfad. Ergebnisse mit einer anderen
Policy, Slotzahl oder eingefrorenen Runtimequelle können nicht als Wiederholung
weiterverwendet werden. Die bestehenden Artefakt-, Eingabe-, Funktions- und
Messgrenzen bleiben erhalten. Neue Messungen benötigen einen neuen vorbereiteten
Ausgabestand. Historische Messwerte werden nicht überschrieben.

Im abschließenden lokalen Lauf bestanden 129 gezielte Tests für Runtime,
erzeugten Runner, Daueraufnahme, Metadaten, Dispatcher und Eingabeverträge.
Ein bekannter prozessnamespaceabhängiger Baseline-Test war ausgeklammert.
Dazu gehören Buffer-Wiederverwendung, geordnete Ausgabe,
Warmup-Drain, Pipelineende, Fehler und Timeout sowie erzeugte H10- und DeepX-Runner
mit jeweils 1100 vollständig ausgewerteten Stub-Aufgaben. Der DeepX-Test benutzt
die tatsächlichen Generic-Eingabe- und Ausgabeadapter mit einem SDK-Stub.
Der Lauf umfasst auch Energievertragsprüfungen und asynchrone Daueraufnahmen
beider Backends mit dem echten Template. Es wurde keine Accelerator-Hardware benutzt.
Damit sind weder ein Geschwindigkeitsgewinn noch eine bessere Vorhersage der
Native-Rangfolge belegt. Die kleine Hardwareabnahme und die anschließende getrennte
Neumessung stehen noch aus. Keine Änderung an Version oder Releasefreigabe.


## 21.10.2026 — Bereits abgeschlossene historische Qualityresultate für technische Provenienz finden

Die enge Supplementkorrektur erweitert ausschließlich den vorhandenen Provenienzschreiber record_technical_central_projection. Falls dessen bisherige aktuelle Quellen keinen exakten Treffer liefern, darf er einen bereits vorhandenen, indexgebundenen completed-Resultatsidecar derselben ursprünglichen Requestidentität aus einem früheren normalen Versuch finden. Zugelassen sind nur der exakte bekannte Modell-/quality_inputs-Pfad, eine echte andere 32-stellige Sessionidentität, passende Requestquelle und identische Request-SHA sowie vollständige Gleichheit des erwarteten Resultatobjekts. Größe und tatsächlicher Datei-SHA werden über den bestehenden strikten Reader verifiziert. Alte failed-Sidecars und die historische Summary werden nicht umgeschrieben.

Damit kann der normale Join die bereits im Versuch347 abgeschlossenen ursprünglichen 50 Qualityergebnisse regulär weiterverwenden, nachdem die drei technischen Beobachtungen übernommen wurden. Es werden weder Statistik, Bootstrap noch Inferenz oder Hardwaremessungen wiederholt. Die bestehenden Regeln für negative QC-Kontexte, Interfaceblocking, Originaltimingzeilen und alle weiteren Request-/Native-/Quellen-/Writer-/Mergeprüfungen bleiben unverändert. Ein normaler No-op erzeugt weiterhin keinen künstlichen zusätzlichen Provenienzeintrag.

Der tatsächliche externe Nachweis führte zwei unveränderte normale Joins mit jeweils 47 normalisierten Treffern und drei separaten Full-Companions aus, schrieb und validierte die drei technischen Provenienzen jeweils vollständig und setzte die normale Summary vor Veröffentlichung mit exakt 50 unveränderten Originalergebnissen und null fehlgeschlagenen Resultaten zusammen. Die tatsächliche Live-Summary bleibt dabei ehrlich 53; alle Terminaldateien blieben bytegleich. Dieser Softwarenachweis ist kein wissenschaftlicher Qualitäts-PASS.

Diese Adoption ändert ausschließlich Supplement und diese Dokumentation sowie die beiden regulär gebauten Quellmanifeste. Runner, Transport a669, Accuracygate, Suite-/Casebytes, Profil, Sollmatrix, Protokolle, Toleranzen und CPU-Referenz bleiben unverändert. Sie verlangt die neuen tatsächlichen Terminalpins von Attempt04/Session574463, normale releasedOwnership und den unveränderten FirstAnchor347. BCC-Pins werden weder ersetzt noch für die neue Session wiederverwendet. Der anschließende normale Attempt05 muss weiterhin drei Technical preserve, null Technical execute und 50 kanonische QC-Ergebnisse ohne pending ausweisen; fehlende Generic-/Native-/Completed-Arbeit bleibt offen. Es gibt keinen neuen Finalize-only- oder Local-only-Runner.
## 21.10.2026 — Gesammelte negative technische QC-Kontexte regulär erhalten

Die Supplementkorrektur validiert den No-Timing-Nachweis am kanonischen Writerfeld measured_variants. Sie verarbeitet die drei bereits regulär gesammelten BCC-N5000-Beobachtungen über den vorhandenen strikten Retained-Pfad, ohne neue Hardwareaufnahme. Der tatsächlich geschlossene BCC-Stand mit 53 zentralen Ergebnissen bleibt unverändert gebunden. Die vorhandene completed_quality-/pending_quality-Auswahl verwendet vollständig geprüfte technische Bindings, um exakt die 50 ursprünglichen kanonischen Ergebnisse auszuwählen; die drei zusätzlichen Statistik-Sidecars bleiben unveränderte Diagnostik. Ungebundene gleiche oder anders formatierte Spiegel bleiben unzulässig.

Bei identischen Timinginputs gilt der bisherige Projektionspfad unverändert. Bei einer abweichenden Probe ist ausschließlich ein konsistenter kanonischer FLOAT-NumericFAIL unter dem unveränderten ursprünglichen QC-Vertrag zulässig. Er verlangt denselben ersten Datasetframe, denselben nichtleeren Inputfile-SHA, dieselbe kanonische Preprocessingidentität und dasselbe Backend-Encoding. Der neue negative Vergleich wird ausdrücklich als separater QC-Kontext mit beiden Inputbindungen erhalten. Die komplette alte Timing-, Guard- und QC-Zeile bleibt unverändert, bestehendes Interfaceblocking bleibt true. Shape-, Mapping-, unklare oder positive Ergebnisse werden über diesen Weg nicht zugelassen; der neue QC-Kontext validiert die alte Timingprobe nicht.

Die ausschließlich normale Completion-Relocation zweier output_contracts.json-Deklarationspfade wird durch den vorhandenen Transport-Runbezug, indexgebundene unveränderte Containerbytes und genau einen ursprünglichen Vertragsrecord mit identischer SHA belegt. Alle übrigen Rezept-, Quell-, Request-, Native-, Engine-, Collector- und N5000-Loaderguards bleiben aktiv. Der vollständige echte Closed-Run-Defaultreader ist belegt: drei Technical preserve, null execute, exakt 50 ursprüngliche QC-Ergebnisse, pending=0; die gespeicherte Summary bleibt ehrlich 53. Es werden keine Reports oder Requests umgeschrieben und keine negativen Ergebnisse zu PASS geändert.

Diese Sourceadoption ändert ausschließlich Supplement und diese Dokumentation sowie die beiden normal gebauten Quellmanifeste. Transport a669, Runner, Suite-/Casebytes, Accuracygate, eingefrorenes Profil, Sollmatrix, Toleranzen und CPU-Referenz bleiben unverändert. Neue tatsächliche BCC-Terminalpins und normal freigegebener Owner bleiben Pflicht; Sourceanchor 347 und eigenes BCC-Originalbackup bleiben getrennt erhalten. Die nachfolgende normale CLI muss weiterhin drei Retained-Beobachtungen, null Technical-Ausführung und 50 kanonische QC-Ergebnisse ausweisen. Fehlende 27 Generic-Obligationen werden damit nicht zu fertigen Messungen erklärt; die Korrektur schafft keinen Local-only-Modus.
## 21.10.2026 — Vorhandene Full-Quality-Companions im selektiven Transport bewahren

Die selektive Setup-Projektion transportiert die vorhandene Full-Quality-Endpoint-ID nur dann, wenn die zugehörige Qualityarbeit tatsächlich fehlt. Bei bereits vollständig belegtem Companion bleibt die Endpoint-ID leer; der vorhandene Suite-Parser erhält damit wieder einen konsistenten vorhandenen Controlvertrag.

Die Änderung betrifft ausschließlich zwei Zeilen in execution_binding.py. Laufplan, 84 Generic- und 159 nominale Zellen, 20 Cuts, N5000/B1000, Timing-/Native-Protokolle, Suite-/Casebytes, Runner und Accuracygate bleiben unverändert. Bereits abgeschlossene Qualitybeobachtungen und Originalbelege bleiben erhalten. Neue technische Revalidierungsergebnisse dürfen ausschließlich über den bestehenden gebundenen Workflowpfad entstehen.

Die Sourceadoption setzt den echten normalen Abschluss der Session 2fa89442a11643e09cb387ebe65da58c, ihre eigene PASS-Terminalclosure und regulär freigegebenen Owner voraus. Der ursprüngliche Supplement-Sourceanchor 34774441feb54fc6ad66c890981469b4 wird nicht neu aufgebaut oder verschoben. Der Versuch besitzt daneben sein normales eigenes Originalbackup. Vorhandene Belege zur 50er Qualitykohorte, CPU-Referenz und den archivierten Sourceepochengruppen werden weiterverwendet; keine neue Statistik oder Messung wird durch die Sourceadoption ausgelöst.

## 21.10.2026 – Reports: bestehendes hartes Hailo-Interfaceveto erhalten

Die technische Reporting-Eligibility berücksichtigt das bestehende aktuelle Hailo→TRT-Interfaceblocking, bevor strukturelle Gültigkeit eine positive Eligibility erzeugen kann. Explizites Blocking oder unaufgelöste Interfaceevidenz bleibt ein technisches Veto. Ein ausdrücklich nicht blockierender aktueller Vertrag bleibt auch nach normaler frischer technischer Projektion maßgeblich; ein historisches runtime_contract_decision=fail allein wird nicht zu einem neuen Veto.

Der gezielte Fix ändert keine Accuracy-, reference_close-, Struktur-/Numerik- oder Reportingpolicy und keine Toleranz, kein Profil oder Protokoll. Die fünf tatsächlichen technischen Negativzeilen bleiben negative Ergebnisse. Synthetische positive Projektionsfixtures sind Softwarebelege und kein Hardware-PASS. Die gleichzeitig übernommene Transportkorrektur und der Erhalt aller abgeschlossenen Quality- und Messdaten sind im folgenden Eintrag beschrieben.

## 20.10.2026 – Supplement: ausschließlich remote fähige Completion-Rollen übertragen

Der bestehende Scheduler überträgt `policy_excluded` aus Generic und `local_statistics` aus Qualitycompanion nicht mehr an Remote-Worker. Beide Rollen bleiben vollständig im unveränderten Controllerplan und in der Sollmatrix. Die vorhandenen strikten Controller- und Suiteparser werden nicht erweitert; unbekannte Actions und falsche Rollen bleiben Fehler. Eine leere Remoteauswahl bleibt ausdrücklich `--selective-completion-rows []`, damit kein gewöhnlicher Scope als Fallback startet.

Der konkrete erste Supplementversuch 347 scheiterte an diesem Parservertrag vor Suiteupload und Case-Workerstart. Für die drei technischen Fälle wurde dabei keine neue N5000-Prediction erzeugt. Der externe Kandidat prüfte beide tatsächlichen Payloads durch beide unveränderten Parser: H8 23 auf 17 und H10 24 auf 18 Zeilen, je genau fünf Policy- und eine lokale Statistikrolle entfernt. Technical3, die bestehende reine Composed-Auswahl und der Erhalt bereits gültiger Part2-Timings bleiben unverändert. Dreizehn gezielte Softwaretests und der unabhängige Review bestanden; kein Hardware-PASS daraus.

Die source-only Übernahme erfolgt erst nach dem tatsächlichen normalen Abschluss und der freigegebenen Ownership von Sitzung347. Deren FirstAnchor bleibt vollständig bytegleich, ebenso die abgeschlossenen zentralen Qualitätsergebnisse, Originalpredictions und alle bestehenden Messungen. Ein anschließender normaler Supplementlauf verwendet seine vorhandene completed_quality- und Attempt-Reuselogik; gültige positive und negative Ergebnisse werden nicht neu bewertet oder durch Erfolgsmarkierungen ersetzt. Suite-, Case-, CPUreferenz-, Runner- und Supplementquellen werden für diesen Filter nicht geändert.

## 20.10.2026 – Supplement: Vergleichsdomäne, Fallberichte und fehlende Composed-Arbeit

Der normale Supplement-Selektor erhält H10 `canonical_float32`/`auto` als echte Float-Beobachtung. Der nachgewiesene b031-Numeric-FAIL wird dadurch bewahrt; unbekannte Domänen werden nicht als Raw-Fehler zugelassen. Die drei belegten Raw-UINT8-Komparatorfälle verwenden bei fehlendem separatem Diagnostikbericht die bereits indexgebundene vollständige normale Aggregatzeile mit ihrem vorhandenen `source_row_sha256`. Alle drei Originalbericht-Verbraucher lesen dieselbe eindeutige Bindung; Request-, Native-, Quell- und Fresh-Prüfungen bleiben streng.

Der normale Collector kopiert `case_validation_report` ohne Diagnostik-Größenlimit. Logs und andere Diagnostik behalten 2 MiB. Bei einer ehrlichen partiellen Setupgruppe genügt ausschließlich ein fehlerfreier Copybeleg mit ausdrücklich anderem fehlendem Run und dem vollständigen eindeutigen eigenen kanonischen Body, der allen Feldern der beiden bytegleichen Reportkopien entspricht. Die Gruppe bleibt partiell; anschließende CPUreferenz-, N5000-, Artefakt-, Cache-, Quell- und Guardprüfungen bleiben erforderlich.

Softwarebeleg extern: 56 gezielte Regressionen und derselbe unabhängige Lauf bestanden. Reine normale Planfunktion auf unverändertem Run: drei technische Wiederholungen, b031 bewahrt. Kein Hardware- oder Workflow-PASS aus diesen Prüfungen. Technische Wiederholungen erzeugen je N5000 neue Composed-Kandidaten in eigenen Attempts, bei null Timingloops und ausschließlich Cacheprüfung; alte N5000 und Performance bleiben erhalten. Der gesamte Supplementlauf enthält zusätzlich 27 normale fehlende Generic-Aufgaben vor `evaluate_quality` und ist deshalb nicht pauschal compilerfrei. Runner, Profil, Auswahl, Messprotokoll und bestehende CPUreferenzquellen bleiben unverändert.

Die vorhandene selektive Transportprojektion übergibt normale auszuführende Hailo-Split-Obligationen als die bereits unterstützte physische Variante `composed`. Die logischen Split-Zeilen, Profile und Sollmatrix bleiben unverändert; Full-, technische Quality-Source-, Preserve- und andere Backend-Zeilen bleiben bytegleich. Der bestehende Required-Scope-Vertrag `reference_measurement_variants` bindet eine vollständige Split-Messung ausdrücklich an Composed, während alleinige Part1-/Part2-Timings unvollständig sind. Damit wählt die unveränderte normale HEF-Verfügbarkeitsprüfung bei den aktuellen 25 fehlenden Split-HEFs keine erneuten isolierten Part2-Timings. Zwei fehlende Full-Fälle bleiben ebenfalls offen. Bei tatsächlich vorhandener HEF-Verfügbarkeit wird nur der fehlende Composed-Fall ausgewählt; es entsteht kein permanenter Supportausschluss. Suite-, Case- und CPUreferenzquellen werden hierfür nicht geändert. Enger Transportbeleg: 22 reine Tests einschließlich der tatsächlichen 27 Vorher-/Nachher-Varianten und Cache-Availability-Kontrollen bestanden; keine Hardwareausführung.

## 20.10.2026 — YOLO26x: gebundene DeepX-Full-Planmetadaten und regulärer Supplement-Quality-Stopp

Nach regulärem Abschluss und Freigabe der Management-CPU-Referenzoperation ist der getrennt geprüfte Controller-Runner für die vorhandene Kampagne übernommen. Der vorangegangene DeepX-Full-Versuch scheiterte vor allen drei Wiederholungen mit explicit_full_benchmark_plan_source_conflict: Der normale Standalone-Plan enthielt nachträgliche Metadaten, während benchmark_set.json.plan seine ursprüngliche Projektion behielt. Die neue Native-Stagingbehandlung gilt ausschließlich für angefordertes DeepX Full im A4-Modus. Sie rekonstruiert die vollständige normale Standalone-Projektion unabhängig aus eingefrorenem Profil, Modell, Analysekopien und Required-Run-Scope; gemeinsame Wertkonflikte, unbekannte Ergänzungen und widersprüchliche Aliase bleiben blockiert. Nur der frische Native-Plan wird anschließend aus der unveränderten BenchmarkSet-Autorität geschrieben und über bestehende Transfermetadaten und beide Remote-SHA-Prüfungen gebunden. Der strikte Native-Guard bleibt unverändert.

Kanonische Suite, Referenz, erhaltener Quellworkspace, DXNN, alte Fehlerberichte und vorhandene Messungen werden durch diese Sourceadoption nicht verändert. Die sechs bereits übernommenen Quelldateien bleiben als vorherige Source-Epoche erhalten; workflow/runner.py wechselt von 419122de auf 7f45b426 und workflow/supplement.py von 226ea791 auf 33162092. Referenz-Restore, Scheduler und Supplement-Admission sind unverändert. Ein neuer Referenzlauf folgt daraus nicht. Force-Rebuild bleibt false; die 20 Splits, N5000, B1000 und Seed20260710 bleiben erhalten.

Die gezielten externen Fixtures und der unabhängige Schlussreview liegen unter operations/resume_20261019_controller/native_deepx_full_route_candidate_20261020 sowie operations/resume_20261019_evidence/native_deepx_full_route_independent_review_20261020.json. Der echte Drei-Wiederholungs-Steuerfluss erreicht im Fixture je 100 Warmup- und 1000 Main-Aufgaben bis zur ausdrücklich ersetzten SDK-Schedulergrenze; run_model -l50 bleibt im Native-Messmodus übersprungen. Dies belegt keine ausgeführte Hardwaremessung oder neue wissenschaftliche Gültigkeit. Der normale Supplementplan, tatsächliche Artefakt-/Qualitätsguards und einzelne Native-/Energy-Repetitionen bleiben gesondert zu prüfen. Die 27 Hailo-Speicherfälle bleiben offen.

Die bestehende archivprofilgesteuerte Supplement-CLI erhält --stop-after mit den vorhandenen ALL_STAGES-Namen. Der bestehende options_from_run-Constructor bleibt unverändert; nur ein expliziter Aufrufwert setzt das bereits volatile options.stop_after. Plan-only nennt den Stopp und inventarisiert weiterhin die vollständige Sollmatrix. Energy-Parents lehnen diesen Workflow-Control ab. Der hier vorgesehene evaluate_quality wird im Supplement neu projiziert und erreicht im tatsächlichen global_barrier-Modus den normalen Stopp vor Native. Bei allgemein wiederverwendeten Stages kann der bestehende frühe Return dagegen die Stopprüfung umgehen; dafür wird keine neue Garantie behauptet. Der normale Cleanup-/Sealing-/Ownershippfad bleibt unverändert. CLI-Parser-/Optionswiring und unveränderte Planrows sind mit neun inerten Fixtures unabhängig geprüft; die tatsächliche Fortsetzung und ihre Qualitätsentscheidungen sind damit nicht ausgeführt.

## 20.10.2026 — YOLO26x: begrenzte Korrekturen für vorhandene Kampagne

Im Auftrag yolo26x_final_thesis_20splits_20261018 sind nach regulärem Ende der Messphase sechs zuvor getrennt geprüfte Quelldateien übernommen. Die fehlende Management-CPU-Referenz wurde durch die Verwechslung des Producer-Kontexts mit einer vorhandenen Referenz vor Inferenz blockiert. Der korrigierte Harness erhält N5000, B1000 und Seed20260710. Der Hailo-Schnittstellencheck dequantisiert ausschließlich tatsächlich rohe UINT8-Ausgaben vor dem Float-Vergleich; vorhandene Float-Negativbefunde und Qualitätsgrenzen bleiben erhalten.

Der normale Supplement-Pfad erhält eine eng belegte 2.95.5-Kompatibilitätsaufnahme, strenges frühes Referenz-Restore, gebundene technische Cache-only-Revalidierung, Descriptortransport und den bestehenden zentralen Quality-Join. Originale Vorhersagen, Messungen und Fehlversuche werden dadurch nicht als gültig umetikettiert. Vor der Übernahme bestanden 52 unabhängige Adapterprüfungen sowie fokussierte Produktpfad-, Descriptor-, Collector- und Spiegelprüfungen; deren genaue Belege stehen im Auftragsordner operations/resume_20261019_controller/post_terminal_reference_repair_independent_review und hailo8_interface_domain_candidate_20261020/normal_retry_candidate. Geräte-Inferenz, N5000-Referenzerzeugung und neue gültige Messungen sind zum Zeitpunkt dieser Quellübernahme noch nicht abgeschlossen.

Die reguläre Hailo-Build-Fortsetzung bleibt wegen fehlendem lokalem Arbeitsraum offen. Unabhängige CPU-Referenz/Qualitätsarbeit wird vorgezogen; der frühere unveränderte-Source-Build-Adapter darf nach dieser Übernahme nicht unverändert gestartet werden. Konfiguration, 20 Splits, Version2.95.5, Force-Rebuild=false und die vorhandene schmutzige Git-Baseline bleiben erhalten. Quelloriginale, Manifeste, Gitstatus und genaue Adoption werden extern unter operations/resume_20261019_controller/source_adoption_actual_* gesichert. Dies ist eine Softwarekorrektur, kein Abschluss der Kampagne und kein Hardware-PASS.

## 16.10.2026 — DeepX Native A4: Abnahme, 75 Punkte und Ergebnislieferung abgeschlossen

Die installierte profilgesteuerte CLI hat alle 14 ursprünglichen Abnahmezellen und alle 75 vorgewählten Punkte (68 Splits, sieben Fulls) vollständig abgeschlossen. Die 14 kompatiblen Abnahmekohorten sind enthalten und wurden nicht doppelt aufgenommen. Ausgewählt sind insgesamt 225 unabhängige Performanceprozesse mit je 100 Warmups und 1.000 Haupttasks sowie 225 gültige 60-s-Native-Energieaufnahmen ohne Workload-Warmup. Die zusätzlichen 61 Punkte benötigten 183 Performanceprozesse und 234 Energiecapturestarts; 51 Fehlcaptures bleiben erhalten. Zusammen mit der Abnahme sind 279 Energiecapturestarts mit 225 gültigen und 54 fehlgeschlagenen Aufnahmen dokumentiert. Native A4, Batch 1, eine Engine, maximal vier offene DeepX-Aufträge und Split-FIFO 3 einschließlich serieller TensorRT-Taskcompletion gelten unverändert.

Die normale Ergebnisvorbereitung und die unabhängige Prüfung des tatsächlichen Archiv-/Importziels sind erfolgreich. Genau 75 von 450 Ergebniszeilen wurden aktualisiert; die übrigen 375 Zeilen einschließlich ihrer Quellen und Repeats sowie alle 450 Qualitätsselektionen sind exakt erhalten. Historische S1-Ergebnisse, Fehlversuche, gültige langsame Messungen und wissenschaftliche Einschränkungen bleiben bestehen. Die drei additiven Archive umfassen 19.394 unterschiedliche Originaldateien mit 6.844.644.708 Bytes. Die normale Current-Aktivierung ist mit freigegebenem Writerlock abgeschlossen; die anschließende Kontrolle bestätigt NOOP_VERIFIED. Die bisherige Analyse bleibt ausdrücklich historisch; eine neue Auswertung wurde nicht behauptet.

Nach dem Messabschluss wurden die kleine Korrektur des normalen Split-Ratenlesers und die enge Zuordnung vorhandener Energie-Repeats im Reportadapter geprüft. Gültige Messungen wurden dafür nicht wiederholt und Originaldaten nicht verändert. Die finale Dokumentations- und Manifestaktualisierung erhält Version 2.95.5 und denselben Produktcode. Der vollständige Bericht einschließlich tatsächlicher Quellen-, Test-, Archiv- und Aktivierungsbelege liegt unter /home/kmika/Reports/deepx_295_gui_acceptance_remeasurement_20261015/20261015T130753Z/REPORT.md. Alle nachfolgenden Arbeitsstände bleiben als historische Zwischenstände erhalten; insbesondere frühere Desktopblockaden und Zwei-Versuchs-Grenzen gelten nicht mehr für diesen abgeschlossenen Auftrag.

## 16.10.2026 — 75 Native-A4-Punkte vollständig aufgenommen; reguläre Übernahme wird abgeschlossen

Die korrigierte Beauftragung wurde über die installierte profilgesteuerte CLI ausgeführt. Alle 14 ursprünglichen Abnahmezellen sind vollständig; ihre kompatiblen Kohorten wurden in den 75 Punkten (68 Splits, sieben Fulls) wiederverwendet. Die sieben Modellläufe sind terminal und geschlossen. Ausgewählt sind 225 unabhängige Performanceprozesse mit je 100 Warmups/1.000 Haupttasks und 225 gültige 60-s-Native-Energieaufnahmen ohne Workload-Warmup. Für die 61 übrigen Punkte entstanden 183 neue Performanceprozesse und 234 Capturestarts mit 183 gültigen Aufnahmen und 51 erhaltenen Fehlaufnahmen. Die Abnahme enthält zusätzlich 45 Capturestarts mit 42 gültigen und drei fehlgeschlagenen Aufnahmen. RegNet und ResNet vervollständigten ausschließlich fehlende Energieslots durch normales Resume; ihre ursprünglichen Teilabschlüsse bleiben erhalten.

Die abschließende Ressourcenprüfung meldet keine bekannten aktiven Worker oder offenen Ressourcen; alle beobachteten Freigaben sind abgeschlossen. Der während sämtlicher Hardwaremessungen eingefrorene installierte Quellstand (2.95.5, 2282 Release-Dateien) bestand die finale Prüfung. Danach scheiterte die normale Übernahme an der vorhandenen Bindung eines dreiteiligen Split-Ergebnisses an nur einen repräsentativen Rawreport. Die kleine Korrektur im normalen Ratenleser öffnet die drei bereits zugelassenen Prozesse über PREPARED, Import und Originalbelege erneut. Die gesammelte originale YOLO26-Outputdeklaration wird über denselben vorbereiteten Auftrag aufgelöst. Zähler, Zeitintervalle, IDs und bestehende Statistik werden erhalten; widersprüchliche Belege bleiben unzulässig. Qualitätsurteile, wissenschaftliche Ausschlüsse und gespeicherte Partial-Befunde werden nicht aufgewertet.

Softwareprüfung des endgültigen Kandidaten: 35 fokussierte Tests bestanden; unabhängige Prüfung 82/82 ohne Auslassungen, einschließlich Full-/Generic-/Hailo-Regressionen und vier tatsächlicher Outputfamilien. Die normalen Messprogramme und ursprünglichen Messdateien bleiben unverändert. Diese nachträgliche Lesekorrektur erfordert keine Wiederholung gültiger Messungen. Installation, abschließende Übernahme, Archivprüfung und Current-Aktivierung werden im Auftragsreport mit ihren tatsächlichen Ergebnissen dokumentiert. Zum Stand dieses Abschnitts ist die Veröffentlichung noch ausstehend.

## 16.10.2026 — Reguläre CLI-Abnahme vollständig; fehlende Native-Energieslots gezielt fortsetzen

Für den Auftrag `deepx_295_gui_acceptance_remeasurement_20261015` ist gemäß korrigierter Nutzeranweisung die installierte profilgesteuerte CLI der reguläre Produktstart. Anforderungen an Desktop, Fenster, Tk-Bedienung oder grafische Startbelege sind aufgehoben. Tatsächliche CLI-Starts werden als CLI ausgewiesen; ältere GUI-Berichte bleiben historische Belege.

Alle ursprünglichen 14 Full-/Split-Zellen der sieben Modelle sind numerisch bestanden und enthalten je drei unabhängige Performanceprozesse mit 100 Warmups/1.000 Haupttasks sowie drei gültige Native-A4-Energieaufnahmen mit 60-s-Vertrag und null Workload-Warmups. Insgesamt sind 42 Performanceprozesse und 42 gültige Energieaufnahmen aus 45 tatsächlichen Captures belegt; die drei Fehlaufnahmen bleiben erhalten. Der ursprüngliche b027-Numerik-PASS wurde übernommen. Die bereits integrierten CLI-/Resume-, Energiezulassungs-, Checkpoint-Umgebungs- und Full-Stagingkorrekturen verwenden weiterhin die gemeinsamen Produktworker und normalen Ergebnisleser.

Die 75-Punkte-Neuaufnahme läuft regulär mit gruppierten Modellprofilen. MobileNet ist mit 18/18 Punkten abgeschlossen. Der erste RegNet-Durchlauf `deepx_2951_native_refresh_regnet_x_1_6gf_20261016_004656` endete am 16.10.2026 terminal partial mit Artefakt-Closurepass: 19/21 Punkte vollständig, alle 57 frischen Performanceprozesse abgeschlossen, 71 Capturestarts mit 54 gültigen und 17 verworfenen Aufnahmen. Nur b008 repeat2 und b037 repeat0/1 fehlen. Originaler Teilabschluss, Budgets, gültige Slots und Fehlaufnahmen wurden vor Resume vollständig kopiert und byteweise geprüft. Die normale Ressourcenbeobachtung bestätigt beendete Besitzer, freie physische Ressourcen und alle 362 Capture-Antworten RELEASED.

Die kleine Resume-Korrektur erhält im bestehenden Aufgabenbudget die ursprünglichen STOPs, Zähler, Ketten und gültigen Slots. Ein ausdrückliches normales Resume kann nach nachgewiesen abgeschlossenem reinem Sampleverlust eine begrenzte weitere Capture-Serie ausschließlich für fehlende Slots eröffnen. Aufgabenbudget und Native-Zulassung prüfen denselben Vertrag; jede fehlende Position erhält die gewöhnliche Aufnahme plus zwei Retries. Die Quellengrenze von 20 Transportfehlern, Ressourcenprüfungen, Kalibration, Sensorfenster und Abschlussprüfungen bleiben unverändert. Collector und Workloadprogramme sind bytegleich; Generic und Hailo werden nicht geändert.

Softwareprüfung: 140 fokussierte Tests beim Autor; unabhängige Prüfung mit 160 fokussierten Tests und zwei portablen normalen Collector-Fällen. Diese Prüfungen sind kein Hardware-PASS der Korrektur. Die tatsächliche Nachaufnahme der drei fehlenden Slots und die übrigen fünf Modelljobs stehen zum Installationszeitpunkt noch aus. Die installierte Version bleibt 2.95.5. Aktueller tatsächlicher Verlauf und vollständige Belege: `/home/kmika/Reports/deepx_295_gui_acceptance_remeasurement_20261015/20261015T130753Z/STATUS.md`. Der finale 75-Zeilen-Import und die Archivaktivierung erfolgen erst nach vollständiger Prüfung.

## 15.10.2026 — Echter Split-PASS, Full-Pfadkorrektur 2.95.5

Die abschließende unabhängige Softwareprüfung von 2.95.5 besteht mit 32 Integrationsprüfungen und neun Kontext-/Erhaltungsprüfungen. Drei ursprüngliche Full-Performanceverträge werden über ihre bestehenden vorbereiteten Aufträge und Originaldateien zusammengeführt; der normale Matrixleser behält alle drei Taskzahlen und Zeitintervalle. Die Energieplanung verwendet den tatsächlich ausgewählten Vertrag. Die portable YOLO26-BN6-Auswertung erhält die unveränderte, bereits bereitgestellte Outputdeklaration. Alle strikten Quellen-/Artefaktprüfungen bleiben bestehen; ein Software-PASS ersetzt keine Hardwareabnahme. Der Stand wird vor dem nächsten ausschließlich Full betreffenden GUI-Versuch regulär archiviert und installiert.

2.95.4 wurde regulär installiert. Die zusätzliche Full-Sperre wurde am 15.10.2026 um 14:08:53 UTC regulär anhand 77 unveränderter Originalbelege und frischer Besitzerprüfung freigegeben. Der anschließende echte GUI-Job `evaluation-workflow-20261015_161110-a4b5f53deedb4590b09da31e1917292f` / Run `deepx_native_a4_gui_acceptance_mobilenet_v3_large_numerical_continuation_20261015_161156` schloss MobileNet b027 als `numerical_passed` ab. Full blieb wegen `full_command_contract_deepx_artifact_role_path_mismatch` failed. Beide Rawreports melden 32 Referenzen, 32 gemischte A4-Aufträge und 64 Smoke-Tasks; Geräte- und Workerabschluss sind jeweils belegt. Die getrennte Ressourcenbehandlung verhinderte eine unbegründete neue Sperre.

2.95.5 gleicht die A4-Full-Veröffentlichung vorbereiteter Eingaben an den vorhandenen kanonischen Pfadvertrag an und erhält ihre normale vollständige Abholung. Die GUI wurde am 15.10.2026 um 14:19:20 UTC nach terminalem Job und bestätigtem Artefaktabschluss regulär geschlossen. Der b027-PASS ist im vollständigen Abnahmeprofil gebunden; ein separates normales Reparaturprofil enthält ausschließlich Full. Kein gültiger PASS wird wiederholt, kein alter Fehler überschrieben. Reguläre Performance und Energie stehen weiterhin aus.

## 15.10.2026 — Full-Suite-Bereitstellung nach echtem 2.95.3-GUI-Lauf

Die ursprüngliche b027-Sperre wurde am 15.10.2026 um 13:36:15 UTC regulär anhand unveränderter Originalbelege und aktueller Besitzerprüfung freigegeben. 2.95.3 wurde normal installiert und über die echte GUI gestartet (Job `evaluation-workflow-20261015_153810-48bbab6d8cad4abc8bd86a8f1a70df70`). Full erreichte die serielle DeepX-Vorbereitung, danach wählte die Suite mangels separatem Benchmark-Plan das falsche ORT-Profil. Der fehlende gemessene A4-Abschluss löste die Schutzsperre aus; b027 wurde vor Admission blockiert. GUI-Job terminal, Anwendung regulär geschlossen. 2.95.4 korrigiert diesen konkreten Bereitstellungsweg; kein unveränderter Hardware-Retry, keine Toleranzlockerung, keine neue Freigabeanforderung. Belege und weiterer Verlauf im aktuellen Report `20261015T130753Z`.

Die A4-Stagingkorrektur stellt den vorhandenen eingebetteten Suite-Plan unverändert separat und die exakt gebundene Outputautorität bereit. Der direkte A4-Full-Runner prüft diese Auswahl vor seinen Runtime-Children und übergibt den Plan ausdrücklich. Frische S1-Voraussetzungen, Generic und Hailo bleiben erhalten. Der integrierte gezielte Testsatz besteht mit 139 PASS; eine unabhängige Prüfung mit 74 PASS. Zusätzlich bestehen sieben Tests mit den originalen Metadaten aller sieben Modelle und je zwei tatsächlichen Suite-Auswahlen für Numerik und Performance. Diese Prüfungen ersetzen die Runtime an ihrer SDK-Grenze und sind kein Hardware-PASS. Die separate normale Recovery und die nächsten echten GUI-Aufnahmen bleiben an ihren tatsächlichen Belegen zu bewerten.

# Fortsetzung15.10.2026: 2.95.3, normale Recovery und technische Reparaturen

Die neueste Nutzeranweisung erweitert den laufenden Auftrag `deepx_295_gui_acceptance_remeasurement_20261015`: Die belegte abgeschlossene b027-Inferenz soll regulär recoverbar sein; Ergebnisgültigkeit und Ressourcenabschluss sind getrennt zu prüfen. Konkrete technische Korrekturen mit Regression und normaler Bereitstellung dürfen weitere dokumentierte Versuche auslösen. Die alte Zwei-Versuchs-Lebenszeitgrenze wird nicht als Blocker fortgeführt; historische Versuche, negative Ergebnisse und gültige Slots bleiben erhalten. Keine neue Budget-/Registrysoftware, kein Force-Unlock, unveränderte Capture-Retries und echte STOPs.

Aktueller Fortsetzungsreport: `/home/kmika/Reports/deepx_295_gui_acceptance_remeasurement_20261015/20261015T130753Z`; ursprüngliche Lieferung `/home/kmika/Reports/deepx_295_gui_acceptance_remeasurement_20261015/20261015T093320Z` bleibt als damaliger Stand erhalten. 2.95.3 bindet die dafür notwendige normale Produktkorrektur nachvollziehbar. Das Ziel bleibt die echte GUI-Abnahme aller sieben Full-/Split-Paare und anschließend unmittelbar die75-Punkte-Neuaufnahme mit jeweils drei Performance- und drei gültigen Native-Energierepeats, vollständigem kompatiblem UID-Reuse und normalem Import/Archiv. Softwareprüfung, tatsächliche Recovery und tatsächliche Hardwareläufe werden im aktuellen Report getrennt nachgewiesen. Die folgenden älteren Abschnitte beschreiben ihre damaligen engeren Autorisierungen und bleiben historisch erhalten.

# Folgeauftrag 15.10.2026: 2.95.2, echte GUI ausgeführt; Hardwareabnahme blockiert

Auftrag `deepx_295_gui_acceptance_remeasurement_20261015`; vollständige aktuelle Belege unter `/home/kmika/Reports/deepx_295_gui_acceptance_remeasurement_20261015/20261015T093320Z`. Die alte eng belegte MobileNet-pre-inference-Quarantäne wurde am 15.10.2026 um 10:21:06 UTC über die normale Produktoperation unter den bestehenden exklusiven Locks regulär aufgelöst. Alte Rohreports, fehlende historische SDK-Zähler, Fehler, Budgets und Quarantänebelege bleiben erhalten.

Der regulär installierte und archivierte 2.95.1-Stand wurde anschließend über das echte Evaluation-Panel der tatsächlichen Tk-GUI gestartet: genau ein MobileNet-Modelljob und die beiden noch zulässigen technischen Ersatzversuche. Job `evaluation-workflow-20261015_122605-c0e4c504322d4cabbb2329221bc83c9b`, Run `deepx_native_a4_2951_gui_acceptance_mobilenet_v3_large_numerical_20261015_122654`. Ergebnis: terminal partial, beide Produktzellen failed. Full scheiterte im Staging vor Workerstart. b027 meldet im unveränderten Worker 32 serielle, 32 gemischte und 64 Smoke-Abschlüsse mit drain=true; die normale Lifecycleprüfung lehnte einen fehlenden expliziten seriellen Abschlusszähler ab und quarantänisierte den DUT erneut. Die Produktzelle wird nicht nachträglich als PASS ausgegeben.

2.95.2 korrigiert die zwei konkreten Schnittstellenfehler: Full reicht das exakt zum versiegelten ursprünglichen Benchmark und deklarierten Inputcase gehörende lokale Case-Metadatum durch die normale Bereitstellung in den frischen Full-Baum; der Split-Worker zählt Ein-/Rückkehr des seriellen Referenz-SDK-Aufrufs explizit. Der strenge Reader und die enge historische Recovery bleiben unverändert. Gezielte Regressionsprüfungen und unabhängige Reviews sind im Auftragsreport ausgewiesen. Die beiden vorherigen Vorbereitungsfixes (Venv-invocation_path, aktueller Workerprüfwert bei frischem A4) bleiben bestehen.

Beide MobileNet-Budgets sind mit je zwei Versuchen ausgeschöpft. Die neue Quarantäne nach tatsächlich begonnener Inferenz fällt nicht unter die alte pre-inference-Recovery und bleibt erhalten. Zwölf Abnahmezellen sind ungestartet, null reguläre Performanceprozesse, null Energiecaptures, null vollständige neue Kohorten. Softwarestand und Hardwareabnahme werden getrennt bewertet; 2.95.2 ist ohne erneute zulässige Hardwareausführung nicht hardwareabgenommen.

Die sieben normalen GUI-Neuaufnahmeprofile binden genau 75 ursprüngliche UIDs. Normales Resume erhält gültige Slots; Übernahme verlangt vollständige kompatible reale GUI-3+3-Kohorten. Der finite Native-Refresh-Selektor ist getrennt von der unveränderten Abnahmebeschränkung auf einen Split je Modell. Nach vollständiger Abnahme ist die Kampagne bereits beauftragt; aktuell verhindern Quarantäne und erschöpftes Numerikbudget ihren Start. Keine privaten Messwrapper, Modell-/Engine-Rebuilds, Generic-/Hailo-Neumessungen, neue Registry oder historischen Umdeutungen. Datasetgeneration, Sprite-Transfer und Current-Aktivierung wurden nicht ausgeführt. Die eigene GUI wurde nach terminalem Abschluss regulär geschlossen.

Der breite Softwarelauf vor GUI-Ausführung ergab 587 passed und zwei Fehler in bereits auf dem archivierten 2.95.0-Stand identisch fehlschlagenden Hailo-Fault-Injection-Fixtures. Deren leere Qualitätsvoraussetzungen blockieren vor dem zu injizierenden Transportfehler; Gesamtstatus bleibt partial/scientific_ready=false. Diese Befunde werden offengelegt, nicht durch Guardlockerung beseitigt. Abschließende 2.95.2-Installation, Releaseprüfung und fokussierte Testergebnisse stehen in REPORT.md/VALIDATION.md des neuen Auftragsreports. Der nachfolgende 2.95.0-Abschnitt bleibt als damaliger historischer Stand erhalten.

Die abschließende gezielte Release-/Installer-/Full-Staging-/Split-/Generic-Hailo-Prüfung für 2.95.2 bestand mit 71 Tests und einer bestehenden Deprecation-Warnung. Alle 14 frisch aus dem finalen Produkt vorbereiteten Zellen binden die aktuellen Worker, korrekte Venv-Aufrufpfade und sämtliche YOLOv7-Heads; die reine Budgetprüfung verweigert beide verbrauchten MobileNet-Versuche ausdrücklich. Offlinevorbereitung und Softwaretests haben keine neuen Hardwareversuche oder Captures erzeugt.

# ONNX Splitpoint – fortlaufender Arbeitsstand

## 2.95.0: DeepX Native A4 (15.10.2026, Software installiert; Hardwareabnahme blockiert)

Der neue Produktauftrag bindet Full und alle unterstützten Native DeepX→TRT-Splits an dieselbe A4-Policy. Gemeinsame Submit-/Wait-/Slotverwaltung, sichere Outputübernahme, FIFO 3 und serielle TRT-/Taskcompletion ersetzen modellabhängige Schedulingentscheidungen. Fehlende Modusfelder alter Jobs bleiben historisch S1. Vorhandene lokale Refresh16-Korrekturen bleiben erhalten.

Die Produktadapter und Regressionen prüfen kontinuierliche A4-Nachfüllung, vier SDK-konforme Gesamtpuffer mit typisierten Multioutput-Ansichten, sichere letzte Outputs, Warmup-/Main-Trennung und Fehlerdrain. Die normale Split-Dispatch-Ebene startet für drei Repeats drei frische Prozesse. GUI-/CLI-Profile, Prepared-/Resume-Verträge und reguläre Performance-/Energieimporte tragen den expliziten Modus; passende S1-/A4-Kohorten werden getrennt gehalten. Der finite Abnahmecontroller erhält einen DeepX-Full-Einstieg neben den bisherigen Splits.

Softwareprüfung und begrenzte Abnahme werden unter `~/Reports/deepx_native_a4_v295_20261015/20261015T071759Z` geführt. Die Abnahmematrix umfasst 14 vorgewählte Zellen; Hardware-PASS wird erst nach tatsächlicher Ausführung vergeben. Die spätere 75-Punkte-Neuaufnahme bleibt Prepare-only.

Die Full-Integration verarbeitet die tatsächlich gebundenen SDK-Multioutputs mit vier getrennten Gesamtpuffern und den vorhandenen Klassifikations-/Detektorendpunkten. Generierte Generic-/Hailo-Bundles, Backendrequests und Profile wurden vor/nach der Änderung verglichen; drei frische generierte Generic-CPU-Läufe liefern gleiche fachliche Ergebnisse. Alle 14 ausgewählten Zellen bestehen die normale Offlinevorbereitung. Der begrenzte numerische Einstieg verwendet dieselbe Vorbereitung, Ressourcenreservierung und Transportstrecke; er eröffnet keine Energieaufnahme und liefert keinen wissenschaftlichen Performance-PASS. Installation und tatsächliche Hardwareergebnisse werden im Auftragsreport gesondert geführt.

327 neue 2.95-Tests bestanden vor der regulären Installation. Der erste normale MobileNet-Phase-A-Auftrag wurde tatsächlich ausgeführt: Full scheiterte vor dem Workerstart an einer falsch aufgelösten Python-Aufrufbindung; b027 scheiterte vor der SDK-/TRT-Konstruktion am erhaltenen historischen Runnerprüfwert. Die beiden Bindungen sind korrigiert (104 gezielte Tests bestanden). Die konservativ gesetzte physische DUT-Quarantäne bleibt bestehen, da das vorhandene Capture-Recovery diesen numerischen Fall nicht unterstützt. Keine weitere Hardwarearbeit, keine Performanceprozesse oder Energiecaptures; zwölf Zellen bleiben ungestartet. Die Softwarelieferung ist kein Nachweis vollständiger physischer A4-Abnahme. REPORT.md, ACCEPTANCE_RESULTS.csv, STATUS.md und RESUME.md halten die Grenzen und Originalbelege fest; alte Ergebnisse, STOPs und lokale Vorarbeiten sind erhalten.

## Refresh16: Native-Energiebeobachtungen regulär fortsetzen (13.10.2026)

Die normale `--previous-results`-Fortsetzung verwarf einen versiegelten Native-
Split-Beobachtungspfad, weil sie dessen bereits gültigen `admission_scope` nicht
an denselben Qualitätsbindungsprüfer wie der Erstlauf übergab. Beide produktiven
Source-/Resource-Einstiege prüfen nun zuerst die ursprüngliche versiegelte
Zulassung und reichen ausschließlich deren authentifizierten Scope weiter.
Identität, Commands, Quellen, Cohort und Hashbindungen bleiben unverändert;
negative Qualitätsbefunde erhalten keine positive Claimzulassung.

58 gezielte Offline-Regressionen bestanden auf der bestehenden installierten
Umgebung. Belege und ursprüngliche Sources liegen unter
`~/Reports/refresh16_20261013/energy_prepare/split_resume_install/`. Die neue
begrenzte Messung verwendet weiterhin die regulären Sensor-, Runtime- und
Resultconsumer; diese Softwarekorrektur ist kein eigener Hardwareabnahmebeleg.
Version 2.94.0 bleibt bestehen; keine Compiler-, Modell- oder Decoderänderung.

## Begrenzter Revision-3-Deltaabschluss: Recovery nach sauberem Folgelauf (12.10.2026)

Die normale wissenschaftliche Resume-Übergabe eines bereits physisch freigegebenen
alten STOP lehnte eine zwischenzeitlich sauber abgeschlossene Hardwareabnahme auf
demselben DUT und derselben Sensorquelle ab. Für Controllerressourcen war diese
Folge bereits unterstützt. Dieselbe strikte Prüfung gilt jetzt für genau die vier
ursprünglichen Ressourcen: normale Released-Metadaten, passende physische Ressource,
gleicher Controller, tote Besitzer, keine Quarantine und eine spätere, nicht zukünftige
Freigabezeit. Alle Flocks bleiben während der Accounting-Übergabe gehalten; die normale
Live-Zulassung vor dem nächsten Capture bleibt erforderlich.

Alte STOPs, ungültige Quellenabschlüsse, gültige Repeats, Zähler und Messkriterien
werden dadurch nicht geändert. 120 bestehende Recoveryregressionen und 42 gezielte
Tests mit temporären echten Locks bestanden. Ein erster Testaufruf scheiterte am
fehlenden externen Ausgabe-Elternordner; der korrigierte Lauf bestand. Unabhängige
Diffprüfung ohne blockierenden Befund. Diese Softwareprüfung ist kein Hardwarebeleg;
die anschließenden zulässigen wissenschaftlichen Deltaaufnahmen werden im bestehenden
Report unter operations/token_saving_batch_20261009/delta_abschluss_20261012 dokumentiert.
Version 2.94.0 bleibt bestehen; keine neue Toolabnahme oder Releasekampagne.

Der anschließende normale Recoveryversuch legte zwei weitere Bindungsprobleme offen:
Die Nachbuchung erhält den historischen Raw-STOP und schreibt den aktuellen Fehler
in die fortlaufende Autorisierung; deren Wiedereintritt erkennt jetzt beide belegten
Darstellungen. Eine bereits für einen anderen Workflow erteilte physische Freigabe
wird unverändert wiederverwendet. Die ausdrückliche wissenschaftliche Fortsetzung
erhält eine getrennte Autorisierung mit exakter Parent-/Operations-/Quellen-/Budget-
und Limitbindung. Das geprüfte Profil wird unverändert versiegelt; spätere Recovery
erhält frühere Autorisierungen ausschließlich in ihrer zugehörigen Historie.
23 zusätzliche gezielte Tests bestanden, insgesamt 185 unterschiedliche Regressionen.
Die unabhängige Prüfung umfasst Folge-Recovery, Profiländerung während der Übergabe
und die Trennung beider Autorisierungen im normalen CLI-/Backendweg.

Der vorbereitete wissenschaftliche Generic-Controllerrefresh berücksichtigt zusätzlich
die seit v2.92.0 ergänzten administrativen Recovery-/Admission-CLI-Bindungen und
`execution_binding.py`. Dessen einzige Änderung betrifft die Auflösung expliziter
Benchmark-Warmups; dieser Generic-Energiepfad benutzt nur die unveränderte
`physical_dut_key`-Funktion und seine eingefrorenen Runner-/Aufrufverträge. Die
wissenschaftlichen Aufruf- und Runtimeverträge werden weiterhin geprüft; frühere
Controllerbindungen bleiben im normalen Refresh-Receipt erhalten. Bei strikt
ungestarteten Parents wird ausschließlich die eingebettete Collector-CLI-Bindung
in einer erhaltenen Plankopie aktualisiert, nicht deren wissenschaftlicher Vertrag.
Sieben gezielte
Regressionen bestätigen den Metadatenrefresh ohne Hardware-/Budgetänderung sowie
die Ablehnung geänderter Runner, Eingaben, Dauer und Wiederholungszahl sowie
fremder Collectorbindungen und bereits versuchter Captures; sechs
bestehende Parentflow-Prüfungen bestehen ebenfalls. Reale Deltaaufnahmen sind
separat im Operationsstand ausgewiesen, nicht durch diese Tests behauptet.

Der wissenschaftliche Generic-Originalabgleich prüft weiterhin jedes eingefrorene
Snapshotbyte und die tatsächlich gemessenen DUT-Bundles. Seit der Funktionsabnahme
geänderte GUI-, Profil-/Dispatcher- und Releasemetadaten erhalten eng benannte,
unabhängig geprüfte lokale Bindungen im bestehenden Quellsupplement; ihre
vorhandenen Modell- und Energieaufrufe werden nicht neu konstruiert. Änderungen
an expliziten Null-Warmups sind allgemein semantisch relevant, aber in diesen
bereits vorbereiteten wissenschaftlichen Generic-Aufrufen nicht wirksam. Die
zusätzlichen sechs UI-/Dispatcher-Metadatenbindungen erfordern ausdrücklich das
unveränderte Originalbundle; zwei administrative Controllerbindungen bleiben in
der bestehenden Controllerklasse. Zwanzig gezielte Regressionen prüfen diese
zulässigen Fälle und die unverändert strengen negativen Snapshot-, Quellen-,
Scope- und Bundlebedingungen. Keine geänderte Numerik oder aufgeweichte
Messwert-/Qualitätsgrenze wird dadurch zugelassen. Originalquellen, Vorversionen
und die konkrete Abgrenzung stehen im begrenzten Delta-Operationsbereich.

Die erste neue Generic-Kohorte wurde regulär mit drei gültigen Repeats beendet,
aber ihr installierter Ergebnisreader erkannte die vorher belegte Collector-
Metadatenhistorie noch nicht. Der normale GUI-/CLI-Import prüft jetzt auch diese
exakte, vor jedem Capture erzeugte Historie und ihre erhaltenen Parent-/Plan-
Hashes. Ausschließlich die bereits geprüfte Collectorbindung wird zum Vergleich
in einer Plankopie angeglichen; originale Pläne, numerische Verträge und
Messkriterien bleiben erhalten. Neun gezielte Produkttests einschließlich acht
negativer Bindungsfälle sowie 24 bestehende Refresh-/Import-/Parentflowtests
bestanden (33 insgesamt). Der reale erste Fall wird ausschließlich aus seinen
vorhandenen drei Aufnahmen erneut importiert; daraus entsteht kein Messauftrag.

## H8-Zusatzaufnahme mit Paketmitschnitt (06.10.2026)

Genau eine Zusatzaufnahme für repeat:2 ausgeführt; die zwei gültigen Repeats
wurden ohne neue Aufnahme verifiziert. Quellenende regulär bestätigt, aber
7.936 Samples (124 Datagramme) fehlen im Commandfenster. Der Controller-pcap
zeigt dieselbe 4,000052-s-Lücke von 16:16:00,142 bis 16:16:04,142 MESZ;
2.566 Pakete, null Kernel-Drops, null Collector-Socketdrops. Alle empfangenen
Samplepaket-Zähler und gespeicherten Indizes stimmen überein. Ursache vor bzw.
am Controller-Mitschnittpunkt weiter offen; kein Firmwaredefekt behauptet.
Die Messung bleibt nach unveränderten Regeln ungültig: 2/3 gültige Repeats,
5/9 Starts, 3/20 Transportfehler, Ein-Start-Grenze 1/1 verbraucht.

Ein lokaler Adapterfehler durch Python -u vor dem Skriptpfad trat vor jeglichem
Collectorstart auf und wurde korrigiert. Reguläre saubere Parentübergabe auch
für null Operationen, ausschließlich mit unverändertem Budget-Handoffbeleg.
202 Softwaretests PASS; keine GUI-Abnahme. Vollständiges argv und env pro
Subprozess dokumentiert, keine dauerhaften Umgebungsänderungen.
Alle 202 ursprünglichen Remote-Runtimedateien, 112 alte Capturedateien und
bestehende Performance erhalten. Kein Build/Rollout, keine längeren Timeouts.
Aktuelle Ressourcen und Runlock regulär freigegeben, historische STOP-Belege
erhalten, keine aktiven Messprozesse. Kein weiterer automatischer Versuch.
Profiling, Full-Fix und finale Matrix offen, da der dritte Vorher-Repeat fehlt.
Belege: Reportordner yolov7-full-fix-20261005/H8_ENERGIE_STATUS.md,
energy/h8_before/targeted_20261006/packet_capture_analysis.json und
validation/h8_targeted_resume/post-measurement.xml.
Die folgenden Abschnitte dokumentieren frühere Zwischenstände.

## Gezielte H8-STOP-Korrektur (06.10.2026, nachmittags)

Software abgeschlossen: negativer Capture wird auch bei Broker-STOP dauerhaft
verbucht und die STOP-Exception weitergegeben. Die beleggebundene idempotente
Nachbuchung schließt beide historischen Fehler ab: vier verbrauchte Starts,
zwei Transportfehler, zwei gültige Repeats; alle ursprünglichen Limits bleiben.
Resume-Recovery bindet explizit den letzten fehlgeschlagenen Parent einschließlich
Journal, Operation, Ressourcen, Runlock und Budget. Alte Grants bleiben erhalten.
Die zusätzliche Freigabedatei nach einem maschinell gültigen Repeat entfällt.
196 gezielte Tests PASS, einschließlich echtem lokalen Broker mit Fake-Collector
und temporären Sperren sowie vollständiger Resume-Runlock-Übergabe. Keine GUI-Abnahme.

Die vier physischen H8-Ressourcen wurden über die reguläre API freigegeben;
der Runfence bleibt bis zur regulären Parentübergabe bestehen. Null neue
Collectorstarts in diesem Folgeauftrag. 112 originale Capturedateien, gültige
Performance und alle 202 ursprünglichen Remote-Runtimedateien erhalten.
Hardware und Gesamtauftrag offen: Der erlaubte einzelne Zusatzversuch für
repeat:2 wartet auf den angeforderten Controller-Paketmitschnitt. Lokal fehlen
RAW-Socketrechte; SSH und Schreibzugriffe funktionieren. Keine Hardware-
Nichterreichbarkeit behauptet. Nach gültigem dritten Repeat selbstständig zum
ursprünglichen Profiling-, Full-Fix- und Matrixauftrag zurückkehren.
Belege und genauer Folgebefehl: Reportordner yolov7-full-fix-20261005,
H8_ENERGIE_STATUS.md und energy/h8_before/targeted_20261006/ready_command.json.
Der folgende ältere Abschnitt dokumentiert den vorherigen Zwischenstand.

## H8-Vorherenergie – Recovery und begrenztes Resume (06.10.2026)

Bestehender Worktree und Reportordner sind im tatsächlichen Kontext schreibbar;
SSH zu H8 funktioniert. Der externe Energieparent kann seinen ursprünglichen
Journalpfad jetzt an die reguläre Vier-Ressourcen-Recovery anbinden. Ein expliziter
Resume übernimmt das vorhandene Campaignbudget unter den bestehenden Sperren.
Ein sauber beendeter Parent kann nach gebundener Terminal-/Cleanupprüfung an einen
Nachfolger übergeben; historische STOPs, Versuche, gültige Repeats und Limits
bleiben erhalten. Keine generelle Freigabe ungeklärter Quellenabschlüsse.

Software: abschließend **170 PASS** für Recovery, Resume, Budget und Adapter;
zusätzlich **5 PASS** mit dem unveränderten registrierten R6-Collector gegen
Loopback. Das ist keine GUI-Abnahme. Ein initialer Testaufruf scheiterte am noch
fehlenden lokalen pytest-Ausgabe-Elternordner; der korrigierte Aufruf bestand.
Ein neuer Adapter-Nachcheck verwechselte zunächst Vertrags- und Datei-Hash;
das wurde mit eigener Regression korrigiert, der gültige Repeat wiederverwendet.

Hardware: drei neue Collectorstarts, davon zwei vollständige gültige 60-s-Repeats
mit 1,677943913 und 1,602798440 J/Task. Der dritte Versuch endet nach 92.672
lückenlosen Samples ohne Endpaket: letztes Datagramm 46,400 s nach GO, anschließend
5,098 s Stille; null Socketdrops, kein lokaler Cancel, sauberer Writerabschluss.
Quelle und DUT sind weiterhin erreichbar. Sender und Netzweg sind ursächlich
noch nicht getrennt; lokaler Paketmitschnitt und Systemjournal sind mangels
Berechtigung nicht verfügbar. Der dritte Repeat bleibt gesperrt. Kein Timeout-
oder Budgetanstieg, kein weiterer Versuch, kein Runtime-Rollout, kein Build.

Alle 202 gesicherten Remote-Runtimedateien sind nach den Messungen weiterhin
bytegleich. Drei gültige Vorher-Performance-Repeats und alle alten Fehlerbelege
bleiben erhalten. Belege: `/home/kmika/Reports/yolov7-full-fix-20261005/`, insbesondere
`energy/h8_before/resume_control/udp_analysis.json`, `ENERGY_REPEATS.csv`,
`validation/h8_energy_resume/final.xml` und `H8_ENERGIE_STATUS.md`.
Der ursprüngliche Gesamtauftrag ist wegen der offenen Vorherenergie unvollständig.

## YOLO-Min3 – selektive Qualityquelle und erster H8-Anschluss (03.10.2026)

Der erhaltene Fehlversuch übergab zwölf `quality_source=execute` neben zwölf
`generic=preserve`. Beide Auswahlstellen der tatsächlichen Remote-Suite
ignorierten die Qualitypflicht. Die Templatequelle lässt jetzt genau passende
Modell-/Setup-/Casepflichten für `composed` zu. Der gebundene Qualityaufruf nutzt
die vorhandene CPUreferenz und erzeugt Kandidatenrecords ohne Generic-Timing,
Streaming oder Build. Vollständig erledigte und fremde Fälle bleiben gesperrt;
die explizite Generic-Completion behält ihre ursprünglichen 100/1000/3.

Der konkrete zweite Wiedereintritt erkennt die noch nie publizierte Nativebasis
nur anhand des original gebundenen Fehlercheckpoints und derselben eingefrorenen
Kohorte. Fehlende vorhandene Messbelege bleiben Fehler. Die bestehende physische
Setupqueue hält weitere Qualityquellen bis zur normalen N5000/B1000-Auswertung,
Publication und Native-Bindung von H8/YOLO11l/b120 zurück. Die Native-Startbarriere
bleibt unverändert wirksam. Keine Reserveaktivierung oder neue Auswahl.

Softwarebelege und inkrementeller Sourcecheckpoint liegen extern unter
`$HOME/Reports/thesis20-yolo-min3-augmentation-20261002_222904/repairs/quality-selector-20261003_113308/`.
Die unveränderte produktive Suite reproduzierte zunächst den Fehler. Die
normale neu erzeugte Suite durchläuft beide Selektoren für alle zwölf Identitäten;
nur SDK-/Inferenzblätter werden in der Softwareprüfung ersetzt. Runtime- und
Schutzregressionen prüfen insbesondere null Generic-Zeitmessung und strikte
Request-/Artefaktbindungen. Das ist keine Hardware- oder normale GUI-Abnahme.

Ein einziger selektiver produktiver Anschluss ist nach abschließender Prüfung und
Installation autorisiert. Tatsächliche Qualityabschlüsse, Nativebindungen,
Messfortschritt und gegebenenfalls der konkrete neue Abbruch werden ausschließlich
im neuen externen Attempt dokumentiert. Keine automatische Fix-/Neustartschleife,
kein Commit/Push/Release; abgeschlossene Basis und 192er-Ergänzung bleiben erhalten.

## THESIS20 – gezielte Generic-Completed-Task-Ergänzung (02.10.2026)

Der ausdrücklich autorisierte Anschlussauftrag ergänzt exakt die 192 gebundenen
F04-Identitäten außerhalb des abgeschlossenen Primärruns. Der selektive Einstieg
`python -m onnx_splitpoint_tool.workflow.generic_completed_task` verwendet die
bestehenden SSH-/Owner-/Ressourcenlocks und den generierten Generic-P1/P2-Runner.
Je Fall drei frisch initialisierte Prozesse, je 100 vollständig abgeschlossene
Warmups und 1000 gemessene Taskabschlüsse. Timer: bereite Worker freigeben bis
letzter materialisierter Task; bestehende Handoff-/Queuepolitik bleibt erhalten.
Die ursprüngliche vorbereitete Native-Eingabe wird wiederholt; H8 FLOAT32-/UINT8-
und H10-Quantisierungsverträge werden unverändert berücksichtigt. Funktionale
Kontrolle und Dump-I/O liegen außerhalb des Timers, erforderliche Completion darin.

Plan, aktuelle Native-/Qualitybindung und kleine Originaloutputbelege sind für
alle 192 Fälle geprüft. Historische Rohoutput-FPS, neue Completed-FPS und gemessene
Native-FPS bleiben getrennt. Reporter erhält Accuracyverluste und Development-/
Screeningrollen; Ranggruppen verwenden den vorhandenen Ranker und Mindestumfang.
Kein neuer Qualitäts-, Native-, Energie- oder Compilerauftrag. Primärdaten und
vorbestehende Auswertungskorrekturen bleiben erhalten; keine Release-/Gitfreigabe.

Softwareabnahme: **108 PASS** in der abschließenden fokussierten Prüfung von Plan,
echtem generiertem Runner, selektivem Dispatcher, Writer/Parser und Reporter.
Zusätzlich zuvor **9 PASS** portable bestehende Completionregressionen. Ein früherer
breiterer Legacy-Testversuch scheiterte an 13 fehlenden historischen Fixturedateien;
diese wurden nicht erfunden. Doppelte/gesperrte Einstiege dürfen bestehende Reports
nicht verändern, Runtimeidentität und Ausgabepfade werden vor Dispatch geprüft.

Read-only-Geräteprüfung: alle drei Ownerprüfungen frei, je Setup 256 benötigte
gebundene Remote-Dateien vorhanden; keine bekannten Widersprüche der verfügbaren
historischen Power-/Clockkonfigurationen. Unbekannte historische Werte bleiben
unbekannt. Dies ist keine reale Inferenzabnahme. Die autorisierten sechs Startfälle
und der anschließende Rest laufen erst mit dem gesicherten Endstand; reale Belege,
Restbilanz und automatischer Abschlussbericht werden ausschließlich extern unter
`$HOME/Reports/thesis20-generic-completion-192-20261002/` fortgeschrieben.

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

### 2026-09-29 – THESIS20: begrenzte Produktkorrekturen und selektive Ergänzung

Ausgangscommit `182092216dfaf4f3ad36460a883098548c83bd8f`, Gitstatus vor Beginn
sauber. Maßgeblich ist der vollständig gelesene Vervollständigungsauftrag vom
29.09.; der vorhandene Run bleibt die Kampagne. Die lokale Projektion der
Originalpläne ergibt exakt 140 Grenzen, 399 vorhandene Genericresultate,
37 gespeicherte negative Buildfälle und 152 weitere Generic-Kandidaten.
414 abgeschlossene zentrale Qualityresultate einschließlich negativer
Accuracyentscheidungen bleiben erhalten; sechs Full-Companions fehlen.
Native umfasst 213 erhaltene Messungen, 228 technisch nicht unterstützte
Multi-Input-Fälle und 15 Split- plus sechs TensorRT-Full-Lücken. Kein Backfill.

**AP A/E – Software:** Manifestquellen lösen alle 280 regulären ONNX-Dateien
der 140 Originalgrenzen korrekt auf; Baselinefehler b064/b136 reproduziert.
Echte Konflikte und fehlende deklarierte Quellen bleiben Fehler. Die
398.315.499-Byte-Native-Summary wurde vollständig streamend in eine
53.509.218-Byte-ZIP-Datei exportiert, CRC/SHA geprüft; Python-Speichermaximum
2.376.974 Bytes. Originaldatei unverändert. Das vorhandene Replay-Metadatenlimit
steigt gezielt auf 512 MiB, ohne neuen Export-/Registrymechanismus.

**AP B – Software:** Der normale Nativepfad filtert anhand der tatsächlichen
Part2-Eingänge vor Voraussetzungen und Dispatch. Neue Projektionen trennen
228 `not_supported` von 21 ausführbaren Lücken. Alte Messzeilen behalten ihre
Version und Payloads. Kontrollierte Prozessblätter prüfen Wiederanwendung,
Zusammenführung und Full-Untermenge. Die reguläre Restage-Vertragsableitung
für TensorRT-Full-Hotloops verwendete irrtümlich Hailo-Rollen; der enge Fix
verwendet deren gebundene Eingabedateien. Alle 213 alten Energiequellen lassen
sich lokal ableiten, 563/563 benötigte lokale Artefaktpfade existieren. Das ist
keine Remote-Cachevalidierung und keine Hardwareabnahme.

**AP C – Software:** Der vollständige normale Energie-Plancheck auf den
Originalbelegen besteht offline für 213 Zeilen. Die 177 identischen
Provenienzfehler sind mit der Baseline reproduziert und durch konfliktgeprüfte
Propagation vorhandener Duplikate behoben. Identitäts-, Command-, Modell- und
Attestierungs-Negativkontrollen bleiben negativ. Der lokale Validator
veröffentlicht nach 953,74 s mit rc=0 einen vollständigen 462-Zeilen-Bericht;
sämtliche 213 vorhandenen Messoutputs bestehen Tensor-/Strukturprüfung und sind
technisch ausführbar. 207 haben exakt zugeordnete zentrale Qualityevidenz;
sechs vorhandenen Vendor-Full-YOLO26-Zeilen fehlt diese bis zur Genericergänzung.
Zwei dieser sechs zeigen lokale semantische Fehler. Diese Befunde bleiben
erhalten; daraus wird keine neue Performancemessung abgeleitet. Die ursprünglichen
169 Teilberichte und alten Kennzeichnungen wurden nicht ersetzt.

**AP D – Software integriert:** Neuer CLI-Einstieg
`onnx-splitpoint-complete-evaluation` führt über den vorhandenen Workflow.
Er übernimmt unveränderte archivierte Optionen und prüft den bisherigen
Resumevertrag. Pläne und Status werden vor neuen Projektionen gezielt in
Sitzungshistorie kopiert. Genericdispatches betreffen nur offene Identitäten;
neue Raw-Ergebnisse erhalten eigene Attempts. Der reguläre Normalizer/Reporter
führt alte und neue Zeilen samt ursprünglichen Quellen zusammen. Geprüfte
kompatible Artefakte lösen keinen Compilerdispatch aus. Bereits normalisierte
Messungen und anfragenlokale fertige Qualityresultate überleben einen Abbruch
vor Veröffentlichung des Gesamtberichts. Die normale Zulassung akzeptiert die
3728 bestehenden Hailo-Publikationslinks strukturell, ohne HEF-Inhalte erneut
zu hashen; unsichere Links und veränderte Verträge bleiben abgelehnt.

Energie verwendet die vorhandenen Rowjournale und Kampagnenbudgets für komplette
Zeilen sowie verifizierte einzelne Replikate. Kontrollierte normale CLI-Prüfung:
3→4 Planzeilen, eine wiederverwendet/drei ausgeführt; zweite Anwendung alle vier
wiederverwendet/null Starts. Teilreplikate: eines erhalten/zwei ergänzt, danach
null Starts. Der normale Wiederherstellungspfad umfasst auch den Ergänzungsmodus.
Der zweite vollständige Offline-Energiecheck mit abgeschlossenem Validator und
Original-`--previous-results` besteht: 213/213 Verträge, rc=0, 97,27 Sekunden,
keine Wrapperstarts. Die vollständige normale Qualityqueue auf dem Originalrun
plant exakt null neue Aufträge für die 414 abgeschlossenen Resultate.

Konsolidierte lokale Integrationsprüfung: **104 PASS**, zwei Dependencywarnungen,
12,26 Sekunden. Neue A–D-Regressionen, normaler Generic-/Native-/Energiepfad,
Reentry-/Zulassungsnegativkontrollen und bestehende Restage-/DeepX-Regressionen.
Separat Energie/Validator/Budgets: **187 PASS**, drei auf unveränderter Baseline
reproduzierte alte v2727-Testfehler deselektiert. Native-Statusregressionen:
78 PASS; Native-Voraussetzungen 102 PASS. Eine frühere breitere Nativegruppe
hatte 107 PASS/2 alte Fehler (historische Versionsliste und Full-Imagemap-
Testerwartung); diese wurden nicht durch Produktänderungen kaschiert.
Die Testgruppen überlappen; ihre Zahlen sind keine additive Gesamtsuite.
Ausführliche Befehle und Resultate liegen in den lokalen Paketberichten unter
`/tmp/thesis20-local-validation`, `/tmp/thesis20-native-package-report.json`
und `/tmp/thesis20-final-integration.xml`.

Abschließende strikte Native-Energie-Reuseprüfung: 3 PASS; null neue Starts,
vier wiederverwendete Zeilen und kein fälschlicher Strict-Fehler. Normale
Editable-Installation auf 2.91.1 aktualisiert, 210 CLI-Einstiege, Abhängigkeiten
unverändert. Versions-/Mirror-/lokale Smokeprüfung: 4 PASS in 1,56 Sekunden.
Source-Release im getrennten `/tmp`-Quellstaging und Hauptinstallation geprüft:
2091 Quelldateien, acht neue, keine entfernten; 43 Nutzerprofile bleiben lokal.
Der Releasebuilder wurde nicht durch Entfernen lokaler Log-/Venv-Symlinks
bedient: der erste direkte Release-Build im Installationsbaum lehnte diese
korrekt ab; die normale Installed-Prüfung akzeptiert die geschützten Laufzeit-
wurzeln. Source-Manifest und SHA256SUMS gehören zur bestehenden Releasepflege.

**Hardware/GUI:** Null SSH-/Compiler-/Collector-/Workloadstarts. Kein kompletter
Benchmarklauf, keine G1/G2-Runde, keine neue Energie- oder Qualitykampagne.
Reale normale GUI-Hardwareabnahme ausdrücklich nicht durchgeführt. Alle lokalen
Replay-/pytest-/Exportausgaben liegen unter `/tmp`, nicht im Quellbaum oder im
Originalrun. Patchversion 2.91.1 wird lokal vorbereitet; bestehende Tags bleiben
unverändert, keine Veröffentlichung. Verbleibend ist ausschließlich die spätere
freigegebene Ergänzung: 152 Generic-Kandidaten, sechs Full-Quality-Companions,
15 Native-Splits/sechs TensorRT-Full sowie bis zu 234 Energiequellen. Keine
Garantie auf Qualitäts-PASS, keine neue G1/G2-Freigaberunde. Konkreter CLI-Befehl
und lesbare Bilanz stehen in `docs/RELEASE_2.91.1.md`.

### 2026-09-29 – THESIS20: Preflight-Vertragsfehler und zweiter Eintritt

Auftrag `AUFTRAG_THESIS20_RESUME_PREFLIGHT_FIX_20260929.md` vollständig gelesen.
Ausgangscommit weiterhin `182092216dfaf4f3ad36460a883098548c83bd8f`; alle
vorhandenen uncommitteten 2.91.1-Änderungen bleiben erhalten. Der gescheiterte
produktive Ergänzungsversuch bleibt als Historie bestehen.

**Ursache und Fix:** Je YOLO26m/s 16 geänderte Metadatendateien, keine fehlende
Indexbindung. Echte Writer im frischen Preflight ändern Zeitstempel und den
DeepX-Probeoverride; die anschließende Build-Reuseprüfung verlangte trotzdem
die alten Bytes. Für zwölf Dateien je Modell liegen gebundene Originalkopien
vor, vier fehlten im alten Sicherungsglob. Für alle 32 Dateien ist die alte
SHA256-/Größenbindung mit ausschließlich diesen Änderungen reproduzierbar.
Die Ergänzung prüft jetzt genau diese Writerfelder gegen den ersten erhaltenen
Originalindex. Alle übrigen Inhalte bleiben strikt gebunden; keine pauschale
JSON-Freigabe, kein Neuschreiben von Receipts, keine Originalwiederherstellung.
Neue Sicherungen erfassen die bisher ausgelassenen Metadaten; Sitzungen halten
ihren ursprünglichen Quellenpfad fest. Fehlende frühere Belege dürfen nicht
durch einen neueren Snapshot ersetzt werden.

Zusätzliche Prüfungen erhalten ursprüngliche Resultatidentitäten und verhindern
Nachmessungen aufgrund unvollständiger Ergebnisprojektionen. Neue gültige
Ergebnisse bleiben ergänzbar. Die vorhandenen Logs erreichen die CLI sofort.
Messzeiträume bleiben historisch, neue Reuse-/Attemptzeiten werden separat
ausgewiesen; bereits vermischte Jobzeiten werden aus Original-StageResults
korrekt dargestellt.

**Verwaltungsprofil:** Wiederholte Registrierung kleiner Jobprojektionen
prüfte jedes Mal sämtliche 257.313 Indexpfade. Tatsächliche Registrierung
125,461 s ohne Profiler; derselbe normale isolierte Reuse nach Wegfall der
redundanten Registrierung 0,691/0,624 s. Nachfolgende Quellenänderung wird
weiterhin abgelehnt. Es gibt keinen neuen In-Memory-/Persistenzcache;
Inhaltsprüfungen und normale terminale Inventarisierung bleiben erhalten.

**Erhalt und Restplan:** Die fünf maßgeblichen Ergebnisaggregate stimmen mit
der ersten Sicherung in SHA256 und Größe überein. Einzelidentitäten und
Zeilenbelege bestätigen 399 Generic, 414 Quality einschließlich 27
`accuracy_loss`, 213 Native und 37 negative Builds. Der aktuelle Restplan
entspricht exakt dem ersten Supplementplan: 152 Generic, sechs
Quality-Companions und 21 Native offen, 228 Native-Multi-Input nicht anwendbar.
Die vorhandene Energiedatei ist eine blockierte Projektion mit null Starts.

**Softwareprüfung:** Abschließender fokussierter Lauf 176 PASS in 16,57 s,
zwei vorhandene Dependencywarnungen. Inklusive echter Stage-/Preflightwriter/
Build-Reuse-/Dispatcherübergänge, Reentry und Negativkontrollen. Zusätzlich
alle 83 tatsächlich gespeicherten Stufenentscheidungen lesend geprüft:
73 Wiederverwendungen, zehn ausdrücklich vorgesehene Ergänzungs-/Projektions-
stufen, kein Vertragsfehler. Vorher scheitern beide Originalmetadaten-Replays;
nachher und nach Preflight-Abbruch werden jeweils exakt 158 offene Identitäten
in sechs kontrollierten Dispatcheraufrufen ausgewählt. Zehn Negativkontrollen
an echten Metadaten werden abgewiesen. Eine verkürzte Native-Summary bei
unverändertem Matrixstatus sowie fehlende oder inhaltsveränderte Original-
Messberichte werden vor erneuter Disposition gesperrt. Archivierte Original-
Metadaten und benötigte Messberichte werden gegen ihre bestehenden Bindungen
geprüft, einmal bei Eintritt; keine neue Vollbaumprüfung. Die Nativeprüfung
benötigte lokal 35,889 s, die separate Genericprüfung mit Indexladen 19,802 s.
Historische Remoteberichtlabels verwenden ausschließlich die vorhandene
lokale Reporter-/Transferzuordnung. Die abschließenden Replay-, Installations- und
Integritätsergebnisse stehen im dauerhaften privaten Abschlussbericht.

**Hardware/GUI getrennt:** Keine produktive Ergänzung, kein SSH-, Compiler-,
Collector- oder Workloadstart. Keine neue Hardwaremessung, Normal-GUI-
Hardwareabnahme, vollständige Offline-Qualityrechnung oder G1/G2-Kampagne.
Alle Prüfungen arbeiten lokal beziehungsweise mit kontrollierten äußeren
Blättern. Fortgesetzt wird später derselbe Run mit dem vorhandenen
`onnx-splitpoint-complete-evaluation --run-dir …`-Einstieg aus den Release Notes.

### 2026-09-30 – THESIS20 Restabschluss, AP E: Restplan und Attemptanzeige

Auftrag `AUFTRAG_THESIS20_RESTABSCHLUSS_20260930.md` vollständig gelesen;
Ausgangscommit `182092216dfaf4f3ad36460a883098548c83bd8f`, bestehende uncommittete
2.91.1-Arbeit erhalten. Dauerhafte Quellsicherung und Abnahmebelege unter
`$HOME/Reports/thesis20-restabschluss-20260930`.

Der normale lokale Plan liest die ursprünglichen Splitmanifest-Ausschlüsse:
588 = 527 erhaltene Genericzeilen + 37 gespeicherte negative Builds + 24
`hailo_yolo26_static_part1_guard`-Policyfälle. Keine Generic-Neudisposition,
kein Backfill, Part2-only bleibt getrennt. 479 abgeschlossene Qualityresultate
werden erhalten; die 69 fehlgeschlagenen Requests mit vorhandenen Vorhersagen
sind lokale Statistikarbeit, auch die drei betroffenen Full-Companions.
Native: 222 erhalten, 228 nicht unterstützt, zwölf unterstützte Lücken.

CLI/Stufenbericht, Dashboard und wissenschaftlicher Bericht lesen den Fehler
des maßgeblichen Workflowattempts: 222 geprüft, 142 PASS/80 FAIL, darunter
59 CMake-Lücken, 14 Hailo-Receipt- und sieben DeepX-Manifestabweichungen.
Evidenzpfad und Startzähler werden angezeigt; die historische 213er-Projektion
wird weder überschrieben noch als aktueller Messversuch ausgegeben.

Software: 15 vorhandene fokussierte Workflow-/Dispatchtests PASS und vier neue
Restplan-/Dashboardregressionen PASS. Originaldaten-Plancheck zeigt die obigen
Einzelidentitäten. Keine GUI-Hardwareabnahme oder Messung in diesem Paket.
Weitere AP-Abnahmen und gezielte Installation folgen im selben Auftrag.

**AP E, normaler Wiedereintritt:** Der echte CLI-Einstieg deckte zusätzlich
auf, dass reguläre Hailo-Publikationslinks im neuen
`native_producers/attempts/<session>/<backend>/<model>/benchmark_set`-Layout
nicht zugelassen wurden. Der enge Pfadzweig verwendet unverändert den
vorhandenen Bundleverifier; keine pauschale Symlinkfreigabe. Echter Publisher,
Attemptkopie und manipulierter Zeiger sind geprüft. Ebenso geprüft ist der
normale Native→Energie-Stufenübergang mit kontrollierter Prozessgrenze:
aktueller Preflightfehler, null neue Performancearbeit, historische Ergebnis-
bytes unverändert, alte Ergebnisse getrennt von aktuellem Fehlerpfad.
Fokussiert acht Tests PASS; kombinierte neue Übergänge zunächst 24 PASS.

**AP A – echte lokale Qualityabnahme:** Der Referenzjob wurde vor dem
Harnessrefresh geplant, nach Budgetzulassung las er einen Mischstand mit
sieben neuen und 13 alten Runnern. Rekonstruierte Schedulebindung
`131fbf1ec68ed489f28bb4245d926d16cf144d606c3a0a17011a788249354fac`,
gespeicherter Referenzstatus
`84a335f9cedd7558169f8efa066031a233b587ebefe6bc3212fc81987a00a6f9`.
Der historische RAMwert war nicht persistiert; die Schedulebindung ist aus
Originalquellen rekonstruiert. Alter/neuer Runner unterscheiden sich nur in
der Release-Identitätszeile; CPU verwendet b003, bereits im neuen Anteil.
Der Generatorstatus bindet diesen b003-Runner und seine Referenzrecords.

Refresh liegt jetzt vor Schedule in allen Setupmodi; Admission prüft die
Quellenbindung erneut. Reentry stellt die Zuordnung nur aus exakt belegter
Originalquelle her. Der wiederhergestellte Referenzworkspace ist symlinkfrei
und besteht die normale Run-Pfadzulassung. Drei originale YOLO26s Vendor-Full-
Requests wurden mit gespeichertem CPU-/Kandidatenbestand, N5000/B1000,
Seed20260710, vier Workern und einem aktiven Request berechnet: H8 und H10
`accuracy_loss`, DeepX `reference_close`; alle technisch `completed`.
Laufzeiten nach Intake 193,69/87,85/96,57 Sekunden. Originalfehlresultate,
Referenzstatus und Originalaggregate unverändert. Bestand jetzt 479 erhalten
plus drei neue = 482 abgeschlossen, davon 34 Accuracyverluste; 66 Requests
bleiben lokale Statistikarbeit. Abgeleitete Summary im bestehenden
`reports/attempts/quality_local_20260930`-Reporterpfad.

**Installation, Software:** Controller per vorhandenem stdlib-only PEP376-
Editable-Refresh unter exklusivem Plattforminterlock auf 2.91.2 aktualisiert;
210 Konsoleneinstiege, Abhängigkeiten erhalten. Erster Refreshaufruf wurde
vor Änderung wegen fehlender `--require-entrypoint`-Deklaration des angeforderten
Smoke-Einstiegs abgelehnt; korrigierter vollständiger Aufruf rc=0.
Konsolidierte lokale Regression: 325 PASS, zwei vorhandene Dependencywarnungen,
39,37 Sekunden. Keine produktive Messung oder normale GUI-Hardwareabnahme.

**AP B/C – Native Full und lokale Nachbindung:** Kampagnen-ID und physischer
Attemptroot werden getrennt aus den Dispatcherargumenten geprüft; Kampagnenroot
bleibt kompatibel. Beide Runtimepfade (TRT/Vendor) und Ressourcenmirrors sind
synchron. Der normale Writer→Transferpayload→Runtime-Verbraucher ist auch mit
einem anderen fehlerhaften Modell geprüft. Ein neuer Fullrunner erhält beim
späteren Dispatch einen eigenen Dateinamen pro Attempt im vorhandenen scripts-
Verzeichnis; alte Energieverträge behalten ihre globale ursprüngliche Runnerdatei.
Heute kein Remote-Runner-Update.

Die Vendor-Metadatenarbeit hängt nicht mehr von neuen Performancejobs ab.
Sechs echte Vendor-Full-Originalmessungen wurden durch den normalen lokalen
Validator-main und bestehende zentrale Writer/Loader mit der abgeleiteten
Qualitysummary nachgebunden: Erst- und Zweiteintritt jeweils rc=0, sechs
`exact_identity_match`. YOLO26s H8/H10 bleiben `semantic_ok=false`.
Ursprüngliche Native-Summary (222 Messungen), Fullcommands und Validierungs-
belege bleiben unverändert. Die abgeleiteten Ergebnisse liegen im vorhandenen
`reports/native_validation/attempts/local_binding_20260930`-Reporterpfad.
Externe Inferenzblätter wurden gesperrt; ursprüngliche semantische Reports
verwendet. 21 Negativkontrollen an echten Belegen bestehen (Run/Setup/Modell/
Input/Source-Modell/kompiliertes Artefakt/Claimmirror, jeweils drei Backends).
Final fokussiert 71 Tests PASS; ein alter Taskbudget-/FS-Testfehler wurde auf
dem vollständigen Ausgangscheckpoint reproduziert und getrennt dokumentiert.

**AP D – reale Artefakt-only-Energieabnahme, keine Messung:** Der bestehende
Restagepfad wählte nur Mindestrollen aus, während Split-/Full-Verbraucher die
vollständigen gebundenen Artefaktmengen prüfen. Er umfasst jetzt alle zusätzlich
benötigten runlokalen Dateien sowie separat gebundene Boundarymetadaten;
Installationscode, Interpreter und externe Geräte-Caches bleiben ausgeschlossen.
Archivierte Native-Binaries erhalten beim vorhandenen atomaren Upload das
erforderliche Ausführungsrecht. Kein neuer Restager und kein Compilerlauf.

Je ein realer Originalfall der drei Fehlerklassen wurde vor Wiederbereitstellung
mit dem normalen Gerätepreflight reproduziert. Die 14 Hailo-Receipt- und sieben
DeepX-Manifestmeldungen `sha256_mismatch` waren tatsächlich fehlende Dateien:
Der vorhandene Fullguard benutzt denselben Fehlertext für Abwesenheit und
Byteabweichung. Alle Originalbytes lagen exakt im lokalen Nativearchiv vor;
keine Sollprüfsumme wurde verändert, keine Serialisierung rekonstruiert.

Auf .102/.104/.145 wurden nach read-only Eigentümerprüfung über den bestehenden
konfliktgeprüften Pfad 216 fehlende Dateien ergänzt: 59 CMake-Dateien,
59 erzeugte C++-Quellen, 56 erhaltene Native-Binaries, 14 Hailo-Receipts,
sieben Benchmarkmanifeste und 21 Original-ONNX-Dateien. Insgesamt
1.618.872.882 Bytes; null Ersetzungen vorhandener Dateien/Backups. Die tatsächliche
Artefakt-only-Vorprüfung bestand danach mit **222 PASS / 0 FAIL**.

Der zweite normale Eintritt bestätigt 608 exakte Remote-Dateibindungen,
null erneute Übertragungen/Restages und nochmals **222 PASS / 0 FAIL**.
Zusätzlich wurden die Ausführungsrechte aller 56 Native-Binaries real geprüft.
Beide Eintritte erhalten fallgenau dieselben 222 ursprünglichen Vertrags-
identitäten und unveränderten Command-Dateibytes; sämtliche wissenschaftlichen
und Messparameter bleiben unverändert. Wrapper-, Collector-, Workload-,
Compiler- und neue Geräteinferenzstarts: null. Die Messquelle .176 wurde nicht
berührt; keine produktive Messung oder normale GUI-Hardwareabnahme.

Software: konsolidiert 120 PASS, drei ausgeschlossene historische Tests,
3,83 Sekunden. Die drei alten Quellrekonstruktionsfehler sind auf dem initialen
Sourcecheckpoint reproduziert (15 PASS/3 FAIL in der ergänzenden Suite):
erwartete alte CPP-SHA `99dfa223…`, aktueller Generator `bff08d6f…`.
THESIS20 benötigt diesen Fallback nicht; seine exakten archivierten Bytes
wurden verwendet. Die unabhängige fokussierte Restage-/Preflightgruppe besteht
mit 105 PASS. Diese Gruppen überlappen.

Dauerhafte Belege: `$HOME/Reports/thesis20-restabschluss-20260930/energy/`
(`AP_D_report.json`, Originalfehler, Soll-/Istbelege, Tests, beide Eintritte).
Der zweite Eintritt und die zugehörigen abgeleiteten Berichte liegen außerdem
im ursprünglichen Run unter
`reports/native_energy_measurements/attempts/thesis20_artifact_only_20260930_second`.
Er ist ausdrücklich Artefakt-only-Abnahme und keine produktive Workflowexecution;
der ursprüngliche gescheiterte Attempt und alte Energieberichte bleiben erhalten.
Die 222 technisch geprüften Fälle sind eine künftige erste Energieaufnahme,
keine Wiederholung bereits gemessener Energie. Nach den zwölf fehlenden
Native-Performancefällen sind bis zu 234 Energiezeilen möglich; frische normale
Vorprüfungen bleiben vor dem späteren Messstart erforderlich.

**Gesamtabgleich und Lieferung 2.91.2:** Der installierte normale CLI-Einstieg
`onnx-splitpoint-complete-evaluation --run-dir … --plan-only` besteht. Der
fallgenaue Abschlussabgleich bestätigt 527 Genericidentitäten, 479 ursprüngliche
Qualityresultate (447 reference_close/32 accuracy_loss) und 222 unveränderte
Native-Messungen einschließlich der neun neuen YOLO26m-Splits. Sechs Original-
aggregate beziehungsweise Fehlbelege sind byteunverändert. 482 Qualityresultate
sind einschließlich der drei lokalen Ergänzungen fertig, 66 bleiben offen.
Die drei offenen TensorRT-Full-Companions gehören zu diesen 66; sie sind andere
Identitäten als die fertiggestellten Vendor Fulls. Keine Quellenänderung war
für diesen abschließenden Identitätsabgleich nötig.

Finale A-Generatorprüfung 12 PASS, B/C 71 PASS, D 120 PASS; die Gruppen
überlappen mit der konsolidierten 325er-Regression. Ausführlicher Bericht,
Restplan, Ausgangssicherung und finaler Sourcecheckpoint liegen dauerhaft unter
`$HOME/Reports/thesis20-restabschluss-20260930`. Alle Test-/Reportausgaben
außerhalb des Quellbaums. Kein Commit/Push; bestehende Nutzeränderungen erhalten.
Software und echte Artefakt-only-Abnahme sind dokumentiert; produktive normale
GUI-Abnahme und Messung bleiben ausdrücklich aus. STOP vor dem späteren
bewussten Fortsetzungsstart desselben Runs.

## THESIS20 2.91.2 – Generierungs-Reentry, 30.09.2026

**Ursache und enger Produktfix:** Der neue Fehlversuch 8c017… scheiterte vor dem
Cache-Preflight an fünf `content_changed` bei YOLO26m; YOLO26s hat denselben
belegten Zustand. Die erste Sicherung 6411… ist korrekt ausgewählt. Vier
Plan-/Suitealiase wurden jedoch im bereits ausgeführten Ergänzungsversuch
36c943… durch physische Scopeprojektion und Runtimefinalisierung fortgeschrieben.
Der ausführbare Python-Diff ergänzt den selektiven Dispatch und entspricht
exakt dem installierten Template. Vollständige JSON-Felddiffs, Python-Diffs,
Writerchronologie und exakt gebundene Originalkopien sind im Bericht gesichert.

Der vorhandene Supplementguard erlaubt ausschließlich vollständig reproduzierte
Writeroutputs: ursprüngliche JSONbytes und globaler Scope bleiben original
gebunden; die bestehende Projektion muss die aktuellen Bytes exakt ergeben.
Python muss den bestehenden aktuellen Index und den kanonischen Writeroutput
erfüllen. Die erste Sicherungsbasis bleibt unverändert; keine Rücksetzung von
Arbeitsdateien, keine Neuetikettierung alter Messquellen. Künftige normale
Metadatensicherungen nehmen den Harness mit auf. Die bereits restaurierte
YOLO26s-CPU-Referenz und ihr Workspace werden nicht verändert.

**Zusätzlicher belegt notwendiger Reentryfix:** Zwei echte kleine
YOLO26-Hardware-Smoke-Artefaktsätze reproduzieren nach erstem erfolgreichen
Reuse und Abbruch einen zweiten Fehler: Die gebundene alte
`resume_decision.json` wurde vom Reusewriter überschrieben, während ein
abgebrochener Workflow keine terminale Indexfinalisierung ausführt. Neue
Supplement-Reuseentscheidungen werden jetzt im bestehenden aktuellen
Attemptverzeichnis geschrieben. Die frühere Entscheidung, ihr Indexeintrag
und das abgeschlossene Stage-Ergebnis bleiben unverändert. Dies erfordert
keine neue Freigabe oder nachträgliche Neuberechnung erwarteter Prüfsummen.

**Software:** 26 gezielte neue Regressionen PASS, gesamte betroffene
2.91.1-/2.91.2-Gruppe 190 PASS / zwei Dependencywarnungen; Gruppen
überlappen. Echte Writer und normaler Stage-Wiedereintritt sind enthalten.
Auch gemeinsam mit aktuellen Indexeinträgen veränderte Modelle, Auswahl,
Precision, Datensätze, Endpunkte, Budgets und Pythoncode werden abgelehnt.

**Zusammenhängende lokale Startkette:** Der normale installierte CLI-Einstieg
wird in einem isolierten Dateisystem mit identischen Originalpfaden geprüft;
große Originaldaten und historische Attempts sind nur lesend eingebunden.
Guard, Basiswahl, Writer und Modellpipeline bleiben unverändert. Nur äußere
Statistikausführung wird am Dispatch kontrolliert gestoppt; Netzwerk und
Hardwarestarts sind gesperrt. Vorher-/Nachher-/Wiedereintrittsprotokolle und
der endgültige Abnahmestatus werden nach der finalen Quellinstallation unter
`$HOME/Reports/thesis20-generate-reentry-20260930/ABSCHLUSSBERICHT.txt`
festgehalten. Ein Planaufruf oder die 190 Softwaretests ersetzen diese Abnahme
nicht.

**Produktionsgrenze:** 527 Genericzeilen, 482 abgeschlossene Qualityresultate,
222 Native-Messungen erhalten; Rest 66 lokale Statistikrequests und zwölf
Native-Performancefälle. Die kontrollierte lokale Abnahme führt keine neue
Generic-Inferenz, Kompilierung, Native-/Energiemessung oder normale
GUI-Hardwareabnahme aus. Die frühere zweimalige
222er-Artefakt-only-Abnahme bleibt ein eigener Nachweis. Keine Änderung der
43 Nutzerprofile, Venv, Geräteinstallation oder wissenschaftlichen Kriterien;
kein Commit, Push oder Tagwechsel.

**Autorisierter Produktionsstart:** Der ausdrückliche Nutzernachtrag vom 30.09.
ersetzt den bisherigen Startstopp und ist im lokalen Arbeitsauftrag festgehalten.
Nach erfolgreicher finaler Installation und zusammenhängender Startkettenprüfung
einschließlich Wiedereintritt wird genau eine Fortsetzung desselben Runs mit dem
installierten CLI gestartet. Vorher gelten die bestehenden Prozess-/Ownerprüfungen;
eine vorhandene passende Fortsetzung darf nicht verdoppelt werden. Nur offene
Statistikrequests, höchstens zwölf Nativefälle und danach die erstmalige
Energieaufnahme der ausführbaren Planzeilen sind freigegeben. Zwischenzeitlich
fertige Arbeit wird übernommen; Auswahl, Messumfang und Retrybudget bleiben
unverändert. Sitzung/PID, neues Konsolenlog, späterer Exitcode und der lesend
belegte Fortschritt werden im externen Abschlussbericht gesichert. Kein Warten
auf Kampagnenende, keine Änderungen oder parallelen Gerätetests während des Laufs,
keine automatische Fix-/Neustartschleife. Ein Prozessstart ist kein Mess-PASS.

### 01.10.2026 — Energie-Previous-Results nach abgeschlossenem Ergänzungslauf

Der vorherige produktive Lauf hat 548/548 Qualityresultate technisch beendet
(509 reference_close,39 accuracy_loss) sowie 234 erfolgreiche Native-
Performancefälle und 228 not_supported bilanziert; failed/missing jeweils0.
527 Genericzeilen bleiben erhalten. Der aktuelle Energieplan umfasst234 Zeilen,
vor diesem Reparaturpaket waren sämtliche Energiestartzähler0.

Belegte Ursache: Der ausgewählte alte213er-Bericht aus dem Originalsnapshot
des Attempts ee04… bindet kanonische lokale Contractpfade. Der Supplement-
Publikationswriter hatte aber plan/* eines fehlgeschlagenen früheren Attempts
36c943… auf diese Aliase kopiert; dessen Preflights behalten die ursprünglichen
Attemptpfade. Die historischen Skriptbytes stimmen exakt mit der vorhandenen
Originalindexbindung überein, die technischen Contractbytes bleiben identisch.
Aktuelle Aliasbytes stammen bereits vom späteren ee04-Fehlversuch und wurden
nicht als ursprüngliche Fehlerbytes ausgegeben.

Enger Fix im vorhandenen Energie-Previous-Results-Pfad: Nur vollständig
nachgewiesene, vor dem Start blockierte Kohorten benötigen keinen Vergleich
mit den veränderlichen alten ausführbaren Planaliasen. Vier exakte Nullzähler,
Ergebnis-/Ledgerkohorte, abwesende Ausgaben und fehlende Budgetketten werden
geprüft. Vorhandene Rowjournale werden über ihre bestehenden Digest-/Vertrags-
prüfungen gelesen, ohne das Manifest neu zu schreiben. Autoritative gestartete
Rows, Messungen und Teilreplikate verhindern diese Ausnahme. Technische und
Campaignverträge, Qualityprovenienz, Ausführungsumgebung, vollständige
Energieidentitäten und aktuelle Commandbindungen bleiben strikt.

Softwareabnahme: Der reale213→234-Übergang scheitert mit dem gesicherten
Ausgangscode am gemeldeten lokalen Contractbinding und besteht mit dem Fix.
Neue Schutzregressionen sowie bestehende Energie-Resume-/Provenienztests und
Atomic-Checkpointtests PASS; exakte Ergebnisse im externen Bericht.
Die finale integrierte Produktabnahme wird nach der letzten Installation am
normalen CLI mit isolierter Originalpfadabbildung durchgeführt: vorhandene
Stufen übernehmen,234er-Plan, Previous Results, echte Artefakt-/Owner-/Cohort-
preflights und kontrollierter äußerer erster Energiemessstart. Sie ist von
Softwaretests, normaler GUI-Abnahme und produktiver Aufnahme getrennt.

Nutzerfreigabe: Nach erfolgreicher Installation/Abnahme einmalig denselben
Run produktiv in tmux fortsetzen, neue Konsole und Exitcode unter Reports.
Keine erneute Generic-/Quality-/Nativearbeit, keine Änderung von Auswahl,
FS/command60s×3 oder Retrybudgets. Während Produktion keine Code-/Profil-/
Cacheänderungen und keine parallelen Gerätetests. Kein automatischer Neustart.
Erster tatsächlicher Energiefortschritt, Sitzung/PID, Erhaltungsbestand und
etwaige Grenzen werden rein lesend dokumentiert unter
$HOME/Reports/thesis20-energy-reentry-20261001/ABSCHLUSSBERICHT.txt.
Kein Commit/Push, keine SDK-/Venv- oder Nutzerprofiländerungen.

### 02.10.2026 — THESIS20: lokale Auswertungs- und Projektionskorrekturen

Der produktive Lauf ist abgeschlossen. Maßgeblicher Anschlussauftrag und
Einzelbelege: `$HOME/Reports/THESIS20_ERGEBNISAUDIT_20261002`.
Die frühere Startfreigabe gilt nicht für diesen Anschlussauftrag: keine
Workflowfortsetzung, Inferenz, Compiler-, Performance- oder Energieaufnahme.
Originalmessungen und historische Fehlversuche bleiben unverändert; neue
Berichte liegen ausschließlich unter
`$HOME/Reports/thesis20-auswertung-20261002_110429`.

F01: 47 Hailo10→TRT-Qualitätszeilen werden vor der TensorRT-Unterstringprüfung
der richtigen Producerfamilie zugeordnet. 168 echte TRT-Zeilen des normalen
Reporters einschließlich der 21 Full-Companions bleiben unverändert. F06:
Die vorhandenen Entscheidungen
reference_close/accuracy_loss werden regulär projiziert. Exakte ursprüngliche
Compile-/Part1-Policybelege ergeben 588 Sollfälle = 527 Messfälle + 37 Buildfehler
und 24 Policyausschlüsse, 0 offene Qualityfälle und 29 bisher fehlende H10-
Terminalprojektionen. Keine künstlichen Messwerte oder erfolgreichen Qualityjobs.

F02: Der Finalreport unterschied eine alte ungültige Authority nicht korrekt
von einer echten Run-ID-Abweichung und behielt die alte negative Entscheidung
nach Aktualisierung der Authority bei. Die neue Projektion prüft die bestehenden
Command-/Consumer-/Request-/Manifest-/Payloadbindungen erneut; 186 alte
Ablehnungen werden nachvollziehbar ersetzt, ihre vorherige Projektion bleibt
sichtbar. F03: Die sechs vorhandenen TRT-Full-Outputs liegen im regulären
Supplementverzeichnis außerhalb des alten literal benchmark_set-Pfads.
Der lokale Validator kann sie jetzt über `--existing-outputs-only` gegen exakt
gebundene gespeicherte Full-Referenzen prüfen, ohne ORT-/Probe-Fallback.
Originalpolicy, Modell-, Input-, Completion- und Vergleichsverträge bleiben
verbindlich. Beide YOLO26s-Vendor-Full-Negative bleiben negative Ergebnisse.

F05: Energieexport aus den 234 aufgenommenen Native-Beobachtungen statt aus
527 energielosen Genericzeilen. Kalibrierte FS-Eingangsenergie ohne Idleabzug
bleibt primär; 21 TRT-Full-Zusatznormalisierungen stehen separat. Aufgenommen,
gültig, deskriptiv paarbar, semantisch passend und wissenschaftlich freigegeben
werden getrennt gezählt. 384 deskriptive Quotienten setzen passende physische
und zeitliche Messbereiche voraus; kein bestehendes Claim-Gate wird gelockert.
Development-/Unknown-/Screeningrollen werden mit Originalquellen ausgewiesen.
Die acht DeepX-Vertragsunterschiede bleiben ausdrücklich dokumentiert.

F04/W01–W03: Alle 527 ursprünglichen Generic-Per-Case-Berichte wurden gelesen.
499 Splits messen Logits/P2, keine vollständige Taskzeit; 28 Fulls besitzen
Completion. Ergänzende Gleichendpunktmessungen wären für die bestehende
192er-Native-Schnittmenge erforderlich, sind lediglich als bedingte Fallliste
benannt und nicht freigegeben. Keine Erweiterung der 20 stratifizierten Grenzen.
YOLOv7 bleibt mit einer Grenze je Setup für Rangkorrelationen zu klein.
MobileNet/DeepX b043 bleibt ein belegter Accuracyverlust. Alle Wiederholungen
der fünf Native- und drei Energiewarnungen bleiben erhalten; 5 % CV ist keine
neue Policy. Kein konkret belegter Fehler rechtfertigt hier einen automatischen
Retry oder das Entfernen ungünstiger Wiederholungen.

Softwareprüfung: F01/F06 abschließend 67 PASS; F02/F03 104 PASS und elf reale lokale
Negativmutationen; F05 68 PASS; F04 drei PASS. Unabhängige Reviews bestanden.
Neun historische Cross-Runner-Fixturefehler und ein alter Collectionfehler
wurden mit dem gesicherten Ausgangscode identisch reproduziert und nicht
außerhalb des Auftrags repariert. Die normalen Validator-/Finalreport-CLIs
und die vorhandenen Metadatenbuilder laufen erfolgreich mit schreibgeschütztem
Primärbestand. Abschließender normaler ScientificReplay in
`scientific-corrected-v2`: Exitcode 0, alle integrierten Fallabnahmen PASS.
Das Quality-Topfeld ist konsistent evaluated, die wissenschaftliche Freigabe
bleibt blockiert. Energie: 234 aufgenommen/gültig, 702 gültige Replikate,
384 deskriptive und 372 semantisch passende Paare, 0 Claims. Der Erhaltungs-
vergleich bestätigt alle 437.974 Primärdateien; 527 Generic-, 548 Quality-,
234/702 Native- und 234/702 Energieergebnisse einschließlich der drei ungültigen
physischen Vorversuche bleiben erhalten. Der erste Replay bleibt separat
als Beleg der zuletzt korrigierten Topfeldinkonsistenz bestehen. Einzelbefunde,
Grenzen, genaue Tests, Gitstatus und bedingte nicht freigegebene Zusatzmessliste
stehen im externen `ABSCHLUSSBERICHT.txt` und `korrigierte_falltabellen/`.
Keine normale GUI-/Hardwareabnahme, keine Messfreigabe, kein Commit/Push.

## 2026-10-02 – Separate THESIS20-YOLO-Min3-Augmentierung

Neuer ausdrücklicher Auftrag nach dem abgeschlossenen 192er-Completionlauf
(192 Fälle, 576 Replikate, 576.000 Taskabschlüsse; Exitcode 0). Dieser Bestand
und die abgeschlossene Hauptkampagne sind ausschließlich Originalquellen.
Quellcheckpoint und sämtliche neuen Arbeits-/Ergebnisdateien liegen unter
`$HOME/Reports/thesis20-yolo-min3-augmentation-20261002_222904`.

Die graphbasierte Zusatzwahl ergibt YOLO11l b120, YOLO26s b038 sowie
YOLOv7 b044/cat_1 und b066/mul_20 auf H8, H10 und DeepX: zwölf Primärfälle.
YOLO26m erhält keine neue Grenze. Die ursprünglichen Graphidentitäten und
jeweils ein tatsächlicher P2-Eingang sind lokal belegt. B008/B008/B030 sind
vorab ausgewiesene optionale Standbygrenzen, keine bereits zugelassenen
Messaufträge. Eine nachträgliche Änderung versiegelter akzeptierter Scopes
wird nicht als Reserveaktivierung verwendet.

Neue begrenzte Planung und Workflowverknüpfung verwenden die vorhandenen
Profil-, Generator-, Build-, Quality-, Native-, Completion- und Energiepfade.
Explizite Mixed-composed-Profile behalten N5000/B1000 und erzeugen keine
zusätzlichen Fullperformancefälle. Vorhandene CPU-Referenzen bleiben mit
ihrer Original-Run-ID und ihren Originalbytes gebunden; der neue Generic-
Writer kann daraus zentrale neue Kandidatenrequests erzeugen, ohne eine
CPU-Full-Inferenz oder eine erfundene numerische Rohoutputprüfung auszuführen.
Der vorhandene lokale Nativevalidator kann passende gespeicherte Split-
CPUreferenzen gegen Modell-, Input-, Decoder- und Completionverträge prüfen.

Die Native-Startbarriere führt höchstens einen festgelegten neuen Startfall
je Setup aus und verbindet ihn mit dem vorhandenen Generic-Completionrunner.
Fertige Replikate bleiben erhalten. Controller-Broker, Ressourcenlocks,
Quarantänen, Command-/Qualitybindungen und das gemeinsame Energiefehlerbudget
bleiben aktiv. Ein expliziter kontrollierter Native-Wiedereintritt kann streng
gebundene fertige Leaves aus einem partiellen Augmentierungscheckpoint
übernehmen; daraus entsteht keine automatische Reparatur-/Neustartschleife.

Der Zusatzreporter trennt Basis, Zusatzkohorte und augmentierte Vereinigung
einschließlich negativer Ergebnisse und tatsächlicher Gruppengrößen. Neue
Energie wird nur mit passend gebundenen alten Fullreferenzen verglichen.
Development/screening und post_planned bleiben sichtbar; kein Hold-out-Claim.
Normale Native- und Buildterminals ohne Completion bleiben mit ihrem Originalbeleg in der Endbilanz sichtbar, ohne eine fertige Gesamtmessung zu behaupten.

Softwareabnahme und tatsächliche Hardwareausführung werden getrennt im
externen Lieferverzeichnis dokumentiert: `software/`, `selection/`, `quality/`,
`reporter/` sowie die späteren Start-, Fortschritts- und Terminaldateien.
Dieser Quellarbeitsstand ist vor dem produktiven Start erstellt; Softwaretests
belegen keinen Hardware-PASS. Keine zusätzliche normale GUI-Abnahme, kein
öffentlicher Release, Commit, Push oder Tag in diesem Arbeitspaket.

## 2026-10-03 – begrenzte YOLO-Min3-Qualityfreigabe und expliziter Wiedereintritt

Der erste Zusatzlauf endete vor Native-Endmessungen: Seine neun Remoteaufrufe
hatten trotz N5000 in den eingefrorenen Plänen kein autoritatives Validation-
Budget und keine mitverpackte Quelle. Die zwölf heruntergeladenen Fallberichte
enthalten keine Kandidatenrecords. Die leere Queue wurde als skipped projiziert;
die unveränderte Native-Startbarriere lehnte die fehlende neue Qualitybindung ab.
Die drei lokalen Quellen und CPUreferenzen sind mit jeweils 5000 exakten
Bildidentitäten und Originalannotation vorhanden und werden weiterverwendet.

Der enge Reparaturweg verbindet das explizite Mixed-Profil mit den bestehenden
Remote-/Paketparametern, prüft die gebundene Quelle vor Upload und beim Consumer
und fordert für jede aktive neue Primäridentität einen regulär abgeschlossenen
Request. Modell, neue Grenze, Setup, Precision, ursprüngliche Referenz und
N5000/B1000/Seed bleiben geprüft. Negative Accuracyentscheidungen bleiben gültige
negative Ergebnisse. Die Qualitymetadaten der neuen Requests ergänzen vorhandene
Genericzeilen, ohne ihre alten Timing- oder strukturellen Aussagen umzuwerten.
Die anschließende Completion bindet den neuen Kandidatenrohbericht für ihr
Request-/Runtimerezept; die alte Proxyperformancequelle bleibt ausdrücklich
separat. Der normale Completionbinder prüft weiterhin alle sieben Dimensionen,
den exakten Request und den Nativecommand samt Artefakten.

Der kleine Augmentierungseinstieg besitzt einen expliziten --resume mit neuem
--attempt-dir. Er validiert den tatsächlichen fehlerhaften Quality-Nonstart,
verwendet den bestehenden selektiven Ergänzungsweg und hält den alten
Koordinatorcheckpoint unverändert. Neue Nativecheckpoints und die unveränderliche
Invocationkonfiguration liegen im normalen Supplementattempt. Neue Download-,
Konsolen- und Abschlussdateien sind getrennt; konkrete Koordinatorfehler gelangen
auch ins Kindlog, den Hauptstufenbefund und den Abschlussbericht.

Der bestehende Mixed-Runner kann die fehlenden Split-Qualityoutputs nicht über
seinen Full-only-Qualitymodus erzeugen. Seine dabei unvermeidbaren kurzen
Composed-/Streamingmessungen sind ausdrücklich Beifang des Kandidatenlaufs;
die zwölf vorhandenen Genericmessungen bleiben erhalten. Keine Basis-, CPU-Full-
oder 192er-Wiederholung, keine neue Auswahl, kein Force, keine Budgetänderung.

Reparaturbelege, Softwareprüfungen, Sourcecheckpoint und der tatsächliche spätere
Hardwarefortschritt werden getrennt unter
`$HOME/Reports/thesis20-yolo-min3-augmentation-20261002_222904/repairs/quality-release-20261003_064548`
und dem neuen produktiven `attempts/`-Verzeichnis dokumentiert. Die Softwareprüfung
ersetzt ausdrücklich nur Hardware-/SDKblätter und ist kein Hardware-PASS.
Keine zusätzliche allgemeine GUI-Abnahme, kein Commit/Push/öffentlicher Release.

Die verbundene Softwareprüfung fand zusätzlich zwei tatsächlich ursächliche
Verknüpfungen: Ein Native-P2-gebundener Composed-Request muss zugleich seine
vorhandene Genericproducer-Identität tragen. Writer und zentraler Reader prüfen
jetzt beide Belege einschließlich identischer Runtimebytes und Original-P2-Quelle.
Der H10-Originalreferenzvergleich prüft beide Herkunftsattestationen unabhängig
gegen ihre unveränderten Vertragsdateien, bevor er ihre fachlichen Inhalte
vergleicht. Ein unterschiedlicher Suitepfad ist keine fachliche Decoderänderung;
abweichende Primitive und beschädigte Provenienz bleiben ausdrücklich negativ.
Beide Befunde stammen aus dem normalen verbundenen Writer-/Consumer-/Validatorweg,
nicht aus einer nachträglichen Freigabe von Ergebnisflags.

Der anschließende normale Paarimport deckte eine weitere konkrete
Readerlücke auf: Das Backendfeld des Generic-Composed-Producervertrags ist
eine Richtung (z. B. deepx_to_trt), kein einzelnes TensorRT-Gerät. Der
Normalizer validiert diesen bestehenden Vertrag einschließlich Richtung und
beider Stufen und projiziert danach den ersten Producer. Explizite
Widersprüche bleiben negativ; reine TensorRT-Fulls und TRT→TRT bleiben
TensorRT. Die strikten Cross-Runner-Konfliktprüfungen wurden nicht gelockert.

Der gemeinsame Completionimport fand außerdem eine zyklische Startvoraussetzung:
Das Native-Crossbackend-Gate verlangt zwei Backends je Modell/Grenze, während
die drei eingefrorenen Starter jeweils zunächst nur einen besitzen. Die weiteren
Nativefälle liegen absichtlich hinter der Startbarriere. Der Completionreporter
prüft für das technische Generic-/Native-Paar deshalb dessen tatsächliche
Command-, Input-, Output- und Completionverträge sowie die drei gültigen
Genericreplikate über die vorhandenen Verifier. Der historische Crossbackendwert
bleibt sichtbar; die Quality- und wissenschaftlichen Claimgates behalten ihn
unverändert. Die Native-Startbarriere fordert weiterhin ein regulär belegtes
technisches Paar. Falsche Identitäten, Vertragsprimitive, SHA-Bindungen oder
fehlende Originaldateien dürfen diese Voraussetzung nicht erfüllen. Die gezielte
Softwareabnahme und der erneute Wiedereintritt mit dem abschließend installierten
Stand werden im oben verlinkten Reparaturbereich getrennt vom späteren echten
Messfortschritt dokumentiert.

## 2026-10-03 – Native-Cancel nach abgeschlossener Zusatzquality

Die zwölf neuen N5000/B1000-Requests sind vollständig abgeschlossen
(zwölf reference_close, keine technischen Fehler). Kandidaten, Requests,
Ergebnisse und ursprüngliche CPUreferenzen werden unverändert übernommen.
Der Native-Fehler um 13:15:03 entstand vor dem ersten Messstart: Der neue
Workflow hatte seinen Controlzustand noch nicht veröffentlicht, sodass der
Native-Consumer den historischen SIGINT-Marker der vorherigen Sitzung las.

Der normale Workfloweintritt veröffentlicht nun unter seinem bestehenden
Control-Lock den aktuellen Zustand, sobald exklusiver Runbesitz und sämtliche
Resumeprüfungen erfolgreich sind. Der vorhandene Writer bleibt maßgeblich;
Cancelereignisse, Consumerprüfungen und historische Abbruchbelege bleiben
erhalten. Ein aktueller Cancel muss weiterhin jeden Native-Messstart sperren.
Der explizite Wiedereintritt wird am tatsächlichen neuen Nonstartcheckpoint
und seinen ursprünglichen Verträgen geprüft; der erste fehlgeschlagene
Qualitycheckpoint bleibt ein eigener historischer Beleg.

Softwareabnahme mit kontrolliert ersetzten Messblättern, installierter
Sourcecheckpoint und tatsächliche produktive Nativeabschlüsse werden separat
unter
`$HOME/Reports/thesis20-yolo-min3-augmentation-20261002_222904/repairs/native-cancel-20261003_134048`
dokumentiert. Keine erneute Qualityrechnung, Kandidateninferenz, Grenzauswahl
oder Force-Rebuild; keine Wiederholung der Haupt- oder 192er-Kampagne.
Nach erfolgreicher Abnahme ist genau eine dauerhafte selektive Fortsetzung
der offenen Zusatzmessungen autorisiert. Keine automatische Reparatur- oder
Neustartschleife, kein Commit/Push/Release und keine allgemeine GUI-Abnahme.

## 2026-10-03 – Generic-Completion nach dem ersten Nativeabschluss

Der vorige Attempt endete nach gültiger H8/YOLO11l/b120-Nativeperformance
(drei Wiederholungen, Median 43.7662981538428 FPS) am Referenzloader der
ersten Generic-Completion. Die zwölf N5000/B1000-Qualityresultate bleiben
vollständig erhalten. Die originale CPUreferenz ist gültig; der Completionmodus
hatte ihren ausschließlich für Dataset-Quality bestimmten Parameter aus dem
früheren Quellenaufruf übernommen, obwohl Zentralmodus und Bildzahl bereits
abgeschaltet waren. Nur dieser aktive Parameter wird beim Moduswechsel geleert.
Originalkonfiguration und strenger Qualityloader bleiben unverändert; die
Completion verwendet weiterhin ihre eigene gebundene Native-Outputreferenz.

Die gespeicherten H8-Outputs bestehen die normale lokale Semantikprüfung,
wenn der vorhandene zentrale Quality-Join die Modellbindung rechtzeitig vor
dem Outputs-only-Validator bereitstellt. Genau diese Reihenfolge ist korrigiert.
Messwerte, Outputs, Eingaben, Endpunkte, Toleranzen und Commands ändern sich
nicht. Eine inzwischen erfolgreiche lokale Semantikprüfung ersetzt keine
historischen Referenzannotationen oder Claimgates der bereits geplanten
Completion. Alle zugrunde liegenden Referenzbindungen bleiben strikt gleich.

Der Wiedereintritt muss den tatsächlich gemessenen Nativeleaf über dessen
ursprünglichen Checkpoint und Command erhalten; er ist kein Nonstart mehr.
Neue Coordinatorcheckpoints und der korrigierte Completionrunner erhalten
eigene Attemptpfade. Alte Runner, fehlgeschlagene Attempts, gültige Replikate
und der erste Genericfehler bleiben erhalten. Zuerst ist ausschließlich die
fehlende H8-Completion freigegeben; die bestehende Startbarriere entscheidet
danach regulär über die übrigen Startfälle und den Rest einschließlich Energie.
Der normale Resume-Pfad erkennt dabei auch die vom Native-Writer erzeugten
Hailo-Bundlelinks unter `native_producers/variants`; die bestehende strenge
Bundleprüfung bleibt maßgeblich, Links und Zielartefakte bleiben unverändert.

Gezielte Regressionen, endgültige lokale Installation, zusammenhängende
Softwareabnahme und tatsächlicher produktiver Fortschritt werden getrennt
unter
`$HOME/Reports/thesis20-yolo-min3-augmentation-20261002_222904/repairs/generic-reference-20261003_151204`
gesichert. Kein allgemeiner GUI-/Hardwaretest und kein erneuter Basis- oder
192er-Lauf. Nach erfolgreicher Abnahme genau ein dauerhafter tmux-Start;
keine automatische Reparatur-/Neustartschleife, kein Commit/Push/Release.

## 2026-10-03 – Vorhandenes YOLO-Startpaar regulär projizieren

H8/YOLO11l/b120 besitzt jetzt je drei gültige Native- und Generic-Completion-
Wiederholungen. Beide Originalmessungen und alle zwölf N5000/B1000-
Qualityresultate bleiben erhalten. Der normale lokale Paarreplay reproduzierte
zunächst die fehlende Rohidentität am strengen Native-Qualityconsumer. Der
Paaradapter ergänzt nur fehlende Identitätsfelder aus dem bereits gegen die
vollständige Paaridentität geprüften Command; explizite Aliaswidersprüche
bleiben Fehler. Für den tatsächlichen Fast-/Oracle-Nachweis verwendet er die
vorhandenen strengen Attestierungs-, Output- und Persistenzprüfer. Fast- und
Hotloop-Beobachtungsrelationen bleiben verschieden und werden nicht umetikettiert.

Der zusätzliche Precisionkonflikt entstand aus einem beschreibenden
Dispatcherlabel in einer erwarteten Matrixzeile ohne Precision. Der normale
Reportadapter ergänzt diese fehlende Angabe aus den geprüften Command- und
Consumerbindungen. Explizite Precision/Contract-Angaben, cast/dequant-
Unterschiede sowie mehrdeutige unterschiedliche Plankontexte bleiben negativ.
Die Originalreports und ihre alten negativen Projektionen bleiben erhalten.

Der echte lokale Paarconsumer besteht mit 23/23 technischen Checks ohne neue
Inferenz: Native 43.7662981538428 FPS, Generic 35.33776805174142 FPS. Quality-
und wissenschaftliche Claimgates bleiben getrennt und werden nicht angehoben.
Der bestehende enge Ergänzungseintritt schützt nun auch diesen genau belegten
ersten vollständig gemessenen Paarzustand und seinen ursprünglichen Nativeleaf;
keine allgemeine Resume- oder automatische Retryfreigabe.

Gezielte Tests, endgültige Installation, normale Coordinator-Startbarriere auf
geschützter Arbeitskopie, lokale Energieplanung und produktiver Restfortschritt
stehen getrennt unter
`$HOME/Reports/thesis20-yolo-min3-augmentation-20261002_222904/repairs/pair-projection-20261003_175306`.
Nach bestandener installierter Barriere sind genau elf Nativefälle, elf
Generic-Completionfälle und zwölf Energiezeilen einmalig freigegeben. Keine
Neumessung des ersten Paars, keine neue Quality, Auswahl oder Force-Rebuild;
keine allgemeine GUI-/Hardwareabnahme, kein Commit/Push/Release.

## 2026-10-03 – Vertragliche Eingabeauflösung für vorhandene YOLO-Paare

Nach dem ersten erfolgreichen H8-Paar sind auch H10/YOLO26s/b038 Native
und Generic-Completion mit je drei gültigen Wiederholungen abgeschlossen.
Der unveränderte lokale Consumer reproduzierte den H10-Fehler
`reason="'prepared_input'"`: Der Paaradapter erwartete einen festen
Artefaktnamen, obwohl der Originalvertrag seinen NPY-Eingang ausdrücklich
über `entries[0].artifact_name=prepared_input_00` bindet.

Der bestehende Planresolver rekonstruiert nun auch für die Paarprüfung die
deklarierte Eingabezuordnung. Vollständige Original-/Runtimebindung, Format,
Form, Datentyp, Slotfolge und deklarierte Speicherordnung werden geprüft.
Der bisher im Runner enthaltene reine Arrayloader liegt im vorhandenen
Completionmodul; Generator und Paaradapter rufen dieselbe Funktion auf.
Keine Artefaktumbenennung, Änderung gebundener Bytes oder neue Registry.
Der lokale vollständige Consumer akzeptiert beide vorhandenen Zusatzpaare
mit jeweils 23/23 technischen Checks. DeepX wird anhand einer tatsächlichen
Originalbeobachtung der abgeschlossenen 192er-Ergänzung geprüft; dadurch
entsteht kein Ergebnis für einen noch ungemessenen Zusatzfall.

Der vorhandene enge Wiedereintritt erhält jetzt den konkret belegten Zustand
mit zwei abgeschlossenen festgelegten Startfällen und ihren verschiedenen
ursprünglichen Coordinatorbindungen. Die bisherigen Vertrags-, Qualitäts-,
Replikat- und Abbruchprüfer bleiben maßgeblich; keine allgemeine Resume- oder
automatische Reparaturfreigabe. Die finale installierte Softwareabnahme muss
beide normalen Startbarrieren ohne Messblätter passieren und unmittelbar vor
dem nächsten DeepX/YOLOv7/b044-Dispatch kontrolliert enden.

Belege, gezielte Regressionen, Installation und tatsächlicher Produktivstatus:
`$HOME/Reports/thesis20-yolo-min3-augmentation-20261002_222904/repairs/pair-input-contract-20261003_191121`.
Danach ist genau eine dauerhafte Fortsetzung mit zehn Nativefällen, zehn
Generic-Completionfällen und zwölf Energiezeilen autorisiert. Beide fertigen
Paare und zwölf Qualityresultate bleiben erhalten. Auswahl, Messparameter,
Retrybudget und wissenschaftliche Rollen bleiben unverändert; Force aus,
keine neue Quality, kein Commit/Push/Release oder allgemeiner Hardwaretest.


## 2026-10-03 – H8-Dreistufen-Helper im bestehenden Native-Sync

Der beendete YOLO-Min3-Attempt `pair_input_contract_20261003_192749` erhält elf erfolgreiche Nativefälle/33 Wiederholungen, drei Generic-Completionfälle/neun Wiederholungen und zwölf Qualityresultate. H8/YOLOv7/b066 endet vor Messausführung mit `concurrent_three_stage_helper_missing`; keine Energieaufnahme. Der tatsächliche H8-Runner löst den unveränderten Helper relativ als Skript-Sibling auf. Der vorhandene Updater übertrug diesen Helper und dessen Ressourcen nicht.

Die einzige Produktänderung erweitert die H8-Transferliste in `scripts/update_evalset_native_producers.py` und ihrem Paketmirror um den vorhandenen Helper sowie zwölf unmittelbar benötigte Python-/C++-/JSON-Ressourcen. Alle Transfers benutzen die bestehenden SHA-/Capabilityprüfer. Keine Helper-, Oracle-, Referenz-, Precision-, Grenz-, Messparameter- oder Modellartefaktänderung. Die vorhandene Paketclosure enthält die benötigten Import- und Source-Snapshotmodule bereits.

Software: tatsächlichen vorherigen Transferzweig bis zum Helperfehler reproduziert; korrigierten Transfer in Quell- und Paketpfad bis zum unveränderten Parent-Commandgenerator geprüft; fehlende/manipulierte Übertragungen bleiben negativ. Der reale H8-Pfad wurde zunächst rein lesend ebenfalls als fehlend bestätigt. Gezielte Ergebnisse, finale Installation und Remote-Nachweis stehen im Reparaturbericht unter `$HOME/Reports/thesis20-yolo-min3-augmentation-20261002_222904/repairs/h8-three-stage-helper-20261003_211543`. Keine normale GUI-/Hardwaremessabnahme behauptet.

Startreife bleibt gesondert blockiert: der bestehende Wiedereintritt kennt den nun terminalen Native-Abschlussfehler nicht; gültige fehlgeschlagene Koordinator-/Leaf-Checkpoints werden regulär als Fehler übernommen. Außerdem bindet der mitgelieferte Dreistufen-Oracle zwei Prozessorquellen an historische SHA, die nicht dem aktuellen Toolbestand entsprechen. Kein Zurücksetzen dieser Checkpoints, kein Ändern der Sollhashes und keine Abschaltung der Prüfung. Der negative Originalversuch und alle fertigen Daten bleiben erhalten. Keine neue Resume-Architektur, kein Commit/Push/Release.

## 2026-10-03 – H8/b066: Oraclebindung und konkrete Reststartgrenzen

Der vorhandene Canary verwendet jetzt für den eingefrorenen Prozessorvertrag den bereits bestehenden `verify_frozen_postprocess_contract`. Die vorher doppelte reine Quellbyteprüfung lehnte dessen vorhandene, versionsgebundene Kompatibilität ab. Historische Sollhashes, Referenzen, Decoder/NMS, Fastcallback und C++ bleiben unverändert. Die ausgewählten Toolquellen müssen weiterhin mit den tatsächlich geladenen Implementierungen übereinstimmen. Zehn neue und neun betroffene vorhandene Softwaretests bestehen; die normale Canary-main passiert die echte Oracleprüfung bis zur kontrollierten Artefakt-/Runtimegrenze.

Die originalen v2.78.2-Quellen und Sentinelheads wurden aus vorhandenen Archiven SHA-verifiziert. Alter und aktueller Oracle sowie Fastdecoder liefern daraus auf dem Controller exakt dieselben neun Detektionen. Die gespeicherten ARM-Floats weichen auf x86 auch mit Originalquellen um einzelne ULP ab; keine gelockerte Toleranz und keine neue ARM-/Hardwareabnahme werden daraus abgeleitet.

Für den einzigen fehlgeschlagenen Fall wird der vorhandene Singlecase-Updater mit neuer Artefaktnamespaceidentität geprüft. Der bisherige terminale Workflow und seine historischen Fehler bleiben erhalten; kein neuer Resume- oder unvollständiger Produktions-CLI-Zweig wird eingeführt. Die tatsächliche installierte Eintrittsprüfung und deren kontrollierte Grenze werden separat dokumentiert.

Der produktive Reststart bleibt durch konkrete unmittelbar anschließende Unvereinbarkeiten gesperrt: Der Helper prüft die aktuelle Engine e93f81da…, übergibt dem Canary aber dessen historische Engine ddf2e26a…. Sein früher Rücksprung umgeht die aktuellen Quality-/Command-/Preparedinput-/Ergebniswriter. Eine echte historische Hardwarefixture durch den normalen aktuellen Reporter liefert deshalb keine FPS-Replikate und keinen gültigen Nativecommandvertrag. Außerdem umfasst der unveränderte C++-Throughput Bildlesen und Letterbox, während die festgeschriebene Generic-Completion vorbereitete Eingaben ohne Vorverarbeitung verlangt. Diese Grenzen werden weder durch angepasste Sollhashes noch durch erfundene Ergebnisfelder oder Zeitkorrekturen übergangen.

Belege, Installation und endgültiger Status: `$HOME/Reports/thesis20-yolo-min3-augmentation-20261002_222904/repairs/h8-restabschluss-20261003_214340`. Elf Nativefälle/33 Wiederholungen, drei Genericfälle/neun Wiederholungen, zwölf Qualityresultate und alle Originalkampagnen bleiben erhalten. Rest: ein Nativefall, neun Genericfälle, zwölf Energiezeilen. Kein produktiver Start, keine Messung oder Modellkompilierung, keine GUI-Abnahme, kein Commit/Push/Release.

## 2026-10-04 – Autorisierter YOLO-Restabschluss in zwei Blöcken

Der neue Auftrag `AUFTRAG_YOLO_RESTABSCHLUSS_20261004.md` autorisiert die drei zusammenhängenden b066-Korrekturen und den anschließenden Restlauf. Der begrenzte Einstieg `workflow.yolo_augmentation --finish-rest` verwendet bestehende Consumer, Einzelfall-Updater, Prozess-/Ressourcenschutz und Energie-CLI. Block 1 übernimmt elf Nativefälle und drei fertige Genericpaare; er wählt acht offene Genericfälle und elf Energiezeilen, unabhängig vom negativen H8/b066-Leaf. Block 2 erhält einen eigenen Native-Attempt und Toolpfad, danach genau seine Generic- und Energiezeile. Die Zwölfer-Sollkohorte und der historische Fehler bleiben unverändert sichtbar. Es gibt keinen automatischen Neustart und keine allgemeine Resumeänderung.

Der neue b066-Kampagnenpfad bindet die aktuelle qualitätsgeprüfte HEF/Engine bis zum C++-Aufruf. Eine reguläre ungemessene Vertragsprobe bestimmt die tatsächliche Decodersemantik; sie ersetzt keine Qualityauswertung. Das gebundene HWC-UINT8-Eingabebeispiel wird vor Warmup und Messzeit geladen. Drei bereite Worker starten das echte monotone Performancefenster; jeweils 100 vollständig abgearbeitete Warmups liegen davor, jeweils 1000 abgeschlossene Tasks und der vollständige Drain werden erfasst. Der historische Canary behält seinen eigenen V1-Pfad und seine Originalmetadaten. Decoder/NMS, Precision, Grenze und Modellartefakte ändern sich nicht.

Der Concurrentpfad bedient nun die normalen Command-, Eingabe-, Output-, Completion- und Replikatwriter. Der Finalreport liest drei unabhängige Wiederholungen; Energie verwendet die tatsächlich gebundene Laufzeitbibliothek nach normalem Preflight. Der neue Energie-Marker enthält den vollständigen erforderlichen Completionnachweis, während umfangreiche Latenz-/Replikatdiagnosen zusätzlich im vollständigen Roh-JSON bleiben. Kein Loglimit, Claim- oder Provenienzgatter wird gelockert. Die zusätzliche Referenzoption des Einzelfall-Updaters verwendet den bestehenden outputs-only-Validator und die vorhandenen Originalreferenzen, ohne Full-ORT- oder N5000-Neuberechnung.

Softwareabnahme und echte Gerätebelege bleiben getrennt: normale Callbackprobe auf einer isolierten Arbeitskopie mit drei erhaltenen und acht ausgewählten Fällen; tatsächliche Updater-main bis zum kontrollierten äußeren Transportstopp; gekoppelte Parent-/C++-/Writer-/Finalreport-/Energieprüfung mit SDK-Doubles einschließlich eines echten zeitgesteuerten 60-Sekunden-Softwareloops. Falsche Artefakte/Eingaben, falsche Zeitumfänge, fehlende Wiederholungen und aktueller Cancel bleiben negativ. Der tatsächliche Remote-Artefaktpreflight besteht für alle elf Block-1-Zeilen ohne Workload oder Collector. Zwei zusätzliche ältere GUI-Testabweichungen betreffen unveränderte Quellen und sind gesondert belegt; eine normale GUI-Abnahme wird nicht behauptet.

Finale Installation, Compile-only-Prüfung auf H8, genaue Testresultate, Quellsicherung und tatsächlicher Produktivfortschritt werden unter `$HOME/Reports/thesis20-yolo-min3-augmentation-20261002_222904/repairs/restabschluss-20261004_061539` dokumentiert. Eine SDK-Softwareprobe ist kein Hardware-PASS. Vor der produktiven Phase werden alle Quellen eingefroren; danach läuft genau ein dauerhafter sequenzieller Attempt mit normalen Gates und bestehenden Energie-Retries. Keine neuen Modellcompiles, kein Force, keine erneute fertige Messung, kein Commit/Push/Release.

## 2026-10-04 – Restenergie: tatsächlicher Quellen-STOP und erhaltene Teilresultate

Der Attempt `restabschluss_20261004_065600` endete um 08:09 CEST nach elf vollständigen Generic-/Nativepaaren (je 33 Wiederholungen), zwölf Qualityresultaten und sechs vollständigen Energiezeilen mit 18 gültigen Replikaten. Die originale Partialdatei und ihr Rowjournal sind maßgeblich. Der normale Energieconsumer kann diese Quelle jetzt ausdrücklich lesen; eine vorhandene finale Datei behält Vorrang. Der Augmentierungsreport übernimmt die vorhandenen gültigen Zeilen auch bei unterbrochenem oder vorzeitig blockiertem Folgeattempt. Originalberichte werden nicht überschrieben. Physisch: 23 Collectorstarts, davon 18 gültig, vier abgeschlossen ungültig und einer mit unbestätigtem Abschluss; wissenschaftliche Freigaben bleiben null.

H10/YOLOv7/b044 enthält kein gültiges Teilreplikat. Die erste Aufnahme hat 7.872 verlorene Samples im Commandfenster; die zweite meldet `source closed verified=false` nach Socket-Silence/OS-Fehler 11 ohne Protokollende. Dauerhaft belegter lokaler Collector-Cleanup ist keine Quellenbestätigung. Der Quellenfehler `campaign_source_completion_unresolved` wird vor dem nachfolgenden `rest cleanup is not proven` gesondert erhalten. Vier physische Quarantänen und die zusätzliche Run-Quarantäne sind konkrete, weiterhin gültige Schutzbelege.

Der vorhandene Recovery-Einstieg erhält ausschließlich einen expliziten Adapter für das tatsächlich geschriebene Restattempt-Journal und dessen Terminalbelege. Plan, Run, Session, Parent-PID, Exitcode und Cleanup werden gebunden; der historische Runstatus `partial` bleibt unangetastet. Betreiberhandlung beziehungsweise ausdrückliche Neuversuchsfreigabe, alle vier Ressourcenlocks und die bisherigen Owner-/STOP-Prüfungen bleiben verpflichtend. Der begrenzte Restanschluss verwendet den normalen Previous-Results-Consumer für sechs erhaltene und fünf offene Block-1-Energiezeilen, ohne die elf fertigen Performancepaare zu dispatchen. Eine Runquarantäne kann nur nach exakt belegter physischer Recovery über den bestehenden exklusiven Lockcallback übernommen werden.

Software: 116 gezielte Recovery-/Ownershiptests, 79 Reporterregressionen und 58 Restanschlussprüfungen bestanden (getrennte, teilweise überlappende Gruppen). Die normale Projektion aus den tatsächlichen Originalen belegt 6/18 und elf technische Paare. Eine isolierte Kopie des tatsächlichen Kampagnenbudgets durch den normalen Budgetconstructor und Admissionpfad bestätigt weiterhin `campaign_source_interrupted_chain_unresolved`; keine Budgetgrenze ist erschöpft und kein Zähler wurde zurückgesetzt. Physische Recovery allein hebt diese zusätzliche Unterbrechung nicht auf. Ein allgemeiner Budget-Resumezustand wurde nicht eingeführt.

Aktueller Produktivstart bleibt blockiert: tatsächliche Quellenwiederherstellung ist nicht belegt, die unterbrochene Budgetchain bleibt negativ. Keine Recoveryfreigabe, Messung, Kompilierung, Prozessbeendigung oder neue tmux-Sitzung durch diese Prüfung. Controller und drei DUTs wurden über den bestehenden Transport nur lesend auf Konkurrenz geprüft. Keine normale GUI-/Hardwaremessabnahme behauptet. Korrigierte Berichte, genaue Testergebnisse, Quellenabschluss, gesicherte Originale und lokaler Installationsstand: `$HOME/Reports/thesis20-yolo-min3-augmentation-20261002_222904/repairs/energy-source-rest-20261004_103311`. Kein Commit/Push/Release.

## 2026-10-04 – Einmalige Quellenfreigabe, budgetierter Diagnosetest und Restenergie

Der Nutzer autorisiert ausdrücklich einen neuen Versuch trotz des bekannten unbestätigten Quellenendes an der gebundenen H10-Quelle. Dies behauptet keinen Reset, Powercycle oder bestätigten historischen Abschluss und gilt nicht für zukünftige Quellenfehler. Der vorhandene ausdrückliche Recoveryweg bleibt maßgeblich. Unter dem bestehenden Runlock, den vier physischen Recoverylocks und dem Budgetflock wird dessen genaue alte Capturebindung zusätzlich an genau einen neuen lebenden Workflowparent gebunden. Alte Chains, Stopgründe, gültige Resultate, Zähler und Grenzen bleiben erhalten. Neue Abbrüche und Quellenfehler werden gesondert wirksam; eine fremde Sitzung erhält diese Freigabe nicht.

Der vorhandene `energy_measurement_cli.py test-setup` nutzt den normalen GUI-Messpfad mit genau einer Aufnahme und ohne Diagnose-Retry. Er kann nun die vorhandenen Kampagnenbudgetargumente übernehmen und kennzeichnet den Test ausdrücklich als Diagnose. Der enge Restanschluss `--test-recovered-energy-source` verlangt den konkreten Vorgängerattempt, führt zuerst diesen Test aus und prüft normale Rohdaten-/Commandfensterbindung, tatsächliche Samples, Protokollende, Collectorcleanup und Ressourcenfreigabe. Erst danach folgt innerhalb desselben geschützten Parents die vorhandene Restfolge. Ein fehlgeschlagener Test startet keine Restmessung und keinen automatischen Neuversuch.

Die Auswahl bleibt sechs erhaltene Energiezeilen/18 gültige Replikate und fünf offene Block-1-Energiezeilen, danach H8/YOLOv7/b066 Native, Generic-Completion und Energie. Elf fertige Performancepaare und zwölf Qualityresultate bleiben übernommen; keine neue Haupt- oder 192er-Kampagne, keine Modellkompilierung oder geänderten wissenschaftlichen Messparameter. Die Diagnose zählt separat im bestehenden Budget und ist kein 60-Sekunden-Replikat.

Quellsicherung, gezielte Softwareprüfungen, installierter Eintritt mit tatsächlichen Plan-/Budgetbelegen sowie der davon getrennte echte Test und Produktivstatus stehen unter `$HOME/Reports/thesis20-yolo-min3-augmentation-20261002_222904/repairs/energy-authorized-reentry-20261004_135903`. Reale Test- oder Messerfolge werden ausschließlich aus den dortigen Aufnahmebelegen berichtet; keine allgemeine GUI-Abnahme. Alle Quellen werden vor der produktiven Messphase abgeschlossen. Kein Commit, Push oder Release.

## 2026-10-04 – Autorisierter H8-Energierest mit begrenzten Wiederholungen

Der Folgeauftrag autorisiert den regulären neuen Aufnahmeversuch für den konkret unterbrochenen H8/YOLO26s/b038-Capture an der gebundenen H8-Quelle, ohne Resetbehauptung und ohne zusätzlichen GUI-Diagnosetest. Der letzte Attempt erhält neun vollständige Energiezeilen/27 Replikate sowie eine weitere gültige b038-Wiederholung; die nächste b038-Aufnahme mit unbestätigtem Quellenende bleibt negativ. Elf fertige Native-/Genericpaare, zwölf Qualityresultate, Hauptkampagne und 192er-Ergänzung bleiben erhalten.

Der vorhandene explizite Recovery-/Budgetweg erhält die tatsächliche H8-Bindung und eine historisierte Limitänderung ausschließlich für H8/yolo26s/b038 sowie H8/yolov7_paper/b044 und b066: fünf Retries, höchstens sechs Versuche je logischem Replikat und 18 Chains je Zeile, zehn Sekunden Pause vor erneutem Versuch; Quellenfehlerlimit 20 unverändert. Alte Chains und ihr Verbrauch zählen weiter. Die vorherige .176-Freigabe wird nicht umgebunden. Der historische reine Quellen-Nichtstart von b044 wird nur innerhalb derselben exakt gebundenen Freigabe berücksichtigt; neue Quellenfehler bleiben sperrend.

Die eingefrorene normale Plan-/Previous-Results-Prüfung bleibt maßgeblich. Erst die normale tatsächliche Messaufrufvorbereitung übernimmt die autorisierten H8-Limits aus dem bestehenden Budget. Der Collector erhält dieselbe Policy; sein Retrykind hat keine eigenen Retries und verwendet dasselbe logische Budget. Vorhandene gültige Wiederholungen werden über den normalen Originalbelegloader übernommen. Ein neuer unbestätigter Quellenabschluss wird nicht automatisch durch höhere Retrygrenzen freigegeben.

Nach gezielter Software- und installierter Eintrittsprüfung ist genau ein dauerhafter Restattempt freigegeben: fehlende b038-Energie, H8/b044-Energie, danach der bereits implementierte H8/b066-Native-/Generic-/Energiepfad. Keine Modellbuilds, neue Qualitykampagne, allgemeine Resumeentwicklung oder Code-Fixschleife. Quellenabschluss vor der Messung; währenddessen nur lesende Fortschrittskontrolle. Tatsächliche Software-, Budget-, Hardware- und Endbilanzbelege werden getrennt unter `$HOME/Reports/thesis20-yolo-min3-augmentation-20261002_222904/repairs/h8-energy-rest-20261004_170518` gesichert. Kein Commit, Push oder Release in diesem Arbeitspaket.

## 2026-10-04 – Konsolidierte Quellenlieferung 2.92.0

Der anschließende Abschlussauftrag veröffentlicht den zusammengehörigen Toolstand
als `2.92.0` / `v2.92.0`. Vollständiger vorheriger Quellcheckpoint, Gitstatus,
Diffs und 43 separat geschützte lokale Profile sind extern gesichert.
Historische Messquellen und fehlgeschlagene Attempts werden nicht umbenannt oder
dem neuen Release zugeschrieben. Die ursprüngliche Entwicklungs-/Screeningrolle,
postgeplante Zusatzkohorte und Grenzen der Runtimevergleichbarkeit bleiben erhalten.

Releaseidentität, Updater, README und Versionsregeln sind konsistent fortgeschrieben.
Die enge H8-Recoveryprüfung bindet private Source-/DUT-Adressen nun aus den
ursprünglichen vier Ressourcenbesitzern und dem exakten Captureauftrag.
Modell-/Setupscope, maximale Versuche, Pause, Quellenlimit und neue STOP-Wirkung
bleiben unverändert. Der echte gespeicherte H8-Fall besteht diese Prüfung rein
lesend; fremde Setup-, Capture-, Source-, DUT- und Besitzerbindungen bleiben negativ.
Neue Testquellen verwenden ausdrücklich `THESIS20_FIXTURE_HOME` oder relative
Softwarefixtures. Bereits öffentliche historische Hashfixtures bleiben erhalten.

Gezielte Softwareabnahme: 65 Budget-/Collector-/Retrytests, 304 Completion-/Paar-/
Profil-/Reporter-/Recoverytests sowie alle 17 normalen Quellen-Selektorfälle
bestanden. Die 17 zunächst wegen isoliertem pytest-HOME ausgelassenen Fälle wurden
mit explizitem Originaldatenpfad tatsächlich ausgeführt. Zusätzliche Paket-/Mirror-,
Bytecode-, Installer- und Archivbelege liegen im externen Releasebericht.
Keine erneute Hardware-, Modellcompile-, Quality- oder Energiemessung und keine
vollständige GUI-Jobabnahme gehört zu dieser Quellenlieferung. Normale CLI- und
GUI-Versionsimporte sowie Profilauflösung sind getrennte Softwareprüfungen.

Maßgeblich: [Release 2.92.0](RELEASE_2.92.0.md) und der externe Bereich
`$HOME/Reports/thesis20-final-analysis-release-20261004/tool-release/` mit
Prüfkommandos, SHA-Indizes, Archivlesbarkeit, Privacyinventar und genauer
Veröffentlichungspfadliste. Gitfreigabe und Veröffentlichung erfolgen erst nach
gemeinsamer Abschlussprüfung.


## 2026-10-06 – YOLOv7: gemessener H8-Profilingmeilenstein

Der ausdrücklich vorgezogene Performanceauftrag hat den unveränderten H8-TensorRT-Full-Endpunkt mit 100 fertigen Warmups und 100 fertigen Diagnosetasks profiliert. Erfolgreicher Hardwarelauf, null fehlgeschlagene Completions; 202 Originaldateien vor/nach unverändert, tatsächliche Importpfade und SHA256 gebunden. Keine Energieaufnahme und kein Modell-/Engine-Neubau.

Mittelwerte je fertigem Task: synchronisierte Runtime 14,818 ms, ausschließlich CPU-Postprocessing 64,363 ms; disjunkte Summe 79,181 ms. Innerhalb der Runtime liegen H2D 0,243 ms, TensorRT 14,203 ms und D2H 0,276 ms. CPU-Decoding/Filterung dominiert mit 59,467 ms (75,10 % der disjunkten Hostintervalle), NMS benötigt 1,023 ms. Exklusive Vertrags-/Quellprüfungen plus JSON-Hashing ergeben 2,805 ms; inklusive Unterfunktionen werden nicht doppelt summiert. Das Runnerfenster umfasst 79,287951 ms/Task und damit zusätzlich 0,107432 ms/Task nicht weiter aufgeschlüsselten Rest. Aus dem instrumentierten Diagnoseprofil wird keine finale FPS-Zahl abgeleitet.

Die Instrumentierung wurde eng korrigiert: keine zweite JSON-Serialisierung zur Bytezählung, Fehlerstatus und Sollzählung, getrennte erfolgreiche/fehlgeschlagene Completions, abgeglichene Warmups, Modulherkunft und keine unbelegten Tensorlayout-Kandidatenzahlen. Sieben gezielte lokale Regressionen bestanden. Die echten Profilbelege sind unter `$HOME/Reports/yolov7-full-fix-20261005/performance_first_20261006/PROFILE_MEASURED.md` und `profile_costs.json` ausgewertet. Der Profilbefund begründet die isolierte Prüfung des vorhandenen Decoderfixes; Parität und uninstrumentierte Kandidaten-FPS sind gesonderte nachfolgende Ergebnisse. Energie bleibt ein offener eigener Arbeitspunkt; keine neue allgemeine GUI-Abnahme.

## 2026-10-06 – YOLOv7: isolierter H8-Fix gemessen

Der vorhandene Fix mit gebundener Prüfung vor/nach dem Lauf und numerisch exaktem Sparse-Decoding wurde ohne weitere Produktcodeänderung isoliert eingesetzt. 67 einschlägige Tests bestanden; zwei erhaltene Canary-Rawdatensätze ergeben exakt gleiche Original-/Kandidatenergebnisse (9 bzw. 67 Detektionen). Das ist kein neuer vollständiger 32-Bild-Replay. Alle drei Originalrepeats, drei Kandidatenrepeats und das Originalprofil bewahren darüber hinaus dasselbe gespeicherte finale Full-Sentinelergebnis mit 23 Detektionen (Artefakt-SHA `a5f87b03861886fa6637b9d06abb05e61a101862445b77539f92982ebe59f0be`).

Drei separate uninstrumentierte Prozesse mit je 100 Warmups und 1.000 vollständigen Tasks messen auf H8 51,781526 / 51,793755 / 51,723216 FPS. Median 51,781526 gegenüber dem gültigen Originalmedian 14,389955 FPS, Verhältnis 3,59845. Engine, vorbereiteter Input, numerische Verträge und Vergleichsendpunkt bleiben gleich. Originalruntime und isolierte Kandidatenruntime wurden vor/nach gegen ihre jeweils 202 Dateihashes geprüft. MAXN_SUPER, CPU-Governor und aufgezeichnete CPU-/GPU-Takte sind gleich; Temperaturmomentaufnahmen und deren Grenzen stehen im Vergleichsbericht. Keine Energieaufnahme, kein Commit/Push, kein Engine-Neubau. Die übrige Performance-Matrix wird seriell weitergeführt; maßgeblich ist `$HOME/Reports/yolov7-full-fix-20261005/performance_first_20261006/H8_ORIGINAL_CANDIDATE.md`.

## 2026-10-06 – YOLOv7: Performance-Matrix abgeschlossen, Energie offen

Alle 15 Performance-Konfigurationen sind mit je drei Wiederholungen und 100 Warmups/1.000 fertigen Tasks belegt: 18 neue Full-Repeats in getrennten Prozessen und 27 erhaltene Split-Repeats. TensorRT-Mediane H8/H10/DeepX: 51,781526 / 51,480597 / 47,319602 FPS; Accelerator-Full-Mediane: 31,477269 / 12,314299 / 22,945769 FPS. TensorRT ist jeweils die schnellste Full-Alternative. Alle 18 gespeicherten finalen Full-Ergebnisartefakte stimmen exakt mit den jeweiligen Originalen überein; dies ist kein neuer COCO-Replay.

Fehlende historische Hailo-Vertragsdateien und das DeepX-Input-Sidecar samt fehlendem RGB-Inputdump wurden bytegleich ausschließlich in isolierten Report-/Suitepfaden wiederhergestellt. Fehlversuche bleiben dokumentiert. Die tatsächlichen DeepX-Produkt-/Suite-Inputloader bestanden vor der erfolgreichen Messreihe auf dem Zielsystem. Originalruntimes bleiben unverändert (202/191/191 Dateien), 48 Artefaktbindungen und Ressourcenfreigaben sind geprüft. Sieben Profilregressionen und 67 bestehende Decoder-/Bindungstests bestanden, vier Runner-Mirrors sind bytegleich. Kein Build, Commit, Push, gemeinsamer Runtime-Rollout oder Energiestart.

Energie bleibt bei zwei von drei gültigen H8-Vorher-Repeats, fünf Starts und drei Transportfehlern; STOP und Budgets sind erhalten. Aktueller Bericht, vollständige Matrix, Rohresultate, Befehle und aktuelle Reviewpaketkennung stehen unter `$HOME/Reports/yolov7-full-fix-20261005/STATUS.md` und `performance_first_20261006/RESULT_REPORT.md`.


### 2026-10-06: YOLOv7 Split-/Orin-Abgleich begonnen

Neuer expliziter Auftrag vollständig gelesen; Report `/home/kmika/Reports/yolov7-full-fix-20261005/splits_orin_20261006_222316`. Vorhandener Worktree und Messungen gesichert. Reguläre exklusive SSH-Ressourcenprüfung und lokaler Schreibtest erfolgreich. Gebundene DeepX-Splitrunner und Quellen von beiden historischen Roots gesichert, vier zentrale historische SHA256 identisch. H10/DeepX wählen im gebundenen Stand strict/dense Completion; H8 Fast-Oracle. Neue Diagnose/Repair/Nachmessung noch in Arbeit. Alle drei L4T36.4.7/TRT10.3/NumPy1.26.4/MAXN_SUPER; Rootrechte für jetson_clocks --show fehlen, kein Verbindungsfehler. Keine Energiestarts, keine Parameter-/Budgetänderung.

### 2026-10-06: Split-Completion repariert und gezielt abgenommen

Der reguläre H10-/DeepX-Dispatcher und alle erforderlichen Runner-Mirrors wählen für den attestierten YOLOv7-Rawhead-Vertrag jetzt die vorhandene Fast-Completion. Sie verwendet den bereits exakten Sparse-Decoder; Decode, vollständige Finitprüfung und NMS bleiben vor dem Taskabschluss. Vertrag, Modus und tatsächliche importierte Quellen werden vor/nach gebunden. Der Oracle prüft den eigenen letzten gemessenen Output außerhalb des Timers; er wird als Postflight-Sentinel ausgewiesen. Andere Modellfamilien behalten strict; Klassifikation ist ein separater Pfad. Es gab keine Schedulingänderung und keinen gemeinsamen Geräte-Rollout.

H10 b066 Vorher/Nachher auf demselben Orin: Completion 63,599 → 6,614 ms, synchronisierte TRT-Runtime 8,943 → 8,860 ms. Das ist instrumentierte Diagnose, keine finale FPS-Serie. Im ursprünglichen DeepX-Profil belegen 69,528 ms Completion ebenfalls den Hostengpass. H8 b066 zeigt bereits die schnelle, mit TRT überlappende Completion (3,386 ms). Gemessene Abschnitte und Overlap stehen in `SPLIT_STAGE_PROFILE.csv` des neuen Reports.

142 gezielte Tests bestanden; drei bestehende Tests auf das private NumPy-2-Attribut `np._core` bleiben als lokale NumPy-1.26-Umgebungsgrenze offen. Fünf gespeicherte Rawsets (zwei bestehende Canarys und drei neue Originalprofile) ergeben exakte Strict-/Bound-/Fast-Parität. Die drei unveränderten TRT-Full-Kontrollen bestätigen die vorhandenen Full-Mediane innerhalb 0,3 %. Die abschließende neunfache Splitserie läuft seriell über einen Hardwareworker, je Fall drei unabhängige Prozesse mit 100 Warmups/1000 fertigen Tasks. Die jeweilige unabhängige Downstream-Abnahme wird dauerhaft im Report nachgeführt. Keine erneute Quality-Kampagne oder allgemeine GUI-Abnahme.

### 2026-10-06: Split-/Orin-Abgleich abgeschlossen

Alle neun Splitkonfigurationen sind mit 27 getrennten Prozessen, je 100 Warmups und 1000 fertigen Tasks, sowie unabhängigem Rawreplay und regulärer Downstream-Abnahme belegt. Mediane b009/b044/b066: H8 42,851710 / 47,800201 / 112,492794 FPS; H10 51,357901 / 64,500526 / 53,350448 FPS; DeepX 33,866133 / 48,790077 / 30,594221 FPS. Die gültigen 18 Full-Repeats bleiben nach drei separaten Kontrollen erhalten. Neu sind die neun Splitfälle; die sechs Fullfälle gehören zur wiederverwendeten Kohorte. Accuracy-loss von H8-/H10-Accelerator-Full bleibt sichtbar, Fullreferenz wird vor Qualitätsfilterung gewählt.

Ein zusätzlicher enger H8-Commandvalidator-Fix erlaubt für den bekannten b066-Concurrent-Producer eine oder drei Wiederholungen bei unverändert 100/1000. 17 Commandtests und acht normale Qualityconsumer-Regressionen bestanden nach den vorherigen 142 Tests; zusätzlich acht lokale Roh-Latenzprüfungen und drei Instrumentierungsprüfungen. Testläufe sind getrennt dokumentiert, nicht als disjunkte Gesamtmenge addiert. Die separate H8-Runtime hat einen eigenen Quellenstand; erfolgreiche H10-/DeepX-Runtimes wurden nicht überschrieben.

Fehlende historische b009-Eingangsmetadaten auf den Geräten wurden bytegleich aus dem Archiv ausschließlich in isolierten Pfaden bereitgestellt. Ein ausgelassener H8-Prozessparameter für die vorhandenen Hailo-Sites und der alte H8-Commandvalidator verursachten erhaltene Fehlstarts. Ein danach erfolgreicher H8-Prozess wurde nach Korrektur des lokalen Latenzformat-Readers ohne Hardwarewiederholung übernommen; seine ursprünglichen Quellen-, Prozess- und Telemetriebelege bleiben gebunden. Alle Fehlbelege bleiben erhalten.

Die H8-Pythonfälle sind langsamer als historisch. b044 zeigt einen von rund 3,22 auf 5,34 ms verlängerten seriellen Completion-Tail bei nahezu gleicher TRT-Zeit. Eine separate 100/100-Diagnose misst die beiden vollständigen Finitprüfungen zusammen mit 1,842 ms pro Task innerhalb des instrumentierten Tails. Die jetzige Runtime erhält diese Prüfungen; eine spätere Zusammenführung wäre ein neuer zu validierender Stand. Für b066 überlappt der Tail mit der Pipeline. Profilzeiten ersetzen keine Final-FPS.

Orin-NX16GB, L4T36.4.7, aktive MAXN_SUPER-Definition sowie erfasste CPU-/GPU-Takte stimmen überein. Lüfterprofile und geladene CUDART-Versionen unterscheiden sich; EMC bleibt wegen fehlender Rootrechte unbekannt. Die Full-Restdifferenz ist gemessen, aber keiner einzelnen Konfigurationsursache zugeordnet. Keine Clock-/Lüfter-/Paketänderung, kein Modell-/Engine-Build und kein gemeinsamer Rollout.

Energie bleibt unverändert STOP: keine Starts, kein Budgetreset, keine Zuordnung alter Joulewerte zu neuen FPS. Die 15 optimierten Full-/aktuellen Splitvarianten stehen als konkrete Restliste auf not_measured/NA. Ergebnis, Paperexport, englischer Papertext, unveränderte historische Belege und das eigenständig prüfbare Reviewpaket sind unter `/home/kmika/Reports/yolov7-full-fix-20261005/splits_orin_20261006_222316` dokumentiert. Der reguläre Code im Worktree ist umgesetzt; außerhalb dieses Worktrees erfordert seine Nutzung die normale Versionsübernahme/Deployment. Keine allgemeine GUI-Abnahme, kein Commit/Push.


### 2026-10-07: Revision 3 – Archiv und fokussierte Softwarekorrekturen

Neuer Reportroot `$HOME/Reports/native-fairness-rev3-20261007`. Der Sprite-Altbestand bleibt ausschließlich lesend; keine Datenlöschung, Verschiebung oder Speicherbereinigung. Der vorhandene Reviewprüfer bestätigt 6.281 Payload-Dateihashes, 27 Splitrepeats, 18 Fullrepeats und drei getrennte Fullkontrollen. Das Native-Inventar enthält genau 204 Splits und 42 Fulls; alle 246 Commandbindungen und 204 historischen Generic-Completion-Ergebnisse sind zugeordnet. Die weitere Archivprüfung ist eine dokumentierte Metadaten-/Artefaktprüfung, keine Vollhashprüfung aller Modellpayloads.

Erreichbarer Generic-Streamingpfad: verlustfreier Queueabschluss und tatsächlich fertige Counts; allgemeine Split-/Multi-Tensor-Funktion bleibt erhalten. Nur drei durch frühere unbedingte Rückgaben unerreichbare DeepX-P2-Duplikate entfernt. Explizite DeepX-Domänen und keine standardmäßige scorebasierte P2-Mappingwahl. Timingauswahl unabhängig von Backend/Cut; gemeinsame tatsächliche RGB-Vorbereitung. Native und Generic verwenden kompatible gebundene Completion mit nötigem Decode/NMS im Timer und strengem Postflight-Oracle. Die doppelte YOLOv7-Finitprüfung ist bei unveränderten Heads zusammengeführt; verworfene NaN/Inf bleiben Fehler. Der reguläre DeepX-Full-Suitepfad und erzeugte Runner-Bundles sind einbezogen. H10 Full bindet explizite inflight-Werte; Standard bleibt effektiv1, Kontrolle8 bekommt eigene Identität. H8 ist synchron.

Reporting trennt Raw/geschätzt/gemessen/Completed sowie Eingaben, Quality und Energie-Betriebspunkte. Der Student-t-Fallback verwendet bestehende geprüfte Collectorquantile; alle 246 historischen A/B-Container hatten bereits korrekte Quantile, keine Intervallnachmessung erforderlich. Fokussierte Tests und tatsächliche gespeicherte Rawset-Paritäten sind getrennt im Report gebunden; keine GUI-Abnahme.

Pixelprüfung: alle 36 Detektor-Splits an bestätigte gleiche vorbereitete Raster gebunden; alle 168 Klassifikations-Splits an von der gemeinsamen Vorbereitung abweichende Raster gebunden. Eine Dateihashänderung allein begründet keinen Neulauf. Nachmessungen und Deployment folgen mit einmalig eingefrorenen Quellen; bislang null neue Hardware-/Collectorstarts. Energie älterer Implementierungen wird nicht mit neuen FPS gepaart.


### 2026-10-07: Revision 3 – aktuelle Nachmessungen und Energiepfadbefund

15 priorisierte Native-Punkte und neun Generic-Completed-Paare sind mit je drei tatsächlichen Prozessen, 100 Warmups und 1000 fertigen Tasks erfasst. Ausgewählter H10-Accelerator-Full: acht Slots nach gebundener Diagnose und vorab festgelegter Auswahlregel; Ein-Slot-Werte bleiben getrennte Kontrolle. Quellen, Eingaben, Einzelwerte und normale technische Gates sind im aktuellen Reportroot erhalten.

Die erste b009-Energiegruppe belegt neun sensorisch gültige Repeats aus zehn Starts. H8 besteht die normale frische Completionprüfung. Die normale wissenschaftliche Prüfung lehnt DX/H10 korrekt ab: der Energiezweig wählt noch strict statt der Performance-Fast-Completion und gibt keine passenden frischen Marker aus. Eng begrenzter Energiepfadfix in Abnahme; bis dahin keine Zuordnung dieser Joulewerte zu neuen FPS. Alle Budgets, Fehlversuche und alten Quellen bleiben erhalten. Keine allgemeine GUI-Abnahme, kein Build oder Systemupdate.

Die Scopeprüfung ergibt237 tatsächlich betroffene Native-Punkte und neun Wiederverwendungskandidaten; fehlende aktuelle Messwerte bleiben not_measured. Die notwendigen weiteren Messungen, Qualitytransfers und finalen Ableitungen sind noch nicht abgeschlossen. Verbindlicher Zwischenstand: `$HOME/Reports/native-fairness-rev3-20261007/STATUS.md` und `RESULT_REPORT.md`. Neue Daten ausschließlich im datierten Schwesterarchiv, keine Archivänderung oder Speicherbereinigung.

### 2026-10-07: Revision 3 – Generic-Completed-Energie, Softwareabnahme

Der bestehende Generic-Completed-Executor unterstützt jetzt eine ausdrücklich gewählte Daueraufnahme ab 60 s mit Admissionstopp und verlustfreiem Drain. Die festen Performancevorgaben bleiben 100 Warmups/1000 Tasks. Energie verwendet tatsächlich fertiggestellte Tasks, frischen normalen Preflight, gebundene Marker und den bestehenden Collectorimport. Drei getrennte Prozesse bleiben drei physische Quellen mit unveränderten Rohindizes. Der Report berechnet mean(E/N) und mean(N/E) getrennt.

Energie und zugehörige Performance binden identische Eingaben, Completion, Scheduling, Quellen sowie tatsächlich beobachteten Interpreter/venv, relevante Laufzeitumgebung und CPU-Affinität. Der gemeinsame Campaign-Checkpoint erhält stabile Generic-Completed-Fallidentitäten, drei logische Wiederholungen und die bestehenden Grenzen von 9 Messketten, 2 Wiederholungsversuchen und 20 Transportfehlern. Source-STOP und akzeptierte physische Messketten werden geprüft; keine impliziten Wiederholungen oder Budgetresets.

Genau sechs Produktdateien nach Vorher-SHA-Prüfung integriert; sechs portable Regressionstestdateien ergänzt. Isolierte Abnahme: 182 Tests, zusätzliche unabhängige Reviewproben und 36 gezielte Tests gegen den integrierten Worktree einschließlich regulär erzeugtem Bundle bestanden. Belege und Quellbindungen liegen unter `$HOME/Reports/native-fairness-rev3-20261007/generic_energy_extension`. Die zusätzliche Schleifensteuerung ist keine behauptete AST-identische Performanceimplementierung: Die vorhandenen neun Generic-Messpunkte bleiben erhalten, neue Energiezuordnung erfordert neue Performanceprozesse desselben Bundles.

Diese Abnahme ist ausschließlich Softwareprüfung. Kein Hardware-/Collectorstart, kein Modell-/Engine-Build, kein DUT-Deployment, keine GUI-Abnahme und keine Archivänderung. Der konkrete Generic-Captureparent wird getrennt vorbereitet; Hardware bleibt bis zur regulären Freigabe gesperrt.

### 2026-10-08: Revision 3 – kleine Fairnesskorrekturen und konkrete STOP-Grenzen

Der ergänzende Finalisierungsauftrag wird im bestehenden Worktree und Reportroot fortgesetzt. Die 80 bereits geänderten Quell-/Testdateien wurden gegen den Review-Snapshot vom 7. Oktober abgeglichen; vorhandene Änderungen bleiben erhalten. Der Exact-DeepX-Aufruf prüft jetzt bei jeder Verwendung die gebundenen Namen, Shapes und Dtypes vor dem SDK. Die wissenschaftliche Generic-Admission prüft explizite Layout-/Domänen-/Backendverträge vor dem Timer; kein Payloadhash oder Wertecache pro Task. H10 persistiert das tatsächlich gewählte Eingabelayout bereits während der Vorbereitung, ohne Änderung der Tensorbytes. Mirror- und Cold-Sync-Abnahme sowie unabhängiges Gegenreview sind dokumentiert; dies ist noch kein DUT-Deployment oder Hardware-PASS.

Die 20 originalen historischen Raw-Zeilen und ihr exakt gebundener Runner wurden tatsächlich im Archiv geöffnet. Allgemeine Raw-/Multi-Tensor-Pfade bleiben erhalten; nur die bereits entfernten 197 nachweislich unerreichbaren DeepX-Zeilen bleiben entfernt. Die vorhandene LEGACY_GENERIC_USAGE-Tabelle wurde präzisiert. Originale wurden weder gelöscht noch umetikettiert.

Die Offline-Energieprüfung führte das exakt ursprüngliche power_calculations-Binary 18-mal mit den ursprünglichen Argumenten und separaten Ausgabezielen aus: alle results.yaml bytegleich, neun Primärleistungsvektoren identisch, unabhängige Trapezintegration maximal 8,28e-11 J abweichend. Die drei zugelassenen Full-Energiegruppen und zwölf Student-t-Intervalle bleiben unverändert. Die rohe ADC-Konversion wurde nicht unabhängig reproduziert; vorhandene externe Sampleindexprüfungen bleiben als solche gekennzeichnet.

Der präzise Ressourcenscope ist enger als eine allgemeine Hardwaresperre: H10-DUT und sämtliche Sensorcaptures bleiben durch den ursprünglichen STOP beziehungsweise NIC-/Capturequarantänen gesperrt. Normale H8-/DeepX-Performance und SourceSync sind nach finaler Quellenadmission und aktueller Besitzprüfung zulässig. Budgets, STOPs und die verlangte ausdrückliche Einzelfreigabe für H10 bleiben unverändert. Der finale gemeinsame Freeze, reale Deployments und tatsächlich neue Messungen werden gesondert in STATUS/RESUME im bestehenden Reportroot fortgeschrieben.

Die begrenzte reguläre H8-SDK-Inspection hat inzwischen alle sechs repräsentativen Split-/Full-HEFs tatsächlich geöffnet und ihre vorhandenen Identitäten bestätigt. Input-QuantInfo, Originalcompiler und Qualityruntime belegen die standardisierte ImageNet-Domäne. Der bisherige Native-Classifier-Split konvertiert hingegen nur RGB/255; Full-H8 übergibt rohe UINT8-Werte statt des entsprechend quantisierten ImageNet-Eingangs. Die tatsächliche Vertragsinventur erfasst 56 H8-Classifier-Splits (17 MobileNet, 20 RegNet, 19 ResNet), ausschließlich C++-VStreams/FLOAT32, sowie drei Vendor-Fulls. Diese konkreten numerischen Adapter werden vor dem gemeinsamen Freeze korrigiert; kein HEF-/Engine-Neubau und keine Änderung der Detectionnumerik. Für die drei Fulls entfällt damit die bisherige Wiederverwendung als aktueller unveränderter Eingabepfad, während ihre Originalmessungen erhalten bleiben. Sechs andere Classifier-Full-Wiederverwendungen bleiben gültig. Die Nachmessmatrix weist daher vor weiteren Aufnahmen 225 neue Native-Punkte aus; zusätzliche begrenzte Quellenzulassungsbestätigungen werden getrennt gezählt.

Die H8-Korrektur ist nun implementiert und mit 68 fokussierten Tests abgenommen: kanalrichtiges ImageNet-FLOAT32 im C++-/Generic-Splitadapter, einmalige SDK-Quantisierung anhand des konkreten HEF im Full-Vorbereitungspfad, explizite Metadaten und unveränderte Detection-/H10-Numerik. Die reale CPU-Quantisierungsprobe im installierten HailoRT 4.20.0 besteht für alle sechs HEFs; keine Geräteinferenz wurde daraus abgeleitet. Ein normaler C++-Wrapperbuild ist wegen der tatsächlich geänderten Hostoperation erforderlich, anschließend erlaubt die vorhandene native-executable-Option dessen gebundene Wiederverwendung mit jeweils eigenen Caseausgaben. HEFs und Engines werden nicht neu gebaut. Die normale H8-Energieadmission lehnt die neun alten Repeatverträge wegen des Runnerwechsels ab; damit sind zusätzlich zu den sechs DX/H10-Bestätigungen drei begrenzte H8-Kohorten erforderlich. Alle alten FPS bleiben Beobachtungen ihrer ursprünglichen Quellen, ohne Pooling oder nachträgliche Neuversiegelung.

Die normalen aktuellen Consumer wurden mit dem vorhandenen Bestand erneut ausgeführt: 15 Native-Messpunkte, sechs weiterhin zulässige historische Full-Wiederverwendungen, 225 fehlende neue Native-Punkte, neun Generic-Punkte/technische Paare und 195 fehlende Generic-Punkte. Der zuvor verlorene Native-Qualityjoin im Genericexport ist korrigiert; Generic-Quality-/Claimgates bleiben mangels vollständiger numerischer Transferbindung offen. Drei aktuelle Full-Energiepunkte mit neun Repeats bestehen weiterhin die normalen exakten Zuordnungsprüfungen. Neue Hardware-Performance, SourceSync und Wrapperbuild stehen zum Zeitpunkt dieser Softwareintegration noch aus.


### 2026-10-08: Revision 3 – finaler Runtimefreeze und H8-Controllerreferenz

Runtime-/Input-/Completionstand `8346d43d526221bd9d138943122206b658d56cbf` ist mit 241 Dateien eingefroren und über den regulären Pfad auf allen drei DUTs bestätigt. Die neun aktuellen YOLOv7-Native-Splitkohorten und sechs Generic-Bestätigungen auf DeepX/H8 wurden mit jeweils drei getrennten Prozessen und 100 Warmups/1000 abgeschlossenen Tasks aufgenommen. Weitere fehlende Gruppen werden im bestehenden Revision-3-Plan fortgesetzt. Die drei H8-Classifier-Vendor-Fulls sind nach dem belegten ImageNet-/HEF-Inputfix tatsächlich neu gemessen; vorhandene HEFs/Engines bleiben erhalten. Genau ein notwendiger C++-Hostwrapperbuild erfolgte im bestehenden Pfad. Maßgebliche laufende Counts und Resume-Befehle stehen im vorhandenen Reportroot, nicht in diesem Quelljournal.

Im reinen Controllerprüfer wurde ein davon unabhängiger Reportfehler behoben: H8-Full-Classifier-Quantisierungscodes wurden zuvor als RGB-Pixel für die CPU-Referenzvorbereitung gelesen. Der Reader verwendet nun die separat gebundenen kanonischen RGB-Bytes nach Prüfung des eigentlichen Runtime-Tensors und der vorhandenen Modell-, Bild-, Preprocessing- und Domänenbindungen. Fehlende oder widersprüchliche Metadaten scheitern vor dem Referenzaufruf. Alle neun tatsächlichen Manifeste liefern jetzt den erhaltenen kanonischen ImageNet-CPUfeed. 48 gezielte Tests und unabhängiger Review bestanden. Validator und Regressionstest gehören nicht zur 241-Dateien-Runtime; deren Bytes sind unverändert. Dieser Reportfix begründet weder einen zweiten Runtimefreeze noch erneute Hardwaremessungen. Beleg: `operations/h8_classifier_domain_20261008/CONTROLLER_REFERENCE_FIX_20261008.json`.

Die ausdrücklich freigegebene einmalige H10-TRT-Full-Energiefortsetzung endete nach dem ersten Repeat mit 7936 fehlenden Samples im ausgewerteten Fenster. Sofortiger STOP, kein zweiter Repeat und kein automatischer Retry; neue Quellen-/Prozessfreigabe ist belegt, der historische unbestätigte Quellenabschluss wird dadurch nicht rückwirkend bestätigt. Budgets und Originale bleiben erhalten. H8-Capture bleibt durch die ursprüngliche ungelöste Zero-Start-Kette eingeschränkt; ein Source-STOP ist keine pauschale Sperre aller unabhängigen Performanceläufe.

Ein echter eingefrorener BN6-Full-Fehler betrifft acht geplante Fälle: `postflight_oracle` wird vom normalen strikten Attestationprüfer als zusätzliches Feld abgelehnt. Der erste DeepX-Fall ist als Fehlversuch erhalten; die anderen identischen Pfade werden nicht auf Verdacht gestartet. Die drei H10-Generic-Bestätigungen sind separat an der normalen Prepared-Input-Pfadbindung blockiert. Quality-/Claim-/Energieausschlüsse bleiben in den normalen Consumers sichtbar; Sourceprüfung, lokale Tests, Hardware-Performance und wissenschaftliche Zulassung werden weiterhin getrennt ausgewiesen.


### 2026-10-08: Revision 3 – lokale Berichtsverträge und vollständige Wiederholungsreihen

Die weitere Korrektur betrifft ausschließlich lokale Controller-Verbraucher. Der originale Runtime-/Input-/Completionfreeze `8346d43d526221bd9d138943122206b658d56cbf`, sein 241-Dateien-Snapshot, die tatsächlich ausgeführten DUT-Dateien und das finale Generic-Bundle bleiben unverändert. Das bestehende `REPORTING_SOURCE_SUPPLEMENT.json` bindet die konkreten lokalen Abweichungen an diese Originalquellen. Der normale lokale Sourcecheck lässt ausschließlich die drei einzeln geprüften und exakt gebundenen Reporterdateien abweichen; alle anderen lokalen Messquellen sowie sämtliche ursprünglichen Snapshot-/DUT-/Bundleprüfungen bleiben strikt. Eine Berichtskorrektur löst keine erneute Hardwaremessung aus.

Der Finalreport prüft bei BN6 jede ursprüngliche Deklaration, jeden Prozessvertrag, die gebundenen Payloads und jeden einzelnen Count-/Zeitendpunkt. Unterschiedliche Wiederholungsverzeichnisse dürfen dabei den nachgewiesenen gemeinsamen Betriebspunkt nicht zerstören. Die bestehende Aggregation bewahrt alle drei tatsächlichen FPS-Werte, ihren normalen Median und ihr Intervall; weder originale Contracts noch deren Identitäten werden umetikettiert. Der Berichtsmirror ist mit dem Hauptreporter abgeglichen. Die bisher negative Ableitung mit fehlenden Samples bleibt erhalten; ein regulärer neuer Bericht verwendet dieselben Originalprozesse.

Der Crossreport liest die bestehenden aktuellen Qualitätsentscheidungen und unterscheidet den physischen Ausgabetensorvertrag vom tatsächlich gemessenen Completed-Task-Endpunkt. Alle ursprünglichen Wiederholungsattestationen und deren Quellen bleiben erforderlich. Das Full-Quality-Binding erhält bei den konkret verifizierten Hailo-Raw-Heads den originalen gespeicherten Sourcehash und prüft die separate physische Signatur streng; ein leerer oder widersprüchlicher physischer Hash bleibt ungültig. Der Interfacevalidator verwendet die bereits exakt gebundenen kanonischen Tensor- und Runtime-Aliasnamen für die drei belegten H10-Singletonfälle.

Gezielte Positiv-/Negativtests, normale Prüfungen tatsächlicher Originalkohorten und unabhängige Source-/Ergebnisreviews sind unter dem bestehenden Revision-3-Reportroot gebunden. Die regulären finalen Qualitäts-, Performance- und Energiezulassungen werden erst aus den tatsächlichen normalen Consumeroutputs übernommen. Gültige `accuracy_loss`-Beobachtungen, semantische Ausschlüsse und fehlende N5000-Inputgleichheit bleiben sichtbar. Insbesondere bleibt der eigenständige BN6-Full-Fehler im tatsächlich ausgeführten Remoteprüfer ein technischer Ausschluss; diese lokale Berichtskorrektur behebt ihn nicht. Die maßgeblichen laufenden Zahlen und nächsten zulässigen Schritte stehen in `STATUS.md`, `RESULT_REPORT.md` und `RESUME_STATE.json` des Reportroots.


### 2026-10-08: Revision 3 – unveränderte BN6-Deklarationsquellen erhalten

Die abschließende Gesamtkomposition erkannte einen zu breiten Zweig im neuen lokalen Berichtspfad: Sechs vorhandene DeepX-YOLO26-Splitkohorten besitzen bereits je drei identische Deklarations-/Completionverträge ohne `repNN` in der Deklarationsquelle. Die erneute Rekonstruktion verlangte dort unnötig eine lokale Stagefilebindung und blendete gültige Originaldurchsätze aus. Der bestehende BN6-Rekonstruktionszweig wird nun ausschließlich betreten, wenn die vorhandene Pfadnormalisierung eine der beiden bereits verifizierten Deklarationsquellen tatsächlich verändert. Unveränderte Quellen behalten den bisherigen strikten Originalvertragspfad. Der Mirror enthält dieselbe Korrektur; Runtimefreeze, Originalreports, DUTs und Generic-Bundle bleiben unverändert.

Die 29 vorhandenen fokussierten Tests bestanden. Eine neue Shared-Source-Fixture scheiterte zunächst an mehrfacher Verzeichnisanlage; nur die Fixture wurde korrigiert und ihr einzelner Test anschließend erfolgreich ausgeführt. Zusätzlich bestehen 18 Prüfungen realer gespeicherter Kohorten: sechs DeepX mit exakt erhaltenen FPS-/Sample-/Median-/Intervall-/Zeitwerten und zwölf Hailo mit weiterhin gültiger normaler Projektion. Keine Hardwaremessung oder N5000-Inferenz wurde wiederholt. Source, Testprotokolle und unabhängige Review stehen in `operations/finalisation_native_20261008/bn6_controller_reporting/unchanged_origin/`; die Gesamtkomposition wird anschließend regulär neu abgeleitet.

### 2026-10-09: Revision 3 – begrenzte Generic-Consumer- und Planfortsetzung

Zwölf vorhandene H8/H10-YOLO26-Kohorten verloren ihre Native-Raten im Crossreader, weil die reduzierte Generic-Projektion unterschiedliche originale repNN-Completionverträge erneut als eine unveränderte Serie prüfte. Der Reader verwendet jetzt ausschließlich den bereits gebundenen normalen Native-Report und dessen vorhandene Originalbelegprüfung. Physischer Endpunkt und Completed-Task-Vergleich bleiben getrennt; die abgeleitete Validierung erhält die geprüften Completion-/Operating-Point-Belege derselben drei Prozesse. Kein freier FPS-Fallback und keine aus FPS erfundene Requestlatenz.

26 fokussierte Regressionen bestanden, darunter alle zwölf tatsächlichen normalen Vergleiche und negative Rate-/Source-/Endpoint-Fixtures. Eine zusätzliche Plan-/Generic-STOP-Probe bestand nach der unabhängigen Reviewkorrektur; sie überlappt den 26er-Lauf. Die früheren Testfehler bleiben als XML erhalten. Der bestehende Matrix-Updater korrigiert den Fortschritt nach vollständigem Native-/Generic-Regime:14 semantische Statusfehler und204 falsch aus Native übernommene Generic-Quellbindungen wurden berichtigt. Scopeentscheidungen und Messdaten bleiben erhalten. Der laufende Gesamt-Current-Abschluss entscheidet die finalen Counts; ein Hardware- oder Energie-PASS wird daraus nicht abgeleitet.

Runtimefreeze8346/241Dateien, DUTs, Engines und Budgets bleiben unverändert. Die vorhandenen14 DX-Energiekandidaten sind nur Metadaten-PASS; kein Parent wurde vorbereitet und keine Messung gestartet. H10-STOP und verbrauchter Einmalgrant sowie H8-Zero-Start bleiben bestehen. Gemeinsamer Diff, Test-XML und Review liegen unter `operations/rev3_three_points_20261009/` im Reportroot.


### 2026-10-10: Revision 3 – Generic-Resume und gezielte Energieübernahme

Der normale Generic-Parent bindet die tatsächlich erworbenen Transferressourcen und legt je explizitem Resume neue Attemptpfade an. Er erhält Originalparent, wissenschaftliche Eingaben und kampagnenweites Case-/Repeatbudget. Gültige kompatible Quellen werden über den normalen Energieimport übernommen; fehlende Repeats bleiben in ihren ursprünglichen logischen Slots. Deskriptoren werden vor dem Consumerimport gespeichert und ältere Quellen ohne Manifest anhand ihrer originalen Start-/Prozess-/Download-/Planbelege rekonstruiert. Der erste tatsächlich gespeicherte Repeat wurde ohne Hardwarestart normal reimportiert; die anschließenden beiden Aufnahmen führte der Hauptagent aus.

Der normale Current-Consumer übernimmt das vollständige gespeicherte Ergebnis über LATEST_ATTEMPT. Sein gezielter Energiepfad erhält die anderen 203 Generic-Zeilen, vorhandene Performance-/Qualitygates und Crossrunner-/Repeatbelege; ein separater vollständiger Export bleibt für neue Performance-/Qualitybefunde verfügbar. Der erste Fall wurde normal in Current, Matrix und Plan übernommen: drei gültige Repeats, Energie zugelassen und Claim weiterhin ausgeschlossen. 30 Resume-/Ownershiptests, 24 Batchtests, 30 Current-/Importtests und 15 Tests für die exakt bestehende H10-Follow-up-Gruppe sowie unveränderte Casezulassung bei kumulativen Currentupdates bestanden. Der reale H10-Plan erreicht die erhaltenen normalen Source-/Kohortengates; eine parallel geänderte lokale Reportingquelle muss dort noch ausdrücklich gebunden werden. Keine Hardwarestarts, Deployments oder Budgetmutationen durch diese Integration. Belege stehen unter operations/token_saving_batch_20261009/continuous_20261010 im bestehenden Reportroot.


### 2026-10-10: Revision 3 – gezielte laufende Abschlusskorrekturen

Die fortlaufende Kampagnenautorisierung bleibt an die vorhandenen Fälle, Quellen und Parents gebunden. Originale STOPs, Fehlversuche und kumulative Starts werden erhalten. Generic verwendet seine eigenen Completed-Task-Aufnahmen und den tatsächlich gemessenen Ausführungskontext; die Wiederaufnahme verändert weder Messkommando noch wissenschaftliche Eingaben. Der isolierte DeepX-Qualitätspfad erfasst den aktuellen direkten RGB-uint8-Eingang und übernimmt vorhandene gültige Native-Prozessbelege ausschließlich nach nachgewiesener numerischer und semantischer Bindung. Alte Float32-Rückwandlungsbefunde werden nicht umgedeutet.

Der Native-Observer erkennt nun auch den tatsächlich gestarteten H10-TRT-Splitworker. Die ersten drei Aufnahmen ohne diese Workerzuordnung bleiben final ungültig. Beide Native-Einstiegspunkte behalten bei Wiederverwendung gültiger Rows den ursprünglichen Capturepräfix; historische Baselines außerhalb dieses Präfix bleiben streng geprüfter Vergleichskontext. Die BN6-Berichtskorrektur bindet die belegten ursprünglichen Materialisierungen und unveränderten Decode-/NMS-Verträge statt Sourceänderungen pauschal als neuen Messbetriebspunkt zu behandeln.

Die gezielten Regressionen und tatsächlichen Normalimports sind im bestehenden Reportroot dokumentiert. Neue portable Tests prüfen Capturepräfix und Worker-Erkennung an beiden Einstiegspunkten sowie die belegten Decoder-/Materialisierungskorrekturen. Die laufende Revision 3 ist noch nicht insgesamt abgeschlossen; aktuelle Gruppen, normale Current-Pointer, Archive und nächste Schritte stehen in den bestehenden STATUS-/RESUME-Dateien.


### 2026-10-11: Revision 3 – autorisierte Sampleverlust-Retries

Reine Sampleverluste bleiben ungültig. Die vorhandene fortlaufende Autorisierung erlaubt gezielte endliche +3-Serien für fehlende logische Slots nach geprüftem Sourceabschluss und freigegebenen Ressourcen. Historische STOPs, Originalversuche und kumulative Zähler bleiben erhalten; wissenschaftliche Kriterien und DUT-Runtime bleiben unverändert. Der initiale Slotabschluss berücksichtigt eine verbrauchte Sourcequote vor der nächsten normalen Admission.

Der normale Reader prüft wiederverwendete gültige Retryversuche anhand ihrer exakten Originalchain, des unveränderten physischen Reports und der bestehenden Auswahlhistorie im Originalaggregate. Keine Auswahl nach Messwerten und keine neue Aufnahme zur Readerkorrektur. 37 Budget-, 17 Recovery- und gezielte Collectorregressionen sowie fünf Initialslot- und 20 Reader-/Reuseprüfungen bestanden. Tatsächlich: b085 unter Erhalt seiner zwei Originalrepeats geschlossen; b132 mit drei gültigen Slots regulär bestätigt. Die bestehende Restqueue läuft weiter. Aktuelle Imports, Worker und ausstehender Archiv-/Paper-/Publicabschluss stehen im vorhandenen Reportroot; Revision 3 ist noch nicht insgesamt abgeschlossen.


Der Generic-Controller erhält seine drei exakten Einzelerfassungen und verwendet für reine bestätigte Sampleverluste den bestehenden normalen Parent-Resume. Ein aktiver Broker/Budgethook prüft Freigabe und Ressourcen; erst nach komplettem Parentexit und normalem Cleanup folgen höchstens drei getrennte Resumes mit fünf Sekunden Pause. Gültige Slots, Current-/LATEST-Attemptpfade und die unveränderte wissenschaftliche Zulassung bleiben erhalten. 27 gezielte Series-/Abbruch-/Bindungs-/Reuse-/Limitregressionen bestanden; der reale DeepX-Mobile-b001-Resume wurde über dieselbe Restqueue gestartet.


### 2026-10-11: Revision 3 – normaler Generic-Recoveryhandoff

Die bestehende physische Vierressourcen-Recovery bindet jetzt auch das genaue Generic-Parentlayout einschließlich Original-/Resume-Plan, terminalem Journal und PREPARED. Die normale Budgetübergabe verwendet den vorhandenen Generic-Unterordner; der externe Adapter übernimmt dieselben freigegebenen Vorgänger und den bestehenden Runfence-Callback wie Native. Historische Quellenabschlüsse, Fehlversuche, Zähler und wissenschaftliche Kriterien bleiben erhalten. Der Controllerrefresh erlaubt ausschließlich die bereits gebundene geänderte Recoverydatei.

53 gezielte Tests und die unabhängige Prüfung bestanden. Die konkrete H8-RegNet-b001-Operation capture-6844f7b6f43a412aaba7c5d9bae73146 wurde anschließend normal physisch freigegeben, mit null neuen Captures und regulärem Sourcebudgethandoff. Der historische Quellenabschluss bleibt unbestätigt; fünf ungültige Versuche bleiben erhalten. Die Kampagne ist weiter offen. Current: 165 dokumentierte Native-/157 finale Punkte sowie 20 normale Generic-Energiepunkte; laufender Zustand und Belege stehen im vorhandenen Reportroot.


### 2026-10-11: Revision 3 – vorhandene Energieparents im regulären Produkteinstieg

`onnx-splitpoint-complete-evaluation --run-dir <vorbereiteter Energieparent>` verwendet den exakt gespeicherten Native- oder Generic-Plan samt unveränderter wissenschaftlicher Zulassung. `--resume` setzt fehlende Repeats fort; `--import-only` übernimmt vorhandene Ergebnisse ohne Collectoraufruf. `--plan-only --output-dir <externes Verzeichnis>` zeigt Auswahl und Quellen ohne Hardwarestart. Eine konkrete vorhandene Recoveryautorisierung wird mit `--recovery-authorization <AUTHORIZATION.json>` über denselben normalen Ressourcenvorgang verarbeitet. Der bestehende EvalRun-Pfad bleibt erhalten.

Die wiederverwendbare Parent-/Resume-/Retrylogik liegt im Toolpaket; historische Reportwrapper delegieren kompatibel. Reine nachgewiesene Sampleverluste erhalten alle gültigen Slots und verwenden normale endliche Retryserien. Originale unbestätigte Quellenabschlüsse bleiben unbestätigt; sichere aktuelle Wiederverwendung erfordert die normale Ressourcenbeobachtung. Importfehler führen nicht zu einer erneuten Hardwareaufnahme. Der bestehende Batch nutzt denselben Produkteinstieg und kann mit `stop --after-current-parent` nach Cleanup und normalem Import pausieren.

Der Generic-Vergleich übernimmt einen vorhandenen qualifizierten Normalvergleich nur bei exakt gebundener Performancequelle, ausgewählter Identität und unverändert gültigen Energieprimärgates. Die belegte fehlende H8-MobileNet-Projektion wird dadurch geschlossen; negative Qualität oder Capturegates werden nicht freigegeben. Keine Änderung an DUT-Runtime, Messfenster, Samples, erstgültiger Auswahl oder kumulativen Startzahlen.

Die zusammenhängende Integration ist gezielt getestet und unabhängig geprüft. Reale Produktabnahme, Current-/Archivstand und verbleibender Gesamtabschluss werden im bestehenden Revision-3-Reportroot dokumentiert. Der Gesamtauftrag ist dadurch noch nicht abgeschlossen.


## 12.10.2026 — Toolabnahme und veröffentlichbarer Produktstand 2.93.0

Eigene Softwarephase; die Restmesskampagne bleibt gespeichert und pausiert. Der bestehende Stand wurde nach Sicherung der abweichenden Benutzerinstallation bytegenau mit dem Worktree konsolidiert. Keine neue Modellgruppe, kein N5000-Lauf, kein DUT-Deployment und kein Artefaktneubau.

GUI-Startcallback und normaler profilgesteuerter CLI-Einstieg werden mit derselben speicherbaren Konfiguration bis zu den tatsächlich materialisierten Generic-/Native-Backendrequests geprüft. Run-mode-Budgets bleiben maßgeblich; Nullwerte in der Profilauflösung und im expliziten Remote-Warmup werden erhalten. Die GUI zeigt einen korrekt quotierten profilgesteuerten Startbefehl.

Native-/Generic-Originaldatenimporte und die gezielte Generic-Current-/CSV-Projektion sind nun paketiert. Gespeicherte historische Importargv behalten ihre Kontrollen und lösen regulär auf den installierten Produktleser auf; ein altes Arbeitsverzeichnis oder PYTHONPATH überschattet ihn nicht. Die historischen Leser, Daten und Belege bleiben erhalten. Neue Native-Resumes verwenden den normalen Collector statt eines privaten Modul-Overlays und prüfen seine installierte Quellenbindung.

Originaldaten-Reprocessing: sechs gespeicherte Parents, 23 Punkte auf H8/H10/DeepX, exakt gleiche Ergebnis-/Entscheidungsfelder. 236 originale JSON-Quellen sowie Queue, Budget und Vergleichsmatrix bleiben unverändert. Das ist ein Reader-/Produktnachweis mit vorhandenen Hardwaredaten, keine neue physische GUI-/Collectorabnahme. Private vollständige Fairness-Neuauswertung bleibt ein eigener Berichtsvorgang.

Gezielte lokale Tests, Installations-/Archivprüfung, regulärer GUI-Start und tatsächlicher Remotecommit werden im bestehenden Operationsbereich unter `tool_acceptance_push_20261012` belegt. Automatisierte Controller-/Signalprüfung und echte Tk-Start-/Editorprüfung werden getrennt von physischer Ausführung ausgewiesen.


### 2026-10-12: endliche neue Hardwareabnahme je Netz und Setup

Die explizite Zusatzbeauftragung begrenzt die Abnahme auf 21 Netz/Setup-Zellen
und je eine vorab ausgewählte unterstützte Splitposition. Native und Generic
Completed erhalten getrennte frische Performance- und Energiebelege; eine
vollständige Splitkampagne und die Revision-3-Restqueue bleiben pausiert.
Die Matrix, Auswahlregel, normalen 100/1000/3-Performancewerte, das vorhandene
60-s-/3-Repeat-FS-command-Energieverfahren und normale Retrygrenzen liegen vor
dem ersten Hardwarestart im bestehenden Abnahmebereich. Fulls bleiben separat
gebundene Referenzen; fehlende Energievergleiche und negative Qualität bleiben
sichtbar. Es wird keine neue N5000-Auswertung gestartet.

Die Produktprüfung zeigte fehlenden Generic-Completed-Dispatch im regulären
GUI-Workflow und eine unzureichende Fallbegrenzung beim bestehenden Suiteimport.
Der neue gespeicherte Abschnitt `hardware_acceptance` läuft deshalb über den
regulären GUI-Startcontroller und denselben profilgesteuerten CLI. Er verbindet
endliche Auswahl und Prepare mit den vorhandenen Runnern, normalen Sperren,
Prozessleases, Collectors und Ergebnislesern in einem neuen EvaluationRun.
Ein eigener endlicher Budgetstand zählt die neue Abnahme, während historische
Messungen, Originalqueues und Originalbudgets erhalten bleiben.

Die konkrete Native-ACQUIRING-Recovery erhält den unbewiesenen historischen
Quellenabschluss und prüft aktuelle Ressourcen unter den normalen Sperren.
140 fokussierte Recoveryprüfungen und 33 gezielte Release-/Installerprüfungen
bestanden; die neue Controller-/Adapterabdeckung wird separat geprüft.
Der reale Hardwareabschluss wird ausschließlich aus den anschließend tatsächlich
ausgeführten GUI-Jobs und normalen Imports berichtet. Der veröffentlichte
2.93.0-Softwareabschluss mit 377 Fällen ersetzt diese Hardwarebelege nicht.

Die erste installierte sichtbare GUI-Ausführung bestätigte Generic Completed
für MobileNet/Hailo8 mit drei neuen Performanceprozessen und drei gültigen
FS-command-Aufnahmen, normalem Import und Ressourcenfreigabe. Native schloss
zunächst 1000 Tasks ab, scheiterte danach aber beim Download an fehlender
Controller-Aktivitätsbindung. Der Runner schrieb zudem seinen Hilfsbericht in
das wiederverwendete Benchmarkverzeichnis. Die gezielte Korrektur bindet die
normalen Transfers und gibt jedem Repeat ein eigenes beschreibbares
Benchmark-Metadatenverzeichnis; Modellartefakte bleiben unverändert gebunden.
Die gültige Generic-Kohorte bleibt erhalten. Die Fortsetzung prüft nur die
41 noch nicht abgeschlossenen Pfade; der zusätzliche Native-Prozess wird im
Abnahmebericht separat gezählt. Die Restkampagne bleibt pausiert.

Die gezielte Native-Fortsetzung lieferte drei gültige frische Performance-
Repeats. Der folgende Energiestart wurde vor dem Collector abgewiesen, weil
diagnostische Rohaufnahmen und erforderliche Umweltbeobachtung bislang
widersprüchlich behandelt wurden. Die Korrektur erlaubt die normale technische
Rohabnahme mit unveränderten Umwelt-/Fenster-/Budgetprüfungen; wissenschaftliche
Claims bleiben falsch, und finale wissenschaftliche Kohorten bleiben gesperrt.
Gültige Performance aus einem abgeschlossenen Abnahmelauf derselben Matrix wird
beleggebunden erhalten; die Energie-Fortsetzung führt weiterhin neue reale
Captures durch. Sie zählt keine alten Prozesse als neue Ausführung.

Generic/MobileNet/Hailo10 lieferte ebenfalls drei gültige frische Performance-
Repeats. Sein erster Capture scheiterte an Empfangsstille ohne Quellenendbeleg.
Die reguläre Recovery akzeptiert jetzt auch den exakt belegten terminalen
Partial-Workflow, prüft dessen unveränderten negativen Quellenabschluss erneut
nach aktueller Ownerbeobachtung und erhält STOP, Fehlversuch und Budget.
Die Fortsetzung zählt diesen ersten Start weiter gegen die bestehenden drei
Versuche je Slot und neun je Pfad. Kein Quellenabschluss wird nachgetragen.
Für Hailo10/YOLOv7 wird der vom normalen Runner benötigte vorhandene Part1-ONNX
zusätzlich schreibgeschützt in das private Metadatenverzeichnis übernommen;
die ursprüngliche Artefaktbindung bleibt erhalten, ohne Build oder Cacheeingriff.
Der tatsächliche Abschluss der 21 Zellen bleibt bis zu den weiteren realen
GUI-Ausführungen offen.

Der erste reale Native-Energieretry auf Hailo8 scheiterte an Empfangsstille
ohne Quellenendpaket. Die normale Ressourcen-Recovery erhält diesen invaliden
Versuch. Für die endliche Fortsetzung prüft ein SHA-gebundener normaler
Admission-Hook vor jeder Collectorreservierung, einschließlich interner Retries,
die alten und aktuellen Starts derselben Matrix: höchstens drei je Slot und
neun je Pfad. Der erste Fehlstart verbraucht eine Slotgutschrift. Die Profile
behalten drei Repeats und maximal zwei Retries. Alte akzeptierte Energie aus
einem anderen Job wird ohne normale Übernahmeprüfung abgewiesen; gültige
Repeats im aktuellen Job bleiben über den bestehenden Reusepfad erhalten.
77 Native-, 17 Coordinator- und 103 relevante Coreprüfungen bestanden;
unabhängige Prüfung bestätigte den begrenzten Fortsetzungspfad.

Generic/MobileNet/Hailo10 hat nach normaler Recovery drei neue gültige
Energieaufnahmen und normale Ergebnisübernahme/Darstellung abgeschlossen.
Die drei bereits gültigen aktuellen Performance-Repeats wurden erhalten,
ohne einen weiteren Performanceprozess zu starten. Der alte Fehlversuch bleibt
zusätzlich gezählt. Die weiteren Zellen werden weiter ausschließlich über
die gespeicherte endliche GUI-Startliste abgearbeitet.

Die anschließende Nutzerkorrektur begrenzt die praktische Abnahme auf genau
21 Modell-/Setup-Kombinationen. Die umfangreiche Startliste wurde regulär
an der Zellgrenze beendet. Acht Kombinationen besitzen bereits gültige neue
Performance- und Energiebelege; diese werden ohne Wiederholung übernommen.
Für die verbleibenden 13 Zellen fordert der explizite Profilmodus
`hardware_acceptance.mode: functional` jeweils genau einen zugewiesenen
Runner, einen vollständigen Prozess mit 100 Warmups und 1000 Tasks sowie
eine gültige 60-s-FS-command-Aufnahme. Zwei reguläre Retries bleiben erlaubt;
die unveränderte große Restkampagne bleibt pausiert.

Der normale GUI-/CLI-Koordinator, Generic-Dispatcher und Energieimport
unterscheiden diese Einzelprobe ausdrücklich von wissenschaftlichen
Dreierkohorten. Numerischer Output gegen den gebundenen Oracle, normaler
Import, Kalibrierung, Taskcount, Drain und Quellenabschluss bleiben Pflicht.
Einzelproben erhalten keine neuen wissenschaftlichen Claims oder
Kohortenstatistiken. Die vorhandenen drei Repeats und sämtliche Fehlversuche
bleiben unverändert erhalten. Der wissenschaftliche Standardpfad bleibt bei
drei Wiederholungen. Neue Full-Baselines und N5000-Auswertungen gehören nicht
zu dieser reduzierten Abnahme. Die installierte Version bleibt 2.94.0.

Die erste reduzierte reale GUI-Ausführung bestätigte ResNet50/DeepX mit
genau einem gültigen Performanceprozess, einem gültigen 60-s-Capture,
numerischer Oracleprüfung und normalem Import. Damit sind neun Zellen
abgeschlossen. YOLOv7/Hailo8 wurde danach noch vor dem SDK-Start abgewiesen:
dem historischen Vertrag für belegte HWC-RGB-uint8-Bytes fehlte die explizite
Zahlenbereichsangabe. Die gezielte Metadatenkorrektur übernimmt `uint8_0_255`
nur bei verifiziertem RGB-Preprocessing und identischem vorbereiteten
Payload-/RGB-Hash. Eingabebytes, Padding, Modellartefakte und Oracle bleiben
unverändert; der Fehlversuch und seine null Capturestarts bleiben erhalten.
Die Fortsetzung umfasst ausschließlich die zwölf offenen Detektorzellen.

Die reduzierte Hardware-Funktionsabnahme wurde am 12.10.2026 um 19:46 Uhr
regulär beendet: 21 von 21 festen Modell-/Setup-Kombinationen besitzen gültige
Performance-, Energie- und numerische Outputbelege sowie normale GUI-/CLI-
Ergebnisübernahme. RegNet verwendet auf den drei Acceleratoren Native, die
übrigen 18 Zellen Generic. Acht bereits vollständig ausgeführte Kombinationen
wurden übernommen; die 13 offenen Zellen erhielten jeweils genau einen neuen
Prozess mit 100 Warmups/1000 Tasks und eine gültige 60-s-Aufnahme. Für diese
13 Aufnahmen war kein Retry nötig. Der YOLOv7/Hailo8-Metadatenfehlversuch vor
SDK-Start bleibt erhalten; nur dessen betroffener Test wurde nach Korrektur
erneut gestartet. Es gab keine neuen Full-Baselines oder N5000-Auswertungen.

Die gesamte frühere und reduzierte Abnahmehistorie bleibt separat belegt:
61 lokal bewiesene Performanceprozesse, 63 Dispatchversuche (62 remote
abgeschlossen) und 65 physische Energiestarts mit 61 gültigen und vier
ungültigen Aufnahmen. Zwei alte unbestätigte Quellenabschlüsse bleiben
unbestätigt; aktuelle Recoveryfreigaben ändern diese negativen Belege nicht.
Diese Summen enthalten die früheren umfangreicheren Läufe und sind kein
zusätzlicher Umfang der reduzierten Fortsetzung.

Der Abschlussbericht führt genau 21 Zeilen sowie getrennte Performance-,
Energie-, Output- und Importprüfungen. Für sechs zugewiesene Betriebspunkte
fehlt eine geeignete historische Energievergleichsaufnahme; Native behält
seine diagnostischen Freigaben und unvollständigen historischen Vertragsfelder.
MobileNet/Hailo8 behält den historischen N5000-Befund `accuracy_loss` trotz
bestandener aktueller numerischer Funktionsprobe. Unterschiede zu bisherigen
FPS-/Energiewerten bleiben mit vorhandener Streuung und belegten Bedingungen
beschrieben, ohne neue Toleranzgrenzen, Kausalbehauptungen oder Einzelproben-
Statistikclaims. Belege liegen unter
`~/Reports/native-fairness-rev3-20261007/operations/token_saving_batch_20261009/hardware_acceptance_20261012/analysis_review/REDUCED_ACCEPTANCE_EVIDENCE.json`
und `analysis_matrix/HARDWARE_ACCEPTANCE_REPORT.txt` im selben Abnahmeverzeichnis.

Gezielte Softwareprüfungen bestanden separat: 20 Coordinator-, 81 Native-,
54 Generic-Regressions- und 25 Eingabedomänenfälle sowie ein normaler
GUI-Workflowtest; alle 13 reduzierten Profile wurden regulär zurückgelesen.
Der finale Quellarchivstand bleibt Version 2.94.0. Gegenüber dem tatsächlich
gemessenen v8-Quellstand werden zum Abschluss nur dieser Arbeitsstand und die
normalen Quellenindizes ergänzt. Die große Revision-3-Restkampagne bleibt
pausiert; nach Git-/Installationsabschluss wird keine weitere Messung gestartet.

## 2026-10-14: Begrenzte DeepX-Sync/Async-Diagnose abgeschlossen

Auftrag: `~/Downloads/CODEX_DEEPX_SYNC_ASYNC_DIAGNOSE_20261014.txt`.
Diagnosecode, Plan, Tabellen, Rohreports und Resumezustand liegen unter
`~/Reports/deepx_sync_async_diagnose_20261014/20261014T173407Z/`.
Abgeschlossene Ursachenklärung und frühere Stufenkohorte wurden wiederverwendet;
b002 blieb Offlinekontrast. Verwendet wurden MobileNet b027 und ResNet50 b045
mit vorhandenen Artefakten auf dem registrierten DeepX-DUT.

Die separate Native-Diagnose erhält S1 und ergänzt S1/A1/A2/A4 mit einer Engine,
vier festen privaten Outputslots, sicherer Requestzuordnung, eigener Boundarykopie
und vorhandenem TRT-Hoststaging. 193 Tests bestanden vor dem Hardwarestart;
die letzte vollständige Diagnosesuite bestand 246 Tests, danach die gezielte
Auswertungsprüfung 30 Tests; 54 bestehende Regressionstests wurden wiederverwendet.
Alle 20 Hardware-Paritäts-/Smokebedingungen und 60/60 Performanceprozesse bestanden.
600.000 Main-Completions, 6.000 drainierte Warm-ups, keine Performance-Ersatzversuche
oder Profile. Vier vor Inferenz abgebrochene Referenzprozesse bleiben erhalten:
Das SDK verlangte explizites NHWC-Batch 1; RGB-Nutzdaten und Toleranzen blieben gleich.
Die unabhängige Vollprüfung und der zusätzliche Vergleich gespeicherter serieller
Referenzen mit den letzten A4-Messausgaben bestanden. Reguläre Lease freigegeben,
keine Quarantäne und kein eigener STOP; Abschlussbeleg siehe STATUS.md.

MobileNet PC: S1 760,853/s, A1 742,267/s, A2 770,125/s, A4 773,405/s (Mediane).
ResNet PC: 220,588/s, 222,197/s, 425,326/s, 567,329/s; wegen thermischer Grenzen
nur deskriptiv. Insgesamt besitzen 21 Prozesse eingeschränkte Zustandsvergleichbarkeit.
Full S1→A4 gewinnt blockweise im Median 2,414× beziehungsweise 2,013×, bei
vergleichbaren Zuständen in allen drei Blöcken. Verbesserter Fullbetrieb bleibt
schneller als der jeweilige Split. Empfehlung: gezielter Native-Full-Async-Patch
als separate Integration; kein pauschaler Split-/Generic- oder Kern-/PCIe-Nachweis.

Dies ist eine Diagnoseabnahme, keine neue Normal-GUI- oder N5000-Abnahme.
Produktive Defaults, Generic, historische Qualitätsurteile und kanonische
Ergebnismatrizen bleiben unverändert. Keine Builds, Energieaufnahme oder neue
Hash-/Registry-Infrastruktur. Bereits vorhandene Energie-Dateiänderungen wurden
nicht verändert; dieser Auftrag ergänzt im Repository ausschließlich diesen Stand.


### 2.95.1 — gezielte Native-Neuaufnahme im normalen GUI-Workflow

Der getrennte `native_refresh`-Selektor bindet die 75 ursprünglichen DeepX-Native-UIDs in sieben normalen Modelljobs; die Abnahme bleibt bei höchstens einem Split je Modell/Setup. Bestehende Prepare-/Numerik-/Performance-/Energie-/Import- und Ressourcenpfade werden verwendet. Vollständige GUI-Abnahmekohorten werden pro UID über echte 3+3-Rohdaten, normale Importbindungen, A4/max4 und den bestehenden Profilsnapshot geprüft; Reuse meldet null neue Aufrufe. Der echte Tk-Queuepfad ergänzt ausschließlich beobachtende GUI-Job-/Fensterdaten im normalen Ausführungssession-Manifest, außerhalb der wissenschaftlichen Resumeoptionen. Alle sieben gespeicherten Profile sind regulär aufgelöst und alle 75 Punkte offline vorbereitet. Das ist Softwareevidenz: keine GUI-/Hardwarestarts durch diese Integration, kein A4-Hardware-PASS und keine Current-Aktivierung. Details/erhaltene Fehlprüfungen: `/home/kmika/Reports/deepx_295_gui_acceptance_remeasurement_20261015/20261015T093320Z/CAMPAIGN_INTEGRATION.md`.


### 2026-10-15: Reguläre CLI-Fortsetzung der DeepX-A4-Abnahme

Die ausdrückliche Nutzerkorrektur erlaubt die installierte profilgesteuerte CLI und hebt sichtbare GUI-Herkunft als Abnahmebedingung auf. Der vorhandene Full-only-Run 165459 ist inzwischen regulär mit Numerik-PASS und Geräte-/Ressourcenabschluss beendet. Der b027-PASS bleibt erhalten; beide Zellen werden nicht erneut numerisch gemessen. Die 14-Zellen-Abnahme und die 75-Punkte-Neuaufnahme verwenden dieselben vorhandenen Profile, Produktworker, Messrezepte und normalen Importe. Die Vollabnahme und Kampagne sind noch offen. Quellen-/Numerik-/Ressourcenprüfungen bleiben unverändert.

Der reguläre CLI-Run `...183413` endete am 15.10.2026 um 18:40:43+02:00 mit Teilstatus und normalem Artefakt-/Ressourcenabschluss. Full und b027 haben jeweils drei gültige Performanceprozesse, noch keine Energieaufnahme. Die Full-Energieplanung wird bei fehlender Vergleichspräzisionsannotation einheitlich mit ihrer bereits versiegelten Kennzeichnung geschrieben; tatsächliche Runtimepräzision und Qualitätsflags bleiben erhalten. Ein gemeinsamer Leser der archivierten Workflowoptionen ermöglicht die explizite profilgesteuerte CLI-Wiederaufnahme desselben Runs; der normale Reader hat alle sechs bestehenden Performance-Slots erneut geprüft. Der unveränderte Teilstand einschließlich fehlerhafter Energiepläne ist vor Resume im Auftragsreport gesichert. Die noch offene Trennung diagnostischer Qualitätsaussagen von technisch gültigen Native-A4-Energiekohorten wird gezielt korrigiert; kein Energie-PASS wird vor tatsächlicher Aufnahme beansprucht.

Die Energiekorrektur ist auf bereits gebundene Native-A4-Abnahmekohorten mit der vorhandenen Capture-Admission begrenzt. Exakte 60-s-/3-Repeat-/FS-/Task-/Fensterbindungen und tatsächlich zugelassene Capture-Belege bleiben Pflicht; diagnostische Qualitätskennzeichnung und explizit falsche Claim-Flags bleiben erhalten. Generische/Hailo-Diagnosekohorten bleiben abgewiesen. Sieben gezielte Writer-/Aggregation-/Reuse-/Importprüfungen und 53 bestehende Regressionen bestanden; sie ersetzen keine physische Energieaufnahme. Die getrennte Full-Präzisionskorrektur bestand 26 gezielte Prüfungen (davon fünf unabhängig wiederholt), die gemeinsame CLI-Resume-Integration 153 Prüfungen sowie die unabhängige Prüfung aller sechs Originalslots.

Der erste reguläre CLI-Resume endete am 15.10.2026 um 18:53:04+02:00 erneut mit Teilstatus, ohne neue Performanceprozesse oder Energieaufnahmen. Die sechs gültigen Prozesse blieben erhalten. Ursache des Checkpointkonflikts war die JSON-bedingt veränderte Reihenfolge zweier Environmentvariablen; die gezielte Korrektur reproduziert beide ursprünglichen Invocation-Hashes und bestand 125 Tests. Beide terminalen Teilstände sind vor weiterer Wiederaufnahme vollständig außerhalb des Quellbaums gesichert. Für die belegten Energiefehler vor Collector-/Workloadstart wird der normale verwaltete Energieversuch unter Erhalt der alten Checkpoints und desselben physischen Budgets neu angesetzt; die normalen Leser folgen dem tatsächlich ausgewählten Versuch. Noch keine neue physische Energieaufnahme und noch keine Ergebnisaktivierung.

Die reguläre CLI-Wiederaufnahme von MobileNet Run183413 ist am 15.10.2026 um 19:20:17+02:00 mit status=ok, technical_status=ok und normalem Artefakt-/Ressourcenabschluss beendet. Full und b027 enthalten jeweils drei erhaltene Performanceprozesse und drei gültige 60-s-Energieaufnahmen. Ein Full-Capture mit verlorenen Sensormesswerten wurde regulär ersetzt und bleibt erhalten (sieben physische Starts insgesamt); keine neuen Performanceprozesse. Beide normalen Kampagnen-Kohortenprüfungen bestanden einschließlich A4/max4, Split-FIFO3, Kalibration, Umgebung und Originalquellen. Danach bestanden RegNet, ResNet und YOLO11l jeweils Full und gespeicherter Split numerisch mit 32/32/64. YOLO26m Run194012 endete regulär mit Teilstatus: b038 PASS; Full scheiterte vor Workerstart lokal am SSH-Argumentlimit (146682 Bytes gegenüber 131072). Die korrigierte Übertragung komprimiert das unveränderte Staging-Programm; die Originaldateien, Modellartefakte, Quellenprüfungen und Prozess-Leases bleiben maßgeblich. Die fehlende Full-Zelle wird nach gezielter Prüfung über ein normales Teilprofil aufgenommen; vorhandene PASS bleiben erhalten. Vollabnahme weiterhin 2/14, Numerik aktuell 9/14; keine Current-Aktivierung.


### 2026-10-18: YOLO26x — begrenzte TensorRT-Cacheprobe-Korrektur

Die reguläre 20-Split-Vorbereitung endete mit normalem Artefaktabschluss. Der TensorRT-Cacheverifier mit 41 Anforderungen überschritt lokal das Linux-Argumentlimit (144.149 Bytes); es wurde dabei kein Remoteprozess gestartet. Die vorhandene vollständige Pythonprüfung wird nun wie beim produktiven Full-Staging zlib/Base64-komprimiert übertragen. Programmbytes, Cacheidentitäten, Receiptprüfung und HIT/MISS/UNKNOWN-Semantik bleiben unverändert; das maximale SSH-Argument beträgt für denselben Payload 81.841 Bytes. 33 gezielte Regressionstests einschließlich tatsächlichem Payload, Bytegleichheit, lokalem execve und HIT/MISS/Tamper sowie der unabhängige Review bestanden. Dies ist ein lokaler Transportfix auf 2.95.5, keine neue Messmethodik oder neue Releaseversion. Ursprünglicher Teilstand und Softwarebelege: `/home/kmika/Reports/yolo26x_final_thesis_20splits_20261018/transport_fix/` und `preflight_before_resume/`. Die vollständige Kampagne wird danach über den normalen CLI-Resume fortgesetzt; Hardware-/Qualitäts-/Energieerfolg ist damit noch nicht behauptet.


### 2026-10-18: YOLO26x — äußere Hailo-Split-Ausgänge

Im normalen 20-Split-Lauf meldeten beide Hailo-Compiler für b031 einen FeatureSplitter-Ausgang ohne eindeutigen Nachfolger. Die vorhandenen äußeren Identity-Adapter werden vom SDK entfernt. Der eng begrenzte Fix materialisiert ausschließlich tool-eigene YOLO26x-Part1-Ausgänge eines statisch bestimmten Kanal-Splits als konstante Slice-Ausgänge. Ursprüngliche Split-Knoten, interne Verbraucher, Gewichte, Ausgabealiase und die gemeinsame 20er-Auswahl bleiben erhalten. Der Fix greift vor dem tatsächlichen Compilerhash im bestehenden Deferred-/Cachepfad; ein eigener Dateisuffix erhält die ursprünglichen fehlgeschlagenen Compilerdateien. Andere Modelle und die unveränderten Compilergraphen von b003/b062 behalten Bytes, Metadaten und Cachekeys. Force-Rebuild bleibt aus.

23 fokussierte Kandidatenprüfungen einschließlich sechs bitgleichen CPU-ORT-Ausgabevergleichen sowie 22 unabhängige Prüfungen bestanden. Ein SDK-Parser-/HEF- oder Qualitäts-PASS für den neuen Adapter ist damit nicht behauptet. Vor der Integration wurde der eigene Controller nach dem versiegelten Hailo-10H-HEF für b062 regulär abgebrochen; drei fertige HEFs blieben erhalten, der bereits angelaufene Hailo-8-b062-Versuch wurde als abgebrochen dokumentiert. Der normale Artefaktabschluss bestand für 10.704 Dateien. Qualitäts-, Performance- und Energiephasen waren noch nicht gestartet. Originalzustand, Fehler, Softwareprüfungen und Resume-Belege liegen unter `/home/kmika/Reports/yolo26x_final_thesis_20splits_20261018/hailo_output_fix/`, `terminal_before_hailo_output_fix_111949/` und `preflight/hailo_b031_diagnosis/`. Die Fortsetzung erfolgt über denselben Run und die reguläre profilgesteuerte CLI; Kalibration 500/opt1, N5000/B1000 und die freigegebenen Messprotokolle bleiben unverändert.


### 2026-10-18: YOLO26x — Zeitstempel beim regulären Scope-Resume

Der anschließende CLI-Resume wurde vor neuen Compiler- oder Messläufen durch den unveränderlichen Model-Scope gesperrt. Der Generator hatte ausschließlich `benchmark_plan.created_at` erneuert; alle 604 Kandidaten, 20 akzeptierten Splitdefinitionen und 84 physischen Generic-Identitäten blieben gleich. Der vollständige eingebettete Planhash behandelte diesen Zeitwechsel als Scopeänderung. Der eigene Lauf wurde während der Cacheprüfung kontrolliert beendet und mit bestandenem Artefaktabschluss als zweiter Checkpoint gesichert.

Der begrenzte Kompatibilitätspfad prüft den vorhandenen Scope und die ursprüngliche Produkt-Session, grenzt die ursprüngliche Planzeit durch deren Generierungsintervall ein und akzeptiert ausschließlich eine Fassung mit exakt dem bereits versiegelten vollständigen Planhash. Danach bleibt der normale vollständige Scopevergleich unverändert Pflicht; bestehende Scopebytes werden nicht umgeschrieben. Andere Planfelder, Cases, Endpoints oder Messparameter können dadurch nicht geändert werden. Die tatsächliche Wiederherstellung wird in den normalen Generate-Stage-Metriken dokumentiert. Unveränderte Legacy-Pläne behalten ihren bisherigen Prüfpfad.

41 gezielte Prüfungen und bestehende Scope-Regressionen bestanden auch gegen die installierten Module, einschließlich tatsächlichem Caller und GenerateStage mit kopierten Metadaten. Der unabhängige Review prüfte zusätzlich 50 Fälle. Dies belegt die Softwarekorrektur; die Kampagne und der neue Hailo-Slice-Compilerpfad benötigen weiterhin ihre tatsächliche Ausführung. Belege: `/home/kmika/Reports/yolo26x_final_thesis_20splits_20261018/scope_resume_fix/` und `terminal_before_scope_resume_fix_111949/`. Drei fertige HEFs, alte Fehler und beide abgeschlossenen Sessions bleiben erhalten. Die Wiederaufnahme erfolgt mit demselben Run und unverändertem Profil über die installierte CLI; Force-Rebuild bleibt aus.

### 2026-10-21: Generic-only übernimmt reguläre CPU-Referenz-Cachetreffer

Die Generic-only-Zulassung akzeptiert neben `completed` den normalen erfolgreichen CPU-Referenzstatus `cache_hit`. Provider, ausschließlich semantische Referenzverwendung, tatsächliche Artefaktbytes und Deskriptoren sowie der vollständige Quality-/Completion-/Dataset-Vertrag werden unverändert geprüft. Ursprüngliche Ergebnisse und Qualitätsurteile werden nicht umgeschrieben. Dies ist eine Controllerkorrektur ohne Änderung der Runtime oder Messarithmetik; sie behauptet keinen neuen Hardware- oder FPS-PASS. Prüfbelege liegen unter `/home/kmika/Reports/generic_detectors20_multitensor_20261021T163124Z/reviews/actual_b039_quality_independent/`.


### 2026-10-22: Große reguläre TensorRT-Cacheproben begrenzen

Die normalen H10-/DeepX-Quality-Vorbereitungen überschritten trotz bestehender Kompression lokal das Linux-Einzelargumentlimit (133.533 beziehungsweise 140.745 UTF-8-Bytes). Die unveränderte Read-only-Prüfung wird bei Bedarf deterministisch auf disjunkte Anforderungsbatches verteilt. Jeder Batch enthält die vollständigen Validator-, Owner-, ABI-, Quellen- und Rezeptmetadaten; das tatsächlich SSH-gequotete Argument einschließlich NUL bleibt bei höchstens 120 KiB. Kleine Proben behalten ihre bisherigen Befehlsbytes. Eine zu große Einzelanforderung bleibt ausdrücklich UNKNOWN, während passende unabhängige Anforderungen weiter geprüft werden. Antworten werden vollständig und eindeutig an Modell, Setup, Rolle und Item gebunden; Reihenfolge sowie HIT/MISS/UNKNOWN bleiben erhalten.

Alle Verifier-Batches teilen das bisherige einmalige Timeoutbudget nach der lokalen Vorbereitung. Pfadauflösung und ABI-Probe behalten ihre eigenen bisherigen Budgets; es gibt keine automatische Wiederholung. Nach einem Transport- oder Antwortfehler startet kein weiterer Read-only-Batch auf diesem Gerät, weil lokales SSH-Ende kein Remote-Ende beweist. Bereits geprüfte Ergebnisse bleiben erhalten; übrige Anforderungen werden mit ihrer Quelle ausdrücklich als nicht ausgeführtes UNKNOWN gemeldet. Erfolgreiche Prüfantworten mit normalen MISS-/UNKNOWN-Zeilen und rein lokale Größenfehler erlauben weiterhin unabhängige Anforderungen. Lokale Produktfixtures prüfen die unveränderten Receipt-/Owner-/Byteprüfungen sowie Teilfehler und Budgetverbrauch. Die tatsächlichen alten H10-/DeepX-Payloads werden ausschließlich offline dekodiert und auf zwei sichere Batches aufgeteilt; kein Remoteprogramm wird dabei ausgeführt. Die 42 Bundle-Dateien einschließlich 37 Python-Runtimequellen bleiben bytegleich zu freeze420_v2. Dies ist eine isolierte Softwarekorrektur, kein Hardware-, Quality- oder FPS-PASS. Belege: /home/kmika/Reports/generic_detectors20_multitensor_20261021T163124Z/reviews/readonly_trt_cache_batching_v1/.

### 2026-10-22: Beleggebundene Recovery verwaister Generic-Quality-Ausführungen

Der explizite Produktbefehl `remote.process_lease recover-generic-orphan` ergänzt ausschließlich den fehlenden Recovery-Übergang für eindeutig zugeordnete, beendete Generic-Quality-Ausführungen. Er bindet die aktuelle Benutzerfreigabe an die ursprünglichen Launcher-, Session-, Profil-, Dispatch- und Unterbrechungsbelege. Unter gemeinsam gehaltenen Run- und physischen DUT-Sperren prüft er lokale Prozessidentitäten, Nachkommen und frische entfernte Ressourcenbeobachtungen. Aktive, fremde oder nicht eindeutig beobachtbare Besitzer bleiben gesperrt. Die existierenden sitzungsgebundenen Remote-Leases werden über ihren regulären Lifecycle geschlossen.

Ein neuer transaktionaler Recoverybeleg erhält die ursprünglichen Manifest-, Owner- und Quarantänebytes. Die alte Ausführung wird als abgebrochen geschlossen; weder ursprünglicher Cleanup-Erfolg noch Messabschluss werden behauptet. Ein Receipt-Guard hält unvollständige oder beschädigte Gruppenfreigaben auch nach Teilabstürzen gesperrt. Runner, Inferenzrechnung, Nachverarbeitung, Messvertrag, Qualitätsregeln und Scheduler bleiben unverändert. 159 gezielte Prüfungen einschließlich bestehender Ressourcen-/Lease-Regressionen bestanden. Dies ist Softwareevidenz, kein neuer Hardware- oder FPS-PASS. Freigabe, Prüfprotokolle und spätere tatsächliche Recoverybelege liegen unter `/home/kmika/Reports/generic_detectors20_multitensor_20261021T163124Z/continuation_v3_20261022/`.

### 2026-10-22: Ressourcenbezogene Generic-FPS-Zulassung und belegter Manifestimport

Die normale Generic-only-FPS-Zulassung ordnet konkurrierende Controller anhand ihrer aktuellen Prozessidentität, tatsächlich gehaltenen Kernel-Runlocks, ursprünglichen Session-, Profil-, Plan- und Ressourcenbindungen zu. Nachweislich unabhängige physische Setups und reine zentrale Statistik können überlappen. Gleiche aktive Hardware, unbekannte Zuordnung, unvollständige Sichtbarkeit oder offene Prozess-Leases bleiben gesperrt. Die entfernte Ownerprüfung und physischen Sperren bleiben verbindlich; die strengeren Standardprüfungen für Native, Energie und Recovery werden nicht geändert. Kein neuer Pool oder Scheduler wird eingeführt.

Der begrenzte Generic-Split-Manifestvergleich erhält beide tatsächlichen Dateihashes und prüft die ursprünglichen, tatsächlich verwendeten Boundary-Metadaten sowie alle sieben Modell-, Runtime- und Receiptbindungen, Cut-Tensoren und Eingaben. Ausschließlich die einzeln belegten Staging-, Wiederverwendungs- und nicht ausgeführten Full-Metadatenunterschiede erhalten eine zusätzliche Herkunftszuordnung im normalen Import. Andere wissenschaftliche oder unbekannte Änderungen bleiben abgewiesen; ursprüngliche Belege und Messungen werden nicht umgeschrieben.

100 gezielte Ressourcen-, Dispatch-, Controller- und Importprüfungen bestanden ohne Abwahl. Dies belegt die Softwarekorrekturen, keinen neuen Hardwareabschluss oder akzeptierten Ergebnisimport. Die historische b177-Messung bleibt an ihren ursprünglichen Quellenstand gebunden; neue Aufträge erhalten einen neuen veröffentlichten Freeze. Prüf- und Ausführungsbelege liegen unter `/home/kmika/Reports/generic_detectors20_multitensor_20261021T163124Z/continuation_v3_20261022/resources_import_repair_v1/`.
