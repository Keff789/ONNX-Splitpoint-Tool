# Release 2.91.2 – THESIS20 Restabschluss

## Lokale Auswertungskorrekturen vom 02.10.

Nach Abschluss aller 234 Energiezeilen werden ausschließlich abgeleitete
Berichte korrigiert. Originale und historische Fehlerberichte bleiben erhalten;
dieser Auftrag enthält keine neue Inferenz oder Workflow-/Messfreigabe.

Die Reporter korrigieren 47 Hailo10-Backendlabels, übernehmen die vorhandenen
Qualityentscheidungen und projizieren 37 originale Buildfehler plus 24
Policyausschlüsse terminal. Die Native-Authorityprojektion verifiziert alte
Ablehnungen erneut mit den bestehenden strengen Bindungsprüfern. Der lokale
Validator unterstützt für die sechs TRT-Full-Fälle vorhandene gespeicherte
Outputs und exakt gebundene Referenzen ohne Inferenzfallback. Neue Optionen
`--existing-outputs-only`, `--existing-reference-validation-summary`,
`--previous-validation-summary` und `--eval-root` ermöglichen die reguläre
Ableitung außerhalb des Original-Runs. Die Übernahme vorhandener Validatorzeilen
verlangt denselben Run, Prüfkontext und dieselbe Policy sowie eindeutige
unveränderte Ergebnis-/Request-/Commandbindungen.

Der Energieexport enthält die tatsächlichen Native-Messwerte. Deskriptive
Split/Full-Quotienten, semantische Bindungen und wissenschaftliche Freigaben
bleiben getrennt. FS-Energie ohne Idleabzug bleibt primär; die 21 vorhandenen
TRT-Zusatznormalisierungen und die ursprünglichen Screening-/Entwicklungsrollen
werden ausdrücklich separat ausgewiesen. Fehlende Generic-Completion-Zeiten
werden als Messgrenze benannt, nicht aus Teilzeiten konstruiert.

Auftrag, Ausgangssicherung, gezielte Tests, reguläre lokale CLI-Ausgaben,
unabhängige Reviews und endgültige Abnahme:
`$HOME/Reports/thesis20-auswertung-20261002_110429`.
Keine Installation, kein Commit/Push und keine Hardwareabnahme dieses Pakets.

## Nachbesserung vom 01.10.: Energie-Previous-Results

Der produktive Ergänzungslauf hat inzwischen alle 548 Qualityresultate und
234 unterstützten Native-Performancefälle abgeschlossen; 228 Nativefälle
bleiben korrekt ausgeschlossen. Der aktuelle Energieplan umfasst 234 Zeilen.
Der anschließende Startabbruch entstand durch einen alten, nie gestarteten
213er-Energiebericht: Seine Planzeilen banden kanonische Commanddateien,
während dorthin kopierte Preflights noch lokale Contractpfade eines anderen
Attempts enthielten. Die ursprünglich gebundenen Skriptbytes und der Writer
sind im externen Ursachenbericht nachgewiesen; der technische Contract blieb
unverändert. Keine Energiezeile war gestartet.

Nur im bestehenden `--previous-results`-Übergang entfällt die historische
Skriptgleichheit für eine vollständig nachgewiesene, vor dem Start blockierte
Kohorte. Erforderlich sind vier exakte Nullzähler, vollständige negative
Ergebnis- und Ledgerzeilen, fehlende Ausführungsverzeichnisse und keine
vorhandenen Campaignketten. Ein vorhandenes Rowjournal wird mit seinen
bestehenden Digest-, Identitäts- und Vertragsprüfungen ausschließlich gelesen;
seine autoritativen Rowdateien müssen ebenfalls ungestartet sein. Unklare,
gestartete oder teilweise gemessene Historien behalten die strenge Prüfung.
Aktuelle Commands, technische Verträge, vollständige Energieidentitäten,
Qualityprovenienz, Kohorte, Ausführungsumgebung und Campaignbudgets bleiben
verbindlich. Reguläres `--resume-existing` bleibt unverändert.

Der tatsächliche 213→234-Übergang reproduziert vor der Korrektur den gemeldeten
Fehler und besteht danach ohne Wiederverwendung erfundener Messungen.
Gezielte neue Regressionen sowie die bestehenden Resume-/Provenienz- und
Rowjournaltests sind bestanden. Die zusammenhängende Abnahme erfolgt nach
finaler Installation über den normalen CLI am isoliert abgebildeten echten
Run einschließlich realer Artefakt-/Owner-/Cohortpreflights; nur der äußere
erste Energiemessstart wird kontrolliert gestoppt. Maßgebliche Ergebnisse,
Erhaltungsbelege, Installation und Produktionsstatus stehen unter
`$HOME/Reports/thesis20-energy-reentry-20261001`.

Der Nutzer hat anschließend genau eine dauerhafte produktive Fortsetzung
autorisiert. Sie startet erst nach erfolgreicher integrierter Abnahme und
erneuter Ownerprüfung. 527 Genericzeilen, 548 Qualityresultate und 234 Native-
Messungen bleiben erhalten; keine erneute Inferenz, Statistik oder Native-
Performance. FS/command60s×3 und die bestehenden Retrybudgets bleiben bestehen.
Während Produktion keine weiteren Quell-/Profil-/Cacheänderungen oder
parallelen Gerätetests; keine automatische Reparatur-/Neustartschleife.

## Nachbesserung: Wiedereintritt nach der Genericergänzung

Der produktive Einstieg vom 30.09. brach bei YOLO26m bereits an
`generate_benchmark_set` ab. Dieselbe Lücke betraf YOLO26s: Die erste
Supplement-Sicherung blieb die richtige Originalbasis, der Guard behandelte
aber vier inzwischen regulär fortgeschriebene Plan-/Suitealiase und den
erneuerten ausführbaren Harness als unveränderliche Generierungsausgaben.

Der bestehende Reusepfad prüft diese Änderungen jetzt gegen die konkreten
Writer: Die physische Projektion aus dem original gebundenen globalen Scope
muss die aktuellen JSONbytes vollständig reproduzieren. Der Harness muss
seine bestehende aktuelle Indexbindung und den exakten installierten
Templateoutput erfüllen. Sonstige Abweichungen bleiben Fehler; Auswahl,
Datensatz, Precision, Endpunkte und Messparameter erhalten keine Ausnahme.
Die erste Quellenbasis wird nicht ersetzt, aktuelle Aliase werden nicht
zurückgesetzt und alte Messungen erhalten keine neue Quellenzuordnung.
Neue Metadatensicherungen enthalten auch den generierten Harness.

Der kontrollierte Abbruch deckte außerdem gebundene ältere Resumeentscheidungen
in bereits fertigen Stufen auf. Neue Supplement-Reuseentscheidungen liegen
deshalb im aktuellen vorhandenen Attemptverzeichnis; die alte
`stages/.../resume_decision.json` bleibt unverändert. Damit hängt ein weiterer
Eintritt nicht von einer erst am erfolgreichen Workflowende ausgeführten
Indexfinalisierung ab.

Die gezielte lokale Regression umfasst 26 Tests, die vollständige betroffene
2.91.1-/2.91.2-Gruppe 190 Tests (überlappend), jeweils PASS. Die zusammenhängende
Prüfung des installierten CLI-Einstiegs auf isolierter Originaldatenabbildung
wird getrennt von diesen Softwaretests dokumentiert: Vorherfehler, alle sieben
Modelle, echter Cache-Preflight-/Reuseübergang, erster lokaler Statistikdispatch
und erneuter Eintritt nach kontrolliertem Abbruch. Maßgebliche Protokolle und
finale Installationsprüfung liegen unter
`$HOME/Reports/thesis20-generate-reentry-20260930`.
Der Bestand bleibt 527 Genericzeilen, 482 fertige Qualityresultate und 222
Native-Messungen; offen bleiben 66 lokale Statistikrequests und zwölf
unterstützte Nativefälle. Die frühere Artefakt-only-Abnahme und eine produktive
Messabnahme sind davon getrennt. Der ausdrückliche Nutzernachtrag vom 30.09.
ersetzt den bisherigen Startstopp: Nach erfolgreicher finaler Installation und
integrierter Prüfung einschließlich Wiedereintritt wird genau eine produktive
Fortsetzung desselben Runs dauerhaft entkoppelt gestartet. Vorher bleiben die
normalen Prozess-/Ownerprüfungen erforderlich. Start und letzter belegter
Fortschritt werden im Abschlussbericht dokumentiert; ein Start gilt nicht als
erfolgreiche Messabnahme. Während des Laufs erfolgen keine weiteren Quell-,
Profil- oder Cacheänderungen und keine parallelen Gerätetests. Keine automatische
Fix-/Neustartschleife; Umfang und konfigurierte Mess-Retries bleiben unverändert.

Dieser Patch vervollständigt die Produktpfade des bestehenden THESIS20-Runs.
Die 527 Genericzeilen, 479 abgeschlossenen Qualityresultate einschließlich
32 Accuracyverlusten und 222 Native-Performancemessungen bleiben erhalten.
37 gespeicherte negative Builds, 24 statische Hailo-Part1-Policyausschlüsse
und 228 nicht unterstützte Native-Multi-Input-Fälle erzeugen keinen neuen
Messauftrag. Kohorte, Precision, Endpunkte und wissenschaftliche Budgets
bleiben unverändert; Force-Rebuild bleibt aus.

Die CPU-Referenzquelle wird nach vollständigem Harnessrefresh gebunden und
vor Zulassung erneut geprüft. Beim Wiedereintritt kann eine gespeicherte
Referenz nur aus ihrer exakt verifizierten ursprünglichen Quellenbasis
wiederverwendet werden. Technisch fehlgeschlagene Statistikrequests bleiben
lokale Arbeit; abgeschlossene negative Quality ist fertige Arbeit.

Native Full prüft logische Kampagnen-ID und physischen Attemptpfad getrennt.
Die Bindungssets werden mit den normalen Writern erzeugt und unverändert
transportiert; falsche Run-, Setup-, Modell-, Input- oder Enginebindungen
bleiben Fehler. Fehlende Vendor-Full-Metadaten werden unabhängig vom Bedarf
neuer Performancejobs nachgeführt. Ursprüngliche Performancemessungen und
lokale semantische Entscheidungen werden dabei nicht überschrieben.

Der bestehende Energie-Restagepfad enthält jetzt sämtliche gebundenen,
runlokalen Rollen des tatsächlichen Artefaktprüfers, einschließlich CMake,
generierter Quellen, Binaries, Hailo-Receipts und DeepX-Manifesten. Es werden
nur exakt belegte Originaldateien über den vorhandenen konfliktgeprüften
Transferpfad ergänzt. Externe Cacheartefakte und Runtimecode bleiben unter
der tatsächlichen Preflightprüfung; kein Compilerersatz und kein neuer
Restager. Neue Fullrunner erhalten im bestehenden Skriptverzeichnis einen eigenen
Dateinamen pro Attempt, damit alte Energieverträge ihre gebundenen
Runnerdateien behalten.

Die Fehleranzeige liest den maßgeblichen Workflowattempt samt konkretem
Evidenzpfad und Startzählern. Die historische Energieprojektion mit 213 Zeilen
bleibt erhalten. Die aktuellen ursprünglichen 80 Artefaktfehler verteilen sich
auf 59 CMake-Lücken, 14 fehlende Hailo-Receipts und sieben fehlende DeepX-
Manifeste. Die Fullprüfung hatte fehlende Dateien als SHA-Abweichung bezeichnet.

## Fortsetzung desselben Runs

Lesender Plancheck:

```bash
$HOME/ONNX-Splitpoint-Tool/.venv/bin/onnx-splitpoint-complete-evaluation \
  --run-dir $HOME/Models/EvaluationRuns/thesis_20splits_n5000_b1000_20260925_20260925_103745 \
  --plan-only --output-dir $HOME/Reports/thesis20-restabschluss-20260930/plan-final
```

Produktiver Fortsetzungsbefehl, durch den Nachtrag nach erfolgreicher finaler
Startkettenprüfung ausdrücklich zum einmaligen Start autorisiert:

```bash
$HOME/ONNX-Splitpoint-Tool/.venv/bin/onnx-splitpoint-complete-evaluation \
  --run-dir $HOME/Models/EvaluationRuns/thesis_20splits_n5000_b1000_20260925_20260925_103745
```

Keine offenen Genericbenchmarks. Unterstützte Performancearbeit: YOLO26s
b003/b021 auf H8/H10/DeepX und TensorRT Full YOLO26m/s auf allen drei Setups,
insgesamt zwölf Fälle. Der Energieplan kann danach von 222 auf höchstens 234
Zeilen wachsen. Dies ist die erste Energieaufnahme dieses Datensatzes.
FS/command60s×3, Retry2/Quellenfehlerbudget20 bleiben unverändert.

Die endgültigen lokalen/integrativen Prüfergebnisse, erlaubten Artefakt-only-
Remoteprüfungen und verbleibenden Grenzen stehen im Arbeitsstand sowie im
dauerhaften Abschlussbericht unter
`$HOME/Reports/thesis20-restabschluss-20260930`. Keine reale GUI-
Messabnahme, kein Compilerlauf und keine produktive Energieaufnahme gehören
zu dieser Installation. Keine SDK-/Venv-Neuinstallation, kein Commit/Push.

Abgeschlossen: drei echte gespeicherte YOLO26s-Full-Qualityrequests lokal,
H8/H10 weiterhin `accuracy_loss`, DeepX `reference_close`. Damit 482 fertige
Resultate und 66 offene lokale Statistikrequests; darunter drei TensorRT-Full-
Companions, keine zusätzlichen Geräteaufträge. Sechs vorhandene Vendor-Fulls
wurden zweimal lokal nachgebunden; negative semantische Beobachtungen erhalten.

Erlaubte reale Artefakt-only-Abnahme: zweimal 222 PASS / 0 FAIL auf den drei
Beschleunigersystemen. 216 fehlende exakt gebundene Originaldateien ergänzt,
keine vorhandenen Dateien ersetzt. Zweiteintritt ohne Übertragung. Wrapper-,
Collector-, Workload-, Compiler- und neue Inferenzstarts sämtlich null.
Dies ist keine produktive Messabnahme; der spätere Einstieg prüft aktuelle
Dateien und Besitz erneut. Messquelle .176 blieb unberührt.

Software: konsolidiert 325 PASS; finale betroffene Zusatzgruppen A 12 PASS,
B/C 71 PASS und D 120 PASS (überlappende Gruppen). Ein vorhandener Budget-
Projektionstest und drei alte C++-Rekonstruktionsfixtures scheitern bereits
auf dem Ausgangscheckpoint; ihre Grenzen sind separat dokumentiert. THESIS20
verwendet die vorhandenen Originalartefakte und benötigt keinen Compilerfallback.
