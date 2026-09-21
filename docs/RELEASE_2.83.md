# Version 2.83

Build: `v2.83-r9b-request-latency`.

Die vorhandenen R1–R5-Fixes bleiben erhalten. Der normale profilgesteuerte Workflow reicht `native_producers.energy.task_budget` an Native-Energieplan, Mess-CLI, Varianten und Fenstervergleichsprobe weiter. Die optionale Policy verwendet den vorhandenen R3-Checkpoint-/flock-Pfad: je Zeile höchstens `Wiederholungen * (1 + max_retries)` begonnene Ketten einschließlich fehlgeschlagener Preflights. Der zusätzliche Probe-Hashpreflight läuft innerhalb dieser Reservierung und behält den bestehenden Remote-Leasepfad.

Beispiel für die neue Policy (ändert weder Wiederholungszahl noch Messdauer):

```yaml
native_producers:
  energy:
    task_budget:
      enabled: true
      max_retries: 1
      max_transport_failures: 2
```

Der Checkpoint liegt im eigenen EvaluationRun als `energy_task_budget.json`. Stabile Energiezeilenkennungen und die physische u.RECS-Adresse verhindern einen Neustart des Etats durch neue Ausgabepfade. Zwei Transportfehler sperren die Quelle über alle Zeilen; ungeklärter Sourceabschluss sperrt sie sofort. Unabhängige Quellen bleiben verfügbar. Gesperrte Zeilen werden ausdrücklich BLOCKED/NOT_RUN gemeldet; wissenschaftliche Import- und Vollständigkeitsgates bleiben unverändert.

Die beibehaltenen Smoke-/Diagnose-Einstiege und die Metadaten des vorhandenen kleinen Abnahmeskripts verwenden die aktuelle Releaseidentität 2.83. Historische 2.82-Testannahmen werden nicht pauschal migriert.

## R6: Fixtureverträge und Empfangsdiagnose

Die 19 offenen Release-, Updater- und Workflowtests verwenden konsistente private 2.83-/Quality-/Leasefixtures. Der Installerparser nutzt stabile Abschnittsgrenzen. Historische Ergebniswerte, Binärfixtures, Negativprüfungen und alle bisherigen Testknoten bleiben erhalten. Die sieben Updater-Fehlerinjektionen prüfen echte rsync-Aufrufzahlen und die Reihenfolge der Verifikationsschritte; Crash-/Exceptiontests weisen den tatsächlichen Szenarioeintritt nach.

Der funktionale Auto/R5-Collector wurde ausschließlich in einer privaten R6-Kopie um eine optionale, begrenzte Datagrammdiagnose ergänzt. Sie erfasst Empfang, exakte Endbytes, Samplecounter, Socketdrops und Writerabschluss mit monotonen Zeiten. Messwerte, Zeitfenster und Empfangsgrenzen bleiben erhalten. Der Originalcollector wurde nicht ersetzt.

Am 15.09.2026 lieferten zwei neue lastfreie Aufnahmen der H8-u.RECS-Quelle jeweils 54.016 lückenlose Samples, echte Protokollenden und sauberes Cleanup. Anschließend wurden genau drei neue YOLOv7-Full-TensorRT-Erstaufnahmen auf orin_nx_hailo8_01 mit je 1 s innerem Lastsoll ausgeführt. Jede meldete exakt 16 abgeschlossene Work Units und bestand das technische Einzelgate. Insgesamt fünf R6-Ketten, keine Retries, keine Modell-/Enginebuilds.

Befund: heute reproduzierbar erfolgreich, frühere Ursache offen. R5 bleibt unverändert ungültig: 45.760 Samples, Gap128 und fehlender Endnachweis. Ein heutiges Endpaket ist keine separate Geräte-Idlequittierung.

Die 1-s-Native-Energieserie bleibt Smoke/Screening; sie ist keine 60-s-Langzeitabnahme. Development/Final Quality wird nicht zu einer vorab eingefrorenen wissenschaftlichen Finalkampagne umdeklariert. Keine Qualitäts-, Modell-, Kalibrierungs- oder Buildvertragsänderung; kein Force-Rebuild. Ein Abendbefehl wird privat vorbereitet, ohne Evaluation, Autostart oder Timer zu starten.

Die verbindlichen finalen Testzahlen, Source-/Binarybindungen, Restgrenzen und Rohdatenpfade stehen im externen `ABSCHLUSSBERICHT_R6.md` und `TEST_RESULTS_R6.json`. Softwareausführbarkeit, kurzer Energie-Smoke und vollständige wissenschaftliche Kampagne bleiben getrennte Aussagen.

## R7: Remote-Admission, Managementabschluss und GUI-Bindung

Lesende Speicherproben erhalten lokale Transport-/Prozessphasen und Remote-Phasenmarker. Ungültige Bytes-/Inode-/Mountfelder sperren den Upload im bestehenden Admissionpfad; Timeout und Kapazitätsmangel bleiben getrennt. Der nachgelagerte Uploadpreflight archiviert Befehl, rc und Prozessbelege lokal. Keine automatische Wiederholung und keine Änderung der wissenschaftlichen Grenzen.

Terminale Remotefehler schließen die Aufnahme neuer Managementarbeit und stoppen wartende beziehungsweise laufende Qualityarbeit vor den Coordinator-Joins. Wiederholte Abschlussaufrufe beobachten denselben Shutdown; Poolreferenzen bleiben für den tatsächlichen Endnachweis erhalten. Ungelöste Worker bleiben quarantänisiert. Die gemeinsame Ausnahme zeigt Primärfehler und konkrete sekundäre Cleanup-Phase.

Der vorhandene Debugexport nimmt ausschließlich begrenzte, redigierte Diagnosebelege der exakt referenzierten fehlgeschlagenen RemoteBenchmarkRuns auf. Fehlende spätere Probe-/Prozessbelege bleiben NOT_RECORDED und machen den Export unvollständig. Die GUI zeigt private Registry, Collector und Kampagnenbudget. Der sichtbare Startsnapshot bindet den Collectorpfad und private Energie-Defaults; eine Änderung vor Start wird abgewiesen. Planung benötigt kein ausführbares Collectorbinary.

Der historische H8-/H10-Timeout vom 15.09.2026 bleibt mangels ursprünglicher Phasenbelege ursächlich offen; kein ENOSPC- oder Collectorfehlernachweis. Neue R7-Tests verwenden kontrollierte lokale Prozesse und fangen den GUI-Start vor Runanlage ab. Keine neue Modell-/Energie-/Hardwareabnahme.

## R8: normale Produktintegration

Neue Standard-Nativebasis 100 Frames, 10 Warmup, eine Performancewiederholung; Final Quality behält 1000/100/3. Energie-Replikate bleiben separat. Schema14 migriert ausschließlich den kopierten Standardvertrag. Moduswerte, Profiloverrides und gefrorene Werte sind gekennzeichnet.

Der vorhandene Konfigurationsbefehl `python -m onnx_splitpoint_tool.run_modes_cli integrate-runtime --collector-source <geprüftes Binary> --backup-dir <neuer Backupordner>` installiert die unveränderten geprüften Collectorbytes idempotent und registriert sie normal. Explizite abweichende Bindungen sind Konflikte. Backups und atomare Produktlocks erhalten Konfigurationsfelder, Rechte und ACLs. Neue energieaktive Profile erben die vorhandene Quellen-/Retrypolicy; Native Full, Split und Fensterprobe verwenden den Collector aus dem Startsnapshot.

Native-/SSH-Vollausgaben liegen redigiert in getrennten Diagnosen. Die Normalansicht zeigt Fortschritt, Warnungen und Abschlussachsen. Hardwareabnahme und bekannte Grenzen werden ausschließlich in `docs/ARBEITSSTAND.md` und im externen R8-Bericht ausgewiesen. Eine Build-ID ist kein GUI-Abnahmebeleg.

## R9A: backendbezogene Auswahl und FPS-Endpunkte

Neue Config-folgende Starts materialisieren die sichtbare endliche Nachrückpolicy. Nur exakte kompatible Hailo-PARSER_UNSUPPORTED-/COMPILE_INFEASIBLE-Belege erlauben den nächsten deterministischen Kandidaten; bereits erfüllte Backends erhalten keine Ersatzmatrix. Eingefrorene Auswahlen und historische Resumes bleiben erhalten. DeepX besitzt bisher keinen entsprechenden negativen Cachepfad und erhält daher bei gewöhnlichen Buildfehlern keinen automatischen Ersatz. Kaltstarts, Auditfälle und backendbezogene Auswahl werden getrennt protokolliert. Setupbezogene TRT-Budgetanteile werden vor Remotedispatch festgelegt, und verbrauchter Startetat wird beim Suite-Austausch erhalten.

Die primäre Native-Rate ist der durch passende Completed-Task-Count-/Zeitdaten belegte Replikatmedian mit eigenem Intervall. P2 und geschätzter Pipelinezyklus werden separat ausgegeben. Fehlende oder widersprüchliche Completion bleibt unavailable; reziproker Durchsatz wird nicht als Einbildlatenz angezeigt. Historische Messdateien werden nur gelesen. Endgültige lokale und echte GUI-Abnahme stehen getrennt im Arbeitsstand und externen Abschlussbericht R9A.

## R9B: gepaarte Requestlatenz

Native Python-Full-/Splitrunner erfassen begrenzte monotone Start-/Endpaare mit stabiler Request-ID pro Wiederholung. Beginn am vorbereiteten Input vor Admission, Ende nach dem bestehenden Hostabschluss und vorhandenen Task-Postprocessing. Admissionwartezeiten zählen mit; Warmup ist ausgeschlossen. Mittelwert, P50/P95 (lineare Interpolation), min/max und vollständige Nenner werden nach Workerabschluss berechnet. Unvollständige, doppelte oder widersprüchliche Paare bleiben ungültig. Gleiche vollständige Wiederholungen werden aus Rohpaaren gepoolt, ohne neues Wiederholungs-CI.

Rohreport, Summary, Ergebniswidget und Export erhalten die Messsemantik. Legacywerte bleiben erhalten. Klassifikationspfade ohne gemessenes Top-k liefern ausschließlich gekennzeichnete Hostoutputlatenz; trtexec-Fullklassifikation bleibt ohne neue Tasklatenz. H8-C++ exportiert vorbereitete Hostoutputpaare, Three-Stage vollständige C++-Zeitpaare über den bestehenden Decode/NMS-Callback. In Three-Stage wird das bestehende Preprocessing unmittelbar vor die Admission verschoben, damit dessen Wartezeit enthalten ist; Queuegrößen und Synchronisationen bleiben unverändert. Diese Reihenfolgeänderung hat hier keine Hardware-/Overheadabnahme. Dauerbasierte Energiearbeit erhält keine neuen Paarpuffer.

R9B-Software-, GUI- und Hardwarecoverage sowie die Grenze des normalen H8-Wrapperbuildpfads sind separat im Arbeitsstand und im externen Abschlussbericht dokumentiert.

Ein einzelnes Replikat behält seine FPS; ein degeneriertes gespeichertes FPS-Intervall wird nicht als neu geschätztes Wiederholungs-CI ausgegeben. Rohreports bleiben unverändert.

R9B-Abschluss: DeepX-Part1 wird aus dem gespeicherten ausgewählten Request cache-only vor der abhängigen TRT-Part2-Prüfung aufgelöst. Fehlbestand und ungültige Identität bleiben gesperrt. H8-C++-Wrapper werden über das bestehende Config-/Commandreceipt an Source, CMake-Optionen, Architektur und Binary gebunden; passende Binaries werden wiederverwendet, Vorbereitung ist separat begrenzt und Messung nutzt --no-build. Reine Classification-Logitraten erscheinen als Hostoutput ohne Task-Postprocessing; Originalreports bleiben unverändert. Hardwareumfang und Restgrenzen stehen im Arbeitsstand.

R9B-Abschlussabnahme: 939 Produkttests und 26 Tk-Fälle PASS. H8-MobileNet einschließlich realem C++-Wrapperbuild/Reuse und beide H8-YOLO26s-Full-Zeilen ausgeführt; Backfill wegen eines auftragslokalen forced_cases-Profilfehlers offen. DeepX-Part1 im echten GUI-Preflight HIT, jedoch bestehende Ablehnung unveränderter FLOAT-Brücken im abhängigen TRT-Part2 noch offen; gleich betroffene Detectionzelle nicht gestartet. Alle Messungen ohne Energie und ohne Modellbuild; genaue Coverage im Arbeitsstand.
