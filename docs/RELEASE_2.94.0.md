# Release 2.94.0 — begrenzte reale Hardwareabnahme im normalen Produktpfad

Paket/GUI `2.94.0`, Build/Workflow `v2.94.0`, Entwicklungslinie `v2.79`.
Der gespeicherte Profilabschnitt `hardware_acceptance` verbindet die explizite
endliche Fallauswahl und Prepare mit dem regulären GUI-Startcontroller und dem
profilgesteuerten CLI. Native und Generic Completed verwenden die vorhandenen
Produkt-Runner, Collector und Ergebnisleser.

Je Netz/Setup wird genau eine vorher gespeicherte unterstützte Splitposition
verwendet. Neue Ausgaben erhalten eigene Pfade; gültige Artefakte und passende
historische Full- und Qualitätsreferenzen behalten ihre ursprüngliche Provenienz.
Eine vollständige Cacheübernahme ersetzt keinen neuen Hardwaretest. Ein
unterstützter Runtimepfad benötigt eigene neue DUT-Prozesse und reale Energiebelege.
Die gespeicherten Warmup-, Task-, Energie-, Repeat- und Retrywerte gelten weiter.
Neue Dataset-Qualitätskampagnen und ein Resume einer vorhandenen Restqueue sind
kein Bestandteil dieses Pfads.

Technische Funktion, Outputprüfung, Energieintegrität und wissenschaftliche
Vergleichszulassung werden getrennt berichtet. Fehlende Vergleichsreferenzen,
negative Modellqualität und historische Ausschlüsse werden erhalten.
Normale Sampleverlust-Retries erhalten die gültigen Slots und zählen alle Starts.
Gemeinsame Capture- und Transferressourcen werden seriell genutzt.

Native-Transfers tragen die normale Setup- und Ausgabebindung des Controllers.
Jeder frische Repeat besitzt ein eigenes Benchmark-Metadatenverzeichnis für
die vom normalen Runner geschriebenen Hilfsberichte. Die versiegelten
Modellartefakte und Eingabebindungen werden dabei weiterverwendet.

Die Energie-Fortsetzung erhält exakt verifizierte gültige Performance-Repeats
aus abgeschlossenen Jobs derselben neuen Abnahmephase. Ursprüngliche Prozesse,
Outputs und Fehlstarts bleiben getrennt belegt; die Fortsetzung misst Energie
neu und rechnet bereits verbrauchte Captureversuche gegen dieselben Grenzen.
Diagnostische technische Energie erhält die erforderliche reale Umweltbeobachtung
und alle bestehenden Messprüfungen, ohne wissenschaftliche Vergleichszulassung.
Finale wissenschaftliche Kohorten akzeptieren weiterhin keine Diagnostikaufnahmen.

Die normale Ressourcen-Recovery erkennt außerdem den exakt gebundenen Native-
Abbruch in ACQUIRING. Sie prüft die gespeicherten Besitzer-, Kommando-, Quellen-
und Budgetbelege sowie aktuelle lokale und DUT-Verfügbarkeit unter den normalen
vier Ressourcensperren. Eine aktuelle Ressourcenfreigabe bestätigt keinen früher
unbewiesenen Quellenabschluss und startet den alten Parent nicht erneut.
Dasselbe gilt für den exakt gebundenen terminalen Partial-Workflow nach einem
Generic-Captureabbruch: Quellenlogs werden vor und nach aktueller Ownerbeobachtung
geprüft; der ursprüngliche STOP und sein invalides Budget bleiben erhalten.

Installation und Veröffentlichung nutzen unverändert das manifestgebundene
Quellarchiv und `scripts/update_source_release.sh RELEASE.zip TOOL_DIR`.
Die bestehende Softwareabnahme 2.93.0 bleibt Referenz; geänderte Pfade werden
gezielt geprüft. Konkrete neue Hardwareausführungen und eventuelle blockierte
Zellen gehören in den jeweiligen Abnahmebericht, nicht in eine pauschale
Software-PASS-Aussage.
