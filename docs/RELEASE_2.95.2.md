# Release 2.95.2 — Full-Staging und serielle DeepX-Abschlusszähler

Softwarestand vom 15.10.2026. Paket/GUI: `2.95.2`; Build/Workflow: `v2.95.2`;
Entwicklungslinie: `v2.79`; Build-Vertragsversion: `2`. Dieser Patch kennzeichnet
lokale Softwarekorrekturen und deren Softwareprüfung. Er ist nicht
hardwareabgenommen. Die Originalarchive und Rohberichte von
[2.95.1](RELEASE_2.95.1.md) und [2.95.0](RELEASE_2.95.0.md) bleiben erhalten.

Die tatsächliche normale GUI-Ausführung unter 2.95.1 ist vollständig beendet
und als `partial` geschlossen. Die 14-Zellen-Abnahme ist fehlgeschlagen: Full
und Split b027 von MobileNet sind im Produkt fehlgeschlagen, zwölf weitere
Zellen wurden nicht gestartet. Beide MobileNet-Zellen haben einschließlich
der ursprünglichen Fehlversuche ihr Budget von zwei Versuchen ausgeschöpft.
Es gab null zugelassene numerische Abnahmezellen, null reguläre
Performanceprozesse, null Energiecaptures und keinen Start der
75-Punkte-Neuaufnahme.

Der Full-Versuch scheiterte vor dem Workerstart. Der normale Stagingpfad
verlangte eine vorhandene Falldeklaration in einem selektiv bereitgestellten
Remote-BenchmarkSet, das diese Datei nicht enthielt. Die exakt gebundenen
ursprünglichen Fallmetadaten sind lokal erhalten. Der seit 2.95.0 eingeführte
Stagingguard wird durch die Übernahme dieser geprüften Metadaten in das eigene
frische Ziel ergänzt. Quellen-, Benchmark- und Fallbindung bleiben strikt;
es gibt keine erfundene Falldeklaration, neue Cut-Auswahl oder Kompilierung.

Der Split-Worker führte tatsächlich 32 serielle Referenzen, 32 gemischte
A4-Aufträge und 64 Smoke-Tasks aus. Sein Rohbericht meldet numerische Parität
und abgeschlossene Tasks. Der Produktpfad lehnte den Abschluss trotzdem ab:
Dem Writer fehlte `serial_reference_calls_completed`, während die gemeinsame
Sicherheitsprüfung den passenden Abschlusszähler verlangte. 2.95.2 ergänzt
den tatsächlich beobachteten Zähler im bestehenden Bericht. Fehlende Zähler,
unterbrochene SDK-Aufrufe und widersprüchliche Lebenszyklusdaten bleiben
fehlschließend; ein Fehler wird nicht durch pauschales Gleichsetzen von
Start- und Abschlusszahlen zum Erfolg erklärt.

Die fehlgeschlagene ursprüngliche Produktprojektion wird nicht nachträglich
zum Hardware-PASS umgeschrieben. Die neue nach Inferenz gesetzte physische
DUT-Quarantäne bleibt erhalten. Eine Quellkorrektur, lokale Tests, ein neues
Release oder ein neues Runverzeichnis geben sie nicht frei und eröffnen
kein neues Versuchsbudget. Weitere Hardwareausführung ist nicht freigegeben.

Die A4-Policy bleibt unverändert: Batch 1, eine Engine pro Prozess, höchstens
vier offene DeepX-Aufträge, Split-FIFO 3 und der bestehende serielle
TensorRT-/Taskconsumer. Generic, Hailo, historische S1-Quellen, Qualitätsurteile,
Messfenster und Sensor-/Retry-/STOP-Regeln bleiben erhalten. Neue Performance
wird nicht mit alter S1-Energie verbunden. Die vorbereiteten Import-,
Archiv- und Generationswege wurden nicht auf einen neuen Datensatz angewendet;
es gibt keine neue Aktivierung des kanonischen Current.

Die genaue Softwaretestbilanz, tatsächliche GUI-/Workerbelege, erhaltenen
Fehlstarts und Sperren stehen unter
`~/Reports/deepx_295_gui_acceptance_remeasurement_20261015/20261015T093320Z`.
Softwareprüfung, rohe numerische Beobachtung, zugelassene Hardwareabnahme,
Performance, Energie, Import und Veröffentlichung bleiben getrennte Urteile.
