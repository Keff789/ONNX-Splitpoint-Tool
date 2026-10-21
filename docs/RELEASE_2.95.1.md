# Release 2.95.1 — DeepX-Vorbereitung, Recovery und reguläre GUI-Fortsetzung

Vorläufiger Softwarestand vom 15.10.2026. Paket/GUI: `2.95.1`;
Build/Workflow: `v2.95.1`; Entwicklungslinie: `v2.79`;
Build-Vertragsversion: `2`. Installation und reale Hardwareabnahme dieses
Fehlerbehebungsstands sind noch offen. Die Originalquellen, Fehlversuche und
Berichte von [2.95.0](RELEASE_2.95.0.md) bleiben erhalten.

Der Ausgangsstand hatte zwei vorbereitungsbedingte MobileNet-Fehlstarts:
Full scheiterte vor dem Workerstart an der Auswahl des aufgelösten System-Python
statt des gebundenen Venv-Aufrufpfads. Split b027 scheiterte am verbliebenen
historischen Runnerprüfwert vor DeepX-/TensorRT-Konstruktion. Die anderen zwölf
Abnahmezellen wurden nicht gestartet. Es gab keinen numerischen Hardware-PASS,
keinen regulären Performanceprozess und keinen Energiecapture. Vier vorhandene
GUI-Profilinspektionen waren keine tatsächlichen GUI-Startausführungen.

2.95.1 erhält die zwei Vorbereitungskorrekturen: Full verwendet den gebundenen
`runtime_python.invocation_path`; frische explizite A4-Aufträge prüfen die
tatsächlich bereitgestellten aktuellen Workerbytes. Historisches S1-Replay
behält seine ursprüngliche Quellenbindung und seinen Runnerguard. Die ergänzte
Lifecycle-/Recoverybehandlung unterscheidet belegte Abbrüche vor Inferenz von
begonnener oder unklarer SDK-Ausführung. Eine Freigabe setzt den exakt gebundenen
Quell-/Fehlerbeleg sowie den aktuellen eigenen Prozess- und Ressourcenabschluss
unter den vorhandenen exklusiven Locks voraus. Fehlende Zähler, allgemeine
Exceptions und eine freie Prozessliste genügen dafür nicht. Die vorhandene
Capture-/Sensor-Recovery und Quarantäne bei unbewiesenem SDK-Abschluss bleiben
streng. Eine unterstützte Recovery ist noch kein Numerik-PASS.

Die reale Fortsetzung erfolgt über die laufende normale Anwendung mit ihren
gespeicherten Profilen, Queue, Runnern, Transporten und Ergebnisimporten. Zuerst
müssen die ursprünglichen sieben Full- und sieben Split-Zellen ihre numerische
32-Bild-Prüfung und den begrenzten Smoke bestehen. Danach folgen je Punkt drei
unabhängige Performanceprozesse mit 100 Warmups und 1.000 fertigen Tasks sowie
drei gültige Native-Energieaufnahmen mit 60 s Workloadanforderung und Warmup 0.
Vorhandene physische Versuche und Retrybudgets bleiben angerechnet.

Erst nach vollständig bestandener echter GUI-Abnahme wird die im Folgeauftrag
autorisierte gezielte Neuaufnahme von 68 Splits und sieben DeepX-Fulls über den
normalen GUI-Workflow fortgesetzt. Vollständige kompatible 3+3-Abnahmekohorten
dürfen nach Prüfung ihrer tatsächlichen UID übernommen werden. Die sieben
optionalen TensorRT-Full-Kontrollen sind separat und zunächst nicht ausgewählt.
Bestehende Generic-/Hailo-Daten, ursprüngliche S1-Kohorten, Qualitätsgrenzen,
Artefakte, Eingaben und Energie-/Commandfenster bleiben erhalten. Neue A4-FPS
werden ausschließlich mit passender tatsächlich gemessener A4-Energie verbunden.

Die neuen Nachweise gehören nach
`~/Reports/deepx_295_gui_acceptance_remeasurement_20261015/20261015T093320Z`.
Recovery, GUI-Ausführung, reale Numerik, Performance, Energieintegrität, normaler
Import, Archiv und wissenschaftliche Vergleichbarkeit werden getrennt bewertet.
Der aktuelle Eintrag ist kein abgeschlossener Release- oder Hardwarebericht;
die dortigen Originalergebnisse bestimmen den endgültigen Status.
