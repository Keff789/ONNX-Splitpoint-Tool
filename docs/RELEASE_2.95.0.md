# Release 2.95.0 — DeepX Native A4 für Full und Splits

Neue DeepX-Native-Profile binden Full und alle bisher unterstützten Splits
einheitlich an A4: Batch 1, eine DeepX-Engine pro Messprozess, echte Async-API
und maximal vier offene Requests. Die gemeinsame geordnete Wait-/Nachfüll-
Implementierung hält Inputs und separate SDK-Outputslots bis zum Abschluss.
Full verarbeitet alle gebundenen Outputs einschließlich vorhandener Decoder.
Splits behalten eine sichere Boundarykopie, FIFO-Tiefe 3 und einen seriellen
TensorRT-/Taskconsumer. Die Policy hängt weder vom Modell noch vom Cut ab.

Generic und Hailo behalten ihre Ausführungs- und Messpolitik. Die SDK-Defaults
einschließlich buffer_count=6, CPU-Teilaufgaben/ORT und Kernmaske bleiben erhalten.
A4 kann die Requestlatenz erhöhen; Durchsatz ist kein Ersatz für Requestlatenz.
Warmups werden vollständig drainiert. Gemessene Count-/Dauerfenster enthalten
Fill, Taskverarbeitung und den vollständigen Drain aller angenommenen Requests.

Der explizite Parameter `--deepx-runtime-mode A4|S1` gilt nur für Native DeepX.
Neue Profile schreiben A4. Alte gespeicherte Jobs ohne Modus behalten ihren
ursprünglichen S1-Vertrag; sie werden nicht mit neuen A4-Repeats oder Energie
aufgefüllt. Der normale Energiepfad bindet denselben ausgeführten Modus.
Historische Ergebnisse und Qualitätsurteile werden nicht umetikettiert.

Der Produktstand wird über die bestehenden Quellmanifest-, Releasearchiv- und
Installationswerkzeuge bereitgestellt. Die Installation startet keine Messung.
Es gibt keine neue Registry, keine neue Hashinfrastruktur und keine Modell-
oder Enginekompilation durch diesen Patch.

Die begrenzte reale Abnahme ist separat dokumentiert unter
`~/Reports/deepx_native_a4_v295_20261015/20261015T071759Z`.
Vorgewählt sind sieben Full-Zellen und sieben Split-Zellen: MobileNet b027,
RegNet b066, ResNet b045, YOLO11 b062, YOLO26m b038, YOLO26s b021 und YOLOv7 b044.
Die Software ist regulär installiert. 327 neue 2.95-Tests sowie gezielte Prüfungen
der letzten Integrationskorrekturen sind bestanden. Die tatsächliche Abnahme ist
blockiert: Beim ersten normalen MobileNet-Auftrag scheiterte Full vor dem
Workerstart an der Auswahl des aufgelösten System-Python-Pfads statt des
ursprünglich gebundenen Venv-Aufrufpfads. Der Split-Worker wies den verbliebenen
2.94-Runnerprüfwert vor der Engine-Erzeugung zurück. Beide Bindungsfehler wurden
anschließend korrigiert und durch 104 gezielte Tests geprüft.

Der ursprüngliche Split-Rohbericht enthält keinen ausdrücklichen Nachweis eines
SDK-freien Frühabbruchs. Der Controller setzte deshalb die vorhandene physische
DUT-Quarantäne. Die unabhängige Quell-/Fehlerprüfung bestätigt den Abbruch vor
der Engine-Erzeugung; das vorhandene Recoveryverfahren unterstützt jedoch nur
anders gebundene Capture-STOPs und kann diesen numerischen Fall nicht freigeben.
Die Sperre und Originalberichte bleiben erhalten. Es gab keine erfolgreiche
numerische Hardwareabnahme, keine regulären Performanceprozesse und keine
Energiecaptures; zwölf weitere Zellen wurden nicht gestartet. GUI-Callbacks
wurden mit echten gespeicherten Profilen inspiziert, noch nicht auf Hardware
ausgeführt. Es wird kein vollständiger Hardware-PASS veröffentlicht.

REPORT.md, ACCEPTANCE_RESULTS.csv und RESUME.md im Auftragsordner dokumentieren
die tatsächlichen Versuche, getrennte Zulassungen und die bedingte Fortsetzung.
Eine passende unterstützte Ressourcenfreigabe ist vor jedem weiteren Start
erforderlich; ein neuer Runname umgeht die physische Sperre nicht.

Die vollständige DeepX-Neuaufnahme wird ausschließlich vorbereitet. Bestehende
Generic-/Hailo-Zeilen und die kanonische Ergebnismatrix bleiben erhalten.
