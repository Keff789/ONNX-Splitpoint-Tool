# ONNX Splitpoint Tool v2.79.32 – Sourceumfang und Nachweisvertrag

Build-/Workflow-ID: `v2.79.32-native-full-failure-closure`.

Basis ist der unveränderte v31-Source mit SHA-256 `cc0bac8f7fe1307f33598dd8156c9149eb693271b388891a20ecd39c329bb67f`, Build `v2.79.31-complete-set-quality-integration`.

Dieser vor der finalen Paketierung geschriebene Sourcebericht beschreibt Änderungen und Prüfumfang. Der tatsächliche Ausführungsstatus steht im `PRUEFBERICHT_2.79.32.md` und den externen Roh-/JUnitberichten des zugehörigen Bundles. Es werden keine Testergebnisse in die bereits geprüften Sourcebytes nachgetragen.

## Schmaler Änderungsumfang

- F1: gemeinsame rollenkorrekte Persistierung negativer Native-Jobs. Vendor-Full und setup-lokales TensorRT Full verwenden die bereits vorhandene Full-Datei; Splitdateien enthalten ausschließlich Splitjobs. Identität, Primärfehler und Nullstartdetails bleiben bis Matrix, Kompaktbericht und Energieausschluss erhalten.
- F2: Full-Runtimeausnahmen erhalten die eindeutige Planidentität und den wirklichen Primärfehler. Reale widersprüchliche Childidentitäten bleiben gesperrt. Fehlversuche erzeugen keine gültigen Replikate oder Messwerte.
- F3: vollständig bestimmte explizite Compilerkontexte werden vor unbenötigten gespeicherten Alternativen ausgewertet. Benötigte mehrdeutige Auswahl und ungültige explizite Pfade bleiben Fehler.
- Release: zentrale Version 2.79.32, neue aktive Smoke-/Acceptance-/Hardware-/Replaywrapper, alle historischen Entrypoints erhalten. Alle v31-Verhaltensgates und v30-DeepX-/Finalisierungsregressionen bleiben Bestandteil des regulären Gates.

Kein neuer Mean/Std-Algorithmus, keine neuen Hash-/Registryebenen, keine neue Parallelität, kein Cachepurge und keine gelockerten Qualitätsmargen. `COMPILE_INFEASIBLE` und `TRANSIENT_INFRASTRUCTURE` bleiben getrennte negative Ergebnisklassen.

## Verbindlicher Prüfpfad

Die fünf neuen v32-Testsuiten prüfen persistierte Fehlerketten, Full-Ausnahmen, Compilerpriorität, zusammenhängende Workflowprojektion und Release-/Updateabschluss. Das gesamte reguläre Gate wird mit echten Pflichtabhängigkeiten aus dem final erzeugten und frisch entpackten Source-ZIP ausgeführt. Skips, Xfails oder fehlende Abhängigkeiten erzeugen keinen PASS. Synthetische Hardwareersatzstellen bleiben ausdrücklich als solche ausgewiesen. Manifest, Archiv-CRC, Source-/Bundleprüfsummen und der komplette ermittelte Remote-Spiegelbestand werden überprüft. Ein tatsächliches 31→32-Update prüft Paketmetadaten, Entrypoints und den bytegleichen Erhalt eigener Profile, Compilerkonfiguration, Kalibrierung, Ergebnissen und synthetischen Caches.

Aktueller Status in diesem vor Paketierung eingefrorenen Dokument: Software- und Installationsgate werden durch die externen finalen Berichte belegt; hier keine vorweggenommene PASS-Aussage.

## Getrennt offene Freigaben

Zielhardwarelauf, DeepX-Produktivpfade, frisches MobileNet-B500, die weiteren Klassifikationsfreigaben, Hailo10/YOLO26 AP6b und eine neue Complete-Set-/Finalkampagne: `NOT_RUN` beziehungsweise mangels Zielhardwareevidence offen. R1/R2 und die historische Complete-Set-Fixture bleiben unveränderte historische Evidenz. 1 s × 3 / 30 s × 3 bleiben Integration; der gesonderte finale Energievertrag bleibt FS/command mit 60 s × 3.

## Im Review reproduzierte Randfälle derselben Fehlerketten

Die F1-Abgrenzung nach bereits begonnenen Aufrufen benötigt zusätzlich die Erkennung der tatsächlich ausgegebenen `[native-full] ... repetition=N/M`-Meldung. Ein nachgelagerter Sammelfehler darf begonnene Versuche nicht als Nullstart melden. Nach erfolgreichem Wiederholen der Ergebnisübertragung wird deren tatsächlicher lokaler Ergebnisbaum auch dem vorhandenen Finalreport übergeben. Beide Fälle werden am Dispatcher beziehungsweise am echten Stagepfad regressiert.

Bei F3 bleibt ein eindeutig gespeichertes individuelles Cacheziel erhalten, auch wenn die explizite Auswahl einen anderen Compiler bestimmt. Mehrere tatsächlich verbleibende Cacheziele verlangen weiterhin eine explizite Auswahl. Hierdurch erzeugt der Vorrangfix keinen neuen Cacheort.

Die zugehörigen Red-/Green-Logs und das vollständige Gate des final entpackten Source stehen außerhalb dieses Sourcearchivs im Lieferpaket unter `verification/`; die gemessenen Testzahlen werden im `PRUEFBERICHT_2.79.32.md` des Lieferpakets angegeben.

## Zusätzlich reproduzierter Installerbefund: kanonische Arbeitsverzeichnisse

Die echte v31-Updatertransaktion entfernte in einer isolierten Fixture `SplitNetworks/`, `Results/` und `EnergyMeasurements/` aus dem Toolordner, obwohl diese durch `workdir.ensure_workdir()` erzeugten Arbeitsverzeichnisse sind und `energy.config.configured_workdir_root()` den Toolordner als Standardrückfall erlaubt. Damit konnten dort vorhandene FS-/Idle-Kalibrierungen, Ergebnisdateien und Splitartefakte beim Sourceupdate nur noch im Updatebackup liegen. Das war eine reale Erhaltungslücke, unabhängig von der separat beobachteten Dateisystembesonderheit der Buildumgebung.

v32 erhält diese drei tatsächlichen Arbeitsverzeichnisse nun in allen drei rsync-Pfaden und in den vorhandenen Installed-Scope-Prüfungen. Release-/Archivprüfungen bleiben strikt; ein gleichnamiger Symlink oder eine reguläre Datei wird weiterhin nicht als erlaubtes Betriebsverzeichnis behandelt. Es entstehen keine neuen Speicherorte oder Artefaktregistrierungen.

Die positive Regression verwendet den tatsächlichen Updater einschließlich rsync, ZIP-/Manifestprüfung und Metadatenrefresh; nur die kleinen importierbaren Paketbodies sind synthetisch und führen keine Hardware aus. Sie prüft die drei zuvor verlorenen Dateien sowie eigene Profile, bestehende `BenchmarkSets/`, `EvaluationRuns/`, Artefakt-/Negativevidence, die Venv und externe konfigurierte Compiler-/Cachedateien. Gegen unverändertes v31 ist der Updater selbst erfolgreich, aber genau die drei neuen Erhaltungsassertionen schlagen fehl. Die tatsächlichen roten/grünen Ausgaben werden im externen Lieferbericht dokumentiert.

Gespeicherte Compilerkonfiguration liegt regulär unter `~/.onnx_splitpoint_tool/build_environments.yaml`, Default-Compilercaches unter `~/Models/BackendArtifacts/deepx` beziehungsweise `~/Models/BackendArtifacts/hailo`. Diese externen vorhandenen Orte bleiben außerhalb der Sourcesynchronisierung; frei erfundene Tool-Unterordner werden nicht pauschal als persistenter State zugelassen.
