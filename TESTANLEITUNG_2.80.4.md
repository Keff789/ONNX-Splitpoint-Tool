# Abnahme v2.80.4

Build: `v2.80.4-hailo8-context-quality-cancel-complete-debug`.

GUI und laufende Workflows vor dem Update beenden. `install_v2804_and_collect_acceptance.sh` verwendet den manifestgebundenen offiziellen Source-Updater und die bestehende Tool-Venv. Er prüft die neue Identität, bewahrt eigene Profile und die Run-Mode-Registry und führt die aktuelle Softwareabnahme aus. `scripts/run_v2804_short_tests.sh` ist der gezielte Wiederholungseinstieg; direkt nach bestandener vollständiger Installationsabnahme ist keine zusätzliche Wiederholung derselben Kurztests nötig.

Der neue `reference_workflow_gate_v2804.py` bereitet eine getrennte Standard-B500-Abnahme für MobileNetV3/b056 und YOLO11l/b062 vor und führt den normalen Workflow einschließlich CPU-Referenz, Consumer, zentraler Quality, Native und Debugabschluss aus. Das Quellprofil und bisherige Ergebnisse bleiben erhalten. Ein unerwarteter Cold Build wird vor der langen Kompilierung als eigener Befund ausgewiesen. Force bleibt AUS, Energie wird für diese begrenzte Runde ausgeschaltet.

Die Hailo8-Overlayauswahl erfolgt im Run-Mode-Editor neben dem GPU-Gerät. Ein vorhandenes Manifest kann explizit ausgewählt und rein lesend geprüft werden. Ein maschinenspezifischer Pfad ist kein Paketdefault. Für einen gültigen Cache-HIT sind weder SDK-Import noch GPU-/Overlay-Vorabtest erforderlich. Ein echter H8-MISS verwendet erst dann den gespeicherten Kontext. Ein vorhandenes Artefakt wird nicht gelöscht oder neu gebaut, um einen Kaltbuildtest zu erzwingen.

**Statusachsen:** vollständige technische Ausführung, Quality PASS/FAIL/INCONCLUSIVE, Benutzerabbruch, Energie und wissenschaftliche Freigabe sind getrennt. Ein vollständig ausgewertetes negatives Hailo-Ergebnis ist kein fehlender Test. COMPILE_INFEASIBLE und TRANSIENT_INFRASTRUCTURE werden getrennt behandelt. Neue Hardwaregates sind vor Ausführung `NOT_RUN`.

Die genauen gebündelten Startbefehle, endgültigen Testzahlen, Quellenverfügbarkeit und Erhaltungsnachweise stehen im beiliegenden Delivery-Bericht. Ein zusätzlicher Final-5000-/Optimierungs-Sweep ist nicht Bestandteil dieses Releasefixes.
