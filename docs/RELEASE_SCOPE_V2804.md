# Releaseumfang v2.80.4

Build: `v2.80.4-hailo8-context-quality-cancel-complete-debug`.

Basis ist das vollständige manifestgeprüfte v2.80.3-FIX5-Source mit allen bisherigen CPU-Referenz-, Remote-Paket-, Deferred-Build-, Native-Nichtstart- und Cachekorrekturen.

Die neue Fassung ergänzt den gespeicherten Hailo8-Dependency-Manifestpfad bis zum tatsächlichen Compilerkind, erhöht die regulären Debugbudgets und unterscheidet vollständig ausgewertete Quality-Requests von Benutzerabbruch und technischen Fehlern. Die normale Debugausgabe soll die vorhandenen gebundenen, dekodierten Vorhersagedaten enthalten, ohne einen separaten Nachsammler zu verlangen.

Hailo balanced/Opt1/B500/Batch8, DeepX EMA/Opt0 und die jeweils gebundene Vorverarbeitung bleiben erhalten. Force bleibt AUS. Ein GPU-Wunsch invalidiert keinen passenden CPU-HEF. Private GPU-Diagnoseartefakte werden nicht automatisch als besseres Modell publiziert. COMPILE_INFEASIBLE bleibt genaue negative Modellevidenz; TRANSIENT_INFRASTRUCTURE bleibt retrybar.

Technische Softwareprüfungen laufen ohne Hardware. Echte Geräteprüfungen des neuen Standes sind bis zur Ausführung des ausgelieferten Normalworkflowstarters `NOT_RUN`. Historische GPU-Compute-, Modellbuild- und Fixed16-Nachweise werden nicht als neue .4-Läufe ausgegeben. Das bestehende negative Hailo-Qualitätsergebnis bleibt FAIL; INCONCLUSIVE bleibt eine eigenständige Entscheidung.

Der vollständige Prüfbericht und die originale Quellenverfügbarkeit werden im Delivery-Bundle dokumentiert. Nicht abrufbare Originalarchive werden nicht durch erfundene Originalfixtures ersetzt.
