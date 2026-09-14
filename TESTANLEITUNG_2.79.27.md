# Testanleitung 2.79.27

Build: `v2.79.27-deepx-output-probe-result-fixes`.

## Reihenfolge

1. GUI schließen und `bash ./install_v27927_and_collect_acceptance.sh` auf Smartmirror2 als `kmika` ausführen. Das Installationsskript ruft die hardwareunabhängige 27er-Abnahme auf.
2. Zuerst die kurze enthaltene DeepX-Full-Ausgabediagnose mit dem vorhandenen 26er-Run verwenden. Sie erfasst Form, Datentyp, Wertebereiche und auffällige Elemente, ohne einen Compiler-Build oder eine neue Messkampagne zu starten. Die genaue Bedienung steht in der Probe-Anleitung des Lieferpakets.
3. Das ausgegebene Diagnose-ZIP auswerten. DeepX Full bleibt bis zur realen Bestätigung **NOT_RUN** für diese Version. Weder kleine Quantisierungsabweichungen noch eine bestimmte Layout-Ursache gelten vorab als bestätigt.
4. Bei einer anschließenden vollständigen Abnahme `bash ./start_deepx_gpu_canary.sh` starten und das Profil `acceptance_profiles/v27927_acceptance_D_YOLO11l_b003_DeepX_GPU.yaml` laden.

## Kriterien für die spätere vollständige Abnahme D

- DeepX Full liefert eine echte Quality-Auswertung und drei gültige Performance- sowie drei Full-System-Energiewiederholungen.
- TRT Full und DeepX→TRT werden ihren tatsächlich gemessenen Endpunkten und eindeutigen Zeilen zugeordnet.
- Ein technischer Full-Fehler bleibt technisch fehlgeschlagen; die primäre Ursache wird angezeigt. Reale Quality-Ergebnisse bleiben von den technischen Statuswerten getrennt.
- Das Debug-Pack enthält die relevanten Ergebnisdateien, Runnerlogs, vorbereiteten Eingabeverträge und Ausgabediagnosen sowie die regulären Energie-Rohdateien.

Bestehende passende HEF/DXNN/Engine-Dateien und `COMPILE_INFEASIBLE`-Evidenz dürfen nicht unnötig neu gebaut werden. Fehlende Artefakte dürfen die vollständigen Profile nach ihrer vorhandenen Policy bauen. `TRANSIENT_INFRASTRUCTURE` bleibt wiederholbar. Der kurze Probe-Lauf kompiliert nichts.

Quality-Grenzen, Full-System-Kalibrierung und Messparallelität bleiben unverändert. Ein Probe-PASS ersetzt weder Full-Quality noch Performance-/Energieabnahme.
