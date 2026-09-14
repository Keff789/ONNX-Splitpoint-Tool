# ONNX-Splitpoint-Tool 2.79.23 – Build- und Testbericht

Build-ID: `v2.79.23-native-reuse-measurement-fixes`

Basis: 2.79.22. Wartungsumfang: TensorRT-Reuse, Native-Laufzeit und
Fehlerweitergabe, Detection-Completion, Full-System-Kalibrierung,
DeepX-Preflight und eindeutige Auswahl-/Cache-Diagnosen. Die Version ergänzt
keine unabhängige Artefaktidentität, keinen neuen Build-Vertrag und keine
zusätzliche Pipeline. YOLOv7-Modell, Splitauswahl und Export werden nicht
gesondert verändert; das verwendete ONNX war laut Nutzer falsch gewählt.

Die Hailo-Negativ-Evidenz aus 2.79.22 bleibt aktiv:
`COMPILE_INFEASIBLE` ist nur mit exakt passender Identität wiederverwendbar;
`TRANSIENT_INFRASTRUCTURE` bleibt wiederholbar.

## Umgesetzte Korrekturen

| Bereich | Änderung im vorhandenen Ablauf |
|---|---|
| TRT-Wiederverwendung | Ein abweichender alter Namespace-Key erreicht bei passender gespeicherter Builder-ABI die bestehenden Einzelprüfungen. Receipt, ONNX-/Engine-/Compilerbytes, Buildkommando, Precision, Workspace und vollständiger Shape-Vertrag bleiben erforderlich. Probe und Migration wählen Kandidaten deterministisch. |
| Erhalt alter TRT-Generationen | Migrationsquellen werden im selben Retention-Schritt nicht entfernt. Umzuschreibende Receipt-/Bridge-Metadaten werden separat kopiert. Partielle Ziele bleiben unberührt. |
| H8 Remote-Start | Die gemeinsame Modulliste enthält nun `native_three_stage.py`; beide Deploymentwege verwenden sie. |
| H10 MobileNet b135 | Eine echte Dimension der Form `[1,1,960]` bleibt erhalten. Nur eine anhand des Vertrags belegte zusätzliche Batchdimension wird entfernt; Formfehler melden Ist und Soll. |
| Detection-Abschluss | `decoded_pre_nms` verwendet die vorhandene NMS-/Rückprojektionsverarbeitung. Three-Stage, TRT Full, Energiepfad und Ergebnisvalidierung behandeln denselben belegten Endpunkt. NMS läuft innerhalb der gemessenen Full-Schleife. |
| Fehleranzeige | Exitcode, Timeout und ursprünglicher Child-Fehler haben Vorrang vor fehlenden Erfolgsfeldern. Alte Ergebnisdateien bleiben erhalten, gelten aber nicht als neuer Erfolg. |
| DeepX | Der tatsächliche Compiler-Python prüft begrenzt GPU-/Torch-Architekturmetadaten. Konkrete Inkompatibilität blockiert einen Cold Build früh. Cache-Lookup erfolgt vor der Bereitschaftssperre; gültige DXNNs bleiben nutzbar. |
| DeepX TRT-Part2 | Ungebundene Läufe ohne erlaubten Build prüfen die exakt verlangte Enginevariante vor dem Child-Start. Fehlende, leere oder ungeeignete Pfade werden ausdrücklich gemeldet; keine Ersatz-Precision. Der Variant-Updater reicht das Boundary-Layout weiter. |
| FS-Kalibrierung | Konfigurierte, passende, verifizierte und angewandte FS-Skalierung wird vom bestehenden Freigabegate anerkannt. Ein bloßer Identitätsfaktor ersetzt keine Kalibrierung. |
| Messdauer | Konfigurierte Dauer und aufgezeichnete Workload-Command-Dauer werden getrennt ausgewiesen. Der Profilwert wird nicht geändert. |
| Auswahl und Preflight | Abstand gleich `min_gap` ist auch im Generator zulässig. Bereits gebaute Artefakte bleiben im historischen MISS-/Policy-Nachweis, zählen aber nicht erneut als ausstehende Cold Builds. |
| Update-Erhalt | Auch tool-lokale Verzeichnisse `artifact_store` und `build_evidence` bleiben beim Update erhalten. Die bestehenden Schutzlisten werden erweitert; Quellcode und unzulässige Symlink-Wurzeln bleiben streng geprüft. |

Die atomare HEF-/Receipt-/Cache-Meta-Publikation und allgemeine Hailo-Negativ-
Evidenz aus 21/22 bleiben enthalten und werden erneut geprüft. Neue
Cachehierarchien, Identitätsverfahren, Dependencies oder Parallelitätsebenen
wurden nicht eingeführt.

## Softwareprüfung

Die gemeinsame Prüfung `scripts/run_v27923_small_acceptance.sh` ergab
**687 PASS, 1 SKIP**. Der übersprungene optionale Test benötigt ONNX Runtime,
das in dieser Offline-Umgebung nicht installiert ist. Release-Smoke,
Quellmanifest, Python-Kompilierung und Shell-Syntax bestehen ebenfalls.

Die Tests enthalten die nachgestellten Fehlerfälle des Nachtlaufs und die
bestehenden Regressionen für Cache-Publikation, Migration, Negativ-Evidenz,
Kalibrierbindung, Energie-Provenienz, Detection-Completion und Native-Start.
Ein tatsächlicher Full-Hotloop wird mit simuliertem GPU-Zugriff durchlaufen.
Das ersetzt keine Messung auf dem Gerät.

Ein unabhängiges Review der TRT-Migration und Detection-Integration fand
keinen neuen blockierenden Fehler. Die prüfungsrelevanten Dateien
`split_export_graph.py`, `model_zoo_manifest.json` und die gelieferten Profile
sind gegenüber 22 bytegleich. Qualitätsgrenzen bleiben bestehen.

```text
OFFLINE_ACCEPTANCE=PASS
PYTEST=687_PASS_1_OPTIONAL_ORT_SKIP
RELEASE_SMOKE=PASS
SOURCE_MANIFEST=PASS
PYTHON_COMPILE=PASS
SHELL_SYNTAX=PASS
HARDWARE_EXECUTION=NOT_RUN
REAL_MULTI_HOST_EVALRUN=NOT_RUN
```

Der Test des konkreten Source-ZIPs beim isolierten Upgrade von 22 auf 23
wird nach dem Paketbau ausgeführt. Sein Ergebnis und die vollständigen Logs
stehen im Lieferpaket in `VERIFICATION_V27923.json` und `verification/`;
damit beziehen sie sich auf die tatsächlich ausgelieferte Archivprüfsumme.

## Verbleibende Grenzen

- Alte TRT-Sonder-Bridges mit nativer Quality-Bindung können bei geändertem
  Namespace weiter `UNKNOWN` bleiben, wenn die vorhandene Migration ihre
  Bindung nicht sicher übertragen kann. Es erfolgt keine ungeprüfte Freigabe.
- Der Erhalt alter Migrationsquellen kann bei ausgeschöpftem Cache-Limit die
  Aufnahme blockieren. Der Grund wird ausdrücklich ausgegeben.
- Eine ungeeignete DeepX-Compilerumgebung wird erkannt, aber nicht automatisch
  umgebaut. Fehlende Prüfevidenz bleibt `unknown`; nur belegte
  Inkompatibilität wird gesperrt.
- Die gemessenen Accuracyverluste von YOLO11l, MobileNet und weiteren Modellen
  sind durch diese Softwaretests nicht behoben. Fehlende Quality-Bindungen
  werden nicht durch unpassende Ergebnisse ersetzt.
- Ein Screening-Run wird nicht nachträglich zur finalen Vergleichskampagne.
  Die zwei berechtigten Energie-Tracefehler und alle Qualitäts-/Vollständigkeits-
  anforderungen bleiben wirksam. Es wurden keine alten Messergebnisse neu
  freigegeben und keine verlorenen Quellen rekonstruiert.
